/**
 * Nikolaus Mayer, 2017 (mayern@cs.uni-freiburg.de)
 */

// Copyright 2014 BVLC and contributors.

/// System/STL
#include <stdint.h>
#include <vector>

/// Boost
#include <boost/thread.hpp>
#include <boost/function.hpp>

/// Caffe and local files
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/layers/data_generation_layer.hpp"

#include "caffe/data_generation/DataGenerator.h"

namespace caffe {

#define Container vector<Blob<Dtype>*>

/**
 * @brief Constructor
 * 
 * @param param Parameters
 */
template <typename Dtype>
DataGenerationLayer<Dtype>::DataGenerationLayer(const LayerParameter& param)
  : Layer<Dtype>(param),
    prefetch_(),
    prefetch_free_(),
    prefetch_full_(),
    data_generator_(param),
    obj_params_generator_(param)
{
  prefetch_.resize(param.data_param().prefetch());

  CHECK_GE(param.data_param().sample().size(), 0) << "No samples defined.";  

  /// Populate prefetching queue with empty buckets
  for (int i = 0; i < prefetch_.size(); ++i) {
    for (int j = 0; j < param.top_size(); ++j) {
      Blob<Dtype> *tmpblob = new Blob<Dtype>();
      prefetch_[i].push_back(tmpblob);
    }
    prefetch_free_.push(&prefetch_[i]);
  }
}


/**
 * @brief Destructor
 */
template <typename Dtype>
DataGenerationLayer<Dtype>::~DataGenerationLayer<Dtype>() {
  this->StopInternalThread();
  data_generator_.Stop(false);

  /// Tidy up
  /// Depublish data buckets and ensure that all are accounted for
  {
    unsigned int total = prefetch_.size();
    while (prefetch_free_.size() > 0) {
      prefetch_free_.pop();
      --total;
    }
    while (prefetch_full_.size() > 0) {
      prefetch_full_.pop();
      --total;
    }
    if (total > 0)
      LOG(INFO) << "Warning: There are " << prefetch_.size() << " allocated"
                << " prefetching buckets, but " << total << " of these could"
                << " not be accounted for during cleanup. This is not fatal,"
                << " but might indicate a memory leak.";
    else if (total < 0)
      LOG(INFO) << "Warning: There are " << prefetch_.size() << " allocated"
                << " prefetching buckets, but " << -total+prefetch_.size() 
                << " were found in the prefetching queues during cleanup."
                << " This is not fatal, but probably bad.";
  }
  
  /// Free raw Blob memory within the buckets
  for (unsigned int i = 0; i < prefetch_.size(); ++i) {
    Container& container = prefetch_[i];
    for (unsigned int j = 0; j < container.size(); ++j) {
      if (container[j])
        delete container[j];
    }
  }
}


/**
 * @brief Initial layer setup
 */
template <typename Dtype>
void DataGenerationLayer<Dtype>::LayerSetUp(const Container& bottom,
                                            const Container& top)
{
  const bool verbose = this->layer_param_.data_param().verbose();
  if (verbose && this->layer_param_.data_param().block_size())
    LOG(INFO) << "Block size: " << this->layer_param_.data_param().block_size();

  const int batch_size = this->layer_param_.data_param().batch_size();

  DLOG(INFO) << "Initializing prefetch";
  data_generator_.setBatchSize(batch_size);
  data_generator_.Start();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";

  /// Look at a data sample and use it to initialize the top blobs
  /// NOTE: This BLOCKS until a data sample is available
  //Container* peek_data = prefetch_full_.peek();
  //if(!output_index_)
  {
    // output only blobs
    assert(top.size() == peek_data->.size());
    top[0]->Reshape({batch_size, 3, DGEN_HEIGHT, DGEN_WIDTH});
    top[1]->Reshape({batch_size, 3, DGEN_HEIGHT, DGEN_WIDTH});
    top[2]->Reshape({batch_size, 2, DGEN_HEIGHT, DGEN_WIDTH});
  }
}

/**
 * Entry point for InternalThread (not called by user)
 * 
 * Fetches an empty bucket from the prefetch_free_ queue, fills it, and
 * pushes it into the prefetch_full_ queue.
 */
template <typename Dtype>
void DataGenerationLayer<Dtype>::InternalThreadEntry() 
{
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Container* container_ptr = prefetch_free_.pop();
      /// Fetch data from reader; stall while no data is available
      load_batch(container_ptr);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        for (unsigned int i = 0; i < container_ptr->size(); ++i)
          (*container_ptr)[i]->data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(container_ptr);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}


/**
 * Fetch a data batch from the internal reader (not called by user)
 * 
 * @param output_ptr Prefetch target into which data is copied. The prefetch
 *                   target is a batch; each Blob in the vector stores multiple
 *                   samples
 */
template <typename Dtype>
void DataGenerationLayer<Dtype>::load_batch(Container* output_ptr)
{
  const int batch_size = this->layer_param_.data_param().batch_size();

  ///
  /// Enqueue new tasks
  ///
  unsigned int total_tasks = data_generator_.m_undone_tasks.size() +
                             data_generator_.m_task_IDs_in_flight.size() +
                             data_generator_.m_finished_tasks.size();
  if (total_tasks < prefetch_.size() * batch_size) {
    for (int i = 0; i < (prefetch_.size()*batch_size-total_tasks); ++i) {
      //LOG(INFO) << "Pushing a new task into DataGenerator";

      DataGenerator::TaskBucket* new_task_ptr = new DataGenerator::TaskBucket();
      /// Background
      {
        DataGenerator::ObjectBlueprint* b = new DataGenerator::ObjectBlueprint();
        b->obj_id = 1;
        obj_params_generator_.generateBackground(b);
        new_task_ptr->background_blueprint = b;
      }
      /// Moving objects
      const int fg_objs = obj_params_generator_.generateNumberOfFgObjects();
      new_task_ptr->object_blueprints.resize(fg_objs);
      for (unsigned int obj_idx = 0; obj_idx < fg_objs; ++obj_idx) {
        DataGenerator::ObjectBlueprint* b = new DataGenerator::ObjectBlueprint();
        b->obj_id = obj_idx+10;
        obj_params_generator_.generateForegroundObject(b);
        new_task_ptr->object_blueprints[obj_idx] = b;
      }
      data_generator_.commissionNewTask(new_task_ptr);
    }
  }

  ///
  /// Retrieve finished tasks and assemble a data batch
  ///

  const Container& output = (*output_ptr);

  /// hardcoded output shape
  int W = DGEN_WIDTH;
  int H = DGEN_HEIGHT;
  output[0]->Reshape({batch_size, 3, H, W});
  output[1]->Reshape({batch_size, 3, H, W});
  output[2]->Reshape({batch_size, 2, H, W});

  /// Reshape output to match source data
  
  for (unsigned int i = 0; i < output.size(); ++i)
    CHECK(output[i]->count());

  /// Fill output
  for (unsigned int i = 0; i < batch_size; ++i)
  {
    /// Fetch one data sample from internal reader
    DataGenerator::Sample one_sample{data_generator_.retrieveFinishedTask()};

    caffe_copy(W*H*3,
               one_sample.image0_ptr->data(),
               output[0]->mutable_cpu_data()+output[0]->offset(i,0,0,0));
    caffe_copy(W*H*3,
               one_sample.image1_ptr->data(),
               output[1]->mutable_cpu_data()+output[1]->offset(i,0,0,0));
    caffe_copy(W*H*2,
               one_sample.flow0_ptr->data(),
               output[2]->mutable_cpu_data()+output[2]->offset(i,0,0,0));

    /// Recycle spent data container for data reading
    one_sample.Destroy();
  }
}



/**
 * @brief Forward_cpu
 * 
 * Fetches a full bucket from the prefetch_full_ queue, uses its contents,
 * and pushes it back into the prefetch_free_ queue.
 */
template <typename Dtype>
void DataGenerationLayer<Dtype>::Forward_cpu(const Container& bottom,
                                         const Container& top) 
{  
  Container* container_ptr = prefetch_full_.pop("Data layer prefetch queue empty");
  const Container& container = (*container_ptr);

  /// Reshape tops and copy data
  for (unsigned int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*container[i]);
    caffe_copy(container[i]->count(), 
               container[i]->cpu_data(),  
               top[i]->mutable_cpu_data());
  }

  /// Recycle spent data container for prefetching
  prefetch_free_.push(container_ptr);
}


template <typename Dtype>
void DataGenerationLayer<Dtype>::Forward_gpu(const Container& bottom,
                                         const Container& top) 
{
  /// Nah fam, we doin this on CPU
  Forward_cpu(bottom, top);
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(DataGenerationLayer, Forward);
#endif

INSTANTIATE_CLASS(DataGenerationLayer);
REGISTER_LAYER_CLASS(DataGeneration);

}  // namespace caffe

