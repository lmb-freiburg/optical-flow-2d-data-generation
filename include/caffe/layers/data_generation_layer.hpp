#ifndef CAFFE_DATA_GENERATION_LAYER_HPP_
#define CAFFE_DATA_GENERATION_LAYER_HPP_

/**
 * Nikolaus Mayer, 2017 (mayern@cs.uni-freiburg.de)
 */

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/data_generation/DataGenerator.h"

namespace caffe {

/**
 * @brief A BATCH of data consists of two BLOBs
 */
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};


/**
 * @brief LMB data-generating layer. Does not conform to the "usual" DataLayer
 *        inheritance hierarchy: It performs as a DataLayer, but includes
 *        all needed functionality that the DataLayer gets from the
 *        BasePrefetchingDataLayer and BaseDataLayer.
 */
template <typename Dtype>
class DataGenerationLayer : public Layer<Dtype>, public InternalThread 
{
public:
  /// Constructor
  explicit DataGenerationLayer(
        const LayerParameter& param);
  /// Destructor
  virtual ~DataGenerationLayer();
  /// Initial setup
  virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {}
  
  virtual inline bool ShareInParallel()    const { return false; }
  virtual inline const char* type()        const { return "DataGeneration"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs()         const { return 1; }

  virtual void Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  
  /// Trivial backward passes for data layers
  virtual void Backward_cpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) {}
  
  
protected:
  /// Entry point for internal prefetching thread
  virtual void InternalThreadEntry();
  /// Fetch data samples from reader_
  virtual void load_batch(vector<Blob<Dtype>*>* output_ptr);
  
  vector<vector<Blob<Dtype>*> > prefetch_;
  BlockingQueue<vector<Blob<Dtype>*>*> prefetch_free_;
  BlockingQueue<vector<Blob<Dtype>*>*> prefetch_full_;
  
  DataGenerator::DataGenerator data_generator_;
  DataGenerator::ObjectParametersGenerator obj_params_generator_;
};


}  // namespace caffe

#endif  // CAFFE_DATA_GENERATION_LAYER_HPP_

