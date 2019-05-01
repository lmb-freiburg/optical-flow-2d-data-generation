
#ifndef QUEUEPROCESSOR_H__
#define QUEUEPROCESSOR_H__

/// System/STL
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

namespace QueueProcessing 
{

  template <class T>
  class QueueProcessor
  {
    typedef std::shared_ptr<T> item_t;

    public:

      QueueProcessor(
            std::function<int(item_t)> func,
            bool start_immediately,
            bool use_threading,
            int number_of_worker_threads,
            bool print_information);

      ~QueueProcessor();

      void Start();

      void Stop(bool wait=true);

      void Finish();

      QueueProcessor& Give(
            item_t item_ptr);
      QueueProcessor& Give(
            T raw_item);

      QueueProcessor& SetMaxQueueLength(
            size_t max_queue_length);


    private:

      void _WatchQueue();

      int _ProcessItem(
            item_t item_ptr);

      void _PushItem(
            item_t item_ptr);

      bool m_running;
      bool m_use_threading;
      bool m_print_information;
      std::function<int(item_t)> m_item_processor_handle;

      size_t m_running_task_ID;
      std::queue<std::pair<size_t, item_t>> m_queue;
      std::vector<size_t> m_task_IDs_in_flight;
      size_t m_max_queue_length;

      std::vector<std::unique_ptr<std::thread>> m_worker_threads;
      std::mutex m_queue_lock;
  };



  /**
   * Implementation
   */

  /// Not pretty but readable
  #define QPT QueueProcessor<T>

  template <class T>
  QPT::QueueProcessor(std::function<int(item_t)> func,
                      bool start_immediately,
                      bool use_threading,
                      int number_of_worker_threads,
                      bool print_information)
    : m_running(start_immediately),
      m_use_threading(use_threading),
      m_print_information(print_information),
      m_item_processor_handle(func),
      m_running_task_ID(0),
      m_max_queue_length(0)
  {
    if (m_use_threading) {
      m_worker_threads.resize(number_of_worker_threads);
      if (start_immediately) {
        for (auto& worker: m_worker_threads) {
          worker.reset(new std::thread(&QPT::_WatchQueue, this));
        }
      }
    }
  }

  
  template <class T>
  QPT::~QueueProcessor()
  {
    Stop(false);
  }
  

  /**
   * Finish all remaining tasks (BLOCKING)
   */
  template <class T>
  void QPT::Finish()
  {
    while (m_queue.size() > 0 or m_task_IDs_in_flight.size() > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }


  template <class T>
  void QPT::Start()
  {
    if (m_running)
      return;

    m_running = true;
    if (m_use_threading) {
      for (auto& worker: m_worker_threads) {
        worker.reset(new std::thread(&QPT::_WatchQueue, this));
      }
    }
  }


  template <class T>
  void QPT::Stop(bool wait)
  {
    if (not m_running)
      return;

    m_running = false;
    /// Deplete queue (will NOT wait until items are processed)
    if (not wait) {
      m_queue_lock.lock();
      while (m_queue.size() > 0) {
        m_queue.pop();
      }
      m_queue_lock.unlock();
    }
    /// Stop worker threads
    if (m_use_threading) {
      for (auto& worker: m_worker_threads) {
        if (worker->joinable()) {
          worker->join();
        }
      }
    }
  }
  

  template <class T>
  QPT& QPT::Give(item_t item_ptr)
  {
    std::lock_guard<std::mutex> lock(m_queue_lock);
    m_queue.push(std::make_pair<size_t, item_t>(m_running_task_ID++, item_ptr));

    return *this;
  }

  template <class T>
  QPT& QPT::Give(T raw_item)
  {
    if (m_use_threading and m_max_queue_length > 0) {
      while (m_queue.size() >= m_max_queue_length)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::lock_guard<std::mutex> lock(m_queue_lock);
    m_queue.push(std::make_pair<size_t, item_t>(m_running_task_ID++,
                                                std::make_shared<T>(raw_item)));

    return *this;
  }


  template <class T>
  QPT& QPT::SetMaxQueueLength(size_t max_queue_length)
  {
    m_max_queue_length = max_queue_length;

    return *this;
  }


  
  template <class T>
  void QPT::_WatchQueue()
  {
    while (m_running or m_queue.size() > 0) {
      if (m_queue.size() > 0) {
        /// Read queue
        std::pair<size_t, item_t> item_pair;
        
        m_queue_lock.lock();
        if (m_queue.size() == 0) {
          m_queue_lock.unlock();
          continue;
        }
        
        try {
          item_pair = m_queue.front();
        } catch(...) {
          m_queue_lock.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        
        /// Pop from queue
        try {
          m_queue.pop();
        } catch(...) {
          m_queue_lock.unlock();
          continue;
        }
        m_task_IDs_in_flight.push_back(item_pair.first);
        m_queue_lock.unlock();
        
        /// Process item
        _ProcessItem(item_pair.second);

        m_queue_lock.lock();
        /// (Erase-remove idiom: http://stackoverflow.com/a/3385251)
        m_task_IDs_in_flight.erase(std::remove(m_task_IDs_in_flight.begin(),
                                               m_task_IDs_in_flight.end(),
                                               item_pair.first),
                                   m_task_IDs_in_flight.end());
        m_queue_lock.unlock();
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }


  template <class T>
  int QPT::_ProcessItem(item_t item)
  {
    return m_item_processor_handle(item);
  }



}  // namespace QueueProcessing


#endif  // QUEUEPROCESSOR_H__

