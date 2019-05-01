/**
 * Nikolaus Mayer, 2017 (mayern@cs.uni-freiburg.de)
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>


#include "caffe/data_generation/WarpFields.h"

#include "thirdparty/CImg/CImg.h"
using namespace cimg_library;

namespace WarpFields {


  /**
   * "Supports" are weighting / influence masks for displacers. A displacer
   * without a mask does not know which pixels it should affect to which degree.
   */
  namespace Supports {

    /**
     * Abstract base class
     */
    SupportBase::SupportBase(float cx, float cy)
      : cx(cx), cy(cy)
    { }

    SupportBase::~SupportBase() 
    { }

    float SupportBase::at_relative(float x, float y) const
    {
      return at(x-cx, y-cy);
    }


    /**
     * Constant support (flat)
     */
    Constant::Constant(float factor)
      : SupportBase(0,0), factor(factor)
    { }

    float Constant::at(float x, float y) const 
    {
      (void)x; (void)y;
      return factor;
    }
      

    /**
     * Isotropic gaussian support (soft gradients)
     */
    Gaussian1D::Gaussian1D(float cx, float cy, float sigma)
      : SupportBase(cx, cy),
        sigma_sq(sigma*sigma), 
        gauss_prefactor(1/std::sqrt(2*M_PI*sigma_sq)),
        normalizer(1/raw_at(cx,cy))
    { }

    float Gaussian1D::raw_at(float x, float y) const
    {
      const float dist_sq{(x-cx)*(x-cx)+(y-cy)*(y-cy)};
      return gauss_prefactor * std::exp(-dist_sq/(2*sigma_sq));
    }

    float Gaussian1D::at(float x, float y) const
    {
      return normalizer * raw_at(x,y);
    }


    /**
     * 2D gaussian support
     */
    Gaussian2D::Gaussian2D(float cx, float cy, 
                 float sigma_x, float sigma_y, float angle)
        : SupportBase(cx,cy),
          a( std::cos(angle)),
          b(-std::sin(angle)),
          c( std::sin(angle)),
          d( std::cos(angle)),
          ratio_x_y(sigma_x/sigma_y),
          sigma_sq(sigma_x*sigma_x), 
          gauss_prefactor(1/std::sqrt(2*M_PI*sigma_sq)),
          normalizer(1/raw_at(cx,cy))
      { }

      float Gaussian2D::raw_at(float x, float y) const
      {
        const float rx =  a*(x-cx) + b*(y-cy);
        const float ry = (c*(x-cx) + d*(y-cy))*ratio_x_y;
        const float dist_sq{rx*rx+ry*ry};
        return gauss_prefactor * std::exp(-dist_sq/(2*sigma_sq));
      }

      float Gaussian2D::at(float x, float y) const
      {
        return normalizer * raw_at(x,y);
      }


  }  // namespace Supports


  /**
   * "Displacers" are functionally described flow field building blocks. A single
   * displacers yields a valid, but boring flow field. Each displacer must be
   * used with a separate "Support" mask. Displacers know exact formulations for
   * their forward and backward flows.
   */
  namespace Displacers {

    /**
     * Abstract base class
     */
    DisplacerBase::DisplacerBase(float cx, float cy)
      : cx(cx), cy(cy), support_ptr(nullptr)
    { }

    DisplacerBase::~DisplacerBase()
    {
      if (support_ptr)
        delete support_ptr;
    }

    DisplacerBase& DisplacerBase::set_support(Supports::SupportBase* new_support_ptr)
    {
      if (support_ptr)
        delete support_ptr;

      support_ptr = new_support_ptr;
      return *this;
    }

    V2 DisplacerBase::flow_at(float x, float y) const
    {
      const V2 raw{raw_flow_at(x, y)};
      const float w{support_ptr->at(x, y)};
      return make_V2(V2_x(raw)*w, V2_y(raw)*w);
    }
    
    V2 DisplacerBase::iflow_at(float x, float y) const
    {
      const V2 raw{raw_iflow_at(x, y)};
      const float w{support_ptr->at(x, y)};
      return make_V2(V2_x(raw)*w, V2_y(raw)*w);
    }

    float DisplacerBase::support_at(float x, float y) const
    {
      return support_ptr->at(x, y);
    }


    /**
     * Zero-flow (static image)
     */
    Identity::Identity()
      : DisplacerBase(0,0)
    { }

    V2 Identity::raw_flow_at(float x, float y) const
    {
      (void)x; (void)y;
      return make_V2(0.f, 0.f);
    }

    V2 Identity::raw_iflow_at(float x, float y) const
    {
      (void)x; (void)y;
      return make_V2(0.f, 0.f);
    }


    /**
     * Translation flow (shift)
     */
    Translation::Translation(float dx, float dy)
      : DisplacerBase(0,0), dx(dx), dy(dy)
    { }

    V2 Translation::raw_flow_at(float x, float y) const
    {
      (void)x; (void)y;
      return make_V2(dx, dy);
    }

    V2 Translation::raw_iflow_at(float x, float y) const
    {
      (void)x; (void)y;
      return make_V2(-dx, -dy);
    }


    /**
     * Rotation flow (twist/vortex/whirl)
     */
    Rotation::Rotation(float cx, float cy, float angular_speed)
      : DisplacerBase(cx,cy),
        omega(angular_speed),
        sin_omega(std::sin(omega)),
        cos_omega(std::cos(omega)),
        sin_nomega(std::sin(-omega)),
        cos_nomega(std::cos(-omega))
    { }

    V2 Rotation::raw_flow_at(float x, float y) const
    {
      const float dx{x-cx};
      const float dy{y-cy};
      const float rot_dx{cos_nomega*dx - sin_nomega*dy};
      const float rot_dy{sin_nomega*dx + cos_nomega*dy};
      return make_V2(rot_dx-dx, rot_dy-dy);
    }

    V2 Rotation::raw_iflow_at(float x, float y) const
    {
      const float dx{x-cx};
      const float dy{y-cy};
      const float rot_dx{cos_omega*dx - sin_omega*dy};
      const float rot_dy{sin_omega*dx + cos_omega*dy};
      return make_V2((rot_dx-dx), (rot_dy-dy));
    }


    /**
     * Zooming flow (expansion/contraction)
     */
    Zoom::Zoom(float cx, float cy, float factor)
      : DisplacerBase(cx,cy),
        factor(factor),
        ifactor(1./factor)
    { }

    V2 Zoom::raw_flow_at(float x, float y) const
    {
      const float dx{x-cx};
      const float dy{y-cy};
      return make_V2(factor*dx-dx, factor*dy-dy);
    }

    V2 Zoom::raw_iflow_at(float x, float y) const
    {
      const float dx{x-cx};
      const float dy{y-cy};
      return make_V2(ifactor*dx-dx, ifactor*dy-dy);
    }

  }  // namespace Displacers




  /**
   * A DisplacementComposer holds a collection of Support-weighed Displacers and
   * adds together their respective flows to compute a final, discretely sampled
   * flowfield. The DisplacementComposer uses a hierarchical diffeomorphism
   * composition approach to compute flowfields with highly accurate inversions.
   */
  DisplacementComposer::DisplacementComposer(int W, int H)
    : W(W), H(H)
  { }

  DisplacementComposer::~DisplacementComposer()
  {
    for (unsigned int i = 0; i < displacer_ptrs.size(); ++i)
      if (displacer_ptrs[i])
        delete displacer_ptrs[i];
  }

  DisplacementComposer& DisplacementComposer::add_displacer(Displacers::DisplacerBase* new_d_ptr)
  {
    displacer_ptrs.push_back(new_d_ptr);
    return *this;
  }

  DisplacementComposer& DisplacementComposer::with_support(Supports::SupportBase* new_s_ptr)
  {
    displacer_ptrs.back()->set_support(new_s_ptr);
    return *this;
  }

  V2 DisplacementComposer::flow_at(float x, float y) const
  {
    V2 flow(0, 0);
    for (const auto& displacer_ptr: displacer_ptrs) {
      V2 dflow = displacer_ptr->flow_at(x, y);
      V2_x(flow) += V2_x(dflow);
      V2_y(flow) += V2_y(dflow);
    }
    return flow;
  }

  V2 DisplacementComposer::iflow_at(float x, float y) const
  {
    V2 flow(0, 0);
    for (const auto& displacer_ptr: displacer_ptrs) {
      V2 dflow = displacer_ptr->iflow_at(x, y);
      V2_x(flow) += V2_x(dflow);
      V2_y(flow) += V2_y(dflow);
    }
    return flow;
  }

  int DisplacementComposer::get_W() const 
  { 
    return W; 
  }

  int DisplacementComposer::get_H() const 
  { 
    return H; 
  }



  FlowField::FlowField()
  { }
    
  /**
   * Generate discrete forward and backward flow fields, given an instance of
   * a DisplacementComposer with flow elements.
   */
  FlowField& FlowField::init_from_DisplacementComposer(const DisplacementComposer& dc)
  {
    const int W{dc.get_W()};
    const int H{dc.get_H()};
    flow.resize(W,H,1,2);
    iflow.resize(W,H,1,2);

    ///
    /// Sample elementary forward and backward flow fields (tiny displacements)
    ///
    cimg_forXY(flow,x,y) {
      V2 pos_flow = dc.flow_at(x,y);
      flow(x,y,0) = V2_x(pos_flow);
      flow(x,y,1) = V2_y(pos_flow);
      pos_flow = dc.iflow_at(x,y);
      iflow(x,y,0) = V2_x(pos_flow);
      iflow(x,y,1) = V2_y(pos_flow);
    }

    CImg<float> tmp_flow(flow);
    CImg<unsigned char> flagged(W, H, 1, 1);
    flagged.fill((unsigned char)0);

    ///
    /// Hierarchically compose the full forward flow field; The binary-tree
    /// approach means that the elementary flow field is integrated 2^iters
    /// times. The result is an approximation (due to sampling), but the
    /// differences are minimal and this is faster by many orders of magnitude.
    ///
    for (int iter = 17; iter > 0; --iter) {
      CImg<float>& from_ref = (iter%2==1 ? tmp_flow : flow);
      CImg<float>& to_ref   = (iter%2==1 ? flow : tmp_flow);
      cimg_forXY(from_ref,x,y) {
        const float fx = from_ref(x,y,0);
        const float fy = from_ref(x,y,1);
        if (x+fx < 0 or x+fx >= W or 
            y+fy < 0 or y+fy >= H) {
          flagged(x,y) = (unsigned char)255;
          to_ref(x,y,0) = fx;
          to_ref(x,y,1) = fy;
          continue;
        }
        const float composite_fx = fx + from_ref._linear_atXY(x+fx,y+fy,0);
        const float composite_fy = fy + from_ref._linear_atXY(x+fx,y+fy,1);
        to_ref(x,y,0) = composite_fx;
        to_ref(x,y,1) = composite_fy;
      }
    }
    ///
    /// Clean up the flow field by flagging (and subsequently setting to NaN)
    /// all pixels whose flows leave the image bounds.
    ///
    cimg_forXY(flow,x,y) {
      if (x+flow(x,y,0) < 0 or x+flow(x,y,0) >= W or 
          y+flow(x,y,1) < 0 or y+flow(x,y,1) >= H) {
        flagged(x,y) = (unsigned char)255;
      }
      if (flagged(x,y)) {
        flow(x,y,0) = std::numeric_limits<float>::signaling_NaN();
        flow(x,y,1) = std::numeric_limits<float>::signaling_NaN();
      }
    }

    tmp_flow = iflow;
    flagged.fill((unsigned char)0);

    ///
    /// Now compose the backward (inverse) flow field (same as above)
    ///
    for (int iter = 17; iter > 0; --iter) {
      CImg<float>& from_ref = (iter%2==1 ? tmp_flow : iflow);
      CImg<float>& to_ref   = (iter%2==1 ? iflow : tmp_flow);
      cimg_forXY(from_ref,x,y) {
        const float fx = from_ref(x,y,0);
        const float fy = from_ref(x,y,1);
        if (x+fx < 0 or x+fx >= W or 
            y+fy < 0 or y+fy >= H) {
          flagged(x,y) = (unsigned char)255;
          to_ref(x,y,0) = fx;
          to_ref(x,y,1) = fy;
          continue;
        }
        const float composite_fx = fx + from_ref._linear_atXY(x+fx,y+fy,0);
        const float composite_fy = fy + from_ref._linear_atXY(x+fx,y+fy,1);
        to_ref(x,y,0) = composite_fx;
        to_ref(x,y,1) = composite_fy;
      }
    }
    cimg_forXY(iflow,x,y) {
      if (x+iflow(x,y,0) < 0 or x+iflow(x,y,0) >= W or 
          y+iflow(x,y,1) < 0 or y+iflow(x,y,1) >= H) {
        flagged(x,y) = (unsigned char)255;
      }
      if (flagged(x,y)) {
        iflow(x,y,0) = std::numeric_limits<float>::signaling_NaN();
        iflow(x,y,1) = std::numeric_limits<float>::signaling_NaN();
      }
    }

    return *this;
  }


  /**
   * Set to zero all tiny flows. This is not strictly necessary, but can clean
   * up a flowfield if a Displacer uses a Support which is nonzero everywhere.
   */
  FlowField& FlowField::clamp_near_zeros()
  {
    const float threshold{1e-3};
    cimg_forXY(flow,x,y) {
      if (std::abs( flow(x,y,0)) < threshold)  flow(x,y,0) = 0.f;
      if (std::abs( flow(x,y,1)) < threshold)  flow(x,y,1) = 0.f;
      if (std::abs(iflow(x,y,0)) < threshold) iflow(x,y,0) = 0.f;
      if (std::abs(iflow(x,y,1)) < threshold) iflow(x,y,1) = 0.f;
    }

    return *this;
  }

  CImg<float> FlowField::get_flow() const 
  { 
    return flow; 
  }

  CImg<float> FlowField::get_iflow() const 
  { 
    return iflow; 
  }



  CropGenerator::CropGenerator(int W, int H, int reuse_same)
    : m_W(W),
      m_H(H),
      m_reuse_same(reuse_same),
      m_reuse_counter(0),
      m_running(false)
  {
  }

  CropGenerator::~CropGenerator()
  {
    Stop();
  }

  CropGenerator& CropGenerator::Start()
  {
    if (m_running)
      return *this;

    m_running = true;
    m_worker_thread_ptrs.resize(10);
    std::cout << "CropGenerator: Spawning ("
              << m_worker_thread_ptrs.size()
              << ") worker threads" << std::endl;
    for (auto& worker: m_worker_thread_ptrs) {
      worker.reset(new std::thread(&CropGenerator::worker_thread_loop, this));
    }

    return *this;
  }

  CropGenerator& CropGenerator::Stop()
  {
    if (not m_running)
      return *this;

    m_running = false;
    /// Stop worker threads
    for (auto& worker: m_worker_thread_ptrs) {
      if (worker->joinable()) {
        worker->join();
      }
    }

    return *this;
  }

  std::pair<CImg<float>,CImg<float>> CropGenerator::get_crop()
  {
    std::pair<CImg<float>,CImg<float>> crop;

    while (m_running) {
      while (m_running and m_finalized_crops_queue.size() == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

      std::lock_guard<std::mutex> LOCK(m_queue_LOCK);
      if (m_finalized_crops_queue.size() == 0)
        continue;
      crop = m_finalized_crops_queue.front();
      ++m_reuse_counter;
      if (m_reuse_counter > m_reuse_same) {
        m_finalized_crops_queue.pop();
        m_reuse_counter = 0;
      }
      //std::cout << "getting crop (" << m_finalized_crops_queue.size()
      //          << "), reuse (" << m_reuse_counter << ")" << std::endl;
      break;
    }
    return crop;
  }

  void CropGenerator::worker_thread_loop()
  {
    /// setup DisplacementComposer and FlowField
    /// loop while m_running
    ///// sleep while m_finalized_crops_queue length is > X
    ///// extract crops until ff is "exhausted"
    ///// replace dc and ff with fresh instances
    /// cleanup

    std::random_device ran_dev;
    std::mt19937 mersenne(ran_dev());
    std::uniform_int_distribution<> displacer_type(0,2);
    std::uniform_real_distribution<> generic_param(-1,1);

    const int W{m_W};
    const int H{m_H};
    const int big_size{std::max(W,H)*3};

    while (m_running) {

      /// Wait while there are many crops in stock
      while (m_finalized_crops_queue.size() > 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      /// Create more flow fields
      {
        std::cout << "CropGenerator: only (" << m_finalized_crops_queue.size() 
                  << ") crops left, gonna make some more!" << std::endl;
        /// Prepare new flow field
        WarpFields::DisplacementComposer dc(big_size, big_size);
        //std::cout << "big_size " << big_size << std::endl;
        const int spacing{200};
        const int isosceles_spacing{(int)(spacing/2. * std::sqrt(3.))};
        const int rows{(dc.get_H()+isosceles_spacing-1)/isosceles_spacing};
        const int cols{(dc.get_W())/spacing};
        for (int yidx = 0; yidx < rows; ++yidx) {
          for (int xidx = 0; xidx < cols; ++xidx) {
            const int x = xidx*spacing + (yidx%2==1 ? spacing/2 : 0) + spacing/2;
            const int y = yidx*isosceles_spacing + spacing/2;

            WarpFields::Displacers::DisplacerBase* displacer_ptr{nullptr};
            switch (displacer_type(mersenne)) {
              case 0: displacer_ptr = new WarpFields::Displacers::Translation(
                                            generic_param(mersenne)*3e-4,
                                            generic_param(mersenne)*3e-4);
                      break;
              case 1: displacer_ptr = new WarpFields::Displacers::Rotation(
                                            x+generic_param(mersenne)*10,
                                            y+generic_param(mersenne)*10,
                                            generic_param(mersenne)*M_PI*2e-6);
                      break;
              case 2: displacer_ptr = new WarpFields::Displacers::Zoom(
                                            x+generic_param(mersenne)*10,
                                            y+generic_param(mersenne)*10,
                                            1+generic_param(mersenne)*2e-6);
                      break;
            }

            WarpFields::Supports::SupportBase* support_ptr = 
                                      new WarpFields::Supports::Gaussian2D(
                                            x+generic_param(mersenne)*10,
                                            y+generic_param(mersenne)*10,
                                            50+generic_param(mersenne)*20,
                                            50+generic_param(mersenne)*20,
                                            generic_param(mersenne)*M_PI);

            dc.add_displacer(displacer_ptr)
              .with_support(support_ptr);
          }
        }

        WarpFields::FlowField ff;
        ff.init_from_DisplacementComposer(dc).clamp_near_zeros();
        const CImg<float> flow  = ff.get_flow();
        const CImg<float> iflow = ff.get_iflow();
        
        /// Sample crops from big flow field
        size_t number_of_crops{0};
        for (int y = H/4; y < big_size-5*H/4; y+=H/3) {
          for (int x = W/4; x < big_size-5*W/4; x+=W/3) {
            //std::cout << "Crop (" << W << "x" << H << ")"
            //          <<  " at (" << x << "," << y << ")" << std::endl;
            CImg<float>  crop =  flow.get_crop(x,y,x+W,y+H);
            CImg<float> icrop = iflow.get_crop(x,y,x+W,y+H);
            {
              std::lock_guard<std::mutex> LOCK(m_queue_LOCK);
              //std::cout << "pushing crop (" << m_finalized_crops_queue.size() << ")" << std::endl;
              m_finalized_crops_queue.push(
                  std::make_pair<CImg<float>, CImg<float>>(std::move(crop),
                                                           std::move(icrop)));
            }
            ++number_of_crops;
          }
        }
        std::cout << "CropGenerator: pushed some fresh crops (" 
                  << m_finalized_crops_queue.size() << ")" << std::endl;
      }

    }

  }


}  // namespace WarpFields
