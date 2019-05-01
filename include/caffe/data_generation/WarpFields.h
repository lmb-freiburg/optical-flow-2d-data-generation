#ifndef WARPFIELDS_H__
#define WARPFIELDS_H__

/**
 * Nikolaus Mayer, 2017 (mayern@cs.uni-freiburg.de)
 */

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "thirdparty/CImg/CImg.h"
using namespace cimg_library;


namespace WarpFields {

  /**
   * We use std::tuple<float,float> as 2-vector class
   */
  typedef std::tuple<float,float> V2;
  #define make_V2 std::make_tuple
  #define V2_x std::get<0>
  #define V2_y std::get<1>


  /**
   * "Supports" are weighting / influence masks for displacers. A displacer
   * without a mask does not know which pixels it should affect to which degree.
   */
  namespace Supports {

    /**
     * Abstract base class
     */
    class SupportBase {
    public:
      SupportBase(
            float cx, 
            float cy);
      virtual ~SupportBase();

      virtual float at(
            float x, 
            float y) const = 0;
      float at_relative(
            float x, 
            float y) const;

    protected:
      const float cx, cy;
    };


    /**
     * Constant support (flat)
     */
    class Constant : public SupportBase {
    public:
      Constant(
            float factor=1.f);

      float at(
            float x, 
            float y) const;
      
    private:
      const float factor;
    };


    /**
     * Isotropic gaussian support (soft gradients)
     */
    class Gaussian1D : public SupportBase {
    public:
      Gaussian1D(
            float cx, 
            float cy, 
            float sigma);

      float raw_at(
            float x, 
            float y) const;
      float at(
            float x, 
            float y) const;

    private:
      const float sigma_sq, gauss_prefactor, normalizer;
    };


    /**
     * 2D gaussian support
     */
    class Gaussian2D : public SupportBase {
    public:
      Gaussian2D(
            float cx, 
            float cy, 
            float sigma_x, 
            float sigma_y, 
            float angle);

      float raw_at(
            float x, 
            float y) const;
      float at(
            float x, 
            float y) const;

    private:
      const float a, b, c, d, ratio_x_y, sigma_sq, gauss_prefactor, normalizer;
    };


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
    class DisplacerBase {
    public:
      DisplacerBase(
            float cx, 
            float cy);
      virtual ~DisplacerBase();

      DisplacerBase& set_support(
            Supports::SupportBase* new_support_ptr);

      V2 flow_at(
            float x, 
            float y) const;
      V2 iflow_at(
            float x, 
            float y) const;

      float support_at(
            float x, 
            float y) const;

      virtual V2 raw_flow_at(
            float x, 
            float y) const = 0;
      virtual V2 raw_iflow_at(
            float x, 
            float y) const = 0;

    protected:
      const float cx, cy;
      Supports::SupportBase* support_ptr;
    };


    /**
     * Zero-flow (static image)
     */
    class Identity : public DisplacerBase {
    public:
      Identity();

      V2 raw_flow_at(
            float x, 
            float y) const;
      V2 raw_iflow_at(
            float x, 
            float y) const;
    };


    /**
     * Translation flow (shift)
     */
    class Translation : public DisplacerBase {
    public:
      Translation(
            float dx, 
            float dy);

      V2 raw_flow_at(
            float x, 
            float y) const;
      V2 raw_iflow_at(
            float x, 
            float y) const;

    private:
      const float dx, dy;
    };


    /**
     * Rotation flow (twist/vortex/whirl)
     */
    class Rotation : public DisplacerBase {
    public:
      Rotation(
            float cx, 
            float cy, 
            float angular_speed);

      V2 raw_flow_at(
            float x, 
            float y) const;
      V2 raw_iflow_at(
            float x, 
            float y) const;

    private:
      const float omega, sin_omega, cos_omega, sin_nomega, cos_nomega;
    };


    /**
     * Zooming flow (expansion/contraction)
     */
    class Zoom : public DisplacerBase {
    public:
      Zoom( float cx, 
            float cy, 
            float factor);

      V2 raw_flow_at(
            float x, 
            float y) const;
      V2 raw_iflow_at(
            float x, 
            float y) const;

    private:
      float factor, ifactor;
    };

  }  // namespace Displacers




  /**
   * A DisplacementComposer holds a collection of Support-weighed Displacers and
   * adds together their respective flows to compute a final, discretely sampled
   * flowfield. The DisplacementComposer uses a hierarchical diffeomorphism
   * composition approach to compute flowfields with highly accurate inversions.
   */
  class DisplacementComposer {
  public:
    DisplacementComposer(
          int W, 
          int H);
    ~DisplacementComposer();

    DisplacementComposer& add_displacer(
          Displacers::DisplacerBase* new_d_ptr);
    DisplacementComposer& with_support(
          Supports::SupportBase* new_s_ptr);

    V2 flow_at(
          float x, 
          float y) const;
    V2 iflow_at(
          float x, 
          float y) const;

    int get_W() const;
    int get_H() const;

  private:
    const int W, H;
    std::vector<Displacers::DisplacerBase*> displacer_ptrs;
  };


  class FlowField {
  public:
    FlowField();
    
    /**
     * Generate discrete forward and backward flow fields, given an instance of
     * a DisplacementComposer with flow elements.
     */
    FlowField& init_from_DisplacementComposer(
          const DisplacementComposer& dc);

    /**
     * Set to zero all tiny flows. This is not strictly necessary, but can clean
     * up a flowfield if a Displacer uses a Support which is nonzero everywhere.
     */
    FlowField& clamp_near_zeros();

    CImg<float> get_flow() const;
    CImg<float> get_iflow() const;

  private:
    CImg<float> flow, iflow;
  };


  /**
   * Asynchronously generate and serve cropped regions from larger flow fields
   */
  class CropGenerator {
  public:
    CropGenerator(int W, int H, int reuse_same=0);
    ~CropGenerator();

    CropGenerator& Start();
    CropGenerator& Stop();

    std::pair<CImg<float>, CImg<float>> get_crop();

  private:
    void worker_thread_loop();

    const int m_W, m_H, m_reuse_same;
    int m_reuse_counter;
    bool m_running;

    std::queue<std::pair<CImg<float>, CImg<float>>> m_finalized_crops_queue;
    std::vector<std::unique_ptr<std::thread>> m_worker_thread_ptrs;
    std::mutex m_queue_LOCK;
  };


}  // namespace WarpFields

#endif  /// WARPFIELDS_H__

