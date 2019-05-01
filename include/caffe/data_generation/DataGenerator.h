#ifndef CAFFE_DATA_GENERATION_H_
#define CAFFE_DATA_GENERATION_H_

/**
 * Nikolaus Mayer, 2017 (mayern@cs.uni-freiburg.de)
 */

/// System/STL
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
/// AGG (Anti-Grain Geometry)
#include "agg_ellipse.h"
#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_renderer_base.h"
#include "agg_renderer_scanline.h"
#include "agg_scanline_u.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_path_storage.h"
#include "agg_conv_curve.h"
#include "agg_conv_transform.h"
#include "agg_span_image_filter_rgb.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_linear.h"
#include "agg_renderer_scanline.h"
#include "agg_span_allocator.h"
#include "agg_rendering_buffer.h"
#include "agg_image_accessors.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/data_generation/SimpleRandom.h"
#include "caffe/data_generation/QueueProcessor.h"
#include "caffe/data_generation/WarpFields.h"

/// CImg (>= 2.0.0)
#include "thirdparty/CImg/CImg.h"

using namespace cimg_library;

#define DGEN_WIDTH 512
#define DGEN_HEIGHT 384

static const int W = DGEN_WIDTH;
static const int H = DGEN_HEIGHT;
static const int BACKGROUND_OBJ_ID = 1;

namespace DataGenerator {

  class Texture {
  public:
    Texture(const std::string& name, CImg<unsigned char>* texture_ptr)
     : m_name(name), m_texture_ptr(texture_ptr)
    { }

    ~Texture();

    CImg<unsigned char>& getTextureRef() const;

    CImg<unsigned char> getRandomizedCrop(int tex_w=W,
                                          int tex_h=H,
                                          float angle=0.f,
                                          float zoom=1.f,
                                          int x_shift=0,
                                          int y_shift=0) const;

    const std::string m_name;
    CImg<unsigned char>* m_texture_ptr;
  };


  class TextureCollection 
  {
  public:
    TextureCollection(const std::string& filepath);
    ~TextureCollection();

    Texture const* getTexturePtr(int raw_random_index) const;

    std::vector<Texture*> m_all_textures;
  };



  CImg<unsigned char> getTransformedTexture(const CImg<unsigned char>& input,
                                            const agg::trans_affine& tf_ref);

  CImg<unsigned char> applyWarpFieldToTexture(const CImg<unsigned char>& input,
                                              const CImg<float>& iflow);



  class MovingObjectBase
  {
  public:
    MovingObjectBase(size_t ID);
    virtual ~MovingObjectBase();

    void setRawTexture(CImg<unsigned char>&& tex_img);
    void setIntrinsicTransform(float alpha, float xs, float ys);
    void setMotion(float alpha, float scale, float xs, float ys);
    void addBackgroundMotion(const agg::trans_affine& bg_motion);

    virtual void renderTransformedTexture();

    template <class VertexSource>
    void draw(VertexSource& vs, bool AA, unsigned int frame_idx);

    /// Render "before" and "after" masks, AA and non-AA versions
    virtual void renderMasks();

    virtual void getPointFlow(float* x, float* y, bool inverse=false) const;

    bool isFinalized() const;
    void markFinalized(bool v=true);

    void setExtraWarpFields(CImg<float>& flow, CImg<float>& iflow);

    /// Debug
    void writeMasksToFiles() const;
   
  //private:
    size_t ID;

    bool m_is_finalized;

    unsigned char* m_scratch;
    std::vector<unsigned char*> m_masks_noAA;
    std::vector<unsigned char*> m_masks_AA;
    agg::trans_affine m_intrinsic_transform;
    agg::trans_affine m_intrinsic_transform_inv;
    agg::trans_affine m_motion;
    agg::trans_affine m_motion_inv;
    std::vector<CImg<unsigned char> > m_textures;

    agg::rendering_buffer m_rbuf;
    agg::pixfmt_gray8* m_pixf_ptr;
    agg::renderer_base<agg::pixfmt_gray8> m_ren_base;
    agg::renderer_scanline_aa_solid<
        agg::renderer_base<agg::pixfmt_gray8> > m_ren_sl;
    agg::scanline_u8 m_sl;
    agg::rasterizer_scanline_aa<> m_ras;

    bool m_has_extra_warp_fields;
    CImg<float> m_extra_warp_field;
    CImg<float> m_extra_warp_field_inverse;
    MovingObjectBase* m_parent_obj_ptr;
  };


  class MovingObjectEllipse : public MovingObjectBase
  {
  public:
    MovingObjectEllipse(size_t ID);

    void setEllipse(float cx, float cy, float rx, float ry, int steps);

    void renderMasks();

    agg::ellipse m_ellipse;
  };


  class MovingObjectPolygon : public MovingObjectBase
  {
  public:
    MovingObjectPolygon(size_t ID);

    void resetPath(float x, float y);
    void lineTo(float x, float y);
    void curve3To(float x_ctrl, float y_ctrl, float x_to, float y_to);
    void finalizePath();

    virtual void renderMasks();

    agg::path_storage m_path;
  };


  class MovingObjectComponentEllipse : public MovingObjectEllipse
  {
  public:
    MovingObjectComponentEllipse();
    void renderTransformedTexture();
  };


  class MovingObjectComponentPolygon : public MovingObjectPolygon
  {
  public:
    MovingObjectComponentPolygon();
    void renderTransformedTexture();
  };


  class MovingObjectComposite : public MovingObjectBase
  {
  public:
    MovingObjectComposite(size_t ID);
    ~MovingObjectComposite();

    void addComponent(MovingObjectBase* component);
    void subtractComponent(MovingObjectBase* component);

    void renderMasks();

    std::vector<MovingObjectBase*> m_components;
    std::vector<bool> m_component_modes;
  };


  class MovingObjectBackground : public MovingObjectPolygon
  {
  public:
    MovingObjectBackground(size_t ID);

    void renderTransformedTexture();

    void renderMasks();

    void getPointFlow(float* x, float* y, bool inverse=false) const;
  };



  /**
   * Asynchronous processing of single objects
   */
  struct UnfinishedObjectContainer {
    MovingObjectBase* obj_ptr;
  };
  int Process_UnfinishedObjectContainer(std::shared_ptr<UnfinishedObjectContainer> container);



  class RenderCore
  {
  public:
    RenderCore();

    void reset();
    bool blitObject(const MovingObjectBase& obj, bool use_AA);
    void computeFlowImage(std::map<size_t, MovingObjectBase*>& objects_map,
                          bool inverse=false);

  //private:
    CImg<unsigned char> frame0, frame1;
    CImg<float> flow0, flow1;
    CImg<size_t> index_image0, index_image1;
  };



  ///
  /// FlyingChairsRandom
  ///

  namespace FlyingChairsRandom
  {
    float baseGauss(float a, float b, float input, float normalize);

    template <class RNG_t>
    class Trigger {
    public:
      Trigger(float threshold, RNG_t RNG);
      Trigger(float a, float b, float threshold, int seed);

      bool operator()();

      float m_threshold;
      RNG_t m_RNG;
    };


    template<typename T>
    class Choice {
    public:
      Choice(std::vector<T>&& options, int seed);

      T operator()();

      std::vector<T> m_options;
      RNG::FixedRangeUniformInt m_RNG;
    };


    class Uniform {
    public:
      Uniform(float a, float b, int seed);

      float operator()();

      RNG::FixedRangeUniformFloat m_rng;
    };


    class Gaussian {
    public:
      Gaussian(float a, float b, int seed);

      float operator()();

      float a, b;
      RNG::FixedMeanStddevNormalFloat m_rng;
    };


    class GaussianSq {
    public:
      GaussianSq(float a, float b, int seed);

      float operator()();

      float a, b;
      RNG::FixedMeanStddevNormalFloat m_rng;
    };


    class Gaussian3 {
    public:
      Gaussian3(float a, float b, int seed);

      float operator()();

      float a, b;
      RNG::FixedMeanStddevNormalFloat m_rng;
    };


    class Gaussian4 {
    public:
      Gaussian4(float a, float b, int seed);

      float operator()();

      float a, b;
      RNG::FixedMeanStddevNormalFloat m_rng;
    };


    class GaussianMeanSigmaRange {
    public:
      GaussianMeanSigmaRange(float a, float b, float mean, float sigma, int seed);

      float operator()();

      float a, b, mean, sigma;
      RNG::FixedMeanStddevNormalFloat m_rng;
    };
  }  /// namespace FlyingChairsRandom



  /// Possible types of moving objects
  enum class ObjType_t {
    Dummy     = 0,
    Ellipse   = 1,
    Polygon   = 2,
    Composite = 3,
  };

  /// Possible types of segments of a polygon outline
  enum class PolySegmentType_t {
    Dummy  = 0,
    Line   = 1,
    Curve3 = 3,
  };



  /**
   * Encapsulate all the parameters needed to create an object
   */
  class ObjectBlueprint {
  public:
    ObjectBlueprint();
    ~ObjectBlueprint();

    int obj_id;
    ObjType_t obj_type;
    /// Intrinsic object transform
    float init_rot;
    float init_scale;
    float init_trans_x, init_trans_y;
    /// Object motion
    float rot;
    float scale;
    float trans_x, trans_y;
    /// Texture stuff
    int tex_id;
    float tex_rot;
    float tex_scale;
    int tex_shift_x;
    int tex_shift_y;
    /// Ellipse object specifics
    float ellipse_scale_x;
    float ellipse_scale_y;
    /// Polygon object specifics
    std::vector<PolySegmentType_t> polygon_segment_types;
    std::vector<float> polygon_segment_x;
    std::vector<float> polygon_segment_y;
    /// Component objects of a composite object
    std::vector<ObjectBlueprint*> composite_component_blueprint_ptrs;
    bool is_additive_component;
    /// Warp object 
    bool do_warpfield_deformation;
  };

  class TaskBucket {
  public:
    TaskBucket();

    unsigned int task_ID;

    bool bad;

    ObjectBlueprint* background_blueprint;
    std::vector<ObjectBlueprint*> object_blueprints;
    
    CImg<float>* result_image0_ptr;
    CImg<float>* result_image1_ptr;
    CImg<float>* result_flow0_ptr;
  };

  class Sample {
  public:
    void Destroy();

    CImg<float>* image0_ptr;
    CImg<float>* image1_ptr;
    CImg<float>* flow0_ptr;
  };


  class DataGenerator 
  {
  public:
    DataGenerator(const caffe::LayerParameter& param);
    ~DataGenerator();

    void setBatchSize(int batch_size);
    void Start();
    void Stop(bool wait=true);
    void Pause();
    void Resume();

    MovingObjectBase* RealizeObjectBlueprint(
          ObjectBlueprint* p, 
          const agg::trans_affine& bg_motion,
          QueueProcessing::QueueProcessor<UnfinishedObjectContainer>& qp,
          MovingObjectBase* parent_obj_ptr=nullptr);

    void Process_TaskBucket(
          TaskBucket* const task_ptr,
          RenderCore& rendercore,
          std::map<size_t, MovingObjectBase*>& objects_map,
          QueueProcessing::QueueProcessor<UnfinishedObjectContainer>& qp);
    

    void WorkerThreadLoop();

    void commissionNewTask(TaskBucket* new_task_ptr);
    bool hasUnfinishedTasks() const;
    bool hasRetrievableFinishedTasks() const;
    Sample retrieveFinishedTask();


    bool m_generator_is_running;

    std::queue<TaskBucket*> m_undone_tasks;
    std::vector<unsigned int> m_task_IDs_in_flight;
    std::queue<TaskBucket*> m_finished_tasks;
    std::mutex m_task_queues__LOCK;
    std::vector<std::thread*> m_worker_threads;

    TextureCollection m_random_textures;
    //TextureCollection m_flickr_textures;

    int m_batch_size;
    int m_first_level_threads;
    int m_second_level_threads;
    bool m_use_AA;

    int MODE;
    WarpFields::CropGenerator* m_crop_generator_ptr;
  };  /// class DataGenerator


  ///
  /// ObjectParametersGenerator
  ///
  using namespace RNG;
  using namespace FlyingChairsRandom;
  class ObjectParametersGenerator {
  public:
    ObjectParametersGenerator(const caffe::LayerParameter& param);
    ~ObjectParametersGenerator();

    void generateBackground(ObjectBlueprint* b);
    void generateForegroundObject(ObjectBlueprint* b);
    int generateNumberOfFgObjects();


    int MODE;
    int RNG_SEED;
    const int int_max;
    ///
    /// Background
    ///
    FixedRangeUniformInt*   RNG_BgTexID;
    Uniform*                RNG_BgInitRot;
    Choice<int>*            RNG_BgInitTransX;
    Choice<int>*            RNG_BgInitTransY;
    Trigger<Uniform>*       RNG_BgRotTrigger;
    GaussianSq*             RNG_BgRot;
    Gaussian4*              RNG_BgTransX;
    Gaussian4*              RNG_BgTransY;
    Trigger<Uniform>*       RNG_BgScaleTrigger;
    Uniform*                RNG_BgInitScale;  /// tex zoom
    GaussianSq*             RNG_BgScale;
    ///
    /// Foreground objects
    ///
    Uniform*                RNG_NumberOfFgObjects;
    Choice<ObjType_t>*      RNG_ObjType;
    FixedRangeUniformInt*   RNG_ObjTexID;
    Uniform*                RNG_ObjInitTransX;
    Uniform*                RNG_ObjInitTransY;
    Gaussian3*              RNG_ObjTransX;
    Gaussian3*              RNG_ObjTransY;
    Uniform*                RNG_ObjInitRot;
    Trigger<Uniform>*       RNG_ObjRotTrigger;
    GaussianSq*             RNG_ObjRot;
    GaussianMeanSigmaRange* RNG_ObjInitScale;
    Trigger<Uniform>*       RNG_ObjScaleTrigger;
    GaussianSq*             RNG_ObjScale;
    /// (Textures)
    FixedRangeUniformInt*   RNG_ObjTexShiftX;
    FixedRangeUniformInt*   RNG_ObjTexShiftY;
    FixedRangeUniformFloat* RNG_ObjTexRot;
    FixedRangeUniformFloat* RNG_ObjTexZoom;
    /// Ellipse object specifics
    Uniform*                RNG_ElliObj_ScaleX;
    Uniform*                RNG_ElliObj_ScaleY;
    /// Polygon object specifics
    FixedRangeUniformInt*   RNG_PolyObj_spokes;
    Uniform*                RNG_PolyObj_dphi;
    Uniform*                RNG_PolyObj_r;
    Uniform*                RNG_PolyObj_ScaleX;
    Uniform*                RNG_PolyObj_ScaleY;
    Trigger<Uniform>*       RNG_PolyObj_CurveTrigger;
    ///
    /// Component objects
    ///
    Uniform*                RNG_CompObjInitTransX;
    Uniform*                RNG_CompObjInitTransY;
    FixedRangeUniformInt*   RNG_CompObiNumberOfComponents;
    Trigger<Uniform>*       RNG_ComponentIsAdditive;
    Uniform*                RNG_ComponentOffset;
    ///
    /// Thin objects
    ///
    Trigger<Uniform>*       RNG_ObjIsExtraThin;
    ///
    /// Nonrigid deformations
    ///
    Trigger<Uniform>*       RNG_ObjDeformsNonrigidly;

    ///
    /// General purpose stuff (for "other stuff")
    ///
    Uniform*                RNG_GenericUniform;
    Trigger<Uniform>*       RNG_GenericTrigger;
  };

}  /// namespace DataGenerator


#endif  // CAFFE_DATA_GENERATION_H_

