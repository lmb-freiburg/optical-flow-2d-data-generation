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
#include <limits>
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
#include "caffe/data_generation/DataGenerator.h"
#include "caffe/data_generation/WarpFields.h"

/// CImg (>= 2.0.0)
#include "thirdparty/CImg/CImg.h"

using namespace cimg_library;

/**
 * Available MODE values:
 *  1 - Axis-aligned rectangular objects; only translation motion
 *  2 - Polygons with straight edges; only translation motion
 *  3 - Round objects; only translation motion
 *  4 - Shapes from 1+2+3; translation and rotation motion
 *  5 - 4 + scaling motion
 *  6 - 5 + objects with complex-shaped holes
 *  7 - 6 + very thin objects ("needle" and "outline" styles)
 *  8 - Shapes from 1+2+3; only translation motion
 *  9 - 7 + nonrigid deformation motions
 * 10 - 7 with halved motion magnitude distributions
 * 11 - 7 with doubled motion magnitude distributions
 * 12 - 7 with thirded motion magnitude distributions
 * 13 - 7 with tripled motion magnitude distributions
 */

namespace DataGenerator {

  ///
  /// Texture
  ///

  Texture::~Texture()
  {
    delete m_texture_ptr;
  }

  CImg<unsigned char>& Texture::getTextureRef() const
  {
    return *m_texture_ptr;
  }

  CImg<unsigned char> Texture::getRandomizedCrop(int tex_w,
                                                 int tex_h,
                                                 float angle,
                                                 float zoom,
                                                 int x_shift,
                                                 int y_shift) const
  {
    const int width{m_texture_ptr->width()};
    const int height{m_texture_ptr->height()};
    if (width >= tex_w and height >= tex_h) {
      return m_texture_ptr->get_shift(x_shift, y_shift, 0, 0, 3)
                           .rotate(angle, 1, 3)
                           .crop(width/2-tex_w/2,
                                 height/2-tex_h/2,
                                 width/2-tex_w/2+tex_w/zoom-1,
                                 height/2-tex_h/2+tex_h/zoom-1, 3)
                           .resize(tex_w, tex_h, -100, -100, 3);
    } else {
      return m_texture_ptr->get_shift(x_shift, y_shift, 0, 0, 3)
                           .rotate(angle, 1, 3)
                           .resize(tex_w, tex_h, -100, -100, 3);
    }
  }



  ///
  /// TextureCollection
  ///

  TextureCollection::TextureCollection(const std::string& filepath)
  {
    std::ifstream infile(filepath);
    if (infile.bad() or not infile.is_open()) {
      throw std::runtime_error("Could not open texture collection");
    }
    std::string imagepath;
    while (not infile.eof()) {
      std::getline(infile, imagepath);
      if (infile.eof()) break;
      CImg<unsigned char>* cimg_ptr = new CImg<unsigned char>();
      cimg_ptr->load(imagepath.c_str());
      cimg_forXY(*cimg_ptr,x,y) {
        std::swap((*cimg_ptr)(x,y,0), (*cimg_ptr)(x,y,2));
      }
      Texture* new_tex = new Texture(imagepath, cimg_ptr);
      m_all_textures.push_back(new_tex);
    }
    infile.close();

    {
      size_t total_size = 0;
      for (unsigned int i = 0; i < m_all_textures.size(); ++i) {
        total_size += m_all_textures[i]->m_texture_ptr->width() *
                      m_all_textures[i]->m_texture_ptr->height() *
                      m_all_textures[i]->m_texture_ptr->spectrum();
      }
      std::cout << "Loaded " << m_all_textures.size() << " textures"
                << " from " << filepath
                << " with a total size of " << total_size/(1024*1024) << " MB."
                << std::endl;
    }
  }

  TextureCollection::~TextureCollection()
  {
    for (unsigned int i = 0; i < m_all_textures.size(); ++i)
      if (m_all_textures[i])
        delete m_all_textures[i];
  }

  Texture const* TextureCollection::getTexturePtr(int raw_random_index) const
  {
    return m_all_textures[raw_random_index % m_all_textures.size()];
  }



  ///
  /// Transform a texture given an affine transformation
  ///
  CImg<unsigned char> getTransformedTexture(const CImg<unsigned char>& input,
                                            const agg::trans_affine& tf_ref)
  {
    /// Load texture image
    CImg<unsigned char> input_texture_image(input);
    /// Resize input texture
    //input_texture_image.resize(W,H,-100,-100,3);

    const int tex_W = input_texture_image.width();
    const int tex_H = input_texture_image.height();
    
    /// Change pixel layout from RR..RGG..GBB..B to RGBRGB..RGB
    input_texture_image.permute_axes("CXYZ");
    
    /// AGG image source wrapper
    agg::rendering_buffer source_rbuf;
    source_rbuf.attach(input_texture_image.data(), tex_W, tex_H, tex_W*3);
    /// 24-bit RGB type
    typedef agg::pixfmt_rgb24 PIX_RGB_T;
    PIX_RGB_T source_img_tmp(source_rbuf);
    /// Reflect-wrapping texture access
    typedef agg::wrap_mode_reflect TEX_WRAP_T;
    typedef agg::image_accessor_wrap<PIX_RGB_T,TEX_WRAP_T,TEX_WRAP_T> IMG_SRC_T;
    /// The final pixel source for rendering
    IMG_SRC_T source_img(source_img_tmp);

    /// Allocate memory for transformed texture output
    /// (the "C,X,Y,Z" dimensions emulate AGG's RGBRGB..RGB format)
    CImg<unsigned char> output_texture(3,tex_W,tex_H,1);
    agg::rendering_buffer target_img;
    target_img.attach(output_texture.data(), tex_W, tex_H, tex_W*3);
    PIX_RGB_T target_pixf_img(target_img);
    agg::renderer_base<PIX_RGB_T> renderer_base(target_pixf_img);

    /// The texture transformation
    agg::trans_affine image_mtx = tf_ref;
    /// Inverse transformation! (backward warping?)
    image_mtx.invert();
    typedef agg::span_interpolator_linear<> TEX_INTERP_T;
    TEX_INTERP_T interpolator(image_mtx);
    agg::span_allocator<agg::rgba8> span_allocator;
    agg::span_image_filter_rgb_bilinear<IMG_SRC_T,TEX_INTERP_T>
        span_generator(source_img, interpolator);

    agg::rasterizer_scanline_aa<> rasterizer;
    agg::scanline_u8 scanline;
    /// Render a polygon which covers the entire input image
    agg::path_storage path;
    path.move_to(0, 0);
    path.line_to(tex_W, 0);
    path.line_to(tex_W, tex_H);
    path.line_to(0, tex_H);
    path.close_polygon();
    rasterizer.add_path(path);

    /// Finally, render the transformed texture
    agg::render_scanlines_aa(rasterizer, scanline, renderer_base, 
                             span_allocator, span_generator);

    /// Change pixel layout from RGBRGB..RGB to RR..RGG..GBB..B
    output_texture.permute_axes("YZCX");

    return output_texture;
  }


  ///
  /// Warp a texture using an inverse flow field
  ///
  CImg<unsigned char> applyWarpFieldToTexture(const CImg<unsigned char>& input,
                                              const CImg<float>& iflow)
  {
    CImg<unsigned char> result(input.width(),
                               input.height(),
                               1,
                               input.spectrum());
    cimg_forXYC(result,x,y,c) {
      result(x,y,c) = input.linear_atXY(x+iflow(x,y,0),
                                        y+iflow(x,y,1),
                                        0,
                                        c,
                                        (unsigned char)0);
    }
    return result;
  }



  ///
  /// MovingObjectBase (virtual base class)
  ///

  MovingObjectBase::MovingObjectBase(size_t ID)
    : ID(ID),
      m_is_finalized(false),
      m_has_extra_warp_fields(false),
      m_parent_obj_ptr(nullptr)
  {
    m_scratch =            new unsigned char[W * H];
    m_masks_noAA.push_back(new unsigned char[W * H]);
    m_masks_noAA.push_back(new unsigned char[W * H]);
    m_masks_AA.push_back(  new unsigned char[W * H]);
    m_masks_AA.push_back(  new unsigned char[W * H]);

    m_rbuf.attach(m_scratch, W, H, W);
    m_pixf_ptr = new agg::pixfmt_gray8(m_rbuf);
    m_ren_base = agg::renderer_base<agg::pixfmt_gray8>(*m_pixf_ptr);
    m_ren_sl = agg::renderer_scanline_aa_solid<
                   agg::renderer_base<agg::pixfmt_gray8> >(m_ren_base);
    m_ren_sl.color(agg::gray8(255));
  }

  MovingObjectBase::~MovingObjectBase()
  {
    if (m_scratch)
      delete[] m_scratch;
    for (size_t i = 0; i < m_masks_noAA.size(); ++i)
      if (m_masks_noAA[i])
        delete[] m_masks_noAA[i];
    for (size_t i = 0; i < m_masks_AA.size(); ++i)
      if (m_masks_AA[i])
        delete[] m_masks_AA[i];

    if (m_pixf_ptr)
      delete m_pixf_ptr;

    /// Do NOT delete m_parent_obj_ptr!
  }

  void MovingObjectBase::setRawTexture(CImg<unsigned char>&& tex_img)
  {
    m_textures.push_back(std::move(tex_img));
  }

  void MovingObjectBase::setIntrinsicTransform(float alpha, float xs, float ys)
  {
    m_intrinsic_transform  = agg::trans_affine();
    m_intrinsic_transform *= agg::trans_affine_rotation(alpha);
    m_intrinsic_transform *= agg::trans_affine_translation(xs, ys);

    m_intrinsic_transform_inv = m_intrinsic_transform;
    m_intrinsic_transform_inv.invert();
  }

  void MovingObjectBase::setMotion(float alpha, float scale, float xs, float ys)
  {
    /// Object transform (its "motion")
    m_motion  = agg::trans_affine();
    m_motion *= agg::trans_affine_rotation(alpha);
    m_motion *= agg::trans_affine_scaling(scale);
    m_motion *= agg::trans_affine_translation(xs,ys);

    m_motion_inv = m_motion;
    m_motion_inv.invert();
  }

  void MovingObjectBase::addBackgroundMotion(const agg::trans_affine& bg_motion)
  {
    /// Ugly special treatment because of special background object
    agg::trans_affine bg_n = agg::trans_affine_translation(-W/2.,-H/2.);
    bg_n *= bg_motion;
    bg_n *= agg::trans_affine_translation(W/2.,H/2.);

    m_motion *= bg_n;

    m_motion_inv = m_motion;
    m_motion_inv.invert();
  }

  void MovingObjectBase::renderTransformedTexture()
  {
    m_textures.push_back(getTransformedTexture(m_textures[0],
                         agg::trans_affine()));
    if (m_has_extra_warp_fields) {
      m_textures.push_back(applyWarpFieldToTexture(
                                  getTransformedTexture(m_textures[0], 
                                                        m_motion),
                                  m_extra_warp_field_inverse));
    } else {
      m_textures.push_back(getTransformedTexture(m_textures[0], m_motion));
    }
  }

  template <class VertexSource>
  void MovingObjectBase::draw(VertexSource& vs, bool AA, unsigned int frame_idx)
  {
    m_ren_base.clear(agg::gray8(0));
    m_ras.reset();
    m_ras.add_path(vs);
    if (AA) {
      m_ras.gamma(agg::gamma_none());
    } else {
      m_ras.gamma(agg::gamma_threshold(0.5)); 
    }
    agg::render_scanlines(m_ras, m_sl, m_ren_sl);
    if (AA) {
      std::memcpy(m_masks_AA[frame_idx], m_scratch, W*H);
    } else {
      std::memcpy(m_masks_noAA[frame_idx], m_scratch, W*H);
    }
  }

  void MovingObjectBase::renderMasks()
  {
    if (m_has_extra_warp_fields) {
      {
        CImg<unsigned char> tmp(m_masks_noAA[1],W,H,1,1,true);
        CImg<unsigned char> transformed = applyWarpFieldToTexture(tmp,
                                                m_extra_warp_field_inverse);
        std::memcpy(m_masks_noAA[1], transformed.data(), W*H);
      }
      {
        CImg<unsigned char> tmp(m_masks_AA[1],W,H,1,1,true);
        CImg<unsigned char> transformed = applyWarpFieldToTexture(tmp,
                                                m_extra_warp_field_inverse);
        std::memcpy(m_masks_AA[1], transformed.data(), W*H);
      }
    }
  }

  void MovingObjectBase::getPointFlow(float* x, float* y, bool inverse) const
  {
    double ix = *x; 
    double iy = *y; 
    float save_x = ix;
    float save_y = iy;

    if (inverse)
      m_motion_inv.transform(&ix,&iy);
    else
      m_motion.transform(&ix,&iy);

    *x = ix-save_x;
    *y = iy-save_y;

    if (m_has_extra_warp_fields and ix>=0 and ix<W and iy>=0 and iy<H) {
      *x += m_extra_warp_field.linear_atXY(ix,iy,0);
      *y += m_extra_warp_field.linear_atXY(ix,iy,1);
    }
  }

  bool MovingObjectBase::isFinalized() const
  {
    return m_is_finalized;
  }

  void MovingObjectBase::markFinalized(bool v)
  {
    m_is_finalized = v;
  }

  void MovingObjectBase::setExtraWarpFields(CImg<float>& flow,
                                            CImg<float>& iflow)
  {
    m_has_extra_warp_fields = true;
    m_extra_warp_field = flow;
    m_extra_warp_field_inverse = iflow;
  }


  /// Debug
  void MovingObjectBase::writeMasksToFiles() const
  {
    {
      CImg<unsigned char> tmp(m_masks_noAA[0],W,H,1,1,true);
      tmp.save("mask_noAA_0.pgm");
    }
    {
      CImg<unsigned char> tmp(m_masks_noAA[1],W,H,1,1,true);
      tmp.save("mask_noAA_1.pgm");
    }
    {
      CImg<unsigned char> tmp(m_masks_AA[0],W,H,1,1,true);
      tmp.save("mask_AA_0.pgm");
    }
    {
      CImg<unsigned char> tmp(m_masks_AA[1],W,H,1,1,true);
      tmp.save("mask_AA_1.pgm");
    }
  }

   

  ///
  /// MovingObjectEllipse
  ///

  MovingObjectEllipse::MovingObjectEllipse(size_t ID) 
    : MovingObjectBase(ID) 
  {}

  void MovingObjectEllipse::setEllipse(float cx, float cy, float rx, float ry, int steps)
  {
    /// Prepare object shape (original and transformed)
    m_ellipse.init(cx,cy,rx,ry,steps);
  }

  void MovingObjectEllipse::renderMasks()
  {
    //agg::trans_affine save = m_motion;
    //save *= m_intrinsic_transform;
    agg::trans_affine save = m_intrinsic_transform;
    save *= m_motion;
    agg::conv_transform<agg::ellipse> ellipse(m_ellipse, m_intrinsic_transform);
    draw(ellipse,  true,  0);
    draw(ellipse,  false, 0);
    agg::conv_transform<agg::ellipse> ellipse_tr(m_ellipse, save);
    draw(ellipse_tr, true,  1);
    draw(ellipse_tr, false, 1);

    MovingObjectBase::renderMasks();
  }



  ///
  /// MovingObjectPolygon
  ///

  MovingObjectPolygon::MovingObjectPolygon(size_t ID)
    : MovingObjectBase(ID) 
  {}

  void MovingObjectPolygon::resetPath(float x, float y)
  {
    m_path.remove_all();
    m_path.move_to(x, y);
  }

  void MovingObjectPolygon::lineTo(float x, float y)
  {
    m_path.line_to(x, y);
  }

  void MovingObjectPolygon::curve3To(float x_ctrl, float y_ctrl,
                float x_to, float y_to)
  {
    m_path.curve3(x_ctrl, y_ctrl, x_to, y_to);
  }

  //void MovingObjectPolygon::curve4To(float x_ctrl1, float y_ctrl1,
  //              float x_ctrl2, float y_ctrl2,
  //              float x_to, float y_to)
  //{
  //  m_path.curve4(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to);
  //}

  void MovingObjectPolygon::finalizePath()
  {
    m_path.close_polygon();
  }

  void MovingObjectPolygon::renderMasks()
  {
    agg::trans_affine save = m_intrinsic_transform;
    save *= m_motion;
    agg::conv_transform<agg::path_storage> path(m_path, m_intrinsic_transform);
    agg::conv_curve<agg::conv_transform<agg::path_storage> > path_curve(path);
    agg::conv_transform<agg::path_storage> path_tr(m_path, save);
    agg::conv_curve<agg::conv_transform<agg::path_storage> > path_curve_tr(path_tr);
    draw(path_curve,    true,  0);
    draw(path_curve,    false, 0);
    draw(path_curve_tr, true,  1);
    draw(path_curve_tr, false, 1);

    MovingObjectBase::renderMasks();
  }



  ///
  /// MovingObjectComponentEllipse
  ///

  MovingObjectComponentEllipse::MovingObjectComponentEllipse() 
    : MovingObjectEllipse(0) 
  { }
  
  void MovingObjectComponentEllipse::renderTransformedTexture() 
  { }



  ///
  /// MovingObjectComponentPolygon
  ///

  MovingObjectComponentPolygon::MovingObjectComponentPolygon() 
    : MovingObjectPolygon(0) 
  { }

  void MovingObjectComponentPolygon::renderTransformedTexture() 
  { }



  ///
  /// MovingObjectComposite
  ///

  MovingObjectComposite::MovingObjectComposite(size_t ID) 
    : MovingObjectBase(ID) 
  { }

  MovingObjectComposite::~MovingObjectComposite()
  {
    for (unsigned int ci = 0; ci < m_components.size(); ++ci)
      if (m_components[ci])
        delete m_components[ci];
  }

  void MovingObjectComposite::addComponent(MovingObjectBase* component)
  {
    m_components.push_back(component);
    m_component_modes.push_back(true);
  }

  void MovingObjectComposite::subtractComponent(MovingObjectBase* component)
  {
    m_components.push_back(component);
    m_component_modes.push_back(false);
  }

  void MovingObjectComposite::renderMasks()
  {
    std::memset(m_masks_AA[0],   0, W*H);
    std::memset(m_masks_AA[1],   0, W*H);
    std::memset(m_masks_noAA[0], 0, W*H);
    std::memset(m_masks_noAA[1], 0, W*H);
    for (unsigned int ci = 0; ci < m_components.size(); ++ci) {
      MovingObjectBase* component = m_components[ci];
      while (not component->isFinalized())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

      if (m_component_modes[ci]) {
        unsigned char*       u =            m_masks_noAA[0];
        unsigned char const* v = component->m_masks_noAA[0];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*(1.f-(1.f-u[i]/255.f)*(1.f-v[i]/255.f)));

        u =            m_masks_noAA[1];
        v = component->m_masks_noAA[1];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*(1.f-(1.f-u[i]/255.f)*(1.f-v[i]/255.f)));

        u =            m_masks_AA[0];
        v = component->m_masks_AA[0];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*(1.f-(1.f-u[i]/255.f)*(1.f-v[i]/255.f)));

        u =            m_masks_AA[1];
        v = component->m_masks_AA[1];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*(1.f-(1.f-u[i]/255.f)*(1.f-v[i]/255.f)));
      } else {
        unsigned char*       u =            m_masks_noAA[0];
        unsigned char const* v = component->m_masks_noAA[0];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*((u[i]/255.f)*(1.f-v[i]/255.f)));

        u =            m_masks_noAA[1];
        v = component->m_masks_noAA[1];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*((u[i]/255.f)*(1.f-v[i]/255.f)));

        u =            m_masks_AA[0];
        v = component->m_masks_AA[0];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*((u[i]/255.f)*(1.f-v[i]/255.f)));

        u =            m_masks_AA[1];
        v = component->m_masks_AA[1];
        for (unsigned int i = 0; i < W*H; ++i)
          u[i] = static_cast<unsigned char>(255.f*((u[i]/255.f)*(1.f-v[i]/255.f)));
      }
    }

    /// No MovingObjectBase::renderMasks call here (already done by components)
  }



  ///
  /// MovingObjectBackground (a specialized MovingObjectPolygon)
  ///

  MovingObjectBackground::MovingObjectBackground(size_t ID) : MovingObjectPolygon(ID)
  {
    resetPath(-2.5*W, -2.5*H);
    lineTo(    2.5*W, -2.5*H);
    lineTo(    2.5*W,  2.5*H);
    lineTo(   -2.5*W,  2.5*H);
    finalizePath();

    setIntrinsicTransform(0.f, W, H);
  }

  void MovingObjectBackground::renderTransformedTexture()
  {
    m_textures.push_back(getTransformedTexture(m_textures[0],
                 agg::trans_affine()));

    if (m_has_extra_warp_fields) {
      m_textures.push_back(applyWarpFieldToTexture(
                                  getTransformedTexture(m_textures[0], 
                 m_intrinsic_transform_inv*m_motion*m_intrinsic_transform),
                                  m_extra_warp_field_inverse));
    } else {
      m_textures.push_back(getTransformedTexture(m_textures[0],
                 m_intrinsic_transform_inv*m_motion*m_intrinsic_transform));
    }

    m_textures[1].crop(W/2.,H/2.,W*3./2.-1,H*3./2.-1,3);
    m_textures[2].crop(W/2.,H/2.,W*3./2.-1,H*3./2.-1,3);
  }

  void MovingObjectBackground::renderMasks()
  {
    std::memset(m_masks_AA[0],   255, W*H);
    std::memset(m_masks_AA[1],   255, W*H);
    std::memset(m_masks_noAA[0], 255, W*H);
    std::memset(m_masks_noAA[1], 255, W*H);
  }

  void MovingObjectBackground::getPointFlow(float* x, float* y, bool inverse) const
  {
    /// Special treatment for the background object (this object has
    /// a larger texture which is cropped in an extra step, but this
    /// invalidates the easy mapping)
    double ix = *x + W/2; 
    double iy = *y + H/2;
    float save_x = ix;
    float save_y = iy;

    m_intrinsic_transform_inv.transform(&ix,&iy);

    if (inverse)
      m_motion_inv.transform(&ix,&iy);
    else
      m_motion.transform(&ix,&iy);

    m_intrinsic_transform.transform(&ix,&iy);

    *x = ix-save_x;
    *y = iy-save_y;

    if (m_has_extra_warp_fields and ix>=0 and ix<2*W and iy>=0 and iy<2*H) {
      *x += m_extra_warp_field.linear_atXY(ix,iy,0);
      *y += m_extra_warp_field.linear_atXY(ix,iy,1);
    }
  }



  ///
  /// Asynchronous processing of single objects
  ///
  
  int Process_UnfinishedObjectContainer(std::shared_ptr<UnfinishedObjectContainer> container)
  {
    container->obj_ptr->renderTransformedTexture();
    container->obj_ptr->renderMasks();
    container->obj_ptr->markFinalized();
    return 0;
  }



  ///
  /// RenderCore
  ///
  
  RenderCore::RenderCore()
  {
    frame0       = CImg<unsigned char>(W,H,1,3);
    frame1       = CImg<unsigned char>(W,H,1,3);
    flow0        = CImg<float>(        W,H,1,2);
    flow1        = CImg<float>(        W,H,1,2);
    index_image0 = CImg<size_t>(       W,H,1,1);
    index_image1 = CImg<size_t>(       W,H,1,1);
    
    reset();
  }

  void RenderCore::reset()
  {
    frame0.fill(static_cast<unsigned char>(0));
    frame1.fill(static_cast<unsigned char>(0));
    flow0.fill(0.f);
    flow1.fill(0.f);
    index_image0.fill(static_cast<size_t>(0));
    index_image1.fill(static_cast<size_t>(0));
  }

  bool RenderCore::blitObject(const MovingObjectBase& obj, bool use_AA)
  {
    /// Use non-AA masks to draw indices
    cimg_forXY(index_image0,x,y){
      if (obj.m_masks_noAA[0][y*W+x] == 255){
        index_image0(x,y) = obj.ID;
      }
    }
    cimg_forXY(index_image1,x,y){
      if (obj.m_masks_noAA[1][y*W+x] == 255){
        index_image1(x,y) = obj.ID;
      }
    }

    /// Use AA masks to draw images
    {
      CImg<unsigned char> tmp((use_AA ? obj.m_masks_AA[0] : obj.m_masks_noAA[0]),
                              1,W,H,1,true);
      CImg<unsigned char> tmp2 = tmp.get_permute_axes("YZCX");
      try {
        frame0.draw_image(0,0, obj.m_textures[1], tmp2, 1, 255);
      } catch (...) {
        return false;
      }
    }
    {
      CImg<unsigned char> tmp((use_AA ? obj.m_masks_AA[1] : obj.m_masks_noAA[1]),
                              1,W,H,1,true);
      CImg<unsigned char> tmp2 = tmp.get_permute_axes("YZCX");
      try {
        frame1.draw_image(0,0, obj.m_textures[2], tmp2, 1, 255);
      } catch (...) {
        return false;
      }
    }

    return true;
  }

  void RenderCore::computeFlowImage(std::map<size_t, MovingObjectBase*>& objects_map,
                                    bool inverse)
  {
    CImg<float>& flow_image = (inverse ? flow1 : flow0);
    CImg<size_t>& index_image = (inverse ? index_image1 : index_image0);

    cimg_forXY(flow_image,x,y){
      size_t idx = index_image(x,y);
      if (idx == 0) continue;

      float xf = x;
      float yf = y;
      objects_map[idx]->getPointFlow(&xf, &yf, inverse);

      flow_image(x,y,0) = xf;
      flow_image(x,y,1) = yf;
    }
  }



  ///
  /// FlyingChairsRandom
  ///

  namespace FlyingChairsRandom
  {
    float baseGauss(float a, float b, float input, float normalize) {
      float sample{input * ((b+a)/2.f-a) / normalize + (b+a)/2.f};
      return ((a<=sample and sample<=b) ? sample : (b+a)/2.);
    }


    template <class RNG_t>
    Trigger<RNG_t>::Trigger(float threshold, RNG_t RNG)
      : m_threshold(threshold), 
        m_RNG(std::move(RNG)) 
    { }

    template <class RNG_t>
    Trigger<RNG_t>::Trigger(float a, float b, float threshold, int seed)
      : m_threshold(threshold), 
        m_RNG(a,b,seed)
    { }

    template <class RNG_t>
    bool Trigger<RNG_t>::operator() () {
      return (m_RNG() < m_threshold);
    }


    template<typename T>
    Choice<T>::Choice(std::vector<T>&& options, int seed)
      : m_options(options),
        m_RNG(0, m_options.size()-1, seed)
    { }

    template<typename T>
    T Choice<T>::operator() () {
      return m_options[m_RNG()];
    }


    Uniform::Uniform(float a, float b, int seed) 
      : m_rng(a,b,seed) 
    { }

    float Uniform::operator()() {
      return m_rng();
    }


    Gaussian::Gaussian(float a, float b, int seed) 
      : a(a), b(b), m_rng(0, 1, seed) 
    { }

    float Gaussian::operator()() {
      return baseGauss(a, b, m_rng(), 3);
    }


    GaussianSq::GaussianSq(float a, float b, int seed) 
      : a(a), b(b), m_rng(0, 1, seed) 
    { }

    float GaussianSq::operator()() {
      float tmp = m_rng();
      tmp = ((tmp > 0) ? std::pow(tmp, 2) : -std::pow(tmp, 2));
      return baseGauss(a, b, tmp, 6);
    }


    Gaussian3::Gaussian3(float a, float b, int seed) 
      : a(a), b(b), m_rng(0, 1, seed) 
    { }

    float Gaussian3::operator()() {
      float tmp = std::pow(m_rng(), 3);
      return baseGauss(a, b, tmp, 10);
    }


    Gaussian4::Gaussian4(float a, float b, int seed) 
      : a(a), b(b), m_rng(0, 1, seed) 
    { }

    float Gaussian4::operator()() {
      float tmp = m_rng();
      tmp = ((tmp > 0) ? std::pow(tmp, 4) : -std::pow(tmp, 4));
      return baseGauss(a, b, tmp, 15);
    }


    GaussianMeanSigmaRange::GaussianMeanSigmaRange(float a, float b, float mean, float sigma, int seed) 
      : a(a), b(b), mean(mean), sigma(sigma), m_rng(0, 1, seed) 
    { }

    float GaussianMeanSigmaRange::operator()() {
      float tmp = m_rng() * sigma + mean;
      return (((a <= tmp) and (tmp <= b)) ? tmp : mean);
    }
  }  /// namespace FlyingChairsRandom



  ///
  /// ObjectBlueprint
  ///
  
  ObjectBlueprint::ObjectBlueprint()
   : obj_type(ObjType_t::Dummy)
  { }

  ObjectBlueprint::~ObjectBlueprint()
  {
    for (unsigned int i = 0; i < composite_component_blueprint_ptrs.size(); ++i) {
      if (composite_component_blueprint_ptrs[i])
        delete composite_component_blueprint_ptrs[i];
    }
  }


  
  ///
  /// TaskBucket
  ///
  TaskBucket::TaskBucket()
    : task_ID(0),
      bad(false),
      background_blueprint(nullptr),
      result_image0_ptr(nullptr),
      result_image1_ptr(nullptr),
      result_flow0_ptr(nullptr)
  { }

  //void TaskBucket::resetBucket(unsigned int next_task_ID) {
  //  task_ID = next_task_ID;
  //  bad = false;

  //  if (background_blueprint)
  //    delete background_blueprint;
  //  background_blueprint = nullptr;
  //  for (unsigned int i = 0; i < object_blueprints.size(); ++i)
  //    if (object_blueprints[i])
  //      delete object_blueprints[i];
  //  object_blueprints.clear();
  //}


  
  ///
  /// Sample
  ///
  void Sample::Destroy() {
    if (image0_ptr) delete image0_ptr;
    if (image1_ptr) delete image1_ptr;
    if (flow0_ptr) delete flow0_ptr;

    image0_ptr = nullptr;
    image1_ptr = nullptr;
    flow0_ptr = nullptr;
  };



  ///
  /// DataGenerator
  ///

  DataGenerator::DataGenerator(const caffe::LayerParameter& param)
    : m_generator_is_running(false),
      m_random_textures(param.data_generation_param().texture_dbases(0)),
      m_batch_size(1),
      m_first_level_threads(param.data_generation_param().first_level_threads()),
      m_second_level_threads(param.data_generation_param().second_level_threads()),
      m_use_AA(param.data_generation_param().use_antialiasing()),
      MODE(param.data_generation_param().mode())
  { }

  DataGenerator::~DataGenerator()
  {
    Stop(false);
  }

  void DataGenerator::setBatchSize(int batch_size)
  {
    m_batch_size = batch_size;
  }

  void DataGenerator::Start()
  {
    if (m_generator_is_running)
      return;

    /// Start WarpFields generator
    if (MODE == 9) {
      std::cout << "Starting CropGenerator..." << std::endl;
      m_crop_generator_ptr = new WarpFields::CropGenerator(DGEN_WIDTH, DGEN_HEIGHT, 2);
      m_crop_generator_ptr->Start();
    }

    /// Spawn workers
    m_worker_threads.resize(m_first_level_threads);
    for (unsigned int i = 0; i < m_worker_threads.size(); ++i) {
      m_worker_threads[i] = new std::thread(&DataGenerator::WorkerThreadLoop,
                                            this);
    }

    Resume();
  }

  void DataGenerator::Stop(bool wait)
  {
    if (not m_generator_is_running)
      return;

    Pause();

    /// Destroy workers
    for (unsigned int i = 0; i < m_worker_threads.size(); ++i) {
      if (m_worker_threads[i]->joinable()) {
        m_worker_threads[i]->join();
        delete m_worker_threads[i];
      }
    }

    /// Destroy WarpFields generator
    if (MODE == 9) {
      std::cout << "Destroying CropGenerator..." << std::endl;
      m_crop_generator_ptr->Stop();
      delete m_crop_generator_ptr;
    }
  }

  void DataGenerator::Pause()
  {
    m_generator_is_running = false;
  }

  void DataGenerator::Resume()
  {
    m_generator_is_running = true;
  }

  MovingObjectBase* DataGenerator::RealizeObjectBlueprint(
              ObjectBlueprint* p, 
              const agg::trans_affine& bg_motion,
              QueueProcessing::QueueProcessor<UnfinishedObjectContainer>& qp,
              MovingObjectBase* parent_obj_ptr)
  {
    MovingObjectBase* base_obj_ptr{nullptr};
    switch (p->obj_type) {
      case ObjType_t::Ellipse: {
        MovingObjectEllipse* obj_ptr;
        if (parent_obj_ptr)
          obj_ptr = new MovingObjectComponentEllipse();
        else
          obj_ptr = new MovingObjectEllipse(p->obj_id);

        obj_ptr->setEllipse(0,0,p->ellipse_scale_x,p->ellipse_scale_y,100);
        base_obj_ptr = obj_ptr;
        break;
      }
      case ObjType_t::Polygon: {
        MovingObjectPolygon* obj_ptr;
        if (parent_obj_ptr)
          obj_ptr = new MovingObjectComponentPolygon();
        else
          obj_ptr = new MovingObjectPolygon(p->obj_id);

        obj_ptr->resetPath(p->polygon_segment_x[0],
                           p->polygon_segment_y[0]);
        for (unsigned int i = 1; i < p->polygon_segment_types.size(); ++i) {
          switch (p->polygon_segment_types[i]) {
            case PolySegmentType_t::Dummy: {
              throw std::runtime_error("PolySegmentType_t::Dummy found, this should have been skipped!");
              break;
            }
            case PolySegmentType_t::Line: {
              obj_ptr->lineTo(p->polygon_segment_x[i],
                              p->polygon_segment_y[i]);
              break;
            }
            case PolySegmentType_t::Curve3: {
              obj_ptr->curve3To(p->polygon_segment_x[i],
                                p->polygon_segment_y[i],
                                p->polygon_segment_x[i+1],
                                p->polygon_segment_y[i+1]);
              ++i;
              break;
            }
          }
        }
        obj_ptr->finalizePath();
        base_obj_ptr = obj_ptr;
        break;
      }
      case ObjType_t::Composite: {
        MovingObjectComposite* obj_ptr = new MovingObjectComposite(p->obj_id);
        if (MODE==9 and p->do_warpfield_deformation) {
          /// This is ugly: Components of a composite object must be able to
          /// access their parent object's warpflows, but they are realized
          /// before the normal control flow assigns the parent's flows. We
          /// must initialize the parent's flow early here!
          CImg<float> warpflow, warpiflow;
          std::tie(warpflow, warpiflow) = m_crop_generator_ptr->get_crop();
          obj_ptr->setExtraWarpFields(warpflow, warpiflow);
        }
        for (unsigned int component_idx = 0; 
             component_idx < p->composite_component_blueprint_ptrs.size();
             ++component_idx) {
          ObjectBlueprint* component_p = p->composite_component_blueprint_ptrs[component_idx];
          MovingObjectBase* component_obj = RealizeObjectBlueprint(component_p, bg_motion, qp, obj_ptr);
          if (component_p->is_additive_component)
            obj_ptr->addComponent(component_obj);
          else
            obj_ptr->subtractComponent(component_obj);
        }
        base_obj_ptr = obj_ptr;
        break;
      }
      default: {
        throw std::runtime_error("(RealizeObjectBlueprint) Bad object type, or not intended in this mode");
      }
    }
    if (parent_obj_ptr) {
      base_obj_ptr->m_parent_obj_ptr = parent_obj_ptr;
    }
    base_obj_ptr->setRawTexture(m_random_textures.getTexturePtr(p->tex_id)
                                                 ->getRandomizedCrop());
    base_obj_ptr->setIntrinsicTransform(p->init_rot,
                                        p->init_trans_x,
                                        p->init_trans_y);
    base_obj_ptr->setMotion(p->rot, p->scale, p->trans_x, p->trans_y);
    base_obj_ptr->addBackgroundMotion(bg_motion);

    if (MODE==9 and p->do_warpfield_deformation) {
      if (base_obj_ptr->m_parent_obj_ptr) {
        CImg<float> warpflow(base_obj_ptr->m_parent_obj_ptr
                                         ->m_extra_warp_field);
        CImg<float> warpiflow(base_obj_ptr->m_parent_obj_ptr
                                          ->m_extra_warp_field_inverse);
        base_obj_ptr->setExtraWarpFields(warpflow, warpiflow);
      } else if (not base_obj_ptr->m_has_extra_warp_fields) {
        CImg<float> warpflow, warpiflow;
        std::tie(warpflow, warpiflow) = m_crop_generator_ptr->get_crop();
        base_obj_ptr->setExtraWarpFields(warpflow, warpiflow);
      }
    }

    qp.Give(UnfinishedObjectContainer{base_obj_ptr});
    return base_obj_ptr;
  }

  void DataGenerator::Process_TaskBucket(
            TaskBucket* const task_ptr,
            RenderCore& rendercore,
            std::map<size_t, MovingObjectBase*>& objects_map,
            QueueProcessing::QueueProcessor<UnfinishedObjectContainer>& qp)
  {
    /// Create and finalize objects
    /// Create background (not using the RealizeObjectBlueprint function)
    ObjectBlueprint* p = task_ptr->background_blueprint;
    MovingObjectBackground* bg_obj_ptr = new MovingObjectBackground(p->obj_id);
    {
      bg_obj_ptr->setRawTexture(m_random_textures.getTexturePtr(p->tex_id)
                                                 ->getRandomizedCrop(
                                                         2*W,2*H,
                                                         p->tex_rot,
                                                         p->tex_scale,
                                                         p->tex_shift_x,
                                                         p->tex_shift_y));
      bg_obj_ptr->setMotion(p->rot, p->scale, p->trans_x, p->trans_y);
      if (MODE==9 and p->do_warpfield_deformation) {
        CImg<float> warpflow, warpiflow;
        std::tie(warpflow, warpiflow) = m_crop_generator_ptr->get_crop();
        warpflow.resize( 2*W,2*H,-100,-100,3);
        warpiflow.resize(2*W,2*H,-100,-100,3);
        warpflow *= 2.;
        warpiflow *= 2.;
        bg_obj_ptr->setExtraWarpFields(warpflow, warpiflow);
      }
      objects_map[bg_obj_ptr->ID] = bg_obj_ptr;
      qp.Give(UnfinishedObjectContainer{bg_obj_ptr});
    }
    /// Create and finalize objects
    for (unsigned int i = 0; i < task_ptr->object_blueprints.size(); ++i) {
      p = task_ptr->object_blueprints[i];
      MovingObjectBase* base_obj_ptr = RealizeObjectBlueprint(p, bg_obj_ptr->m_motion, qp);
      objects_map[base_obj_ptr->ID] = base_obj_ptr;
    }

    qp.Finish();

    /// Render
    for (std::map<size_t,MovingObjectBase*>::iterator it = objects_map.begin();
         it != objects_map.end();
         ++it) {
      if (not rendercore.blitObject(*it->second, m_use_AA)) {
        task_ptr->bad = true;
        break;
      }
    }

    /// Assemble flow image(s)
    rendercore.computeFlowImage(objects_map, false);

    /// Copy results
    task_ptr->result_image0_ptr = new CImg<float>(rendercore.frame0.width(),
                                                  rendercore.frame0.height(),
                                                  1, 3);
    task_ptr->result_image1_ptr = new CImg<float>(rendercore.frame0.width(),
                                                  rendercore.frame0.height(),
                                                  1, 3);
    for (unsigned int c = 0; c < 3; ++c) {
      for (unsigned int y = 0; y < rendercore.frame0.height(); ++y) {
        for (unsigned int x = 0; x < rendercore.frame0.width(); ++x) {
          task_ptr->result_image0_ptr->operator()(x,y,0,c) =
                static_cast<float>(rendercore.frame0(x,y,0,c));
          task_ptr->result_image1_ptr->operator()(x,y,0,c) =
                static_cast<float>(rendercore.frame1(x,y,0,c));
        }
      }
    }
    task_ptr->result_flow0_ptr  = new CImg<float>(rendercore.flow0);
    
    /// Clean up
    for (std::map<size_t,MovingObjectBase*>::iterator it = objects_map.begin();
         it != objects_map.end();
         ++it) {
      delete it->second;
    }
    objects_map.clear();
  }

  void DataGenerator::WorkerThreadLoop()
  {
    /// Setup infrastructure
    RenderCore rendercore;
    std::map<size_t, MovingObjectBase*> objects_map;
    QueueProcessing::QueueProcessor<UnfinishedObjectContainer> qp(
          Process_UnfinishedObjectContainer, false, true,
          m_second_level_threads, true);
    qp.SetMaxQueueLength(50).Start();

    while (m_generator_is_running) {
      if (m_undone_tasks.size() > 0) {
        /// Fetch a new task
        TaskBucket* task_ptr;
        {
          std::lock_guard<std::mutex> LOCK(m_task_queues__LOCK);
          if (m_undone_tasks.size() == 0)
            continue;
          task_ptr = m_undone_tasks.front();
          m_task_IDs_in_flight.push_back(task_ptr->task_ID);
          m_undone_tasks.pop();
        }

        /// Process the task
        Process_TaskBucket(task_ptr, rendercore, objects_map, qp);

        /// Mark task as done
        {
          std::lock_guard<std::mutex> LOCK(m_task_queues__LOCK);
          if (task_ptr->bad) {
            if (task_ptr->result_image0_ptr)
              delete task_ptr->result_image0_ptr;
            if (task_ptr->result_image1_ptr)
              delete task_ptr->result_image1_ptr;
            if (task_ptr->result_flow0_ptr)
              delete task_ptr->result_flow0_ptr;
            delete task_ptr;
          } else {
            m_finished_tasks.push(task_ptr);
          }
          /// (Erase-remove idiom: http://stackoverflow.com/a/3385251)
          m_task_IDs_in_flight.erase(std::remove(m_task_IDs_in_flight.begin(),
                                                 m_task_IDs_in_flight.end(),
                                                 task_ptr->task_ID),
                                     m_task_IDs_in_flight.end());
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void DataGenerator::commissionNewTask(TaskBucket* new_task_ptr)
  {
    std::lock_guard<std::mutex> LOCK(m_task_queues__LOCK);
    m_undone_tasks.push(new_task_ptr);
  }

  bool DataGenerator::hasUnfinishedTasks() const
  {
    return (m_undone_tasks.size() > 0 or m_task_IDs_in_flight.size() > 0);
  }

  bool DataGenerator::hasRetrievableFinishedTasks() const
  {
    return (m_finished_tasks.size() > 0);
  }

  Sample DataGenerator::retrieveFinishedTask()
  {
    Sample sample;
    bool successful = false;
    while (not successful) {
      while (not hasRetrievableFinishedTasks())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

      {
        std::lock_guard<std::mutex> LOCK(m_task_queues__LOCK);
        if (not hasRetrievableFinishedTasks())
          continue;
        TaskBucket* done_task_ptr = m_finished_tasks.front();
        sample = Sample{done_task_ptr->result_image0_ptr,
                        done_task_ptr->result_image1_ptr,
                        done_task_ptr->result_flow0_ptr};
        done_task_ptr->result_image0_ptr = nullptr;
        done_task_ptr->result_image1_ptr = nullptr;
        done_task_ptr->result_flow0_ptr = nullptr;
        delete done_task_ptr;
        m_finished_tasks.pop();
        successful = true;
      }
    }
    return sample;
  }



  ///
  /// ObjectParametersGenerator
  ///
  using namespace RNG;

  ObjectParametersGenerator::ObjectParametersGenerator(const caffe::LayerParameter& param)
  : MODE(param.data_generation_param().mode()),
    RNG_SEED(0),
    int_max(std::numeric_limits<int>::max())
  {
    switch (MODE) {
      case 1: {  /// Spongebob (only axis-aligned boxes)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg rot
        RNG_BgRot                     = new GaussianSq            (0, 0,                              RNG_SEED++); // no bg rot
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg scale
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (1, 1,                              RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Polygon},              RNG_SEED++); // only polygon objs
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (0, 0,                              RNG_SEED++); // no init obj rot
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj rot
        RNG_ObjRot                    = new GaussianSq            (0, 0,                              RNG_SEED++); // no obj rot
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj scale
        RNG_ObjScale                  = new GaussianSq            (1, 1,                              RNG_SEED++); // no obj scale
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 2: {  /// Patrick (only polygons with straight lines)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg rot
        RNG_BgRot                     = new GaussianSq            (0, 0,                              RNG_SEED++); // no bg rot
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg scale
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (1, 1,                              RNG_SEED++); // no bg scale
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Polygon},              RNG_SEED++); // only polygon objs
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj rot
        RNG_ObjRot                    = new GaussianSq            (0, 0,                              RNG_SEED++); // no obj rot
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj scale
        RNG_ObjScale                  = new GaussianSq            (1, 1,                              RNG_SEED++); // no obj scale
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 3: {  /// Sandy (only round shapes)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg rot
        RNG_BgRot                     = new GaussianSq            (0, 0,                              RNG_SEED++); // no bg rot
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg scale
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (1, 1,                              RNG_SEED++); // no bg scale
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse},              RNG_SEED++); // only ellipse objs
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj rot
        RNG_ObjRot                    = new GaussianSq            (0, 0,                              RNG_SEED++); // no obj rot
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj scale
        RNG_ObjScale                  = new GaussianSq            (1, 1,                              RNG_SEED++); // no obj scale
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 4: {  /// Spongebob+Patrick+Sandy + rotations
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.3,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg scale
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (1, 1,                              RNG_SEED++); // no bg scale
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, ObjType_t::Polygon}, RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj scale
        RNG_ObjScale                  = new GaussianSq            (1, 1,                              RNG_SEED++); // no obj scale
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 5: {  /// 4 + scaling
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.3,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.6,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.93, 1.07,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, ObjType_t::Polygon}, RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.8, 1.2,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 6: {  /// 5 + composite objects
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.3,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.6,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.93, 1.07,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite},            RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.8, 1.2,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 7: {  /// 6 + thin objects
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.3,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.6,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.93, 1.07,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite,},           RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.8, 1.2,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 8: {  /// Spongebob+Patrick+Sandy (case 4 without rotations)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg rot
        RNG_BgRot                     = new GaussianSq            (0, 0,                              RNG_SEED++); // no bg rot
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no bg scale
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (1, 1,                              RNG_SEED++); // no bg scale
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, ObjType_t::Polygon}, RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj rot
        RNG_ObjRot                    = new GaussianSq            (0, 0,                              RNG_SEED++); // no obj rot
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 0, 1,                           RNG_SEED++); // no obj scale
        RNG_ObjScale                  = new GaussianSq            (1, 1,                              RNG_SEED++); // no obj scale
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 9: {  /// 7 + nonrigid deformations
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.3,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.6,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.93, 1.07,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite,},           RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.8, 1.2,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 10: {  /// 7 with a different displacement magnitude histogram (all motions HALVED)
                  /// for cases 10-14, TRIGGERs are adjusted by halving/doubling/etc the RATIO of affected:unaffected cases
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.176,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-5*agg::pi/180., 5*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-20, 20,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-20, 20,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.429,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.965, 1.035,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite,},           RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-60, 60,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-60, 60,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.539,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-15*agg::pi/180., 15*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.539,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.9, 1.1,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 11: {  /// 7 with a different displacement magnitude histogram (all motions DOUBLED)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.462,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-20*agg::pi/180., 20*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-80, 80,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-80, 80,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.75,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.86, 1.14,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite,},           RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-240, 240,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-240, 240,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.824,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-60*agg::pi/180., 60*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.824,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.6, 1.4,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 12: {  /// 7 with a different displacement magnitude histogram (all motions THIRDED)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.125,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-3.3*agg::pi/180., 3.3*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-13.3, 13.3,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-13.3, 13.3,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.333,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.976, 1.023,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite,},           RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-40, 40,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-40, 40,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.437,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.437,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.933, 1.066,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      case 13: {  /// 7 with a different displacement magnitude histogram (all motions TRIPLED)
        RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
        RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
        RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.563,                         RNG_SEED++);
        RNG_BgRot                     = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
        RNG_BgTransX                  = new Gaussian4             (-120, 120,                           RNG_SEED++);
        RNG_BgTransY                  = new Gaussian4             (-120, 120,                           RNG_SEED++);
        RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.818,                         RNG_SEED++);
        RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
        RNG_BgScale                   = new GaussianSq            (0.79, 1.21,                        RNG_SEED++);
        RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
        RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, 
                                                                    ObjType_t::Polygon,
                                                                    ObjType_t::Composite,},           RNG_SEED++);
        RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
        RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
        RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
        RNG_ObjTransX                 = new Gaussian3             (-360, 360,                         RNG_SEED++);
        RNG_ObjTransY                 = new Gaussian3             (-360, 360,                         RNG_SEED++);
        RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.875,                         RNG_SEED++);
        RNG_ObjRot                    = new GaussianSq            (-90*agg::pi/180., 90*agg::pi/180., RNG_SEED++);
        RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
        RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.875,                         RNG_SEED++);
        RNG_ObjScale                  = new GaussianSq            (0.4, 1.6,                          RNG_SEED++);
        RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
        RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
        RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
        RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
        RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
        RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
        RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
        RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
        RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
        RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
        RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        RNG_ComponentOffset           = new Uniform               (-20, 20,                           RNG_SEED++);
        RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2,                         RNG_SEED++);
        RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
        RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
        RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
        break;
      }
      default: {
        throw std::runtime_error("BAD MODE");
      }
      //case 1: {
      //  RNG_BgTexID                   = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
      //  RNG_BgInitRot                 = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
      //  RNG_BgInitTransX              = new Choice<int>           ({0,W},                             RNG_SEED++);
      //  RNG_BgInitTransY              = new Choice<int>           ({0,H},                             RNG_SEED++);
      //  RNG_BgRotTrigger              = new Trigger<Uniform>      (0, 1, 0.3,                         RNG_SEED++);
      //  RNG_BgRot                     = new GaussianSq            (-10*agg::pi/180., 10*agg::pi/180., RNG_SEED++);
      //  RNG_BgTransX                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
      //  RNG_BgTransY                  = new Gaussian4             (-40, 40,                           RNG_SEED++);
      //  RNG_BgScaleTrigger            = new Trigger<Uniform>      (0, 1, 0.6,                         RNG_SEED++);
      //  RNG_BgInitScale               = new Uniform               (0.8, 1.2,                          RNG_SEED++);
      //  RNG_BgScale                   = new GaussianSq            (0.93, 1.07,                        RNG_SEED++);
      //  RNG_NumberOfFgObjects         = new Uniform               (16, 24,                            RNG_SEED++);
      //  RNG_ObjType                   = new Choice<ObjType_t>     ({ObjType_t::Ellipse, ObjType_t::Polygon}, RNG_SEED++);
      //  RNG_ObjTexID                  = new FixedRangeUniformInt  (0, int_max,                        RNG_SEED++);
      //  RNG_ObjInitTransX             = new Uniform               (-W/2.-50, W*3./2.+50,              RNG_SEED++);
      //  RNG_ObjInitTransY             = new Uniform               (-H/2.-50, H*3./2.+50,              RNG_SEED++);
      //  RNG_ObjTransX                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
      //  RNG_ObjTransY                 = new Gaussian3             (-120, 120,                         RNG_SEED++);
      //  RNG_ObjInitRot                = new Uniform               (-agg::pi, agg::pi,                 RNG_SEED++);
      //  RNG_ObjRotTrigger             = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
      //  RNG_ObjRot                    = new GaussianSq            (-30*agg::pi/180., 30*agg::pi/180., RNG_SEED++);
      //  RNG_ObjInitScale              = new GaussianMeanSigmaRange(0.2, 2.5, 0.8, 0.8,                RNG_SEED++);
      //  RNG_ObjScaleTrigger           = new Trigger<Uniform>      (0, 1, 0.7,                         RNG_SEED++);
      //  RNG_ObjScale                  = new GaussianSq            (0.8, 1.2,                          RNG_SEED++);
      //  RNG_ObjTexShiftX              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
      //  RNG_ObjTexShiftY              = new FixedRangeUniformInt  (-W/2, W/2,                         RNG_SEED++);
      //  RNG_ObjTexRot                 = new FixedRangeUniformFloat(-agg::pi, agg::pi,                 RNG_SEED++);
      //  RNG_ObjTexZoom                = new FixedRangeUniformFloat(0.5, 2.0,                          RNG_SEED++);
      //  RNG_ElliObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
      //  RNG_ElliObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
      //  RNG_PolyObj_spokes            = new FixedRangeUniformInt  (3, 20,                             RNG_SEED++);
      //  RNG_PolyObj_dphi              = new Uniform               (-10, 10,                           RNG_SEED++);
      //  RNG_PolyObj_r                 = new Uniform               (20, 80,                            RNG_SEED++);
      //  RNG_PolyObj_ScaleX            = new Uniform               (0.5, 2,                            RNG_SEED++);
      //  RNG_PolyObj_ScaleY            = new Uniform               (0.5, 2,                            RNG_SEED++);
      //  RNG_PolyObj_CurveTrigger      = new Trigger<Uniform>      (0, 1, 0.33,                        RNG_SEED++);
      //  RNG_CompObjInitTransX         = new Uniform               (-15, 15,                           RNG_SEED++);
      //  RNG_CompObjInitTransY         = new Uniform               (-15, 15,                           RNG_SEED++);
      //  RNG_CompObiNumberOfComponents = new FixedRangeUniformInt  (1, 7,                              RNG_SEED++);
      //  RNG_ComponentIsAdditive       = new Trigger<Uniform>      (0, 1, 0.5,                         RNG_SEED++);
      //  RNG_ObjIsExtraThin            = new Trigger<Uniform>      (0, 1, 0.2                          RNG_SEED++);
      //  RNG_ObjDeformsNonrigidly      = new Trigger<Uniform>      (0, 1, 0,                           RNG_SEED++);
      //  RNG_GenericUniform            = new Uniform               (0, 1,                              RNG_SEED++);
      //  RNG_GenericTrigger            = new Trigger<Uniform>      (0, 1, 0.5                          RNG_SEED++);
      //  break;
      //}
    }
  }

  ObjectParametersGenerator::~ObjectParametersGenerator()
  {
    delete RNG_BgTexID;
    delete RNG_BgInitRot;
    delete RNG_BgInitTransX;
    delete RNG_BgInitTransY;
    delete RNG_BgRotTrigger;
    delete RNG_BgRot;
    delete RNG_BgTransX;
    delete RNG_BgTransY;
    delete RNG_BgScaleTrigger;
    delete RNG_BgInitScale;
    delete RNG_BgScale;
    delete RNG_NumberOfFgObjects;
    delete RNG_ObjType;
    delete RNG_ObjTexID;
    delete RNG_ObjInitTransX;
    delete RNG_ObjInitTransY;
    delete RNG_ObjTransX;
    delete RNG_ObjTransY;
    delete RNG_ObjInitRot;
    delete RNG_ObjRotTrigger;
    delete RNG_ObjRot;
    delete RNG_ObjInitScale;
    delete RNG_ObjScaleTrigger;
    delete RNG_ObjScale;
    delete RNG_ObjTexShiftX;
    delete RNG_ObjTexShiftY;
    delete RNG_ObjTexRot;
    delete RNG_ObjTexZoom;
    delete RNG_ElliObj_ScaleX;
    delete RNG_ElliObj_ScaleY;
    delete RNG_PolyObj_spokes;
    delete RNG_PolyObj_dphi;
    delete RNG_PolyObj_r;
    delete RNG_PolyObj_ScaleX;
    delete RNG_PolyObj_ScaleY;
    delete RNG_PolyObj_CurveTrigger;
    delete RNG_CompObjInitTransX;
    delete RNG_CompObjInitTransY;
    delete RNG_CompObiNumberOfComponents;
    delete RNG_ComponentIsAdditive;
    delete RNG_ComponentOffset;
    delete RNG_ObjIsExtraThin;
    delete RNG_ObjDeformsNonrigidly;
    delete RNG_GenericUniform;
    delete RNG_GenericTrigger;
  }

  void ObjectParametersGenerator::generateBackground(ObjectBlueprint* b)
  {
    switch (MODE) {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
      case 13: {
        /// Object motion
        b->rot          = ((*RNG_BgRotTrigger)() ? (*RNG_BgRot)() : 0.);
        b->scale        = ((*RNG_BgScaleTrigger)() ? (*RNG_BgScale)() : 1.) ;
        float pre_transx = (*RNG_BgTransX)();
        float pre_transy = (*RNG_BgTransY)();
        b->trans_x      =  std::cos(-b->rot)*pre_transx
                          -std::sin(-b->rot)*pre_transy;
        b->trans_y      =  std::sin(-b->rot)*pre_transx
                          +std::cos(-b->rot)*pre_transy;
        /// Texture stuff (= intrinsics in case of the background object)
        b->tex_id       = (*RNG_BgTexID)();
        b->tex_rot      = (*RNG_BgInitRot)();
        b->tex_scale    = (*RNG_BgInitScale)();
        b->tex_shift_x  = (*RNG_BgInitTransX)();
        b->tex_shift_y  = (*RNG_BgInitTransY)();
        b->do_warpfield_deformation = (*RNG_ObjDeformsNonrigidly)();
        break;
      }
      default: {
        throw std::runtime_error("BAD MODE");
      }
    }
  }

  void ObjectParametersGenerator::generateForegroundObject(ObjectBlueprint* b) 
  {
    switch (MODE) {
      case 1: {
        b->obj_type     = (*RNG_ObjType)();
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();

        switch (b->obj_type) {
          case ObjType_t::Polygon: {
            /// Polygon object specifics
            const unsigned int spokes = 4;
            const float radius = (*RNG_PolyObj_r)();
            const float xscale = radius*(*RNG_PolyObj_ScaleX)();
            const float yscale = radius*(*RNG_PolyObj_ScaleY)();
            b->polygon_segment_x.resize(spokes);
            b->polygon_segment_y.resize(spokes);
            b->polygon_segment_x[0] =  xscale;
            b->polygon_segment_x[1] =  xscale;
            b->polygon_segment_x[2] = -xscale;
            b->polygon_segment_x[3] = -xscale;
            b->polygon_segment_y[0] = -yscale;
            b->polygon_segment_y[1] =  yscale;
            b->polygon_segment_y[2] =  yscale;
            b->polygon_segment_y[3] = -yscale;
            b->polygon_segment_types.resize(spokes);
            b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
            for (unsigned int i = 1; i < spokes; ++i)
              b->polygon_segment_types[i] = PolySegmentType_t::Line;
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/1) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      case 2: {
        b->obj_type     = (*RNG_ObjType)();
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();

        switch (b->obj_type) {
          case ObjType_t::Polygon: {
            /// Polygon object specifics
            const unsigned int spokes =
                  static_cast<unsigned int>((*RNG_PolyObj_spokes)());
            std::vector<float> phi(spokes);
            std::vector<float> r(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              phi[i] = (i*360./spokes + (*RNG_PolyObj_dphi)())*agg::pi/180.;
              r[i]   = (*RNG_PolyObj_r)();
            }
            const float xscale = (*RNG_PolyObj_ScaleX)();
            const float yscale = (*RNG_PolyObj_ScaleY)();
            b->polygon_segment_x.resize(spokes);
            b->polygon_segment_y.resize(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              b->polygon_segment_x[i] = xscale*r[i]*std::cos(phi[i]);
              b->polygon_segment_y[i] = yscale*r[i]*std::sin(phi[i]);
            }
            b->polygon_segment_types.resize(spokes);
            b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
            for (unsigned int i = 1; i < spokes; ++i) {
              b->polygon_segment_types[i] = PolySegmentType_t::Line;
            }
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/2) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      case 3: {
        b->obj_type     = (*RNG_ObjType)();
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();

        switch (b->obj_type) {
          case ObjType_t::Ellipse: {
            /// Ellipse object specifics
            b->ellipse_scale_x = (*RNG_ElliObj_ScaleX)()*50;
            b->ellipse_scale_y = (*RNG_ElliObj_ScaleY)()*50;
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/3) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      case 4:
      case 5:
      case 8: {
        b->obj_type     = (*RNG_ObjType)();
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();

        switch (b->obj_type) {
          case ObjType_t::Ellipse: {
            /// Ellipse object specifics
            b->ellipse_scale_x = (*RNG_ElliObj_ScaleX)()*50;
            b->ellipse_scale_y = (*RNG_ElliObj_ScaleY)()*50;
            break;
          }
          case ObjType_t::Polygon: {
            /// Polygon object specifics
            const unsigned int spokes =
                  static_cast<unsigned int>((*RNG_PolyObj_spokes)());
            std::vector<float> phi(spokes);
            std::vector<float> r(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              phi[i] = (i*360./spokes + (*RNG_PolyObj_dphi)())*agg::pi/180.;
              r[i]   = (*RNG_PolyObj_r)();
            }
            const float xscale = (*RNG_PolyObj_ScaleX)();
            const float yscale = (*RNG_PolyObj_ScaleY)();
            b->polygon_segment_x.resize(spokes);
            b->polygon_segment_y.resize(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              b->polygon_segment_x[i] = xscale*r[i]*std::cos(phi[i]);
              b->polygon_segment_y[i] = yscale*r[i]*std::sin(phi[i]);
            }
            b->polygon_segment_types.resize(spokes);
            b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
            for (unsigned int i = 1; i < spokes; ++i) {
              if ((i < spokes-1) and (*RNG_PolyObj_CurveTrigger)()) {
                b->polygon_segment_types[i] = PolySegmentType_t::Curve3;
                b->polygon_segment_types[i+1] = PolySegmentType_t::Dummy;
                ++i;
              } else {
                b->polygon_segment_types[i] = PolySegmentType_t::Line;
              }
            }
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/4+5+8) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      case 6: {
        /// A component part is pre-marked as "Composite"
        if (b->obj_type == ObjType_t::Composite) {
          do {
            b->obj_type = (*RNG_ObjType)();
          } while (b->obj_type == ObjType_t::Composite);
        } else {
          b->obj_type = (*RNG_ObjType)();
        }
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();

        switch (b->obj_type) {
          case ObjType_t::Ellipse: {
            /// Ellipse object specifics
            b->ellipse_scale_x = (*RNG_ElliObj_ScaleX)()*50;
            b->ellipse_scale_y = (*RNG_ElliObj_ScaleY)()*50;
            break;
          }
          case ObjType_t::Polygon: {
            /// Polygon object specifics
            const unsigned int spokes =
                  static_cast<unsigned int>((*RNG_PolyObj_spokes)());
            std::vector<float> phi(spokes);
            std::vector<float> r(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              phi[i] = (i*360./spokes + (*RNG_PolyObj_dphi)())*agg::pi/180.;
              r[i]   = (*RNG_PolyObj_r)();
            }
            const float xscale = (*RNG_PolyObj_ScaleX)();
            const float yscale = (*RNG_PolyObj_ScaleY)();
            b->polygon_segment_x.resize(spokes);
            b->polygon_segment_y.resize(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              b->polygon_segment_x[i] = xscale*r[i]*std::cos(phi[i]);
              b->polygon_segment_y[i] = yscale*r[i]*std::sin(phi[i]);
            }
            b->polygon_segment_types.resize(spokes);
            b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
            for (unsigned int i = 1; i < spokes; ++i) {
              if ((i < spokes-1) and (*RNG_PolyObj_CurveTrigger)()) {
                b->polygon_segment_types[i] = PolySegmentType_t::Curve3;
                b->polygon_segment_types[i+1] = PolySegmentType_t::Dummy;
                ++i;
              } else {
                b->polygon_segment_types[i] = PolySegmentType_t::Line;
              }
            }
            break;
          }
          case ObjType_t::Composite: {
            const unsigned int parts = (*RNG_CompObiNumberOfComponents)();
            for (unsigned int part_idx = 0; part_idx < parts; ++part_idx) {
              ObjectBlueprint* c = new ObjectBlueprint();
              c->obj_type = ObjType_t::Composite;
              /// Prefill values (most will be overwritten below)
              generateForegroundObject(c);
              /// Intrinsic object transform
              c->init_rot     = b->init_rot;
              c->init_trans_x = b->init_trans_x;
              c->init_trans_y = b->init_trans_y;
              /// Object motion
              c->rot          = b->rot;
              c->scale        = b->scale;
              c->trans_x      = b->trans_x;
              c->trans_y      = b->trans_y;

              if (part_idx == 0) {
                c->is_additive_component = true;
              } else {
                c->init_rot      = (*RNG_ObjInitRot)();
                c->init_trans_x += (*RNG_ComponentOffset)();
                c->init_trans_y += (*RNG_ComponentOffset)();
                switch (c->obj_type) {
                  case ObjType_t::Ellipse: {
                    c->ellipse_scale_x *= 0.2;
                    c->ellipse_scale_y *= 0.2;
                    break;
                  }
                  case ObjType_t::Polygon: {
                    for (unsigned int si = 0; si < c->polygon_segment_x.size(); ++si) {
                      c->polygon_segment_x[si] *= 0.2;
                      c->polygon_segment_y[si] *= 0.2;
                    }
                    break;
                  }
                  default: {
                    throw std::runtime_error("Bad component object type");
                  }
                }
                c->is_additive_component = (*RNG_ComponentIsAdditive)();
              }
              b->composite_component_blueprint_ptrs.push_back(c);
            }
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/6) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      case 7:
      case 10:
      case 11: 
      case 12: 
      case 13: {
        /// A component part is pre-marked as "Composite"
        const bool is_component = (b->obj_type == ObjType_t::Composite);
        do {
          b->obj_type = (*RNG_ObjType)();
        } while (is_component and (b->obj_type == ObjType_t::Composite));
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();

        switch (b->obj_type) {
          case ObjType_t::Ellipse: {
            /// Ellipse object specifics
            b->ellipse_scale_x = (*RNG_ElliObj_ScaleX)()*50;
            b->ellipse_scale_y = (*RNG_ElliObj_ScaleY)()*50;
            if (not is_component and (*RNG_ObjIsExtraThin)()) {
              b->ellipse_scale_x *= 0.05;
            }
            break;
          }
          case ObjType_t::Polygon: {
            /// Polygon object specifics
            const unsigned int spokes =
                  static_cast<unsigned int>((*RNG_PolyObj_spokes)());
            std::vector<float> phi(spokes);
            std::vector<float> r(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              phi[i] = (i*360./spokes + (*RNG_PolyObj_dphi)())*agg::pi/180.;
              r[i]   = (*RNG_PolyObj_r)();
            }
            const float xscale = (*RNG_PolyObj_ScaleX)();
            const float yscale = (*RNG_PolyObj_ScaleY)();
            b->polygon_segment_x.resize(spokes);
            b->polygon_segment_y.resize(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              b->polygon_segment_x[i] = xscale*r[i]*std::cos(phi[i]);
              b->polygon_segment_y[i] = yscale*r[i]*std::sin(phi[i]);
            }
            b->polygon_segment_types.resize(spokes);
            b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
            for (unsigned int i = 1; i < spokes; ++i) {
              if ((i < spokes-1) and (*RNG_PolyObj_CurveTrigger)()) {
                b->polygon_segment_types[i] = PolySegmentType_t::Curve3;
                b->polygon_segment_types[i+1] = PolySegmentType_t::Dummy;
                ++i;
              } else {
                b->polygon_segment_types[i] = PolySegmentType_t::Line;
              }
            }
            if (not is_component and (*RNG_ObjIsExtraThin)()) {
              for (unsigned int i = 0; i < b->polygon_segment_x.size(); ++i) {
                b->polygon_segment_x[i] *= 0.05;
              }
            }
            break;
          }
          case ObjType_t::Composite: {
            if ((*RNG_ObjIsExtraThin)()) {
              ObjectBlueprint* c1 = new ObjectBlueprint();
              c1->obj_type = ObjType_t::Composite;
              generateForegroundObject(c1);
              /// Intrinsic object transform
              c1->init_rot     = b->init_rot;
              c1->init_trans_x = b->init_trans_x;
              c1->init_trans_y = b->init_trans_y;
              /// Object motion
              c1->rot          = b->rot;
              c1->scale        = b->scale;
              c1->trans_x      = b->trans_x;
              c1->trans_y      = b->trans_y;
              c1->is_additive_component = true;
              b->composite_component_blueprint_ptrs.push_back(c1);

              ObjectBlueprint* c2 = new ObjectBlueprint(*c1);
              if (c1->obj_type == ObjType_t::Ellipse) {
                if ((*RNG_GenericTrigger)()) {
                  c2->init_trans_x = b->init_trans_x+(*RNG_CompObjInitTransX)();
                  c2->init_trans_y = b->init_trans_y+(*RNG_CompObjInitTransY)();
                } else {
                  c2->init_trans_x = b->init_trans_x;
                  c2->init_trans_y = b->init_trans_y;
                  c2->ellipse_scale_x *= 0.9;
                  c2->ellipse_scale_y *= 0.9;
                }
              } else /*Polygon*/ {
                /// Intrinsic object transform
                c2->init_trans_x = b->init_trans_x;
                c2->init_trans_y = b->init_trans_y;
                for (unsigned int si = 0; si < c2->polygon_segment_x.size(); ++si) {
                  c2->polygon_segment_x[si] *= 0.9;
                  c2->polygon_segment_y[si] *= 0.9;
                }
              }
              /// Intrinsic object transform
              c2->scale        = b->scale;
              /// Object motion
              c2->rot          = b->rot;
              c2->trans_x      = b->trans_x;
              c2->trans_y      = b->trans_y;
              c2->is_additive_component = false;
              b->composite_component_blueprint_ptrs.push_back(c2);
            } else {
              const unsigned int parts = (*RNG_CompObiNumberOfComponents)();
              for (unsigned int part_idx = 0; part_idx < parts; ++part_idx) {
                ObjectBlueprint* c = new ObjectBlueprint();
                c->obj_type = ObjType_t::Composite;
                /// Prefill values (most will be overwritten below)
                generateForegroundObject(c);
                /// Intrinsic object transform
                c->init_rot     = b->init_rot;
                c->init_trans_x = b->init_trans_x;
                c->init_trans_y = b->init_trans_y;
                /// Object motion
                c->rot          = b->rot;
                c->scale        = b->scale;
                c->trans_x      = b->trans_x;
                c->trans_y      = b->trans_y;

                if (part_idx == 0) {
                  c->is_additive_component = true;
                } else {
                  c->init_rot      = (*RNG_ObjInitRot)();
                  c->init_trans_x += (*RNG_ComponentOffset)();
                  c->init_trans_y += (*RNG_ComponentOffset)();
                  switch (c->obj_type) {
                    case ObjType_t::Ellipse: {
                      c->ellipse_scale_x *= 0.2;
                      c->ellipse_scale_y *= 0.2;
                      break;
                    }
                    case ObjType_t::Polygon: {
                      for (unsigned int si = 0; si < c->polygon_segment_x.size(); ++si) {
                        c->polygon_segment_x[si] *= 0.2;
                        c->polygon_segment_y[si] *= 0.2;
                      }
                      break;
                    }
                    default: {
                      throw std::runtime_error("(generateNumberOfFgObjects/7+10-13/badpart) Bad component object type");
                    }
                  }
                  c->is_additive_component = (*RNG_ComponentIsAdditive)();
                }
                b->composite_component_blueprint_ptrs.push_back(c);
              }
            }
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/7+10-13) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      case 9: {
        /// A component part is pre-marked as "Composite"
        const bool is_component = (b->obj_type == ObjType_t::Composite);
        do {
          b->obj_type = (*RNG_ObjType)();
        } while (is_component and (b->obj_type == ObjType_t::Composite));
        /// Intrinsic object transform
        b->init_rot     = (*RNG_ObjInitRot)();
        b->init_trans_x = (*RNG_ObjInitTransX)();
        b->init_trans_y = (*RNG_ObjInitTransY)();
        /// Object motion
        b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
        b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
        b->trans_x      = (*RNG_ObjTransX)();
        b->trans_y      = (*RNG_ObjTransY)();
        /// Texture stuff
        b->tex_id       = (*RNG_ObjTexID)();
        /// Nonrigid deformations
        b->do_warpfield_deformation = (*RNG_ObjDeformsNonrigidly)();

        switch (b->obj_type) {
          case ObjType_t::Ellipse: {
            /// Ellipse object specifics
            b->ellipse_scale_x = (*RNG_ElliObj_ScaleX)()*50;
            b->ellipse_scale_y = (*RNG_ElliObj_ScaleY)()*50;
            if (not is_component and (*RNG_ObjIsExtraThin)()) {
              b->ellipse_scale_x *= 0.05;
            }
            break;
          }
          case ObjType_t::Polygon: {
            /// Polygon object specifics
            const unsigned int spokes =
                  static_cast<unsigned int>((*RNG_PolyObj_spokes)());
            std::vector<float> phi(spokes);
            std::vector<float> r(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              phi[i] = (i*360./spokes + (*RNG_PolyObj_dphi)())*agg::pi/180.;
              r[i]   = (*RNG_PolyObj_r)();
            }
            const float xscale = (*RNG_PolyObj_ScaleX)();
            const float yscale = (*RNG_PolyObj_ScaleY)();
            b->polygon_segment_x.resize(spokes);
            b->polygon_segment_y.resize(spokes);
            for (unsigned int i = 0; i < spokes; ++i) {
              b->polygon_segment_x[i] = xscale*r[i]*std::cos(phi[i]);
              b->polygon_segment_y[i] = yscale*r[i]*std::sin(phi[i]);
            }
            b->polygon_segment_types.resize(spokes);
            b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
            for (unsigned int i = 1; i < spokes; ++i) {
              if ((i < spokes-1) and (*RNG_PolyObj_CurveTrigger)()) {
                b->polygon_segment_types[i] = PolySegmentType_t::Curve3;
                b->polygon_segment_types[i+1] = PolySegmentType_t::Dummy;
                ++i;
              } else {
                b->polygon_segment_types[i] = PolySegmentType_t::Line;
              }
            }
            if (not is_component and (*RNG_ObjIsExtraThin)()) {
              for (unsigned int i = 0; i < b->polygon_segment_x.size(); ++i) {
                b->polygon_segment_x[i] *= 0.05;
              }
            }
            break;
          }
          case ObjType_t::Composite: {
            if ((*RNG_ObjIsExtraThin)()) {
              ObjectBlueprint* c1 = new ObjectBlueprint();
              c1->obj_type = ObjType_t::Composite;
              generateForegroundObject(c1);
              /// Intrinsic object transform
              c1->init_rot     = b->init_rot;
              c1->init_trans_x = b->init_trans_x;
              c1->init_trans_y = b->init_trans_y;
              /// Object motion
              c1->rot          = b->rot;
              c1->scale        = b->scale;
              c1->trans_x      = b->trans_x;
              c1->trans_y      = b->trans_y;
              c1->is_additive_component = true;
              c1->do_warpfield_deformation = b->do_warpfield_deformation;
              b->composite_component_blueprint_ptrs.push_back(c1);

              ObjectBlueprint* c2 = new ObjectBlueprint(*c1);
              if (c1->obj_type == ObjType_t::Ellipse) {
                if ((*RNG_GenericTrigger)()) {
                  c2->init_trans_x = b->init_trans_x+(*RNG_CompObjInitTransX)();
                  c2->init_trans_y = b->init_trans_y+(*RNG_CompObjInitTransY)();
                } else {
                  c2->init_trans_x = b->init_trans_x;
                  c2->init_trans_y = b->init_trans_y;
                  c2->ellipse_scale_x *= 0.9;
                  c2->ellipse_scale_y *= 0.9;
                }
              } else /*Polygon*/ {
                /// Intrinsic object transform
                c2->init_trans_x = b->init_trans_x;
                c2->init_trans_y = b->init_trans_y;
                for (unsigned int si = 0; si < c2->polygon_segment_x.size(); ++si) {
                  c2->polygon_segment_x[si] *= 0.9;
                  c2->polygon_segment_y[si] *= 0.9;
                }
              }
              /// Intrinsic object transform
              c2->scale        = b->scale;
              /// Object motion
              c2->rot          = b->rot;
              c2->trans_x      = b->trans_x;
              c2->trans_y      = b->trans_y;
              c2->is_additive_component = false;
              c2->do_warpfield_deformation = b->do_warpfield_deformation;
              b->composite_component_blueprint_ptrs.push_back(c2);
            } else {
              const unsigned int parts = (*RNG_CompObiNumberOfComponents)();
              for (unsigned int part_idx = 0; part_idx < parts; ++part_idx) {
                ObjectBlueprint* c = new ObjectBlueprint();
                c->obj_type = ObjType_t::Composite;
                /// Prefill values (most will be overwritten below)
                generateForegroundObject(c);
                /// Intrinsic object transform
                c->init_rot     = b->init_rot;
                c->init_trans_x = b->init_trans_x;
                c->init_trans_y = b->init_trans_y;
                /// Object motion
                c->rot          = b->rot;
                c->scale        = b->scale;
                c->trans_x      = b->trans_x;
                c->trans_y      = b->trans_y;

                if (part_idx == 0) {
                  c->is_additive_component = true;
                } else {
                  c->init_rot      = (*RNG_ObjInitRot)();
                  c->init_trans_x += (*RNG_ComponentOffset)();
                  c->init_trans_y += (*RNG_ComponentOffset)();
                  switch (c->obj_type) {
                    case ObjType_t::Ellipse: {
                      c->ellipse_scale_x *= 0.2;
                      c->ellipse_scale_y *= 0.2;
                      break;
                    }
                    case ObjType_t::Polygon: {
                      for (unsigned int si = 0; si < c->polygon_segment_x.size(); ++si) {
                        c->polygon_segment_x[si] *= 0.2;
                        c->polygon_segment_y[si] *= 0.2;
                      }
                      break;
                    }
                    default: {
                      throw std::runtime_error("(generateNumberOfFgObjects/9/badpart) Bad component object type");
                    }
                  }
                  c->is_additive_component = (*RNG_ComponentIsAdditive)();
                }
                c->do_warpfield_deformation = b->do_warpfield_deformation;
                b->composite_component_blueprint_ptrs.push_back(c);
              }
            }
            break;
          }
          default: {
            throw std::runtime_error("(generateNumberOfFgObjects/9) Bad object type, or not intended in this mode");
          }
        }
        break;
      }
      default: {
        throw std::runtime_error("BAD MODE");
      }
      //case 1: {
      //  b->obj_type     = (*RNG_ObjType)();
      //  /// Intrinsic object transform
      //  b->init_rot     = (*RNG_ObjInitRot)();
      //  b->init_trans_x = (*RNG_ObjInitTransX)();
      //  b->init_trans_y = (*RNG_ObjInitTransY)();
      //  /// Object motion
      //  b->rot          = ((*RNG_ObjRotTrigger)() ? (*RNG_ObjRot)() : 0.);
      //  b->scale        = ((*RNG_ObjScaleTrigger)() ? (*RNG_ObjScale)() : 1.);
      //  b->trans_x      = (*RNG_ObjTransX)();
      //  b->trans_y      = (*RNG_ObjTransY)();
      //  /// Texture stuff
      //  b->tex_id       = (*RNG_ObjTexID)();

      //  switch (b->obj_type) {
      //    case ObjType_t::Ellipse: {
      //      /// Ellipse object specifics
      //      b->ellipse_scale_x = (*RNG_ElliObj_ScaleX)()*50;
      //      b->ellipse_scale_y = (*RNG_ElliObj_ScaleY)()*50;
      //      break;
      //    }
      //    case ObjType_t::Polygon: {
      //      /// Polygon object specifics
      //      const unsigned int spokes =
      //            static_cast<unsigned int>((*RNG_PolyObj_spokes)());
      //      std::vector<float> phi(spokes);
      //      std::vector<float> r(spokes);
      //      for (unsigned int i = 0; i < spokes; ++i) {
      //        phi[i] = (i*360./spokes + (*RNG_PolyObj_dphi)())*agg::pi/180.;
      //        r[i]   = (*RNG_PolyObj_r)();
      //      }
      //      const float xscale = (*RNG_PolyObj_ScaleX)();
      //      const float yscale = (*RNG_PolyObj_ScaleY)();
      //      b->polygon_segment_x.resize(spokes);
      //      b->polygon_segment_y.resize(spokes);
      //      for (unsigned int i = 0; i < spokes; ++i) {
      //        b->polygon_segment_x[i] = xscale*r[i]*std::cos(phi[i]);
      //        b->polygon_segment_y[i] = yscale*r[i]*std::sin(phi[i]);
      //      }
      //      b->polygon_segment_types.resize(spokes);
      //      b->polygon_segment_types[0] = PolySegmentType_t::Dummy;
      //      for (unsigned int i = 1; i < spokes; ++i) {
      //        if ((i < spokes-1) and (*RNG_PolyObj_CurveTrigger)()) {
      //          b->polygon_segment_types[i] = PolySegmentType_t::Curve3;
      //          b->polygon_segment_types[i+1] = PolySegmentType_t::Dummy;
      //          ++i;
      //        } else {
      //          b->polygon_segment_types[i] = PolySegmentType_t::Line;
      //        }
      //      }
      //      break;
      //    }
      //    default: {
      //      throw std::runtime_error("Bad object type, or not intended in this mode");
      //    }
      //  }
      //  break;
      //}
    }
  }
  
  int ObjectParametersGenerator::generateNumberOfFgObjects() 
  {
    return (*RNG_NumberOfFgObjects)();
  }


}  /// namespace DataGenerator





//  size_t sample_idx = 0;
//  while (generator.hasUnfinishedTasks() or 
//         generator.hasRetrievableFinishedTasks()) {
//    Sample result{generator.retrieveFinishedTask()};
//
//    /// Save images 
//    std::ostringstream oss;
//    oss <<"output/"<<std::setw(5)<< std::setfill('0')<<sample_idx<<"-0.ppm";
//    result.image0_ptr->save_pnm(oss.str().c_str());
//    oss.str("");
//    oss <<"output/"<<std::setw(5)<< std::setfill('0')<<sample_idx<<"-1.ppm";
//    result.image1_ptr->save_pnm(oss.str().c_str());
//    oss.str("");
//    oss <<"output/"<<std::setw(5)<< std::setfill('0')<<sample_idx<<"-flow.pfm";
//    result.flow0_ptr->save_pfm(oss.str().c_str());
//
//    result.Destroy();
//    ++sample_idx;
//  }
//  generator.Stop(true);


