
/**
 * Author: Nikolaus Mayer, 2015 (mayern@cs.uni-freiburg.de)
 *
 * Simple RNG
 */

#ifndef SIMPLERANDOM_H__
#define SIMPLERANDOM_H__

/// System/STL
#include <limits>
#include <random>

namespace RNG {

  /** 
   * Base class in separate namespace
   */
  namespace internal {
    class RNGBase 
    {
      public:
        RNGBase(int seed = -1)
        { 
          if (seed >= 0) {
            m_mersenne = std::mt19937(seed);
          } else {
            m_mersenne = std::mt19937(m_randev());
          }
        }

      protected:
        /// Raw randomness source
        std::random_device m_randev;
        /// Mersenne Twister engine 
        std::mt19937 m_mersenne;
    };
  }  // namespace internal


  /**
   * Uniform integer
   */
  class UniformInt : public RNG::internal::RNGBase
  {
    public:
      UniformInt(int seed = -1) : RNGBase(seed) {}

      int operator() (int a = 0, int b = std::numeric_limits<int>::max())
      {
        return std::uniform_int_distribution<>(a, b)(m_mersenne);
      }
  };


  /**
   * Uniform float
   */
  class UniformFloat : public RNG::internal::RNGBase
  {
    public:
      UniformFloat(int seed = -1) : RNGBase(seed) {}

      float operator() (float a = 0, float b = std::numeric_limits<float>::max())
      {
        return std::uniform_real_distribution<>(a, b)(m_mersenne);
      }
  };


  /**
   * Uniform integer with fixed range
   */
  class FixedRangeUniformInt : public RNG::internal::RNGBase 
  {
    public:
      FixedRangeUniformInt(int a = 0, 
                           int b = std::numeric_limits<int>::max(),
                           int seed = -1)
        : RNGBase(seed)
      { m_dist = std::uniform_int_distribution<>(a, b); }

      int operator() () 
      { return m_dist(m_mersenne); }

    private:
      std::uniform_int_distribution<> m_dist;
  };


  /**
   * Uniform float with fixed range
   */
  class FixedRangeUniformFloat : public RNG::internal::RNGBase 
  {
    public:
      FixedRangeUniformFloat(float a = 0.f, 
                             float b = 1.f,
                             int seed = -1)
        : RNGBase(seed)
      { m_dist = std::uniform_real_distribution<>(a, b); }

      float operator() () 
      { return m_dist(m_mersenne); }

    private:
      std::uniform_real_distribution<> m_dist;
  };


  /**
   * Normal distribution
   */
  class NormalFloat : public RNG::internal::RNGBase
  {
    public:
      NormalFloat(int seed = -1) : RNGBase(seed) {}
      
      float operator() (int a = 0, int b = std::numeric_limits<int>::max())
      {
        return std::normal_distribution<float>(a, b)(m_mersenne);
      }
  };


  /**
   * Normal distribution
   */
  class FixedMeanStddevNormalFloat : public RNG::internal::RNGBase
  {
    public:
      FixedMeanStddevNormalFloat(float mean=0.f, float stddev=1.f, int seed = -1)
        : RNGBase(seed)
      { m_dist = std::normal_distribution<float>(mean, stddev); }

      float operator() () 
      { return m_dist(m_mersenne); }

    private:
      std::normal_distribution<float> m_dist;
  };


}  // namespace RNG



#endif  // SIMPLERANDOM_H__

