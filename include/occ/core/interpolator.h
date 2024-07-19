#pragma once
#include <cmath>
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/macros.h>

namespace occ::core {

template <typename T> OCC_ALWAYS_INLINE T lerp(T v0, T v1, T t) {
  return (1 - t) * v0 + t * v1;
}
/**
 * An enum to specify the mapping of the domain of inputs for interpolation.
 *
 * Can help improve precision of linear interpolation while minimizing
 * number of points.
 */
enum DomainMapping {
  Linear,     /**< The typical f(x) mapping */
  SquareRoot, /**< Make f a function of x^2 -> f(x*x) mapping */
  Log,        /**< Make f a function of e^x -> f(e^x) mapping */
};

/**
 * Class for interpolating one-dimensional functions.
 *
 * Templated by the type of data to be interpolated, and the DomainMapping
 * (default = Linear)
 */
template <typename T, DomainMapping mapping = Linear> class Interpolator1D {
public:
  /**
   * Default constructor
   */
  Interpolator1D() {}

  /**
   * Construct an Interpolator1D from a given function.
   *
   * \param f the function to interpolate
   * \param left the lowest (left most) value in the domain
   * \param right the highest (right most) value in the domain
   * \param N the number of points to sample in the domain [left, right]
   *
   * Will evaluate the function f N times, and map the domain values
   * internally based on the specified DomainMapping....
   *
   */
  template <typename F>
  Interpolator1D(const F &f, T left, T right, size_t N)
      : m_domain(N), m_range(N) {
    T l_mapped, u_mapped;

    if constexpr (mapping == Log) {
      l_mapped = std::log(left);
      u_mapped = std::log(right);
    } else if constexpr (mapping == SquareRoot) {
      l_mapped = std::sqrt(left);
      u_mapped = std::sqrt(right);
    } else {
      l_mapped = left;
      u_mapped = right;
    }

    for (size_t i = 0; i < N; i++) {
      T x = l_mapped + i * (u_mapped - l_mapped) / N;

      if constexpr (mapping == Log) {
        x = std::exp(x);
      } else if constexpr (mapping == SquareRoot) {
        x = x * x;
      }
      T y = f(x);
      m_domain(i) = x;
      m_range(i) = y;
    }

    if constexpr (mapping == Log) {
      l_domain = std::log(m_domain[0]);
      u_domain = std::log(m_domain[m_domain.size() - 1]);
    } else if constexpr (mapping == SquareRoot) {
      l_domain = std::sqrt(m_domain[0]);
      u_domain = std::sqrt(m_domain[m_domain.size() - 1]);
    } else {
      l_domain = m_domain[0];
      u_domain = m_domain[m_domain.size() - 1];
    }

    l_fill = m_range[0];
    u_fill = m_range[m_range.size() - 1];

    l_fill_grad = (m_range[1] - m_range[0]) / (m_domain[1] - m_domain[0]);
    u_fill_grad =
        (m_range[m_range.size() - 1] - m_range[m_range.size() - 2]) /
        (m_domain[m_domain.size() - 1] - m_domain[m_domain.size() - 2]);
    m_dx = m_domain.size() / (u_domain - l_domain);
  }

  /**
   * Evaluate the interpolated function at the value provided.
   *
   * \param x the value where the interpolated function should be evaluated.
   *
   * The provided value will be mapped based on DomainMapping.
   * If the provided value falls outside the domain, it will be
   * yield the f(left) and f(right) values from the original
   * construction of the Interpolator1D.
   */
  OCC_ALWAYS_INLINE T operator()(T x) const {
    size_t N = m_domain.size();
    if constexpr (mapping == Linear) {
      T guess = m_dx * (x - l_domain);

      // branchless here seems to be 50% slower
      size_t j = static_cast<size_t>(std::floor(guess));

      if (j <= 0)
        return l_fill;
      if (j >= N - 1)
        return u_fill;
      T t = (x - m_domain[j]) / (m_domain[j + 1] - m_domain[j]);
      return lerp(m_range[j], m_range[j + 1], t);
    } else {

      T dval = x;
      if constexpr (mapping == Log) {
        dval = std::log(x);
      } else if constexpr (mapping == SquareRoot) {
        dval = std::sqrt(x);
      }
      T guess = (dval - l_domain) * m_dx;
      size_t j = static_cast<size_t>(std::floor(guess));
      // linear search after the guess might be required due to domain
      // distortion
      if (j <= 0)
        return l_fill;
      if (j >= (N - 1))
        return u_fill;
      do {
        j++;
      } while (m_domain[j] < x);

      T slope = (m_range[j] - m_range[j - 1]) / (m_domain[j] - m_domain[j - 1]);
      return m_range[j - 1] + (x - m_domain[j - 1]) * slope;
    }
  }

  OCC_ALWAYS_INLINE T gradient(T x) const {
    size_t N = m_domain.size();
    if constexpr (mapping == Linear) {
      T guess = m_dx * (x - l_domain);

      // branchless here seems to be 50% slower
      size_t j = static_cast<size_t>(std::floor(guess));

      if (j <= 0)
        return l_fill_grad;
      if (j >= N - 1)
        return u_fill_grad;
      return (m_range[j + 1] - m_range[j]) / (m_domain[j + 1] - m_domain[j]);
    } else {

      T dval = x;
      if constexpr (mapping == Log) {
        dval = std::log(x);
      } else if constexpr (mapping == SquareRoot) {
        dval = std::sqrt(x);
      }
      T guess = (dval - l_domain) * m_dx;
      size_t j = static_cast<size_t>(std::floor(guess));
      // linear search after the guess might be required due to domain
      // distortion
      if (j <= 0)
        return l_fill;
      if (j >= (N - 1))
        return u_fill;
      do {
        j++;
      } while (m_domain[j] < x);

      return (m_range[j + 1] - m_range[j]) / (m_domain[j + 1] - m_domain[j]);
    }
  }

  Eigen::Array<T, Eigen::Dynamic, 1>
  operator()(const Eigen::Array<T, Eigen::Dynamic, 1> &xs) const {
    static_assert(
        mapping == Linear,
        "Vectorised interpolation only implemented for linear mapping");
    Eigen::Array<T, Eigen::Dynamic, 1> results(xs.size());
    size_t N = m_domain.size();
    Eigen::Array<int, Eigen::Dynamic, 1> js =
        (m_dx * (xs - l_domain)).template cast<int>();
    js = js.min(N - 2); // Clamp js to avoid going out of range

    // Compute weights for interpolation
    Eigen::Array<T, Eigen::Dynamic, 1> weights =
        (xs - m_domain(js)) / (m_domain(js + 1) - m_domain(js));

    // Linear interpolation
    results = (1 - weights) * m_range(js) + weights * m_range(js + 1);

    // Fill values outside domain
    results = (xs < m_domain(0)).select(l_fill, results);
    results = (xs > m_domain(N - 1)).select(u_fill, results);

    return results;
  }

  T find_threshold(T threshold_value) const {
    Eigen::Index index;
    bool found = (m_range < threshold_value).maxCoeff(&index);
    if (found) {
      return m_domain(index);
    } else {
      return u_domain;
    }
  }

private:
  T l_domain, u_domain, l_fill, u_fill, l_fill_grad, u_fill_grad;
  T m_dx;
  Eigen::Array<T, Eigen::Dynamic, 1> m_domain, m_range;
};

} // namespace occ::core
