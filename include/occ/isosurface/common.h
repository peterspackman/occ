#pragma once
#include <occ/core/interpolator.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <utility>
#include <vector>

namespace occ::isosurface {

namespace impl {

template <class Func>
void remap_vertices(const Func &f, const std::vector<float> &v,
                    std::vector<float> &dest) {
  dest.resize(v.size());
  for (int i = 0; i < v.size(); i += 3) {
    dest[i] = occ::units::BOHR_TO_ANGSTROM * v[i];
    dest[i + 1] = occ::units::BOHR_TO_ANGSTROM * v[i + 1];
    dest[i + 2] = occ::units::BOHR_TO_ANGSTROM * v[i + 2];
  }
}

} // namespace impl

struct AxisAlignedBoundingBox {
  Eigen::Vector3f lower;
  Eigen::Vector3f upper;

  inline bool inside(const Eigen::Vector3f &point) const {
    return (lower.array() <= point.array()).all() &&
           (upper.array() >= point.array()).all();
  }
};

using LinearInterpolatorFloat =
    occ::core::Interpolator1D<float, occ::core::DomainMapping::Linear>;

struct AtomInterpolator {
  LinearInterpolatorFloat interpolator;
  Eigen::Matrix<float, 3, Eigen::Dynamic> positions;
  float threshold{144.0};
  int interior{0};
};

struct InterpolatorParams {
  int num_points{8192};
  float domain_lower{0.04};
  float domain_upper{144.0};
};

template <typename F> class BatchFunctor {
public:
  // Perfect forwarding constructor
  template <typename... Args>
  BatchFunctor(Args &&...args) : m_func(std::forward<Args>(args)...) {}

  // Constructor taking existing functor
  BatchFunctor(F &f) : m_func(f) {}

  // Move constructor for functor
  BatchFunctor(F &&f) : m_func(std::move(f)) {}

  inline void batch(Eigen::Ref<const FMat3N> pos,
                    Eigen::Ref<FVec> layer) const {
    m_num_calls += layer.size();
    for (int i = 0; i < pos.cols(); i++) {
      layer(i) = m_func(pos.col(i));
    }
  }

  inline int num_calls() const { return m_num_calls; }

  // Access to underlying functor if needed
  const F &functor() const { return m_func; }
  F &functor() { return m_func; }

private:
  F m_func;
  mutable int m_num_calls{0};
};

// Deduction guide to help with template argument deduction
template <typename F>
BatchFunctor(F &&) -> BatchFunctor<std::remove_reference_t<F>>;

// Helper function to construct BatchFunctor
template <typename F, typename... Args>
auto make_batch_functor(Args &&...args) {
  return BatchFunctor<F>(std::forward<Args>(args)...);
}
} // namespace occ::isosurface
