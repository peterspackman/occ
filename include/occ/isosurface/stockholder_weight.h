#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/isosurface/common.h>
#include <occ/slater/hirshfeld.h>

namespace occ::isosurface {

class StockholderWeightFunctor {
public:
  StockholderWeightFunctor(const occ::core::Molecule &in,
                           occ::core::Molecule &ext, float sep,
                           const occ::slater::InterpolatorParams & = {});

  inline void remap_vertices(const std::vector<float> &v,
                             std::vector<float> &dest) const {
    impl::remap_vertices(*this, v, dest);
  }

  OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
    if (!m_bounding_box.inside(pos))
      return 1.0e8; // return an arbitrary large distance
    m_num_calls++;
    return m_hirshfeld(pos);
  }

  void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const {
    m_num_calls += layer.size();

    int num_threads = occ::parallel::get_num_threads();
    auto inner_func = [&](int thread_id) {
      int total_elements = pos.cols();
      int block_size = total_elements / num_threads;
      int start_index = thread_id * block_size;
      int end_index = start_index + block_size;
      if (thread_id == num_threads - 1) {
        end_index = total_elements;
      }
      for (int pt = start_index; pt < end_index; pt++) {
        Eigen::Vector3f p = pos.col(pt);
        if (!m_bounding_box.inside(p)) {
          layer(pt) = 0.0;
          continue;
        }
        m_num_calls++;
        layer(pt) = m_hirshfeld(p);
      }
    };
    occ::parallel::parallel_do(inner_func);
  }

  OCC_ALWAYS_INLINE FVec3 gradient(const FVec3 &pos) const {
    if (!m_bounding_box.inside(pos))
      return pos.normalized(); // zero normal
    m_num_calls++;
    return m_hirshfeld.gradient(pos);
  }

  inline const auto &side_length() const { return m_cube_side_length; }

  inline Eigen::Vector3i cubes_per_side() const {
    return (side_length().array() / m_target_separation).ceil().cast<int>();
  }

  inline const auto &origin() const { return m_origin; }
  inline int num_calls() const { return m_num_calls; }

  inline void set_background_density(float rho) {
    m_hirshfeld.set_background_density(rho);
  }
  inline float background_density() const {
    return m_hirshfeld.background_density();
  }

  inline const auto &bounding_box() const { return m_bounding_box; }
  inline const auto &weight_function() const { return m_hirshfeld; }

  inline void update_num_calls(int n) const { m_num_calls += n; }

private:
  float m_buffer{8.0};
  Eigen::Vector3f m_cube_side_length;
  Eigen::Vector3f m_origin;
  mutable int m_num_calls{0};
  float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

  occ::slater::StockholderWeight m_hirshfeld;
  AxisAlignedBoundingBox m_bounding_box;
};

template <class Func> class GenericStockholderWeightFunctor {
public:
  GenericStockholderWeightFunctor(Func &func, const occ::core::Molecule &in,
                                  const occ::core::Molecule &ext, float sep)
      : m_func(func), m_target_separation(sep) {

    // Convert positions to bohr and store
    const auto &in_pos = in.positions();
    const auto &ext_pos = ext.positions();
    const auto &in_elements = in.atomic_numbers();
    const auto &ext_elements = ext.atomic_numbers();

    m_internal_positions =
        (in_pos.array() * occ::units::ANGSTROM_TO_BOHR).cast<float>();
    m_external_positions =
        (ext_pos.array() * occ::units::ANGSTROM_TO_BOHR).cast<float>();

    // Store atomic numbers
    m_internal_elements = in_elements.cast<int>();
    m_external_elements = ext_elements.cast<int>();

    // Set up bounding box
    setup_bounding_box();
  }

  inline void remap_vertices(const std::vector<float> &v,
                             std::vector<float> &dest) const {
    impl::remap_vertices(*this, v, dest);
  }

  OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
    if (!m_bounding_box.inside(pos))
      return 1.0e8;

    m_num_calls++;
    return compute_weight(pos);
  }

  void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const {
    m_num_calls += layer.size();
    int num_threads = occ::parallel::get_num_threads();

    auto inner_func = [&](int thread_id) {
      int total_elements = pos.cols();
      int block_size = total_elements / num_threads;
      int start_index = thread_id * block_size;
      int end_index = (thread_id == num_threads - 1) ? total_elements
                                                     : start_index + block_size;

      for (int pt = start_index; pt < end_index; pt++) {
        FVec3 p = pos.col(pt);
        if (!m_bounding_box.inside(p)) {
          layer(pt) = 0.0;
          continue;
        }
        layer(pt) = compute_weight(p);
      }
    };

    occ::parallel::parallel_do(inner_func);
  }

  OCC_ALWAYS_INLINE FVec3 gradient(const FVec3 &pos) const {
    if (!m_bounding_box.inside(pos))
      return pos.normalized();

    m_num_calls++;
    // Numerical gradient calculation
    const float h = 1e-5f;
    FVec3 grad;
    FVec3 pos_p, pos_m;

    for (int i = 0; i < 3; i++) {
      pos_p = pos;
      pos_m = pos;
      pos_p[i] += h;
      pos_m[i] -= h;
      grad[i] = (compute_weight(pos_p) - compute_weight(pos_m)) / (2 * h);
    }
    return grad;
  }

  inline const auto &side_length() const { return m_cube_side_length; }
  inline Eigen::Vector3i cubes_per_side() const {
    return (side_length().array() / m_target_separation)
        .ceil()
        .template cast<int>();
  }
  inline const auto &origin() const { return m_origin; }
  inline int num_calls() const { return m_num_calls; }
  inline const auto &bounding_box() const { return m_bounding_box; }
  inline void update_num_calls(int n) const { m_num_calls += n; }

private:
  float compute_weight(const FVec3 &pos) const {
    // Compute distances using the metric
    FVec d_in = compute_distances(pos, m_internal_positions);
    FVec d_ext = compute_distances(pos, m_external_positions);

    // Handle empty sets
    if (d_in.size() == 0)
      return 0.0f;
    if (d_ext.size() == 0)
      return 1.0f;

    float w_in = m_func(d_in, m_internal_elements);
    float w_out = m_func(d_ext, m_external_elements);
    return w_in / (w_in + w_out);
  }

  FVec compute_distances(const FVec3 &pos, const FMat3N &points) const {
    return (points.colwise() - pos).colwise().norm();
  }

  void setup_bounding_box() {
    float buffer = 8.0f;

    Eigen::Vector3f min_pos = m_internal_positions.rowwise().minCoeff();
    Eigen::Vector3f max_pos = m_internal_positions.rowwise().maxCoeff();

    m_origin = min_pos.array() - buffer;
    m_cube_side_length = (max_pos - m_origin).array() + buffer;

    m_bounding_box.lower = m_origin;
    m_bounding_box.upper = max_pos;
    m_bounding_box.upper.array() += buffer;
  }

  float m_target_separation;
  float m_temperature;
  FMat3N m_internal_positions;
  FMat3N m_external_positions;
  IVec m_internal_elements;
  IVec m_external_elements;
  Func m_func;

  FVec3 m_cube_side_length;
  FVec3 m_origin;
  mutable int m_num_calls{0};
  AxisAlignedBoundingBox m_bounding_box;
};

struct RInvFunc {
  inline float operator()(const FVec &r, const IVec &els) const {
    return (els.cast<float>().array() / r.array().pow(power)).sum();
  }
  float power{6};
};

struct ExpFunc {
  inline float operator()(const FVec &r, const IVec &els) const {
    return (els.cast<float>().array() / (power * r.array()).exp()).sum();
  }
  float power{2};
};

} // namespace occ::isosurface
