#pragma once
#include <Eigen/Core>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/isosurface/common.h>

namespace occ::isosurface {

class RadiusMetric {
public:
  enum class RadiusKind { Unit, VDW, Covalent };
  RadiusMetric(RadiusKind kind = RadiusKind::VDW) : m_kind(kind) {
    initialize_radii();
  }

  FVec compute_distances(const FVec3 &pos, const FMat3N &points,
                         const Eigen::VectorXi &elements) const {
    FVec distances = (points.colwise() - pos).colwise().norm();
    for (int i = 0; i < distances.rows(); i++) {
      float radius = m_radii(elements(i));
      distances(i) /= radius;
    }
    return distances;
  }

private:
  void initialize_radii() {
    m_radii = FVec::Ones(104);
    for (int i = 0; i < 104; i++) {
      float r = 1.0;
      if (m_kind == RadiusKind::VDW) {
        m_radii(i) = occ::core::Element(i).van_der_waals_radius() *
                     occ::units::ANGSTROM_TO_BOHR;
      } else if (m_kind == RadiusKind::Covalent) {
        m_radii(i) = occ::core::Element(i).covalent_radius() *
                     occ::units::ANGSTROM_TO_BOHR;
      }
    }
  }
  RadiusKind m_kind{RadiusKind::VDW};
  FVec m_radii;
};

template <class Metric> class LogSumExpFunctor {
public:
  LogSumExpFunctor(Metric &metric, const occ::core::Molecule &in,
                   const occ::core::Molecule &ext, float sep,
                   float temperature = 0.1f)
      : m_metric(metric), m_target_separation(sep), m_temperature(temperature) {

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

    // Use TBB parallel_for with automatic load balancing
    // Each point processed independently for optimal work distribution
    occ::parallel::parallel_for(0, int(pos.cols()), [&](int pt) {
      FVec3 p = pos.col(pt);
      if (!m_bounding_box.inside(p)) {
        layer(pt) = 0.0;
      } else {
        layer(pt) = compute_weight(p);
      }
    });
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
    FVec d_in = m_metric.compute_distances(pos, m_internal_positions,
                                           m_internal_elements);
    FVec d_ext = m_metric.compute_distances(pos, m_external_positions,
                                            m_external_elements);

    if (d_in.size() == 0)
      return 0.0f;
    if (d_ext.size() == 0)
      return 1.0f;

    // Compute smooth minimum using LSE
    float min_dist_in =
        -m_temperature * logsumexp(-d_in.array() / m_temperature);
    float min_dist_ext =
        -m_temperature * logsumexp(-d_ext.array() / m_temperature);

    return min_dist_ext - min_dist_in;
  }

  float logsumexp(const FVec &x) const {
    float max_x = x.maxCoeff();
    return max_x + std::log((x.array() - max_x).exp().sum());
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
  Eigen::VectorXi m_internal_elements;
  Eigen::VectorXi m_external_elements;
  Metric m_metric;

  FVec3 m_cube_side_length;
  FVec3 m_origin;
  mutable int m_num_calls{0};
  AxisAlignedBoundingBox m_bounding_box;
};

} // namespace occ::isosurface
