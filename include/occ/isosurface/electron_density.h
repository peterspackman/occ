#pragma once
#include <occ/isosurface/common.h>
#include <occ/qm/wavefunction.h>

namespace occ::isosurface {

class ElectronDensityFunctor {
public:
  ElectronDensityFunctor(const occ::qm::Wavefunction &wfn, int mo_index = -1);

  inline void batch(Eigen::Ref<const FMat3N> pos,
                    Eigen::Ref<FVec> layer) const {
    m_num_calls += layer.size();
    if (m_mo_index >= 0) {
      layer = m_wfn.electron_density_mo(pos.cast<double>(), m_mo_index)
                  .cast<float>();
    } else {
      layer = m_wfn.electron_density(pos.cast<double>()).cast<float>();
    }
  }

  inline int num_calls() const { return m_num_calls; }
  void set_orbital_index(int idx);
  inline int orbital_index() const { return m_mo_index; }

private:
  mutable int m_num_calls{0};
  int m_mo_index{-1};
  qm::Wavefunction m_wfn;
};

class MCElectronDensityFunctor {
public:
  MCElectronDensityFunctor(const occ::qm::Wavefunction &wfn, float sep,
                           int mo_index = -1);

  inline void remap_vertices(const std::vector<float> &v,
                             std::vector<float> &dest) const {
    impl::remap_vertices(*this, v, dest);
  }

  inline void batch(Eigen::Ref<const FMat3N> pos,
                    Eigen::Ref<FVec> layer) const {
    m_rho.batch(pos, layer);
  }

  inline const auto &side_length() const { return m_cube_side_length; }

  inline Eigen::Vector3i cubes_per_side() const {
    return (side_length().array() / m_target_separation).ceil().cast<int>();
  }

  inline const auto &origin() const { return m_origin; }
  inline int num_calls() const { return m_rho.num_calls(); }

private:
  void update_region();

  int m_mo_index{-1};
  float m_buffer{5.0};
  FVec3 m_cube_side_length;
  ElectronDensityFunctor m_rho;
  FVec3 m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
  float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

  AxisAlignedBoundingBox m_bounding_box;
};

} // namespace occ::isosurface
