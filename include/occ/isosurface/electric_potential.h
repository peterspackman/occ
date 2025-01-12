#pragma once
#include <occ/isosurface/common.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>

namespace occ::isosurface {

class ElectricPotentialFunctor {
public:
  ElectricPotentialFunctor(const occ::qm::Wavefunction &wfn);

  inline void batch(Eigen::Ref<const FMat3N> pos,
                    Eigen::Ref<FVec> layer) const {
    m_num_calls += layer.size();
    Mat3N dpos = pos.cast<double>().array();
    Vec esp = m_hf.electronic_electric_potential_contribution(m_wfn.mo, dpos);
    esp += m_hf.nuclear_electric_potential_contribution(dpos);
    layer = esp.cast<float>();
  }

  inline int num_calls() const { return m_num_calls; }

private:
  qm::HartreeFock m_hf;
  qm::Wavefunction m_wfn;
  mutable int m_num_calls{0};
};

class MCElectricPotentialFunctor {
public:
  MCElectricPotentialFunctor(const occ::qm::Wavefunction &wfn, float sep = 0.2);

  inline void remap_vertices(const std::vector<float> &v,
                             std::vector<float> &dest) const {
    impl::remap_vertices(*this, v, dest);
  }

  inline void batch(Eigen::Ref<const FMat3N> pos,
                    Eigen::Ref<FVec> layer) const {
    m_esp.batch(pos, layer);
  }

  inline const auto &side_length() const { return m_cube_side_length; }

  inline Eigen::Vector3i cubes_per_side() const {
    return (side_length().array() / m_target_separation).ceil().cast<int>();
  }
  inline const auto &origin() const { return m_origin; }
  inline int num_calls() const { return m_esp.num_calls(); }

private:
  void update_region();
  ElectricPotentialFunctor m_esp;
  float m_buffer{5.0};
  FVec3 m_cube_side_length;
  FVec3 m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
  float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};
  AxisAlignedBoundingBox m_bounding_box;
};

} // namespace occ::isosurface
