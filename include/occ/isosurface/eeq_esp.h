#pragma once
#include <occ/core/molecule.h>
#include <occ/isosurface/common.h>

namespace occ::isosurface {

class ElectricPotentialFunctorPC {
public:
  ElectricPotentialFunctorPC(const occ::core::Molecule &m1);

  void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const {
    m_num_calls += layer.size();
    layer = m_molecule.esp_partial_charges(pos.cast<double>()).cast<float>();
  }

  inline int num_calls() const { return m_num_calls; }

private:
  occ::core::Molecule m_molecule;
  mutable int m_num_calls{0};
};

} // namespace occ::isosurface
