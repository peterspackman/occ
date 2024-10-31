#pragma once

#include <occ/isosurface/electron_density.h>
#include <occ/isosurface/promolecule_density.h>

namespace occ::isosurface {

class MCDeformationDensityFunctor {
public:
  MCDeformationDensityFunctor(
      const occ::core::Molecule &mol, const occ::qm::Wavefunction &wfn,
      float sep, const occ::slater::InterpolatorParams &params = {});

  inline void remap_vertices(const std::vector<float> &v,
                             std::vector<float> &dest) const {
    impl::remap_vertices(*this, v, dest);
  }

  inline void batch(Eigen::Ref<const FMat3N> pos,
                    Eigen::Ref<FVec> layer) const {
    m_rho.batch(pos, layer);
    for (int i = 0; i < pos.cols(); i++) {
      layer(i) -= m_pro(pos.col(i));
    }
  }

  inline const auto &side_length() const { return m_rho.side_length(); }

  inline Eigen::Vector3i cubes_per_side() const {
    return m_rho.cubes_per_side();
  }

  inline void set_isovalue(float iso) { m_pro.set_isovalue(iso); }

  inline const auto &origin() const { return m_rho.origin(); }
  inline int num_calls() const { return m_rho.num_calls(); }

private:
  MCPromoleculeDensityFunctor m_pro;
  MCElectronDensityFunctor m_rho;
};

} // namespace occ::isosurface
