#include <occ/isosurface/deformation_density.h>

namespace occ::isosurface {

DeformationDensityFunctor::DeformationDensityFunctor(
    const occ::core::Molecule &mol, const occ::qm::Wavefunction &wfn,
    const occ::slater::InterpolatorParams &params)
    : m_pro(mol, params), m_rho(wfn) {}

MCDeformationDensityFunctor::MCDeformationDensityFunctor(
    const occ::core::Molecule &mol, const occ::qm::Wavefunction &wfn, float sep,
    const occ::slater::InterpolatorParams &params)
    : m_pro(mol, sep, params), m_rho(wfn, sep) {}

} // namespace occ::isosurface
