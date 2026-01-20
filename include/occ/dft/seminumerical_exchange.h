#pragma once
#include <occ/core/atom.h>
#include <occ/dft/molecular_grid.h>
#include <occ/qm/integral_engine.h>


namespace occ::dft::cosx {

class SemiNumericalExchange {

public:
  SemiNumericalExchange(const qm::AOBasis &, const GridSettings & = {});
  Mat compute_K(const qm::MolecularOrbitals &mo,
                double precision = std::numeric_limits<double>::epsilon(),
                const occ::Mat &Schwarz = occ::Mat()) const;

  Mat compute_overlap_matrix() const;

  const auto &engine() const { return m_engine; }

private:
  std::vector<occ::core::Atom> m_atoms;
  qm::AOBasis m_basis;
  MolecularGrid m_grid;
  mutable occ::qm::IntegralEngine m_engine;

  std::vector<AtomGrid> m_atom_grids;
  Mat m_overlap;
  Mat m_numerical_overlap, m_overlap_projector;
};
} // namespace occ::dft::cosx
