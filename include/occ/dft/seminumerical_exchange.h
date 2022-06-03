#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/dft/grid.h>
#include <occ/qm/basisset.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/mo.h>
#include <occ/qm/shellpair.h>
#include <vector>

namespace occ::dft::cosx {

class SemiNumericalExchange {

  public:
    SemiNumericalExchange(const std::vector<occ::core::Atom> &,
                          const qm::BasisSet &);
    Mat compute_K(qm::SpinorbitalKind kind, const qm::MolecularOrbitals &mo,
                  double precision = std::numeric_limits<double>::epsilon(),
                  const occ::Mat &Schwarz = occ::Mat()) const;

    Mat compute_overlap_matrix() const;

  private:
    std::vector<occ::core::Atom> m_atoms;
    qm::BasisSet m_basis;
    MolecularGrid m_grid;
    mutable occ::qm::IntegralEngine m_engine;

    std::vector<AtomGrid> m_atom_grids;
    Mat m_overlap;
    Mat m_numerical_overlap, m_overlap_projector;
    qm::ShellPairList m_shellpair_list;
    qm::ShellPairData m_shellpair_data;
};
} // namespace occ::dft::cosx
