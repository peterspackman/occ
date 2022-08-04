#pragma once
#include <occ/dft/grid.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/slater/slaterbasis.h>

namespace occ::xdm {

class XDM {
  public:
    struct Parameters {
        double a1{1.0};
        double a2{1.0}; // angstroms
    };

    XDM(const occ::qm::AOBasis &basis);

    double energy(const occ::qm::MolecularOrbitals &mo);

    const Mat3N &forces(const occ::qm::MolecularOrbitals &mo);

    inline const auto &moments() const { return m_moments; }
    inline const auto &hirshfeld_charges() const { return m_hirshfeld_charges; }
    inline const auto &atom_volume() const { return m_volume; }
    inline const auto &free_atom_volume() const { return m_volume_free; }

  private:
    void populate_moments(const occ::qm::MolecularOrbitals &mo);

    occ::qm::AOBasis m_basis;
    occ::dft::MolecularGrid m_grid;
    std::vector<occ::dft::AtomGrid> m_atom_grids;
    std::vector<occ::slater::Basis> m_slater_basis;
    Mat m_density_matrix;
    Mat m_moments;
    Vec m_volume;
    Vec m_volume_free;
    Vec m_hirshfeld_charges;
    double m_energy{0.0};
    Mat3N m_forces;
};

} // namespace occ::xdm
