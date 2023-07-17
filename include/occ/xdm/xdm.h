#pragma once
#include <occ/dft/grid.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/slater/slaterbasis.h>
#include <vector>

namespace occ::xdm {

struct XDMAtomList {
    const std::vector<occ::core::Atom> &atoms;
    const Vec &polarizabilities;
    const Mat &moments;
    const Vec &volume;
    const Vec &volume_free;
};

class XDM {
  public:
    struct Parameters {
        double a1{0.7};
        double a2{1.4}; // angstroms
    };

    XDM(const occ::qm::AOBasis &basis, int charge = 0,
        const Parameters &params = {0.7, 1.4});

    double energy(const occ::qm::MolecularOrbitals &mo);

    const Mat3N &forces(const occ::qm::MolecularOrbitals &mo);

    inline const auto &moments() const { return m_moments; }
    inline const auto &hirshfeld_charges() const { return m_hirshfeld_charges; }
    inline const auto &atom_volume() const { return m_volume; }
    inline const auto &free_atom_volume() const { return m_volume_free; }

    inline const auto &polarizabilities() const { return m_polarizabilities; }

    inline const auto &parameters() const { return m_params; }

  private:
    void populate_moments(const occ::qm::MolecularOrbitals &mo);
    void populate_polarizabilities();

    occ::qm::AOBasis m_basis;
    occ::dft::MolecularGrid m_grid;
    std::vector<occ::dft::AtomGrid> m_atom_grids;
    std::vector<occ::slater::Basis> m_slater_basis;
    Mat m_density_matrix;
    Mat m_moments;
    Vec m_volume;
    Vec m_polarizabilities;
    Vec m_volume_free;
    Vec m_hirshfeld_charges;
    double m_energy{0.0};
    Mat3N m_forces;
    bool m_atomic_ion{false};
    int m_charge{0};
    Parameters m_params{};
};

std::pair<double, Mat3N>
xdm_dispersion_energy(const XDMAtomList &atom_info,
                      const XDM::Parameters &params = {});

std::tuple<double, Mat3N, Mat3N>
xdm_dispersion_interaction_energy(const XDMAtomList &atom_info_a,
                                  const XDMAtomList &atom_info_b,
                                  const XDM::Parameters &params = {});

} // namespace occ::xdm
