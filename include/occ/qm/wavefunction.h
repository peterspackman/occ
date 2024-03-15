#pragma once
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/io/orca_json.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

using occ::Vec;
using occ::io::FchkReader;
using occ::io::FchkWriter;
using occ::io::MoldenReader;
using occ::io::OrcaJSONReader;

struct Energy {
    double coulomb{0};
    double exchange{0};
    double nuclear_repulsion{0};
    double nuclear_attraction{0};
    double kinetic{0};
    double core{0};
    double total{0};
    double ecp{0};

    inline auto operator-(const Energy &rhs) {
        return Energy{
            coulomb - rhs.coulomb,
            exchange - rhs.exchange,
            nuclear_repulsion - rhs.nuclear_repulsion,
            nuclear_attraction - rhs.nuclear_attraction,
            kinetic - rhs.kinetic,
            core - rhs.core,
            total - rhs.total,
            ecp - rhs.ecp,
        };
    }

    inline auto operator+(const Energy &rhs) {
        return Energy{
            coulomb + rhs.coulomb,
            exchange + rhs.exchange,
            nuclear_repulsion + rhs.nuclear_repulsion,
            nuclear_attraction + rhs.nuclear_attraction,
            kinetic + rhs.kinetic,
            core + rhs.core,
            total + rhs.total,
            ecp + rhs.ecp,
        };
    }
    inline std::string to_string() const {
        return fmt::format("{:10s} {:20.12f}\n"
                           "{:10s} {:20.12f}\n"
                           "{:10s} {:20.12f}\n"
                           "{:10s} {:20.12f}\n"
                           "{:10s} {:20.12f}\n"
                           "{:10s} {:20.12f}\n"
                           "{:10s} {:20.12f}\n",
                           "E_coul", coulomb, "E_ex", exchange, "E_nn",
                           nuclear_repulsion, "E_en", nuclear_attraction,
                           "E_kin", kinetic, "E_1e", core, "E_ecp", ecp);
    }
};

struct Wavefunction {
    Wavefunction() {}

    Wavefunction(const FchkReader &);
    Wavefunction(const MoldenReader &);
    Wavefunction(const OrcaJSONReader &);
    Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b);

    static Wavefunction load(const std::string &);
    static bool is_likely_wavefunction_filename(const std::string &);

    inline int multiplicity() const { return abs(n_beta() - n_alpha()) + 1; }
    inline int charge() const {
        size_t c = 0;
        for (const auto &atom : atoms)
            c += atom.atomic_number;
        c -= num_electrons;
        c -= basis.total_ecp_electrons();
        return c;
    }
    inline int n_alpha() const { return mo.n_alpha; }
    inline int n_beta() const { return mo.n_beta; }
    bool is_restricted() const {
        return mo.kind == SpinorbitalKind::Restricted;
    }
    void update_occupied_orbitals();
    void set_molecular_orbitals(const FchkReader &fchk);
    void compute_density_matrix();
    void symmetric_orthonormalize_molecular_orbitals(const Mat &overlap);
    void apply_transformation(const occ::Mat3 &, const occ::Vec3 &);
    void apply_rotation(const occ::Mat3 &);
    void apply_translation(const occ::Vec3 &);

    occ::Mat3N positions() const;
    occ::IVec atomic_numbers() const;

    Vec mulliken_charges() const;

    Vec electron_density(const Mat3N &points) const;
    Mat3N electron_density_gradient(const Mat3N &points) const;

    Vec electron_density_mo(const Mat3N &points, int mo_index) const;
    Mat3N electron_density_mo_gradient(const Mat3N &points, int mo_index) const;

    Vec electric_potential(const Mat3N &points) const;
    Mat3N electric_field(const Mat3N &points) const;

    void save(FchkWriter &);
    bool save(const std::string &filename);

    int num_electrons{0};
    int num_frozen_electrons{0};
    AOBasis basis;
    size_t nbf{0};
    std::vector<occ::core::Atom> atoms;
    MolecularOrbitals mo;
    Mat T, V, H, J, K, Vecp;
    Energy energy;
    bool have_energies{false};
    bool have_xdm_parameters{false};
    Vec xdm_polarizabilities;
    Mat xdm_moments;
    Vec xdm_volumes;
    Vec xdm_free_volumes;
    double xdm_energy{0.0};
};

} // namespace occ::qm
