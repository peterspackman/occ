#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/basisset.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

using occ::Vec;
using occ::io::FchkReader;
using occ::io::FchkWriter;
using occ::io::MoldenReader;

struct Energy {
    double coulomb{0};
    double exchange{0};
    double nuclear_repulsion{0};
    double nuclear_attraction{0};
    double kinetic{0};
    double core{0};
    double total{0};
    void print() const;
};

struct Wavefunction {
    Wavefunction() {}

    Wavefunction(const FchkReader &fchk);
    Wavefunction(const MoldenReader &fchk);
    Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b);

    size_t multiplicity() const { return abs(num_beta - num_alpha) + 1; }
    size_t charge() const {
        size_t c = 0;
        for (const auto &atom : atoms)
            c += atom.atomic_number;
        c -= num_electrons;
        return c;
    }
    size_t n_alpha() const { return num_alpha; }
    size_t n_beta() const { return num_beta; }
    bool is_restricted() const {
        return spinorbital_kind == SpinorbitalKind::Restricted;
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

    void save(FchkWriter &);
    void save_npz(const std::string &);

    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int num_alpha;
    int num_beta;
    int num_electrons;
    BasisSet basis;
    size_t nbf{0};
    std::vector<occ::core::Atom> atoms;
    Mat C, C_occ, D, T, V, H, J, K;
    Vec mo_energies;
    Energy energy;
    bool have_energies{false};
};

Mat symmorthonormalize_molecular_orbitals(const Mat &mos, const Mat &overlap,
                                          size_t n_occ);
} // namespace occ::qm
