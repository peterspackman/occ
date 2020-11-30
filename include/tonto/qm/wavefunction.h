#pragma once
#include <tonto/core/linear_algebra.h>
#include <tonto/qm/basisset.h>
#include <tonto/qm/spinorbital.h>
#include <tonto/io/fchkreader.h>
#include <tonto/io/moldenreader.h>

namespace tonto::qm {

using tonto::MatRM;
using tonto::Vec;
using tonto::io::FchkReader;
using tonto::io::MoldenReader;

struct Energy {
    double coulomb{0};
    double exchange{0};
    double nuclear_repulsion{0};
    double nuclear_attraction{0};
    double kinetic{0};
    double core{0};
    void print() const;
};

struct Wavefunction {
    Wavefunction() {}

    Wavefunction(const FchkReader& fchk);
    Wavefunction(const MoldenReader& fchk);
    Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b);

    size_t n_alpha() const { return num_alpha; }
    size_t n_beta() const { return num_beta; }
    bool is_restricted() const { return spinorbital_kind == SpinorbitalKind::Restricted; }
    void update_occupied_orbitals();
    void set_molecular_orbitals(const FchkReader& fchk);
    void compute_density_matrix();
    void symmetric_orthonormalize_molecular_orbitals(const MatRM& overlap);
    void apply_transformation(const tonto::Mat3&, const tonto::Vec3&);
    void apply_rotation(const tonto::Mat3&);
    void apply_translation(const tonto::Vec3&);

    tonto::Mat3N positions() const;
    tonto::IVec atomic_numbers() const;

    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int num_alpha;
    int num_beta;
    int num_electrons;
    BasisSet basis;
    size_t nbf{0};
    std::vector<libint2::Atom> atoms;
    MatRM C, C_occ, D, T, V, H, J, K;
    Vec mo_energies;
    Energy energy;


};

MatRM symmorthonormalize_molecular_orbitals(const MatRM& mos, const MatRM& overlap, size_t n_occ);
}
