#pragma once
#include "ints.h"
#include "diis.h"
#include "logger.h"
#include "linear_algebra.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <libint2/chemistry/sto3g_atomic_density.h>
#include <tuple>
#include "spinorbital.h"
#include "util.h"

namespace tonto::scf {

using tonto::qm::SpinorbitalKind;
using tonto::qm::expectation;
using tonto::MatRM;
using tonto::util::is_odd;

std::tuple<MatRM, MatRM, double>
conditioning_orthogonalizer(const MatRM &S,
                            double S_condition_number_threshold);

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
std::tuple<MatRM, MatRM, size_t, double, double>
gensqrtinv(const MatRM &S, bool symmetric = false,
           double max_condition_number = 1e8);

template <typename Procedure, SpinorbitalKind spinorbital_kind>
struct SCF {
    SCF(Procedure &procedure, int diis_start = 2)
        : m_procedure(procedure), diis(diis_start) {
        n_electrons = m_procedure.num_e();
        nbf = m_procedure.basis().nbf();
        auto [rows, cols] = tonto::qm::matrix_dimensions<spinorbital_kind>(nbf);
        S = MatRM::Zero(rows, cols);
        T = MatRM::Zero(rows, cols);
        V = MatRM::Zero(rows, cols);
        H = MatRM::Zero(rows, cols);
        F = MatRM::Zero(rows, cols);
        D = MatRM::Zero(rows, cols);
        C = MatRM::Zero(rows, cols);
        orbital_energies = Vec::Zero(rows);
        update_occupied_orbital_count();
    }

    inline int n_alpha() const { return n_occ; }
    inline int n_beta() const { return n_electrons - n_occ; }

    int charge() const {
        double nuclear_charge = 0.0;
        for (const auto &atom : atoms()) {
            nuclear_charge += atom.atomic_number;
        }
        return nuclear_charge - n_electrons;
    }

    int multiplicity() const {
        return n_unpaired_electrons + 1;
    }

    void set_charge(int c) {
        set_charge_multiplicity(c, multiplicity());
    }

    void set_multiplicity(int m) {
        set_charge_multiplicity(charge(), m);
    }

    void set_charge_multiplicity(int chg, unsigned int mult)
    {
        int current_charge = charge();
        bool state_changed = false;
        if(chg != current_charge) {
            n_electrons -= chg - current_charge;
            state_changed = true;
            if(n_electrons < 1) {
                throw std::runtime_error("Invalid charge: systems with no electrons are not supported");
            }
        }
        if(mult != multiplicity() || state_changed)
        {
            state_changed = true;
            n_unpaired_electrons = mult - 1;
            if(is_odd(n_electrons + n_unpaired_electrons)) {
                    throw std::runtime_error(
                        fmt::format(
                            "Invalid spin state for {} electrons: number of unpaired electrons ({}) must have the same parity",
                             n_electrons, n_unpaired_electrons));
            }
        }
        if(state_changed) update_occupied_orbital_count();
    }


    void update_occupied_orbital_count() {
        if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
            n_occ = n_electrons / 2;
            if(is_odd(n_electrons)) {
                throw std::runtime_error(
                            fmt::format(
                                "Invalid num electrons ({}) for restricted SCF: not even",
                                n_electrons)
                            );
            }
        }
        else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
            n_occ = (n_electrons - n_unpaired_electrons) / 2;
            n_unpaired_electrons = n_beta() - n_alpha();
        }
        else if constexpr (spinorbital_kind == SpinorbitalKind::General) {
            n_occ = n_electrons;
        }
    }

    const auto &atoms() const { return m_procedure.atoms(); }

    MatRM compute_soad() const {
        // computes Superposition-Of-Atomic-Densities guess for the molecular
        // density matrix in minimal basis; occupies subshells by smearing electrons
        // evenly over the orbitals compute number of atomic orbitals
        size_t nao = 0;
        for (const auto &atom : atoms()) {
            const auto Z = atom.atomic_number;
            nao += libint2::sto3g_num_ao(Z);
        }

        // compute the minimal basis density
        MatRM D_minbs = MatRM::Zero(nao, nao);
        size_t ao_offset = 0; // first AO of this atom
        for (const auto &atom : atoms()) {
            const auto Z = atom.atomic_number;
            const auto &occvec = libint2::sto3g_ao_occupation_vector(Z);
            for (const auto &occ : occvec) {
                D_minbs(ao_offset, ao_offset) = occ;
                ++ao_offset;
            }
        }

        int c = charge();
        // smear the charge across all shells
        if (c != 0) {
            double v = static_cast<double>(c) / D_minbs.rows();
            for (int i = 0; i < D_minbs.rows(); i++) {
                D_minbs(i, i) -= v;
            }
        }
        return D_minbs * 0.5; // we use densities normalized to # of electrons/2
    }

    void compute_initial_guess() {
        const auto tstart = std::chrono::high_resolution_clock::now();
        if constexpr(spinorbital_kind == SpinorbitalKind::Restricted)
        {
            S = m_procedure.compute_overlap_matrix();
            T = m_procedure.compute_kinetic_matrix();
            V = m_procedure.compute_nuclear_attraction_matrix();
        }
        else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted)
        {
            S.alpha() = m_procedure.compute_overlap_matrix();
            S.beta() = S.alpha();
            T.alpha() = m_procedure.compute_kinetic_matrix();
            T.beta() = T.alpha();
            V.alpha() = m_procedure.compute_nuclear_attraction_matrix();
            V.beta() = V.alpha();
        }
        else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
            S.alpha_alpha() = m_procedure.compute_overlap_matrix();
            T.alpha_alpha() = m_procedure.compute_kinetic_matrix();
            V.alpha_alpha() = m_procedure.compute_nuclear_attraction_matrix();
            S.beta_beta() = S.alpha_alpha();
            T.beta_beta() = T.alpha_alpha();
            V.beta_beta() = V.alpha_alpha();
        }
        H = T + V;
        F = H;
        // compute orthogonalizer X such that X.transpose() . S . X = I
        // one should think of columns of Xinv as the conditioned basis
        // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
        // X)
        // by default assume can manage to compute with condition number of S <=
        // 1/eps
        // this is probably too optimistic, but in well-behaved cases even 10^11 is
        // OK
        double S_condition_number_threshold = 1.0 / std::numeric_limits<double>::epsilon();
        if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            std::tie(X, Xinv, XtX_condition_number) = conditioning_orthogonalizer(S.alpha(), S_condition_number_threshold);
        }
        else {
            std::tie(X, Xinv, XtX_condition_number) = conditioning_orthogonalizer(S, S_condition_number_threshold);
        }

        auto D_minbs = compute_soad(); // compute guess in minimal basis
        libint2::BasisSet minbs("STO-3G", atoms());
        if (minbs == m_procedure.basis()) {
            if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
                D = D_minbs;
            }
            else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
                D.alpha() = D_minbs * (static_cast<double>(n_alpha())/ n_electrons);
                D.beta() = D_minbs * (static_cast<double>(n_beta()) / n_electrons);
            }
            else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
                D.alpha_alpha() = D_minbs * 0.5;
                D.beta_beta() = D_minbs * 0.5;
            }
        } else {
            // if basis != minimal basis, map non-representable SOAD guess
            // into the AO basis
            // by diagonalizing a Fock matrix
            tonto::log::debug("Projecting SOAD into atomic orbital basis...");

            if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
                F += tonto::ints::compute_2body_fock_mixed_basis(
                            m_procedure.basis(), D_minbs, minbs, true,
                            std::numeric_limits<double>::epsilon()
                            );
            }
            else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
                F.alpha() += tonto::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_alpha())/ n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
                F.beta() += tonto::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_beta()) / n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
            }
            else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
                F.alpha_alpha() += tonto::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_alpha())/ n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
                F.beta_beta() += tonto::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_beta()) / n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
            }

            update_molecular_orbitals(F);
            update_density_matrix();

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;
            tonto::log::debug("SOAD projection into AO basis took {:.5f} s", time_elapsed.count());
        }
    }

    void update_density_matrix() {
        if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
            D = C_occ * C_occ.transpose();
        }
        else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            D.alpha() = C_occ.block(0, 0, nbf, n_alpha()) * C_occ.block(0, 0, nbf, n_alpha()).transpose();
            D.beta() = C_occ.block(nbf, 0, nbf, n_beta()) * C_occ.block(nbf, 0, nbf, n_beta()).transpose();
            D *= 0.5;
        }
        else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
            D = (C_occ * C_occ.transpose()) * 0.5;
        }
    }

    void update_molecular_orbitals(const MatRM& fock) {
        // solve F C = e S C by (conditioned) transformation to F' C' = e C',
        // where
        // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
        if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            Eigen::SelfAdjointEigenSolver<MatRM> alpha_eig_solver(X.transpose() * fock.alpha() * X);
            Eigen::SelfAdjointEigenSolver<MatRM> beta_eig_solver(X.transpose() * fock.beta() * X);
            C.alpha() = X * alpha_eig_solver.eigenvectors();
            C.beta() = X * beta_eig_solver.eigenvectors();
            orbital_energies.block(0, 0, nbf, 1) = alpha_eig_solver.eigenvalues();
            orbital_energies.block(nbf, 0, nbf, 1) = beta_eig_solver.eigenvalues();
            C_occ = MatRM::Zero(2 * nbf, std::max(n_alpha(), n_beta()));
            C_occ.block(0, 0, nbf, n_alpha()) = C.alpha().leftCols(n_alpha());
            C_occ.block(nbf, 0, nbf, n_beta()) = C.beta().leftCols(n_beta());
        }
        else {
            Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * fock * X);
            C = X * eig_solver.eigenvectors();
            orbital_energies = eig_solver.eigenvalues();
            C_occ = C.leftCols(n_occ);
        }
    }

    void update_scf_energy(const MatRM& fock) {
        if(m_procedure.usual_scf_energy()) {
            ehf = expectation<spinorbital_kind>(D, fock);
        }
        else {
            ehf = 2 * expectation<spinorbital_kind>(D, H) + m_procedure.two_electron_energy();
        }
    }

    double compute_scf_energy() {
        // compute one-body integrals
        // count the number of electrons
        compute_initial_guess();
        K = m_procedure.compute_schwarz_ints();
        enuc = m_procedure.nuclear_repulsion_energy();
        MatRM D_diff;
        fmt::print("Beginning SCF\n");
        if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
            fmt::print("n_electrons: {}\nn_alpha: {}\nn_beta: {}\n", n_electrons, n_alpha(), n_beta());
        }
        total_time = 0.0;
        do {
            const auto tstart = std::chrono::high_resolution_clock::now();
            ++iter;
            // Last iteration's energy and density
            auto ehf_last = ehf;
            MatRM D_last = D;

            if (not incremental_Fbuild_started &&
                    rms_error < start_incremental_F_threshold) {
                incremental_Fbuild_started = true;
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter - 1;
                next_reset_threshold = rms_error / 1e1;
                tonto::log::debug("Starting incremental fock build");
            }
            if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
                F = H;
                D_diff = D;
            }
            if (reset_incremental_fock_formation && incremental_Fbuild_started) {
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter;
                next_reset_threshold = rms_error / 1e1;
                tonto::log::debug("Resetting incremental fock build");
            }

            // build a new Fock matrix
            // totally empirical precision variation, involves the condition number
            const auto precision_F = std::min(
                        std::min(1e-3 / XtX_condition_number, 1e-7),
                        std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
            F += m_procedure.compute_fock(spinorbital_kind, D_diff, precision_F, K);

            // compute HF energy with the non-extrapolated Fock matrix
            update_scf_energy(H + F);
            ediff_rel = std::abs((ehf - ehf_last) / ehf);

            // compute SCF error
            MatRM FD_comm(F.rows(), F.cols());
            if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
                FD_comm.alpha() = F.alpha() * D.alpha() * S.alpha() - S.alpha() * D.alpha() * F.alpha();
                FD_comm.beta() = F.beta() * D.beta() * S.beta() - S.beta() * D.beta() * F.beta();
            }
            else {
                FD_comm = F * D * S - S * D * F;
            }
            rms_error = FD_comm.norm() / FD_comm.size();
            if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
                reset_incremental_fock_formation = true;

            // DIIS extrapolate F
            MatRM F_diis = F; // extrapolated F cannot be used in incremental Fock
            // build; only used to produce the density
            // make a copy of the unextrapolated matrix
            diis.extrapolate(F_diis, FD_comm);

            update_molecular_orbitals(F_diis);
            update_density_matrix();
            D_diff = D - D_last;

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;

            if (iter == 1) {
                fmt::print("{:>6s} {: >20s} {: >20s} {: >20s} {: >10s}\n", "cycle",
                           "energy", "D(E)/E", "rms([F,D])/nn", "time");
            }
            fmt::print("{:>6d} {:>20.12f} {:>20.12e} {:>20.12e} {:>10.5f}\n", iter,
                       ehf + enuc, ediff_rel, rms_error, time_elapsed.count());
            total_time += time_elapsed.count();

        }   while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));
        fmt::print("{:10s} {:20.12f} seconds\n", "SCF took", total_time);
        fmt::print("{:10s} {:20.12f} hartree\n", "E_nn", enuc);
        fmt::print("{:10s} {:20.12f} hartree\n", "E_k",   expectation<spinorbital_kind>(D, T));
        fmt::print("{:10s} {:20.12f} hartree\n", "E_en",  expectation<spinorbital_kind>(D, V));
        fmt::print("{:10s} {:20.12f} hartree\n", "E_1e",  2 * expectation<spinorbital_kind>(D, H));
        fmt::print("{:10s} {:20.12f} hartree\n", "E_2e",  expectation<spinorbital_kind>(D, F));
        fmt::print("{:10s} {:20.12f} hartree\n", "E_tot", (ehf + enuc));
        return ehf + enuc;
    }

    void print_orbital_energies() {
        if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
            int n_a = n_alpha(), n_b = n_beta();
            int n_mo = orbital_energies.size() / 2;
            fmt::print("\nMolecular orbital energies\n");
            fmt::print("{0:3s}   {1:3s} {2:>16s}  {1:3s} {2:>16s}\n", "idx", "occ", "energy");
            for(int i = 0; i < n_mo; i++)
            {
                auto s_a = i < n_a ? "a" : " ";
                auto s_b = i < n_b ? "b" : " ";
                fmt::print("{:3d}   {:^3s} {:16.12f}  {:^3s} {:16.12f}\n",
                    i, s_a, orbital_energies(i),
                    s_b, orbital_energies(nbf + i)
                );
            }
        }
        else {
            int n_mo = orbital_energies.size();
            fmt::print("\nMolecular orbital energies\n");
            fmt::print("{0:3s}   {1:3s} {2:>16s}\n", "idx", "occ", "energy");
            for(int i = 0; i < n_mo; i++)
            {
                auto s = i < n_occ ? "ab" : " ";
                fmt::print("{:3d}   {:^3s} {:16.12f}\n",
                           i, s, orbital_energies(i)
                           );
            }
        }
    }

    Procedure &m_procedure;
    int n_electrons{0};
    int n_occ{0};
    int n_unpaired_electrons{0};
    int maxiter{100};
    size_t nbf{0};
    double conv = 1e-8;
    int iter = 0;
    double rms_error = 1.0;
    double ediff_rel = 0.0;
    double enuc{0.0};
    double ehf{0.0};
    double total_time{0.0};
    tonto::diis::DIIS<MatRM> diis; // start DIIS on second iteration
    bool reset_incremental_fock_formation = false;
    bool incremental_Fbuild_started = false;
    double start_incremental_F_threshold = 1e-5;
    double next_reset_threshold = 0.0;
    size_t last_reset_iteration = 0;
    MatRM D, S, T, V, H, K, X, Xinv, C, C_occ, F;
    Vec orbital_energies;
    double XtX_condition_number;
    bool verbose{false};
};

} // namespace tonto::scf
