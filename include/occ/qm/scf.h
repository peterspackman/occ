#pragma once
#include <occ/core/diis.h>
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/logger.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/ints.h>
#include <occ/qm/opmatrix.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace occ::scf {

constexpr auto OCC_MINIMAL_BASIS = "mini";
using occ::conditioning_orthogonalizer;
using occ::Mat;
using occ::qm::BasisSet;
using occ::qm::expectation;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::qm::SpinorbitalKind::General;
using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;
using occ::util::human_readable_size;
using occ::util::is_odd;
namespace block = occ::qm::block;

inline double rms_error_diis(const Mat &commutator) {
    return commutator.norm() / commutator.size();
}

inline double maximum_error_diis(const Mat &commutator) {
    return commutator.maxCoeff();
}

namespace impl {

template <typename Procedure, SpinorbitalKind sk>
void set_core_matrices(const Procedure &proc, Mat &S, Mat &T, Mat &V, Mat &H) {
    if constexpr (sk == Restricted) {
        S = proc.compute_overlap_matrix();
        T = proc.compute_kinetic_matrix();
        V = proc.compute_nuclear_attraction_matrix();
    } else if constexpr (sk == Unrestricted) {
        block::a(S) = proc.compute_overlap_matrix();
        block::b(S) = block::a(S);
        block::a(T) = proc.compute_kinetic_matrix();
        block::b(T) = block::a(T);
        block::a(V) = proc.compute_nuclear_attraction_matrix();
        block::b(V) = block::a(V);
    } else if constexpr (sk == General) {
        block::aa(S) = proc.compute_overlap_matrix();
        block::aa(T) = proc.compute_kinetic_matrix();
        block::aa(V) = proc.compute_nuclear_attraction_matrix();
        block::bb(S) = block::aa(S);
        block::bb(T) = block::aa(T);
        block::bb(V) = block::aa(V);
    }
    H = T + V;
}

template <SpinorbitalKind sk>
void set_conditioning_orthogonalizer(const Mat &S, Mat &X, Mat &Xinv,
                                     double &XtX_condition_number) {
    // compute orthogonalizer X such that X.transpose() . S . X = I
    // one should think of columns of Xinv as the conditioned basis
    // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose()
    // . X) by default assume can manage to compute with condition number of
    // S <= 1/eps this is probably too optimistic, but in well-behaved cases
    // even 10^11 is OK
    double S_condition_number_threshold =
        1.0 / std::numeric_limits<double>::epsilon();
    if constexpr (sk == Unrestricted) {
        std::tie(X, Xinv, XtX_condition_number) = conditioning_orthogonalizer(
            block::a(S), S_condition_number_threshold);
    } else {
        std::tie(X, Xinv, XtX_condition_number) =
            conditioning_orthogonalizer(S, S_condition_number_threshold);
    }
}

} // namespace impl

template <typename Procedure, SpinorbitalKind spinorbital_kind> struct SCF {

    SCF(Procedure &procedure, int diis_start = 2)
        : m_procedure(procedure), diis(diis_start) {
        n_electrons = m_procedure.num_e();
        nbf = m_procedure.basis().nbf();
        size_t rows, cols;
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<spinorbital_kind>(nbf);
        S = Mat::Zero(rows, cols);
        T = Mat::Zero(rows, cols);
        V = Mat::Zero(rows, cols);
        H = Mat::Zero(rows, cols);
        F = Mat::Zero(rows, cols);
        D = Mat::Zero(rows, cols);
        C = Mat::Zero(rows, cols);
        Vpc = Mat::Zero(rows, cols);
        orbital_energies = Vec::Zero(rows);
        energy["nuclear.repulsion"] = m_procedure.nuclear_repulsion_energy();
        if (!m_procedure.supports_incremental_fock_build())
            start_incremental_F_threshold = 0.0;
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

    int multiplicity() const { return n_unpaired_electrons + 1; }

    void set_charge(int c) { set_charge_multiplicity(c, multiplicity()); }

    void set_multiplicity(int m) { set_charge_multiplicity(charge(), m); }

    Wavefunction wavefunction() const {
        Wavefunction wfn;
        wfn.atoms = m_procedure.atoms();
        wfn.basis = m_procedure.basis();
        wfn.nbf = wfn.basis.nbf();
        wfn.mo_energies = orbital_energies;
        wfn.D = D;
        wfn.C = C;
        wfn.C_occ = C_occ;
        wfn.num_alpha = n_alpha();
        wfn.num_beta = n_beta();
        wfn.num_electrons = n_electrons;
        wfn.energy.core = energy.at("electronic.1e");
        wfn.energy.kinetic = energy.at("electronic.kinetic");
        wfn.energy.nuclear_attraction = energy.at("electronic.nuclear");
        wfn.energy.nuclear_repulsion = energy.at("nuclear.repulsion");
        if (energy.contains("electronic.coulomb"))
            wfn.energy.coulomb = energy.at("electronic.coulomb");
        if (energy.contains("electronic.exchange"))
            wfn.energy.exchange = energy.at("electronic.exchange");
        wfn.energy.total = energy.at("total");
        wfn.spinorbital_kind = spinorbital_kind;
        wfn.T = T;
        wfn.V = V;
        return wfn;
    }

    void set_charge_multiplicity(int chg, unsigned int mult) {
        int current_charge = charge();
        bool state_changed = false;
        occ::log::debug("Setting charge = {}, multiplicity = {} in scf", chg,
                        mult);
        if (chg != current_charge) {
            n_electrons -= chg - current_charge;
            state_changed = true;
            if (n_electrons < 1) {
                throw std::runtime_error("Invalid charge: systems with no "
                                         "electrons are not supported");
            }
        }
        if (mult != multiplicity() || state_changed) {
            state_changed = true;
            n_unpaired_electrons = mult - 1;
            if (is_odd(n_electrons + n_unpaired_electrons)) {
                throw std::runtime_error(fmt::format(
                    "Invalid spin state for {} electrons: number of unpaired "
                    "electrons ({}) must have the same parity",
                    n_electrons, n_unpaired_electrons));
            }
        }
        if (state_changed)
            update_occupied_orbital_count();
    }

    void update_occupied_orbital_count() {
        if constexpr (spinorbital_kind == Restricted) {
            n_occ = n_electrons / 2;
            if (is_odd(n_electrons)) {
                throw std::runtime_error(fmt::format(
                    "Invalid num electrons ({}) for restricted SCF: not even",
                    n_electrons));
            }
        } else if constexpr (spinorbital_kind == Unrestricted) {
            n_occ = (n_electrons - n_unpaired_electrons) / 2;
            n_unpaired_electrons = n_beta() - n_alpha();
        } else if constexpr (spinorbital_kind == General) {
            n_occ = n_electrons;
        }
    }

    const auto &atoms() const { return m_procedure.atoms(); }

    Mat compute_soad() const {
        // computes Superposition-Of-Atomic-Densities guess for the molecular
        // density matrix in minimal basis; occupies subshells by smearing
        // electrons evenly over the orbitals compute number of atomic orbitals
        size_t nao = 0;
        for (const auto &atom : atoms()) {
            const auto Z = atom.atomic_number;
            nao += occ::qm::guess::minimal_basis_nao(Z);
        }

        // compute the minimal basis density
        Mat D_minbs = Mat::Zero(nao, nao);
        size_t ao_offset = 0; // first AO of this atom
        for (const auto &atom : atoms()) {
            const auto Z = atom.atomic_number;
            const auto occvec =
                occ::qm::guess::minimal_basis_occupation_vector(Z);
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

    void set_initial_guess_from_wfn(const Wavefunction &wfn) {
        m_have_initial_guess = true;
        C = wfn.C;
        C_occ = wfn.C_occ;
        orbital_energies = wfn.mo_energies;
        update_occupied_orbital_count();
        impl::set_core_matrices<Procedure, spinorbital_kind>(m_procedure, S, T,
                                                             V, H);
        F = H;
        impl::set_conditioning_orthogonalizer<spinorbital_kind>(
            S, X, Xinv, XtX_condition_number);
        update_density_matrix();
    }

    void compute_initial_guess() {
        if (m_have_initial_guess)
            return;
        const auto tstart = std::chrono::high_resolution_clock::now();
        impl::set_core_matrices<Procedure, spinorbital_kind>(m_procedure, S, T,
                                                             V, H);
        F = H;
        occ::timing::start(occ::timing::category::la);
        impl::set_conditioning_orthogonalizer<spinorbital_kind>(
            S, X, Xinv, XtX_condition_number);
        occ::timing::stop(occ::timing::category::la);

        occ::timing::start(occ::timing::category::guess);
        auto D_minbs = compute_soad(); // compute guess in minimal basis
        BasisSet minbs(OCC_MINIMAL_BASIS, atoms());
        if (minbs == m_procedure.basis()) {
            if constexpr (spinorbital_kind == Restricted) {
                D = D_minbs;
            } else if constexpr (spinorbital_kind == Unrestricted) {
                block::a(D) =
                    D_minbs * (static_cast<double>(n_alpha()) / n_electrons);
                block::b(D) =
                    D_minbs * (static_cast<double>(n_beta()) / n_electrons);
            } else if constexpr (spinorbital_kind == General) {
                block::aa(D) = D_minbs * 0.5;
                block::bb(D) = D_minbs * 0.5;
            }
        } else {
            // if basis != minimal basis, map non-representable SOAD guess
            // into the AO basis
            // by diagonalizing a Fock matrix
            occ::log::debug(
                "Projecting minimal basis guess into atomic orbital basis...");

            if constexpr (spinorbital_kind == Restricted) {
                F += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs, minbs, true,
                    commutator_convergence_threshold);
            } else if constexpr (spinorbital_kind == Unrestricted) {
                block::a(F) += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs, minbs, true,
                    commutator_convergence_threshold);

                block::b(F) = block::a(F);
            } else if constexpr (spinorbital_kind == General) {
                // TODO fix multiplicity != 1
                block::aa(F) += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs, minbs, true,
                    commutator_convergence_threshold);
                block::bb(F) = block::aa(F);
            }

            update_molecular_orbitals(F);
            update_density_matrix();

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;
            occ::log::debug("SOAD projection into AO basis took {:.5f} s",
                            time_elapsed.count());
        }
        occ::timing::stop(occ::timing::category::guess);
    }

    void update_density_matrix() {
        occ::timing::start(occ::timing::category::la);
        if constexpr (spinorbital_kind == Restricted) {
            D = C_occ * C_occ.transpose();
        } else if constexpr (spinorbital_kind == Unrestricted) {
            block::a(D) = C_occ.block(0, 0, nbf, n_alpha()) *
                          C_occ.block(0, 0, nbf, n_alpha()).transpose();
            block::b(D) = C_occ.block(nbf, 0, nbf, n_beta()) *
                          C_occ.block(nbf, 0, nbf, n_beta()).transpose();
            D *= 0.5;
        } else if constexpr (spinorbital_kind == General) {
            D = (C_occ * C_occ.transpose()) * 0.5;
        }
        occ::timing::stop(occ::timing::category::la);
    }

    void update_molecular_orbitals(const Mat &fock) {
        // solve F C = e S C by (conditioned) transformation to F' C' = e C',
        // where
        // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
        occ::timing::start(occ::timing::category::mo);
        if constexpr (spinorbital_kind == Unrestricted) {
            Eigen::SelfAdjointEigenSolver<Mat> alpha_eig_solver(
                X.transpose() * block::a(fock) * X);
            Eigen::SelfAdjointEigenSolver<Mat> beta_eig_solver(
                X.transpose() * block::b(fock) * X);
            block::a(C) = X * alpha_eig_solver.eigenvectors();
            block::b(C) = X * beta_eig_solver.eigenvectors();
            block::a(orbital_energies) = alpha_eig_solver.eigenvalues();
            block::b(orbital_energies) = beta_eig_solver.eigenvalues();
            C_occ = Mat::Zero(2 * nbf, std::max(n_alpha(), n_beta()));
            C_occ.block(0, 0, nbf, n_alpha()) = block::a(C).leftCols(n_alpha());
            C_occ.block(nbf, 0, nbf, n_beta()) = block::b(C).leftCols(n_beta());
        } else {
            Eigen::SelfAdjointEigenSolver<Mat> eig_solver(X.transpose() * fock *
                                                          X);
            C = X * eig_solver.eigenvectors();
            orbital_energies = eig_solver.eigenvalues();
            C_occ = C.leftCols(n_occ);
        }
        occ::timing::stop(occ::timing::category::mo);
    }

    void update_scf_energy(bool incremental) {

        if (!incremental) {
            occ::timing::start(occ::timing::category::la);
            energy["electronic.kinetic"] =
                2 * expectation<spinorbital_kind>(D, T);
            energy["electronic.nuclear"] =
                2 * expectation<spinorbital_kind>(D, V);
            energy["electronic.1e"] = 2 * expectation<spinorbital_kind>(D, H);
            occ::timing::stop(occ::timing::category::la);
        }
        if (m_procedure.usual_scf_energy()) {
            occ::timing::start(occ::timing::category::la);
            energy["electronic"] = 0.5 * energy["electronic.1e"];
            energy["electronic"] += expectation<spinorbital_kind>(D, F);
            energy["electronic.2e"] =
                energy["electronic"] - energy["electronic.1e"];
            energy["total"] =
                energy["electronic"] + energy["nuclear.repulsion"];
            occ::timing::stop(occ::timing::category::la);
        }
        m_procedure.update_scf_energy(energy, incremental);
    }

    std::string scf_kind() const {
        switch (spinorbital_kind) {
        case Unrestricted:
            return "unrestricted";
        case General:
            return "general";
        default:
            return "restricted";
        }
    }

    double compute_scf_energy() {
        // compute one-body integrals
        // count the number of electrons
        bool incremental{false};
        update_occupied_orbital_count();
        compute_initial_guess();
        K = m_procedure.compute_schwarz_ints();
        Mat D_diff = D;
        Mat D_last;
        Mat FD_comm = Mat::Zero(F.rows(), F.cols());
        update_scf_energy(incremental);
        fmt::print("starting {} scf iterations \n", scf_kind());
        occ::log::info("{} electrons total", n_electrons);
        occ::log::info("{} alpha electrons", n_alpha());
        occ::log::info("{} beta electrons", n_beta());
        total_time = 0.0;

        do {
            const auto tstart = std::chrono::high_resolution_clock::now();
            ++iter;
            // Last iteration's energy and density
            auto ehf_last = energy["electronic"];
            D_last = D;
            H = T + V;
            m_procedure.update_core_hamiltonian(spinorbital_kind, D, H);
            incremental = true;

            if (not incremental_Fbuild_started &&
                diis_error < start_incremental_F_threshold) {
                incremental_Fbuild_started = true;
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter - 1;
                next_reset_threshold = diis_error / 10;
                occ::log::info("starting incremental fock build");
            }
            if (reset_incremental_fock_formation ||
                not incremental_Fbuild_started) {
                F = H;
                D_diff = D;
                incremental = false;
            }
            if (reset_incremental_fock_formation &&
                incremental_Fbuild_started) {
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter;
                next_reset_threshold = diis_error / 10;
                occ::log::info("resetting incremental fock build");
            }

            // build a new Fock matrix
            // totally empirical precision variation, involves the condition
            // number
            const auto precision_F =
                std::min(std::min(1e-3 / XtX_condition_number, 1e-7),
                         std::max(diis_error / 1e4,
                                  std::numeric_limits<double>::epsilon()));

            F += m_procedure.compute_fock(spinorbital_kind, D_diff, precision_F,
                                          K);

            // compute HF energy with the non-extrapolated Fock matrix
            update_scf_energy(incremental);
            ediff_rel = std::abs((energy["electronic"] - ehf_last) /
                                 energy["electronic"]);

            // compute SCF error
            if(diis_error < 0.1) {
                if constexpr (spinorbital_kind == Unrestricted) {
                    const auto &Fa = block::a(F);
                    const auto &Fb = block::a(F);
                    const auto &Da = block::a(D);
                    const auto &Db = block::a(D);
                    const auto &Sa = block::a(S);
                    const auto &Sb = block::a(S);
                    block::a(FD_comm) = Fa * Da * Sa - Fa * Da * Sa;
                    block::b(FD_comm) = Fb * Db * Sb - Sb * Db * Fb;
                } else {
                    FD_comm = F * D * S - S * D * F;
                }
                diis_error = maximum_error_diis(FD_comm);
            }
            else {
                FD_comm = D_diff;
                diis_error = std::abs(energy["electronic"] - ehf_last);
            }

            if (diis_error < next_reset_threshold ||
                iter - last_reset_iteration >= 8)
                reset_incremental_fock_formation = true;

            // DIIS extrapolate F
            Mat F_diis(
                F.rows(),
                F.cols()); // extrapolated F cannot be used in incremental Fock
            F_diis = F;
            // build; only used to produce the density
            // make a copy of the unextrapolated matrix
            diis.extrapolate(F_diis, FD_comm);

            update_molecular_orbitals(F_diis);
            update_density_matrix();
            D_diff = D - D_last;

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;

            if (iter == 1) {
                fmt::print("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}\n", "#",
                           "E (Hartrees)", "|\u0394E|/E", "max|FDS-SDF|",
                           "T (s)");
            }
            fmt::print("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}\n",
                       iter, energy["total"], ediff_rel, diis_error,
                       time_elapsed.count());
            std::cout << std::flush;
            total_time += time_elapsed.count();

        } while (((ediff_rel > energy_convergence_threshold) ||
                  (diis_error > commutator_convergence_threshold)) &&
                 (iter < maxiter));
        fmt::print(
            "\n{} spinorbital SCF energy converged after {:.5f} seconds\n\n",
            scf_kind(), total_time);
        fmt::print("{}\n", energy);
        return energy["total"];
    }

    void print_orbital_energies() {
        if constexpr (spinorbital_kind == Unrestricted) {
            int n_a = n_alpha(), n_b = n_beta();
            int n_mo = orbital_energies.size() / 2;
            fmt::print("\nMolecular orbital energies\n");
            fmt::print("{0:3s}   {1:3s} {2:>16s}  {1:3s} {2:>16s}\n", "idx",
                       "occ", "energy");
            for (int i = 0; i < n_mo; i++) {
                auto s_a = i < n_a ? "a" : " ";
                auto s_b = i < n_b ? "b" : " ";
                fmt::print("{:3d}   {:^3s} {:16.12f}  {:^3s} {:16.12f}\n", i,
                           s_a, orbital_energies(i), s_b,
                           orbital_energies(nbf + i));
            }
        } else {
            int n_mo = orbital_energies.size();
            fmt::print("\nMolecular orbital energies\n");
            fmt::print("{0:3s}   {1:3s} {2:>16s}\n", "idx", "occ", "energy");
            for (int i = 0; i < n_mo; i++) {
                auto s = i < n_occ ? "ab" : " ";
                fmt::print("{:3d}   {:^3s} {:16.12f}\n", i, s,
                           orbital_energies(i));
            }
        }
    }

    Procedure &m_procedure;
    int n_electrons{0};
    int n_occ{0};
    int n_unpaired_electrons{0};
    occ::core::EnergyComponents energy;
    int maxiter{100};
    size_t nbf{0};
    double energy_convergence_threshold = 5e-8;
    double commutator_convergence_threshold = 5e-8;
    int iter = 0;
    double diis_error{1.0};
    double ediff_rel = 0.0;
    double total_time{0.0};
    occ::core::diis::DIIS<Mat> diis; // start DIIS on second iteration
    bool reset_incremental_fock_formation = false;
    bool incremental_Fbuild_started = false;
    double start_incremental_F_threshold = 1e-4;
    double next_reset_threshold = 0.0;
    size_t last_reset_iteration = 0;
    Mat D, S, T, V, H, K, X, Xinv, C, C_occ, F, Vpc;
    Vec orbital_energies;
    double XtX_condition_number;
    bool m_have_initial_guess{false};
};

} // namespace occ::scf
