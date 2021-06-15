#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/logger.h>
#include <occ/core/util.h>
#include <occ/core/diis.h>
#include <occ/qm/ints.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/density_fitting.h>
#include <occ/qm/wavefunction.h>
#include <occ/qm/energy_components.h>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <libint2/chemistry/sto3g_atomic_density.h>
#include <tuple>
#include <optional>

namespace occ::scf {

using occ::util::human_readable_size;
using occ::conditioning_orthogonalizer;
using occ::qm::SpinorbitalKind;
using occ::qm::expectation;
using occ::MatRM;
using occ::util::is_odd;
using occ::qm::BasisSet;
using occ::qm::Wavefunction;

template <typename Procedure, SpinorbitalKind spinorbital_kind>
struct SCF {

    SCF(Procedure &procedure, int diis_start = 2)
        : m_procedure(procedure), diis(diis_start) {
        n_electrons = m_procedure.num_e();
        nbf = m_procedure.basis().nbf();
        size_t rows, cols;
        std::tie(rows, cols) = occ::qm::matrix_dimensions<spinorbital_kind>(nbf);
        S = MatRM::Zero(rows, cols);
        T = MatRM::Zero(rows, cols);
        V = MatRM::Zero(rows, cols);
        H = MatRM::Zero(rows, cols);
        F = MatRM::Zero(rows, cols);
        D = MatRM::Zero(rows, cols);
        C = MatRM::Zero(rows, cols);
        orbital_energies = Vec::Zero(rows);
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

    void set_density_fitting_basis(const std::string& name)
    {
        if(spinorbital_kind != SpinorbitalKind::Restricted) throw std::runtime_error("Density fitting only implemented for RHF");
        BasisSet df_basis(name, m_procedure.atoms());
        BasisSet basis = m_procedure.basis();
        fmt::print("Loaded density-fitting basis-set ({}), {} shells, {} basis functions\n", name, df_basis.size(), occ::qm::nbf(df_basis));
        fmt::print("Storing DF overlap integrals requires {}\n",
                   human_readable_size(occ::qm::nbf(basis) * occ::qm::nbf(df_basis) * occ::qm::nbf(df_basis) * sizeof(double), "B"));
        df_engine.emplace(basis, df_basis);
    }

    void set_charge(int c) {
        set_charge_multiplicity(c, multiplicity());
    }

    void set_multiplicity(int m) {
        set_charge_multiplicity(charge(), m);
    }

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
        wfn.energy.nuclear_attraction = energy.at("nuclear.attraction");
        wfn.energy.nuclear_repulsion = energy.at("nuclear.repulsion");
        if(energy.contains("electronic.coulomb")) wfn.energy.coulomb = energy.at("electronic.coulomb");
        if(energy.contains("electronic.exchange")) wfn.energy.exchange = energy.at("electronic.exchange");
        wfn.energy.total = energy.at("total");
        wfn.spinorbital_kind = spinorbital_kind;
        wfn.T = T;
        wfn.V = V;
        return wfn;
    }

    void set_charge_multiplicity(int chg, unsigned int mult)
    {
        int current_charge = charge();
        bool state_changed = false;
        fmt::print("Charge = {}\nMultiplicity = {}\n", chg, mult);
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

    void set_initial_guess_from_wfn(const Wavefunction &wfn)
    {
        m_have_initial_guess = true;
        C = wfn.C;
        C_occ = wfn.C_occ;
        orbital_energies = wfn.mo_energies;
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
        update_density_matrix();
    }

    void compute_initial_guess() {
        if(m_have_initial_guess) return;
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
        BasisSet minbs("STO-3G", atoms());
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
            occ::log::debug("Projecting SOAD into atomic orbital basis...");

            if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
                F += occ::ints::compute_2body_fock_mixed_basis(
                            m_procedure.basis(), D_minbs, minbs, true,
                            std::numeric_limits<double>::epsilon()
                            );
            }
            else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
                F.alpha() += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_alpha())/ n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
                F.beta() += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_beta()) / n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
            }
            else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
                F.alpha_alpha() += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_alpha())/ n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
                F.beta_beta() += occ::ints::compute_2body_fock_mixed_basis(
                    m_procedure.basis(), D_minbs * (static_cast<double>(n_beta()) / n_electrons), minbs, true,
                    std::numeric_limits<double>::epsilon()
                );
            }

            update_molecular_orbitals(F);
            update_density_matrix();

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;
            occ::log::debug("SOAD projection into AO basis took {:.5f} s", time_elapsed.count());
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

    void update_scf_energy(bool incremental) {

        if(!incremental)
        {
            energy["electronic.kinetic"] = 2 * expectation<spinorbital_kind>(D, T);
            energy["nuclear.attraction"] = 2 * expectation<spinorbital_kind>(D, V);
            energy["electronic.1e"] = 2 * expectation<spinorbital_kind>(D, H);
            energy["nuclear.repulsion"] = m_procedure.nuclear_repulsion_energy();
        }
        if(m_procedure.usual_scf_energy()) {
            energy["electronic"] = 0.5 * energy["electronic.1e"];
            energy["electronic"] += expectation<spinorbital_kind>(D, F);
        }
        m_procedure.update_scf_energy(energy, incremental);
        energy["electronic.2e"] = energy["electronic"] - energy["electronic.1e"];
        energy["total"] = energy["electronic"] + energy["nuclear.repulsion"]
            + energy["solvation.nuclear"] + energy["solvation.surface"]
            + energy["solvation.CDS"];

    }

    std::string scf_kind() const {
        switch(spinorbital_kind) {
        case SpinorbitalKind::Unrestricted: return "unrestricted";
        case SpinorbitalKind::General: return "general";
        default: return "restricted";
        }
    }

    double compute_scf_energy() {
        // compute one-body integrals
        // count the number of electrons
        bool incremental{false};
        update_occupied_orbital_count();
        compute_initial_guess();
        K = m_procedure.compute_schwarz_ints();
        MatRM D_diff = D;
        MatRM D_last;
        MatRM FD_comm = MatRM::Zero(F.rows(), F.cols());
        update_scf_energy(incremental);
        fmt::print("Starting {} SCF iterations (Eguess = {:.12f})\n\n", scf_kind(), energy["total"]);
        if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
            fmt::print("n_electrons: {}\nn_alpha: {}\nn_beta: {}\n", n_electrons, n_alpha(), n_beta());
        }
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
                    rms_error < start_incremental_F_threshold) {
                incremental_Fbuild_started = true;
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter - 1;
                next_reset_threshold = rms_error / 1e1;
                occ::log::debug("Starting incremental fock build");
            }
            if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
                F = H;
                D_diff = D;
                incremental = false;
            }
            if (reset_incremental_fock_formation && incremental_Fbuild_started) {
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter;
                next_reset_threshold = rms_error / 1e1;
                occ::log::debug("Resetting incremental fock build");
            }

            // build a new Fock matrix
            // totally empirical precision variation, involves the condition number
            const auto precision_F = std::min(
                        std::min(1e-3 / XtX_condition_number, 1e-7),
                        std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
            if(df_engine == std::nullopt) {
                F += m_procedure.compute_fock(spinorbital_kind, D_diff, precision_F, K);
            }
            else {
                F = H + (*df_engine).compute_2body_fock_dfC(C_occ);
            }
            /*
            // code for testing DF implementation is working
            {
                BasisSet dfbs("def2-svp-jk", m_procedure.atoms());
                occ::df::DFFockEngine dfe(m_procedure.basis(), dfbs);
                occ::MatRM fock_df = dfe.compute_2body_fock_dfC(C_occ);
                fmt::print("Fock\n:{}\n\n\nFockDF\n{}\n", F, fock_df);
            }*/
            // compute HF energy with the non-extrapolated Fock matrix
            update_scf_energy(incremental);
            ediff_rel = std::abs((energy["electronic"] - ehf_last) / energy["electronic"]);

            // compute SCF error
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
            MatRM F_diis(F.rows(), F.cols()); // extrapolated F cannot be used in incremental Fock
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
                fmt::print("{:>6s} {: >20s} {: >20s} {: >20s} {: >10s}\n", "cycle",
                           "energy", "D(E)/E", "rms([F,D])/nn", "time");
            }
            fmt::print("{:>6d} {:>20.12f} {:>20.12e} {:>20.12e} {:>10.5f}\n", iter,
                       energy["total"], ediff_rel, rms_error, time_elapsed.count());
            total_time += time_elapsed.count();

        }   while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));
        fmt::print("\n{} SCF converged after {} seconds\n\n", scf_kind(), total_time);
        fmt::print("{}\n", energy);
        return energy["total"];
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
    occ::qm::EnergyComponents energy;
    int maxiter{100};
    size_t nbf{0};
    double conv = 1e-8;
    int iter = 0;
    double rms_error = 1.0;
    double ediff_rel = 0.0;
    double total_time{0.0};
    occ::diis::DIIS<MatRM> diis; // start DIIS on second iteration
    bool reset_incremental_fock_formation = false;
    bool incremental_Fbuild_started = false;
    double start_incremental_F_threshold = 1e-5;
    double next_reset_threshold = 0.0;
    size_t last_reset_iteration = 0;
    MatRM D, S, T, V, H, K, X, Xinv, C, C_occ, F;
    Vec orbital_energies;
    double XtX_condition_number;
    std::optional<occ::df::DFFockEngine> df_engine;
    bool m_have_initial_guess{false};
};

} // namespace occ::scf
