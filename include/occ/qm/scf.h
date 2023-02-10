#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/cdiis.h>
#include <occ/qm/ediis.h>
#include <occ/qm/expectation.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/mo.h>
#include <occ/qm/opmatrix.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace occ::scf {

constexpr auto OCC_MINIMAL_BASIS = "sto-3g";
using occ::conditioning_orthogonalizer;
using occ::Mat;
using occ::qm::expectation;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::qm::SpinorbitalKind::General;
using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;
using occ::util::is_odd;
namespace block = occ::qm::block;

namespace impl {

template <typename Procedure, SpinorbitalKind sk>
void set_core_matrices(const Procedure &proc, Mat &S, Mat &T, Mat &V, Mat &H,
                       Mat &Vecp) {

    bool calc_ecp = proc.have_effective_core_potentials();
    if constexpr (sk == Restricted) {
        S = proc.compute_overlap_matrix();
        T = proc.compute_kinetic_matrix();
        V = proc.compute_nuclear_attraction_matrix();
        if (calc_ecp) {
            Vecp = proc.compute_effective_core_potential_matrix();
        }
    } else if constexpr (sk == Unrestricted) {
        block::a(S) = proc.compute_overlap_matrix();
        block::b(S) = block::a(S);
        block::a(T) = proc.compute_kinetic_matrix();
        block::b(T) = block::a(T);
        block::a(V) = proc.compute_nuclear_attraction_matrix();
        block::b(V) = block::a(V);
        if (calc_ecp) {
            block::a(Vecp) = proc.compute_effective_core_potential_matrix();
            block::b(Vecp) = block::a(Vecp);
        }

    } else if constexpr (sk == General) {
        block::aa(S) = proc.compute_overlap_matrix();
        block::aa(T) = proc.compute_kinetic_matrix();
        block::aa(V) = proc.compute_nuclear_attraction_matrix();
        block::bb(S) = block::aa(S);
        block::bb(T) = block::aa(T);
        block::bb(V) = block::aa(V);
        if (calc_ecp) {
            block::aa(Vecp) = proc.compute_effective_core_potential_matrix();
            block::bb(Vecp) = block::aa(Vecp);
        }
    }
    H = T + V + Vecp;
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

    SCF(Procedure &procedure) : m_procedure(procedure) {
        n_electrons = m_procedure.active_electrons();
        n_frozen_electrons =
            m_procedure.total_electrons() - m_procedure.active_electrons();
        fmt::print("{} active electrons\n", n_electrons);
        fmt::print("{} frozen electrons\n", n_frozen_electrons);
        nbf = m_procedure.nbf();
        size_t rows, cols;
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<spinorbital_kind>(nbf);
        S = Mat::Zero(rows, cols);
        T = Mat::Zero(rows, cols);
        V = Mat::Zero(rows, cols);
        H = Mat::Zero(rows, cols);
        F = Mat::Zero(rows, cols);
        Vecp = Mat::Zero(rows, cols);

        mo.kind = spinorbital_kind;
        mo.D = Mat::Zero(rows, cols);
        mo.C = Mat::Zero(rows, cols);
        mo.energies = Vec::Zero(rows);
        mo.n_ao = nbf;

        Vpc = Mat::Zero(rows, cols);
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
        return nuclear_charge - n_electrons - n_frozen_electrons;
    }

    int multiplicity() const { return n_unpaired_electrons + 1; }

    void set_charge(int c) { set_charge_multiplicity(c, multiplicity()); }

    void set_multiplicity(int m) { set_charge_multiplicity(charge(), m); }

    Wavefunction wavefunction() const {
        Wavefunction wfn;
        wfn.atoms = m_procedure.atoms();
        wfn.basis = m_procedure.aobasis();
        wfn.nbf = wfn.basis.nbf();
        wfn.mo = mo;
        wfn.num_alpha = n_alpha();
        wfn.num_beta = n_beta();
        wfn.num_electrons = n_electrons;
        wfn.num_frozen_electrons = n_frozen_electrons;
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
        log::debug("Setting charge = {}, multiplicity = {} in scf", chg, mult);
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
        occ::log::debug("Setting MO n_alpha = {}, n_beta = {}", mo.n_alpha,
                        mo.n_beta);
        mo.n_alpha = n_alpha();
        mo.n_beta = n_beta();
    }

    const auto &atoms() const { return m_procedure.atoms(); }

    Mat compute_soad() const {
        // computes Superposition-Of-Atomic-Densities guess for the molecular
        // density matrix in minimal basis; occupies subshells by smearing
        // electrons evenly over the orbitals compute number of atomic orbitals
        size_t nao = 0;
        bool spherical = m_procedure.aobasis().is_pure();
        for (const auto &atom : atoms()) {
            const auto Z = atom.atomic_number;
            nao += occ::qm::guess::minimal_basis_nao(Z, spherical);
        }

        // compute the minimal basis density
        Mat D_minbs = Mat::Zero(nao, nao);
        size_t ao_offset = 0; // first AO of this atom
        int atom_index = 0;
        const auto &frozen_electrons = m_procedure.frozen_electrons();
        for (const auto &atom : atoms()) {
            const auto Z = atom.atomic_number;
            // the following code might be useful for a minimal
            // basis guess with ECPs
            /*
            double remaining_frozen = frozen_electrons[atom_index];
            */

            auto occvec =
                occ::qm::guess::minimal_basis_occupation_vector(Z, spherical);

            // the following code might be useful for a minimal
            // basis guess with ECPs
            /*
            {
                int offset = 0;
                while (remaining_frozen > 0.0) {
                    double r = std::max(occvec[offset], remaining_frozen);
                    occvec[offset] -= r;
                    remaining_frozen -= r;
                    offset++;
                }
            }

            occ::log::debug("Occupation vector for atom {} sum: {}", atom_index,
                            std::accumulate(occvec.begin(), occvec.end(), 0.0));
            */
            int bf = 0;
            for (const auto &occ : occvec) {
                D_minbs(ao_offset + bf, ao_offset + bf) = occ;
                bf++;
            }
            atom_index++;
            ao_offset += occvec.size();
        }

        int c = charge();
        // smear the charge across all shells
        if (c != 0) {
            double v = static_cast<double>(c) / D_minbs.rows();
            for (int i = 0; i < D_minbs.rows(); i++) {
                D_minbs(i, i) -= v;
            }
        }

        occ::log::debug("Minimal basis guess diagonal sum: {}", D_minbs.sum());
        return D_minbs * 0.5; // we use densities normalized to # of electrons/2
    }

    void set_initial_guess_from_wfn(const Wavefunction &wfn) {
        log::info("Setting initial guess from existing wavefunction");
        m_have_initial_guess = true;
        mo = wfn.mo;
        update_occupied_orbital_count();
        impl::set_core_matrices<Procedure, spinorbital_kind>(m_procedure, S, T,
                                                             V, H, Vecp);
        F = H;
        impl::set_conditioning_orthogonalizer<spinorbital_kind>(
            S, X, Xinv, XtX_condition_number);
        mo.update_density_matrix();
    }

    void compute_initial_guess() {
        if (m_have_initial_guess)
            return;

        log::info("Computing core hamiltonian");
        impl::set_core_matrices<Procedure, spinorbital_kind>(m_procedure, S, T,
                                                             V, H, Vecp);
        F = H;
        occ::timing::start(occ::timing::category::la);
        impl::set_conditioning_orthogonalizer<spinorbital_kind>(
            S, X, Xinv, XtX_condition_number);
        occ::timing::stop(occ::timing::category::la);

        occ::timing::start(occ::timing::category::guess);
        if (m_procedure.have_effective_core_potentials()) {
            // use core guess
            log::info(
                "Computing initial guess using core hamiltonian with ECPs");
            mo.update(X, F);
            mo.update_density_matrix();
            occ::timing::stop(occ::timing::category::guess);
            return;
        }

        log::info("Computing initial guess using SOAD in minimal basis");
        auto D_minbs = compute_soad(); // compute guess in minimal basis
        if (m_procedure.aobasis().name() == OCC_MINIMAL_BASIS) {
            if constexpr (spinorbital_kind == Restricted) {
                mo.D = D_minbs;
            } else if constexpr (spinorbital_kind == Unrestricted) {
                block::a(mo.D) =
                    D_minbs * (static_cast<double>(n_alpha()) / n_electrons);
                block::b(mo.D) =
                    D_minbs * (static_cast<double>(n_beta()) / n_electrons);
            } else if constexpr (spinorbital_kind == General) {
                block::aa(mo.D) = D_minbs * 0.5;
                block::bb(mo.D) = D_minbs * 0.5;
            }
        } else {
            // if basis != minimal basis, map non-representable SOAD guess
            // into the AO basis
            // by diagonalizing a Fock matrix
            log::debug(
                "Projecting minimal basis guess into atomic orbital basis...");
            const auto tstart = std::chrono::high_resolution_clock::now();
            auto minbs =
                occ::qm::AOBasis::load(m_procedure.atoms(), OCC_MINIMAL_BASIS);
            minbs.set_pure(m_procedure.aobasis().is_pure());
            occ::qm::MolecularOrbitals mo_minbs;
            mo_minbs.kind = spinorbital_kind;
            mo_minbs.D = D_minbs;
            F += m_procedure.compute_fock_mixed_basis(mo_minbs, minbs, true);
            mo.update(X, F);
            mo.update_density_matrix();

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;
            log::debug("SOAD projection into AO basis took {:.5f} s",
                       time_elapsed.count());
        }
        occ::timing::stop(occ::timing::category::guess);
    }

    void update_scf_energy(bool incremental) {

        if (!incremental) {
            occ::timing::start(occ::timing::category::la);
            energy["electronic.kinetic"] =
                2 * expectation<spinorbital_kind>(mo.D, T);
            energy["electronic.nuclear"] =
                2 * expectation<spinorbital_kind>(mo.D, V);
            energy["electronic.1e"] =
                2 * expectation<spinorbital_kind>(mo.D, H);
            occ::timing::stop(occ::timing::category::la);
        }
        if (m_procedure.usual_scf_energy()) {
            occ::timing::start(occ::timing::category::la);
            energy["electronic"] = 0.5 * energy["electronic.1e"];
            energy["electronic"] += expectation<spinorbital_kind>(mo.D, F);
            energy["electronic.2e"] =
                energy["electronic"] - energy["electronic.1e"];
            energy["total"] =
                energy["electronic"] + energy["nuclear.repulsion"];
            occ::timing::stop(occ::timing::category::la);
        }
        if (m_procedure.have_effective_core_potentials()) {
            energy["electronic.ecp"] =
                expectation<spinorbital_kind>(mo.D, Vecp);
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
        if (converged)
            return energy["total"];
        // compute one-body integrals
        // count the number of electrons
        bool incremental{false};
        update_occupied_orbital_count();
        compute_initial_guess();
        K = m_procedure.compute_schwarz_ints();
        Mat D_diff = mo.D;
        Mat D_last;
        Mat FD_comm = Mat::Zero(F.rows(), F.cols());
        update_scf_energy(incremental);
        log::info("starting {} scf iterations", scf_kind());
        log::debug("{} electrons total", n_electrons);
        log::debug("{} alpha electrons", n_alpha());
        log::debug("{} beta electrons", n_beta());
        log::debug("net charge {}", charge());
        total_time = 0.0;

        auto write_mat = [](const std::string &filename, const Mat &mat) {
            std::ofstream file(filename);
            if (file.is_open()) {
                file << mat << '\n';
            }
        };
        write_mat("v.txt", V);
        write_mat("t.txt", T);
        write_mat("h.txt", H);

        do {
            const auto tstart = std::chrono::high_resolution_clock::now();
            ++iter;
            // Last iteration's energy and density
            auto ehf_last = energy["electronic"];
            D_last = mo.D;
            H = T + V + Vecp;
            m_procedure.update_core_hamiltonian(mo, H);
            incremental = true;

            if (not incremental_Fbuild_started &&
                diis_error < start_incremental_F_threshold) {
                incremental_Fbuild_started = true;
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter - 1;
                next_reset_threshold = diis_error / 10;
                log::debug("starting incremental fock build");
            }
            if (reset_incremental_fock_formation ||
                not incremental_Fbuild_started) {
                F = H;
                D_diff = mo.D;
                incremental = false;
            }
            if (reset_incremental_fock_formation &&
                incremental_Fbuild_started) {
                reset_incremental_fock_formation = false;
                last_reset_iteration = iter;
                next_reset_threshold = diis_error / 10;
                log::debug("resetting incremental fock build");
            }

            // build a new Fock matrix
            // totally empirical precision variation, involves the condition
            // number
            const auto precision_F =
                std::min(std::min(1e-3 / XtX_condition_number, 1e-7),
                         std::max(diis_error / 1e4,
                                  std::numeric_limits<double>::epsilon()));

            std::swap(mo.D, D_diff);
            F += m_procedure.compute_fock(mo, precision_F, K);
            std::swap(mo.D, D_diff);

            // compute HF energy with the non-extrapolated Fock matrix
            update_scf_energy(incremental);
            ediff_rel = std::abs((energy["electronic"] - ehf_last) /
                                 energy["electronic"]);

            Mat F_diis = diis.update(S, mo.D, F);
            // double prev_error = diis_error;
            diis_error = diis.max_error();
            /*
            bool use_ediis = (diis_error > 1e-1) || (prev_error /
            diis.min_error() > 1.1);

            Mat F_ediis = ediis.update(D, F, energy["electronic"]);
            if(use_ediis) F_diis = F_ediis;
            else if(diis_error > 1e-4) {
                F_diis = (10 * diis_error) * F_ediis + (1 - 10 * diis_error) *
            F_diis;
            }
            */

            if (diis_error < next_reset_threshold ||
                iter - last_reset_iteration >= 8)
                reset_incremental_fock_formation = true;

            mo.update(X, F_diis);
            mo.update_density_matrix();
            D_diff = mo.D - D_last;

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;

            if (iter == 1) {
                log::info("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}", "#",
                          "E (Hartrees)", "|\u0394E|/E", "max|FDS-SDF|",
                          "T (s)");
            }
            log::info("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}", iter,
                      energy["total"], ediff_rel, diis_error,
                      time_elapsed.count());
            std::cout << std::flush;
            total_time += time_elapsed.count();

        } while (((ediff_rel > energy_convergence_threshold) ||
                  (diis_error > commutator_convergence_threshold)) &&
                 (iter < maxiter));
        log::info("{} spinorbital SCF energy converged after {:.5f} seconds",
                  scf_kind(), total_time);
        log::info("{}", energy);
        converged = true;
        return energy["total"];
    }

    Procedure &m_procedure;
    int n_electrons{0}, n_frozen_electrons{0};
    int n_occ{0};
    int n_unpaired_electrons{0};
    occ::core::EnergyComponents energy;
    int maxiter{100};
    size_t nbf{0};
    double energy_convergence_threshold = 1e-6;
    double commutator_convergence_threshold = 1e-5;
    int iter = 0;
    double diis_error{1.0};
    double ediff_rel = 0.0;
    double total_time{0.0};
    occ::qm::CDIIS diis; // start DIIS on second iteration
    occ::qm::EDIIS ediis;
    bool reset_incremental_fock_formation = false;
    bool incremental_Fbuild_started = false;
    bool converged{false};
    double start_incremental_F_threshold = 1e-4;
    double next_reset_threshold = 0.0;
    size_t last_reset_iteration = 0;
    occ::qm::MolecularOrbitals mo;
    Mat S, T, V, H, K, X, Xinv, F, Vpc, Vecp;
    double XtX_condition_number;
    bool m_have_initial_guess{false};
};

} // namespace occ::scf
