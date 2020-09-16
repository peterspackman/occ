#pragma once
#include "ints.h"
#include <tuple>
#include <libint2/diis.h>
#include <Eigen/Dense>
#include <fmt/core.h>

namespace craso::scf
{

    using craso::ints::RowMajorMatrix;

    std::tuple<RowMajorMatrix, RowMajorMatrix, double> conditioning_orthogonalizer(const RowMajorMatrix& S, double S_condition_number_threshold);

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
    std::tuple<RowMajorMatrix, RowMajorMatrix, size_t, double, double> gensqrtinv(const RowMajorMatrix& S, bool symmetric = false, double max_condition_number = 1e8);


    template <typename Procedure>
    struct SCF
    {
        SCF(Procedure &procedure) : m_procedure(procedure), diis(2)
        {
            nelectrons = m_procedure.num_e();
            ndocc = nelectrons / 2;
        }


        void compute_initial_guess()
        {
            const auto tstart = std::chrono::high_resolution_clock::now();
            S = m_procedure.compute_overlap_matrix();
            T = m_procedure.compute_kinetic_matrix();
            V = m_procedure.compute_nuclear_attraction_matrix();
            H = T + V;
            auto D_minbs = m_procedure.compute_soad(); // compute guess in minimal basis
            libint2::BasisSet minbs("STO-3G", m_procedure.atoms());
            if (minbs == m_procedure.basis())
                D = D_minbs;
            else
            {
                // compute orthogonalizer X such that X.transpose() . S . X = I
                // one should think of columns of Xinv as the conditioned basis
                // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
                // X)
                // by default assume can manage to compute with condition number of S <=
                // 1/eps
                // this is probably too optimistic, but in well-behaved cases even 10^11 is
                // OK
                double S_condition_number_threshold =
                    1.0 / std::numeric_limits<double>::epsilon();
                std::tie(X, Xinv, XtX_condition_number) =
                    conditioning_orthogonalizer(S, S_condition_number_threshold);
                // if basis != minimal basis, map non-representable SOAD guess
                // into the AO basis
                // by diagonalizing a Fock matrix
                if(verbose) fmt::print("Projecting SOAD into atomic orbital basis: ");
                F = H;
                F += craso::ints::compute_2body_fock_general(m_procedure.basis(), D_minbs, minbs, true, std::numeric_limits<double>::epsilon());

                // solve F C = e S C by (conditioned) transformation to F' C' = e C',
                // where
                // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
                Eigen::SelfAdjointEigenSolver<RowMajorMatrix> eig_solver(X.transpose() * F * X);
                C = X * eig_solver.eigenvectors();

                // compute density, D = C(occ) . C(occ)T
                C_occ = C.leftCols(ndocc);
                D = C_occ * C_occ.transpose();

                const auto tstop = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> time_elapsed = tstop - tstart;
                if(verbose) fmt::print("{:.5f} s\n", time_elapsed.count());
            }
        }


        double compute_scf_energy()
        {
                // compute one-body integrals
                    // count the number of electrons
            compute_initial_guess();
            K = m_procedure.compute_schwarz_ints();
            enuc = m_procedure.nuclear_repulsion_energy();
            RowMajorMatrix D_diff;
            auto n2 = D.cols() * D.rows();
            RowMajorMatrix evals;

            fmt::print("Beginning SCF\n");
            total_time = 0.0;
            do
            {
                const auto tstart = std::chrono::high_resolution_clock::now();
                ++iter;

                // Last iteration's energy and density
                auto ehf_last = ehf;
                RowMajorMatrix D_last = D;

                if (not incremental_Fbuild_started &&
                    rms_error < start_incremental_F_threshold)
                {
                    incremental_Fbuild_started = true;
                    reset_incremental_fock_formation = false;
                    last_reset_iteration = iter - 1;
                    next_reset_threshold = rms_error / 1e1;
                    if (verbose) fmt::print("Starting incremental fock build\n");
                }
                if (reset_incremental_fock_formation || not incremental_Fbuild_started)
                {
                    F = H;
                    D_diff = D;
                }
                if (reset_incremental_fock_formation && incremental_Fbuild_started)
                {
                    reset_incremental_fock_formation = false;
                    last_reset_iteration = iter;
                    next_reset_threshold = rms_error / 1e1;
                    if (verbose) fmt::print("Resetting incremental fock build\n");
                }

                // build a new Fock matrix
                // totally empirical precision variation, involves the condition number
                const auto precision_F = std::min(
                    std::min(1e-3 / XtX_condition_number, 1e-7),
                    std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
                F += m_procedure.compute_2body_fock(D_diff, precision_F, K);

                // compute HF energy with the non-extrapolated Fock matrix
                ehf = D.cwiseProduct(H + F).sum();
                ediff_rel = std::abs((ehf - ehf_last) / ehf);

                // compute SCF error
                RowMajorMatrix FD_comm = F * D * S - S * D * F;
                rms_error = FD_comm.norm() / n2;
                if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
                    reset_incremental_fock_formation = true;

                // DIIS extrapolate F
                RowMajorMatrix F_diis = F; // extrapolated F cannot be used in incremental Fock
                                   // build; only used to produce the density
                                   // make a copy of the unextrapolated matrix
                diis.extrapolate(F_diis, FD_comm);

                // solve F C = e S C by (conditioned) transformation to F' C' = e C',
                // where
                // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
                Eigen::SelfAdjointEigenSolver<RowMajorMatrix> eig_solver(X.transpose() * F_diis *
                                                                 X);
                evals = eig_solver.eigenvalues();
                C = X * eig_solver.eigenvectors();

                // compute density, D = C(occ) . C(occ)T
                C_occ = C.leftCols(ndocc);
                D = C_occ * C_occ.transpose();
                D_diff = D - D_last;

                const auto tstop = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> time_elapsed = tstop - tstart;

                if (iter == 1) {
                    fmt::print("{:>6s} {: >20s} {: >20s} {: >20s} {: >10s}\n",
                            "cycle", "energy", "D(E)/E", "rms([F,D])/nn", "time");
                }
                fmt::print("{:>6d} {:>20.12f} {:>20.12e} {:>20.12e} {:>10.5f}\n", iter, ehf + enuc,
                       ediff_rel, rms_error, time_elapsed.count());
                total_time += time_elapsed.count();

            } while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));
            fmt::print("SCF complete in {:.6f} s wall time\n", total_time);
            return ehf + enuc;
        }
        Procedure &m_procedure;
        int nelectrons{0};
        int ndocc{0};
        int maxiter{100};
        double conv = 1e-12;
        int iter = 0;
        double rms_error = 1.0;
        double ediff_rel = 0.0;
        double enuc{0.0};
        double ehf{0.0};
        double total_time{0.0};
        libint2::DIIS<RowMajorMatrix> diis; // start DIIS on second iteration

        bool reset_incremental_fock_formation = false;
        bool incremental_Fbuild_started = false;
        double start_incremental_F_threshold = 1e-5;
        double next_reset_threshold = 0.0;
        size_t last_reset_iteration = 0;
        RowMajorMatrix D, S, T, V, H, K, X, Xinv, C, C_occ, F;
        double XtX_condition_number;
        bool verbose{false};
    };
} // namespace craso::scf
