#pragma once
#include "ints.h"
#include "diis.h"
#include "linear_algebra.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <libint2/chemistry/sto3g_atomic_density.h>
#include <tuple>
#include "spinorbital.h"

namespace tonto::scf {

using tonto::qm::SpinorbitalKind;
using tonto::qm::alpha_alpha_block;
using tonto::qm::alpha_beta_block;
using tonto::qm::alpha_block;
using tonto::qm::beta_alpha_block;
using tonto::qm::beta_beta_block;
using tonto::qm::beta_block;

using tonto::MatRM;

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

template <typename Procedure> struct UnrestrictedSCF {
  UnrestrictedSCF(Procedure &procedure, int diis_start = 2)
      : m_procedure(procedure), diis(diis_start) {
    set_charge(0);
    set_multiplicity(1);
    n_electrons = m_procedure.num_e();
    n_alpha = n_electrons / 2;
  }

  int charge() const {
    double nuclear_charge = 0.0;
    for (const auto &atom : atoms()) {
      nuclear_charge += atom.atomic_number;
    }
    return nuclear_charge - n_electrons;
  }

  void set_charge(int c) {
    int current_charge = charge();
    if (c != current_charge) {
      n_electrons -= c - current_charge;
      n_beta = (n_electrons - n_unpaired_electrons) / 2;
      n_alpha = n_electrons - n_beta;
    }
  }

  const auto multiplicity() const { return n_alpha - n_beta + 1; }

  void set_multiplicity(int mult) {
    if (mult != multiplicity()) {
      n_unpaired_electrons = mult - 1;
      n_beta = (n_electrons - n_unpaired_electrons) / 2;
      n_alpha = n_electrons - n_beta;
    }
  }

  const auto &atoms() const { return m_procedure.atoms(); }

  std::pair<MatRM, MatRM> compute_soad() const {
    // computes Superposition-Of-Atomic-Densities guess for the molecular
    // density matrix in minimal basis; occupies subshells by smearing electrons
    // evenly over the orbitals compute number of atomic orbitals
    size_t nao = 0;
    for (const auto &atom : atoms()) {
      const auto Z = atom.atomic_number;
      nao += libint2::sto3g_num_ao(Z);
    }

    // compute the minimal basis density
    MatRM Da = MatRM::Zero(nao, nao);
    size_t ao_offset = 0; // first AO of this atom
    for (const auto &atom : atoms()) {
      const auto Z = atom.atomic_number;
      const auto &occvec = libint2::sto3g_ao_occupation_vector(Z);
      for (const auto &occ : occvec) {
        Da(ao_offset, ao_offset) = occ;
        ++ao_offset;
      }
    }

    int c = charge();

    Da *= 0.5;
    MatRM Db = Da;
    // smear the charge across all shells
    if (n_unpaired_electrons != 0) {
      double v = static_cast<double>(n_unpaired_electrons) / Db.rows();
      for (int i = 0; i < Db.rows(); i++) {
        Db(i, i) -= v;
      }
    }

    if (c != 0) {
      double v = static_cast<double>(c) / Da.rows();
      for (int i = 0; i < Da.rows(); i++) {
        Da(i, i) -= v;
        Db(i, i) -= v;
      }
    }
    return std::make_pair(
        Da, Db); // we use densities normalized to # of electrons/2
  }

  void compute_initial_guess() {
    const auto tstart = std::chrono::high_resolution_clock::now();
    S = m_procedure.compute_overlap_matrix();
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
    T = m_procedure.compute_kinetic_matrix();
    V = m_procedure.compute_nuclear_attraction_matrix();
    H = T + V;
    MatRM Da_minbs, Db_minbs;
    std::tie(Da_minbs, Db_minbs) =
        compute_soad(); // compute guess in minimal basis
    libint2::BasisSet minbs("STO-3G", atoms());

    if (minbs == m_procedure.basis()) {
      Da = Da_minbs * 0.5;
      Db = Db_minbs * 0.5;
    } else {
      // if basis != minimal basis, map non-representable SOAD guess
      // into the AO basis
      // by diagonalizing a Fock matrix
      if (verbose)
        fmt::print("Projecting SOAD into atomic orbital basis: ");
      Fa = H;
      Fb = H;
      Fa += tonto::ints::compute_2body_fock_mixed_basis(
          m_procedure.basis(), Da_minbs, minbs, true,
          std::numeric_limits<double>::epsilon());
      Fb += tonto::ints::compute_2body_fock_mixed_basis(
          m_procedure.basis(), Db_minbs, minbs, true,
          std::numeric_limits<double>::epsilon());
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver_a(X.transpose() * Fa * X);
      Ca = X * eig_solver_a.eigenvectors();
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver_b(X.transpose() * Fb * X);
      Ca = X * eig_solver_a.eigenvectors();
      Cb = X * eig_solver_b.eigenvectors();
      Ca_occ = Ca.leftCols(n_alpha);
      Cb_occ = Cb.leftCols(n_beta);
      Da = Ca_occ * Ca_occ.transpose() * 0.5;
      Db = Cb_occ * Cb_occ.transpose() * 0.5;

      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;
      if (verbose)
        fmt::print("{:.5f} s\n", time_elapsed.count());
    }
  }

  double compute_scf_energy() {
    // compute one-body integrals
    // count the number of electrons
    compute_initial_guess();
    K = m_procedure.compute_schwarz_ints();
    enuc = m_procedure.nuclear_repulsion_energy();
    MatRM Da_diff, Db_diff;
    auto n2a = Da.cols() * Da.rows();
    auto n2b = Db.cols() * Db.rows();

    fmt::print("Beginning SCF\n");
    total_time = 0.0;
    do {
      const auto tstart = std::chrono::high_resolution_clock::now();
      ++iter;

      // Last iteration's energy and density
      auto ehf_last = ehf;
      MatRM Da_last = Da, Db_last = Db;

      if (!incremental_Fbuild_started &&
          rms_error < start_incremental_F_threshold) {
        incremental_Fbuild_started = true;
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter - 1;
        next_reset_threshold = rms_error / 1e1;
        if (verbose)
          fmt::print("Starting incremental fock build\n");
      }
      if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
        Fa = H;
        Fb = H;
        Da_diff = Da;
        Db_diff = Db;
      }
      if (reset_incremental_fock_formation && incremental_Fbuild_started) {
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter;
        next_reset_threshold = rms_error / 1e1;
        if (verbose)
          fmt::print("Resetting incremental fock build\n");
      }

      // build a new Fock matrix
      // totally empirical precision variation, involves the condition number
      const auto precision_Fa = std::min(
          std::min(1e-3 / XtX_condition_number, 1e-7),
          std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
      const auto precision_Fb = std::min(
          std::min(1e-3 / XtX_condition_number, 1e-7),
          std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
      const auto precision_F = std::min(precision_Fa, precision_Fb);
      MatRM Fa_tmp, Fb_tmp;
      std::tie(Fa_tmp, Fb_tmp) = m_procedure.compute_2body_fock_unrestricted(
          Da_diff, Db_diff, precision_F, K);
      Fa += Fa_tmp;
      Fb += Fb_tmp;

      // compute HF energy with the non-extrapolated Fock matrix
      ehf = Da.cwiseProduct(H + Fa).sum() + Db.cwiseProduct(H + Fb).sum();
      ediff_rel = std::abs((ehf - ehf_last) / ehf);

      // compute SCF error
      MatRM FD_comm_a = Fa * Da * S - S * Da * Fa;
      MatRM FD_comm_b = Fb * Db * S - S * Db * Fb;

      MatRM F_diis(Fa.rows() + Fb.rows(), Fa.cols());
      MatRM FD_comm(FD_comm_a.rows() + FD_comm_b.rows(), FD_comm_a.cols());
      F_diis.block(0, 0, Fa.rows(), Fa.cols()) = Fa;
      F_diis.block(Fa.rows(), 0, Fa.rows(), Fa.cols()) = Fb;
      FD_comm.block(0, 0, FD_comm_a.rows(), FD_comm_a.cols()) = FD_comm_a;
      FD_comm.block(FD_comm_a.rows(), 0, FD_comm_b.rows(), FD_comm_b.cols()) = FD_comm_b;

      rms_error = FD_comm.norm() / FD_comm.size();
      if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
        reset_incremental_fock_formation = true;

      // DIIS extrapolate F
      diis.extrapolate(F_diis, FD_comm);

      MatRM Fa_diis = F_diis.block(0, 0, Fa.rows(), Fa.cols());
      MatRM Fb_diis = F_diis.block(Fa.rows(), 0, Fa.rows(), Fa.cols());

      // solve F C = e S C by (conditioned) transformation to F' C' = e C',
      // where
      // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
      Eigen::SelfAdjointEigenSolver<MatRM> eig_alpha(X.transpose() * Fa_diis *
                                                     X);
      orbital_energies_alpha = eig_alpha.eigenvalues();
      Ca = X * eig_alpha.eigenvectors();

      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver_b(X.transpose() *
                                                        Fb_diis * X);
      orbital_energies_beta = eig_solver_b.eigenvalues();
      Cb = X * eig_solver_b.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      Ca_occ = Ca.leftCols(n_alpha);
      Da = Ca_occ * Ca_occ.transpose() * 0.5;
      Da_diff = Da - Da_last;
      // beta
      Cb_occ = Cb.leftCols(n_beta);
      Db = Cb_occ * Cb_occ.transpose() * 0.5;
      Db_diff = Db - Db_last;

      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;

      if (iter == 1) {
        fmt::print("{:>6s} {: >20s} {: >20s} {: >20s} {: >10s}\n", "cycle",
                   "energy", "D(E)/E", "rms([F,D])/nn", "time");
      }
      fmt::print("{:>6d} {:>20.12f} {:>20.12e} {:>20.12e} {:>10.5f}\n", iter,
                 ehf + enuc, ediff_rel, rms_error, time_elapsed.count());
      total_time += time_elapsed.count();

    } while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));
    fmt::print("SCF complete in {:.6f} s wall time\n", total_time);
    double Ek = Da.cwiseProduct(T).sum() + Db.cwiseProduct(T).sum();
    double Een = Da.cwiseProduct(V).sum() + Db.cwiseProduct(V).sum();
    double E_1e = Da.cwiseProduct(H).sum() + Db.cwiseProduct(H).sum();
    double E_2e = Da.cwiseProduct(Fa).sum() + Db.cwiseProduct(Fb).sum();
    fmt::print("{:10s} {:20.12f} seconds\n", "SCF took", total_time);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_nn", enuc);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_k", Ek);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_en", Een);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_1e", E_1e);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_2e", E_2e);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_ee", E_1e + E_2e);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_tot", ehf + enuc);
    return ehf + enuc;
  }

  void print_orbital_energies() {
    int n_mo = orbital_energies_beta.size();
    fmt::print("\nMolecular orbital energies\n");
    fmt::print("{0:3s}   {1:3s} {2:>16s}  {1:3s} {2:>16s}\n", "idx", "occ", "energy");
    for(int i = 0; i < n_mo; i++)
    {
        auto s_a = i < n_alpha ? "a" : " ";
        auto s_b = i < n_beta ? "b" : " ";
        fmt::print("{:3d}   {:^3s} {:16.12f}  {:^3s} {:16.12f}\n",
            i, s_a, orbital_energies_alpha(i),
            s_b, orbital_energies_beta(i)
        );
    }
  }

  Procedure &m_procedure;
  int n_electrons{0};
  int n_alpha{0};
  int n_beta{0};
  int n_unpaired_electrons{0};
  int maxiter{100};
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
  MatRM S, H, T, V, K, X, Xinv;
  MatRM Da, Ca, Ca_occ, Fa;
  MatRM Db, Cb, Cb_occ, Fb;
  Vec orbital_energies_alpha, orbital_energies_beta;
  double XtX_condition_number;
  bool verbose{false};
};

template <typename Procedure> struct RestrictedSCF {
  RestrictedSCF(Procedure &procedure, int diis_start = 2)
      : m_procedure(procedure), diis(diis_start) {
    n_electrons = m_procedure.num_e();
    n_occ = n_electrons / 2;
  }

  int charge() const {
    double nuclear_charge = 0.0;
    for (const auto &atom : atoms()) {
      nuclear_charge += atom.atomic_number;
    }
    return nuclear_charge - n_electrons;
  }

  void set_charge(int c) {
    int current_charge = charge();
    if (c != current_charge) {
      n_electrons -= c - current_charge;
      n_occ = n_electrons / 2;
    }
    if (2 * (n_electrons / 2) != n_electrons)
        throw std::runtime_error(
            fmt::format("Can't set system charge of {} for restricted scf (n_e = {} is not even)", c, n_electrons)
        );
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
    MatRM D = MatRM::Zero(nao, nao);
    size_t ao_offset = 0; // first AO of this atom
    for (const auto &atom : atoms()) {
      const auto Z = atom.atomic_number;
      const auto &occvec = libint2::sto3g_ao_occupation_vector(Z);
      for (const auto &occ : occvec) {
        D(ao_offset, ao_offset) = occ;
        ++ao_offset;
      }
    }

    int c = charge();
    // smear the charge across all shells
    if (c != 0) {
      double v = static_cast<double>(c) / D.rows();
      for (int i = 0; i < D.rows(); i++) {
        D(i, i) -= v;
      }
    }
    return D * 0.5; // we use densities normalized to # of electrons/2
  }

  void compute_initial_guess() {
    const auto tstart = std::chrono::high_resolution_clock::now();
    S = m_procedure.compute_overlap_matrix();
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
    T = m_procedure.compute_kinetic_matrix();
    V = m_procedure.compute_nuclear_attraction_matrix();
    H = T + V;
    auto D_minbs = compute_soad(); // compute guess in minimal basis
    libint2::BasisSet minbs("STO-3G", atoms());

    if (minbs == m_procedure.basis()) {
      D = D_minbs;
    } else {
      // if basis != minimal basis, map non-representable SOAD guess
      // into the AO basis
      // by diagonalizing a Fock matrix
      if (verbose)
        fmt::print("Projecting SOAD into atomic orbital basis: ");
      F = H;
      F += tonto::ints::compute_2body_fock_mixed_basis(
          m_procedure.basis(), D_minbs, minbs, true,
          std::numeric_limits<double>::epsilon());
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * F * X);
      C = X * eig_solver.eigenvectors();
      C_occ = C.leftCols(n_occ);
      D = C_occ * C_occ.transpose();
      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;
      if (verbose)
        fmt::print("{:.5f} s\n", time_elapsed.count());
    }
  }

  double compute_scf_energy() {
    // compute one-body integrals
    // count the number of electrons
    compute_initial_guess();
    K = m_procedure.compute_schwarz_ints();
    enuc = m_procedure.nuclear_repulsion_energy();
    MatRM D_diff;
    auto n2 = D.cols() * D.rows();

    fmt::print("Beginning SCF\n");
    total_time = 0.0;
    double e2e = 0.0;
    double e1e = 0.0;
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
        if (verbose)
          fmt::print("Starting incremental fock build\n");
      }
      if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
        F = H;
        D_diff = D;
        e2e = 0.0;
      }
      if (reset_incremental_fock_formation && incremental_Fbuild_started) {
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter;
        next_reset_threshold = rms_error / 1e1;
        if (verbose)
          fmt::print("Resetting incremental fock build\n");
      }

      // build a new Fock matrix
      // totally empirical precision variation, involves the condition number
      const auto precision_F = std::min(
          std::min(1e-3 / XtX_condition_number, 1e-7),
          std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
      F += m_procedure.compute_2body_fock(D_diff, precision_F, K);

      // compute HF energy with the non-extrapolated Fock matrix
      e2e += m_procedure.two_electron_energy();
      e1e = 2 * D.cwiseProduct(H).sum();
      ehf = e1e + enuc + e2e;

      ediff_rel = std::abs((ehf - ehf_last) / ehf);

      // compute SCF error
      MatRM FD_comm = F * D * S - S * D * F;
      rms_error = FD_comm.norm() / n2;
      if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
        reset_incremental_fock_formation = true;

      // DIIS extrapolate F
      MatRM F_diis = F; // extrapolated F cannot be used in incremental Fock
      // build; only used to produce the density
      // make a copy of the unextrapolated matrix
      diis.extrapolate(F_diis, FD_comm);

      // solve F C = e S C by (conditioned) transformation to F' C' = e C',
      // where
      // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * F_diis *
                                                      X);
      orbital_energies = eig_solver.eigenvalues();
      C = X * eig_solver.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      C_occ = C.leftCols(n_occ);
      D = C_occ * C_occ.transpose();
      D_diff = D - D_last;

      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;

      if (iter == 1) {
        fmt::print("{:>6s} {: >20s} {: >20s} {: >20s} {: >10s}\n", "cycle",
                   "energy", "D(E)/E", "rms([F,D])/nn", "time");
      }
      fmt::print("{:>6d} {:>20.12f} {:>20.12e} {:>20.12e} {:>10.5f}\n", iter,
                 ehf, ediff_rel, rms_error, time_elapsed.count());
      total_time += time_elapsed.count();

    } while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));
    fmt::print("{:10s} {:20.12f} seconds\n", "SCF took", total_time);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_nn", enuc);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_k", D.cwiseProduct(T).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_en", e1e);
    fmt::print("{:10s} {:20.12f} hartree\n", "E_1e", D.cwiseProduct(H).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_2e", D.cwiseProduct(F).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_tot", ehf);
    return ehf;
  }

  void print_orbital_energies() {
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

  Procedure &m_procedure;
  int n_electrons{0};
  int n_occ{0};
  int n_unpaired_electrons{0};
  int maxiter{100};
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


template <typename Procedure> struct GeneralSCF {
  GeneralSCF(Procedure &procedure, int diis_start = 2)
      : m_procedure(procedure), diis(diis_start) {
    n_electrons = m_procedure.num_e();
    n_occ = n_electrons;
  }

  int charge() const {
    double nuclear_charge = 0.0;
    for (const auto &atom : atoms()) {
      nuclear_charge += atom.atomic_number;
    }
    return nuclear_charge - n_electrons;
  }

  void set_charge(int c) {
    int current_charge = charge();
    if (c != current_charge) {
      n_electrons -= c - current_charge;
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
    MatRM D = MatRM::Zero(nao, nao);
    size_t ao_offset = 0; // first AO of this atom
    for (const auto &atom : atoms()) {
      const auto Z = atom.atomic_number;
      const auto &occvec = libint2::sto3g_ao_occupation_vector(Z);
      for (const auto &occ : occvec) {
        D(ao_offset, ao_offset) = occ;
        ++ao_offset;
      }
    }

    int c = charge();
    // smear the charge across all shells
    if (c != 0) {
      double v = static_cast<double>(c) / D.rows();
      for (int i = 0; i < D.rows(); i++) {
        D(i, i) -= v;
      }
    }
    return D * 0.5; // we use densities normalized to # of electrons/2
  }

  void compute_initial_guess() {
    const auto tstart = std::chrono::high_resolution_clock::now();
    auto s_xx = m_procedure.compute_overlap_matrix();
    S = MatRM::Zero(s_xx.rows() * 2, s_xx.cols() * 2);
    S.block(0, 0, s_xx.rows(), s_xx.cols()) = s_xx;
    S.block(s_xx.rows(), s_xx.cols(), s_xx.rows(), s_xx.cols()) = s_xx;

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
    auto t_xx = m_procedure.compute_kinetic_matrix();
    T = MatRM::Zero(t_xx.rows() * 2, t_xx.cols() * 2);
    T.block(0, 0, t_xx.rows(), t_xx.cols()) = t_xx;
    T.block(t_xx.rows(), t_xx.cols(), t_xx.rows(), t_xx.cols()) = t_xx;
    auto v_xx = m_procedure.compute_nuclear_attraction_matrix();
    V = MatRM::Zero(v_xx.rows() * 2, v_xx.cols() * 2);
    V.block(0, 0, v_xx.rows(), v_xx.cols()) = v_xx;
    V.block(v_xx.rows(), v_xx.cols(), v_xx.rows(), v_xx.cols()) = v_xx;
    H = T + V;
    auto D_minbs = compute_soad(); // compute guess in minimal basis
    libint2::BasisSet minbs("STO-3G", atoms());

    if (minbs == m_procedure.basis()) {
      D = MatRM::Zero(S.rows(), S.cols());
      D.block(0, 0, D_minbs.rows(), D_minbs.cols()) = D_minbs;
      D.block(D_minbs.rows(), D_minbs.cols(), D_minbs.rows(), D_minbs.cols()) = D_minbs;
    if (n_occ * 2 <  n_electrons) n_occ += 1;
    } else {
      // if basis != minimal basis, map non-representable SOAD guess
      // into the AO basis
      // by diagonalizing a Fock matrix
      if (verbose)
        fmt::print("Projecting SOAD into atomic orbital basis: ");
      F = H;
      int N = F.rows() / 2, M = F.cols() / 2;
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * F * X);
      C = X * eig_solver.eigenvectors();
      C_occ = C.leftCols(n_occ);
      D = C_occ * C_occ.transpose() * 0.5;
      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;
      if (verbose)
        fmt::print("{:.5f} s\n", time_elapsed.count());
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
        if (verbose)
          fmt::print("Starting incremental fock build\n");
      }
      if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
        F = H;
        D_diff = D;
      }
      if (reset_incremental_fock_formation && incremental_Fbuild_started) {
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter;
        next_reset_threshold = rms_error / 1e1;
        if (verbose)
          fmt::print("Resetting incremental fock build\n");
      }

      // build a new Fock matrix
      // totally empirical precision variation, involves the condition number
      const auto precision_F = std::min(
          std::min(1e-3 / XtX_condition_number, 1e-7),
          std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
      F += m_procedure.compute_2body_fock_general(
            D_diff, precision_F, K
      );

      // compute HF energy with the non-extrapolated Fock matrix
      ehf = D.cwiseProduct(H + F).sum();
      ediff_rel = std::abs((ehf - ehf_last) / ehf);

      // compute SCF error
      MatRM FD_comm = F * D * S - S * D * F;
      rms_error = FD_comm.norm() / FD_comm.size();
      if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
        reset_incremental_fock_formation = true;

      // DIIS extrapolate F
      MatRM F_diis = F; // extrapolated F cannot be used in incremental Fock
      // build; only used to produce the density
      // make a copy of the unextrapolated matrix
      diis.extrapolate(F_diis, FD_comm);

      // solve F C = e S C by (conditioned) transformation to F' C' = e C',
      // where
      // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * F_diis *
                                                      X);
      orbital_energies = eig_solver.eigenvalues();
      C = X * eig_solver.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      C_occ = C.leftCols(n_occ);
      D = C_occ * C_occ.transpose() * 0.5;
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
    fmt::print("{:10s} {:20.12f} hartree\n", "E_k",   D.cwiseProduct(T).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_en",  D.cwiseProduct(V).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_1e",  D.cwiseProduct(H).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_2e",  D.cwiseProduct(F).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_tot", (ehf + enuc));
    return ehf + enuc;
  }

  void print_orbital_energies() {
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

  Procedure &m_procedure;
  int n_electrons{0};
  int n_occ{0};
  int n_unpaired_electrons{0};
  int maxiter{100};
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


template <typename Procedure, SpinorbitalKind spinorbital_kind>
struct SCF {
  SCF(Procedure &procedure, int diis_start = 2)
      : m_procedure(procedure), diis(diis_start) {
    n_electrons = m_procedure.num_e();
    n_occ = n_electrons;
  }

  int charge() const {
    double nuclear_charge = 0.0;
    for (const auto &atom : atoms()) {
      nuclear_charge += atom.atomic_number;
    }
    return nuclear_charge - n_electrons;
  }

  void set_charge(int c) {
    int current_charge = charge();
    if (c != current_charge) {
      n_electrons -= c - current_charge;
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
    MatRM D = MatRM::Zero(nao, nao);
    size_t ao_offset = 0; // first AO of this atom
    for (const auto &atom : atoms()) {
      const auto Z = atom.atomic_number;
      const auto &occvec = libint2::sto3g_ao_occupation_vector(Z);
      for (const auto &occ : occvec) {
        D(ao_offset, ao_offset) = occ;
        ++ao_offset;
      }
    }

    int c = charge();
    // smear the charge across all shells
    if (c != 0) {
      double v = static_cast<double>(c) / D.rows();
      for (int i = 0; i < D.rows(); i++) {
        D(i, i) -= v;
      }
    }
    return D * 0.5; // we use densities normalized to # of electrons/2
  }

  void compute_initial_guess() {
    const auto tstart = std::chrono::high_resolution_clock::now();
    size_t nbf = m_procedure.basis().nbf();
    auto [rows, cols] = tonto::qm::density_matrix_dimensions<spinorbital_kind>(nbf);
    if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
        S = m_procedure.compute_overlap_matrix();
        T = m_procedure.compute_kinetic_matrix();
        V = m_procedure.compute_nuclear_attraction_matrix();
    }
    else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
        S = MatRM::Zero(rows, cols);
        T = MatRM::Zero(rows, cols);
        V = MatRM::Zero(rows, cols);
        alpha_alpha_block(nbf, S) = m_procedure.compute_overlap_matrix();
        alpha_alpha_block(nbf, T) = m_procedure.compute_kinetic_matrix();
        alpha_alpha_block(nbf, V) = m_procedure.compute_nuclear_attraction_matrix();
        beta_beta_block(nbf, S) = alpha_alpha_block(nbf, S);
        beta_beta_block(nbf, T) = alpha_alpha_block(nbf, T);
        beta_beta_block(nbf, V) = alpha_alpha_block(nbf, V);
    }

    // compute orthogonalizer X such that X.transpose() . S . X = I
    // one should think of columns of Xinv as the conditioned basis
    // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
    // X)
    // by default assume can manage to compute with condition number of S <=
    // 1/eps
    // this is probably too optimistic, but in well-behaved cases even 10^11 is
    // OK
    double S_condition_number_threshold = 1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X, Xinv, XtX_condition_number) = conditioning_orthogonalizer(S, S_condition_number_threshold);

    auto D_minbs = compute_soad(); // compute guess in minimal basis
    libint2::BasisSet minbs("STO-3G", atoms());

    if (minbs == m_procedure.basis()) {
      if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
          D = D_minbs;
      }
      else if constexpr(spinorbital_kind == SpinorbitalKind::General) {
          D = MatRM::Zero(rows, cols);
          alpha_alpha_block(nbf, D) = D_minbs;
          beta_beta_block(nbf, D) = D_minbs;
      }
    } else {
      // if basis != minimal basis, map non-representable SOAD guess
      // into the AO basis
      // by diagonalizing a Fock matrix
      if (verbose)
        fmt::print("Projecting SOAD into atomic orbital basis: ");
      F = H;
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * F * X);
      C = X * eig_solver.eigenvectors();
      C_occ = C.leftCols(n_occ);
      D = C_occ * C_occ.transpose() * 0.5;
      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;
      if (verbose)
        fmt::print("{:.5f} s\n", time_elapsed.count());
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
        if (verbose)
          fmt::print("Starting incremental fock build\n");
      }
      if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
        F = H;
        D_diff = D;
      }
      if (reset_incremental_fock_formation && incremental_Fbuild_started) {
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter;
        next_reset_threshold = rms_error / 1e1;
        if (verbose)
          fmt::print("Resetting incremental fock build\n");
      }

      // build a new Fock matrix
      // totally empirical precision variation, involves the condition number
      const auto precision_F = std::min(
          std::min(1e-3 / XtX_condition_number, 1e-7),
          std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
      F += m_procedure.compute_fock(spinorbital_kind, D_diff, precision_F, K);

      // compute HF energy with the non-extrapolated Fock matrix
      ehf = D.cwiseProduct(H + F).sum();
      ediff_rel = std::abs((ehf - ehf_last) / ehf);

      // compute SCF error
      MatRM FD_comm = F * D * S - S * D * F;
      rms_error = FD_comm.norm() / FD_comm.size();
      if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
        reset_incremental_fock_formation = true;

      // DIIS extrapolate F
      MatRM F_diis = F; // extrapolated F cannot be used in incremental Fock
      // build; only used to produce the density
      // make a copy of the unextrapolated matrix
      diis.extrapolate(F_diis, FD_comm);

      // solve F C = e S C by (conditioned) transformation to F' C' = e C',
      // where
      // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
      Eigen::SelfAdjointEigenSolver<MatRM> eig_solver(X.transpose() * F_diis *
                                                      X);
      orbital_energies = eig_solver.eigenvalues();
      C = X * eig_solver.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      C_occ = C.leftCols(n_occ);
      D = C_occ * C_occ.transpose() * 0.5;
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
    fmt::print("{:10s} {:20.12f} hartree\n", "E_k",   D.cwiseProduct(T).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_en",  D.cwiseProduct(V).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_1e",  D.cwiseProduct(H).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_2e",  D.cwiseProduct(F).sum());
    fmt::print("{:10s} {:20.12f} hartree\n", "E_tot", (ehf + enuc));
    return ehf + enuc;
  }

  void print_orbital_energies() {
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

  Procedure &m_procedure;
  int n_electrons{0};
  int n_occ{0};
  int n_unpaired_electrons{0};
  int maxiter{100};
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
