#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/gfn2_periodic_calculator.h>
#include <occ/xtb/periodic_integrals.h>
#include <occ/xtb/repulsion.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

Vec shell_populations(const Mat &PS, const std::vector<int> &bf_to_shell,
                       int n_shells) {
  Vec pop = Vec::Zero(n_shells);
  for (Eigen::Index mu = 0; mu < PS.rows(); ++mu)
    pop(bf_to_shell[mu]) += PS(mu, mu);
  return pop;
}

} // namespace

PeriodicSccResult
run_charge_only_periodic_scc(const PeriodicSystem &sys,
                              const Gfn2Parameters &params,
                              const PeriodicSccOptions &opts) {
  // Build basis, shells, and integrals from the central-cell atoms.
  auto basis = build_aobasis(sys.atoms, params);
  auto shells = build_shell_table(sys.atoms, params);
  const int nbf = static_cast<int>(basis.nbf());
  const int n_shells = static_cast<int>(shells.atom.size());
  const auto bf_to_atom = basis.bf_to_atom();
  const auto bf_to_shell = basis.bf_to_shell();
  const Vec z_sh = shells.ref_occ;

  // Lattice translations for AO/CN/repulsion sums.
  auto images = build_lattice_images(sys.lattice_bohr, opts.real_cutoff);
  // Periodic CN, repulsion, AO blocks, Bloch sums at Γ.
  Vec cn = gfn_coordination_numbers_periodic(sys.atoms, images);
  const double e_rep = repulsion_energy_periodic(sys.atoms, params, images);
  auto S_per_T = periodic_overlap_blocks(sys, params, images);
  auto H0_per_T = periodic_h0_blocks(sys, params, images, S_per_T, cn);
  Mat S = bloch_sum_gamma(S_per_T);
  Mat H0 = bloch_sum_gamma(H0_per_T);

  // Periodic shell-resolved γ via Ewald.
  auto ewald = build_ewald_data(sys, /*tol=*/1e-10, opts.ewald_alpha,
                                 opts.ewald_residual_cutoff);
  Mat J = periodic_klopman_ohno_gamma(sys, shells, params, ewald);

  // Closed-shell electron count (per primitive cell).
  double n_elec_total = 0.0;
  for (Eigen::Index i = 0; i < z_sh.size(); ++i) n_elec_total += z_sh(i);
  n_elec_total -= opts.total_charge;
  if (std::abs(std::round(n_elec_total) - n_elec_total) > 1e-6) {
    throw std::runtime_error(
        "run_charge_only_periodic_scc: non-integer electron count");
  }
  const int n_elec = static_cast<int>(std::round(n_elec_total));
  if (n_elec % 2 != 0) {
    throw std::runtime_error(
        "run_charge_only_periodic_scc: open-shell n_elec=" +
        std::to_string(n_elec));
  }
  const int n_occ = n_elec / 2;

  // EEQ-based initial guess on the central-cell atoms (no PBC for the guess).
  Vec qsh;
  try {
    qsh = eeq_initial_shell_charges(sys.atoms, shells, opts.total_charge);
  } catch (const std::exception &) {
    qsh = Vec::Zero(n_shells);
  }
  double prev_energy = 0.0;
  Vec orbital_energies, orbital_occupations;
  Mat C, P;

  const std::size_t diis_start = 3;
  const std::size_t diis_subspace = 8;
  occ::core::diis::DIIS diis(diis_start, diis_subspace);

  occ::log::info("{:=^72s}",
                  "  GFN2-xTB periodic SCC (Γ-only, charge-only)  ");
  occ::log::info("nbf = {}   n_shells = {}   n_electrons = {}", nbf, n_shells,
                  n_elec);
  occ::log::info("real_cutoff = {:.1f} Bohr   ewald α = {:.4f}   #G = {}",
                  opts.real_cutoff, ewald.alpha, ewald.g_vectors.size());
  occ::log::info("{:>4s}  {:>20s}  {:>12s}  {:>12s}", "iter", "E (Hartree)",
                  "|ΔE|", "max|Δq|");

  bool converged = false;
  int iter = 0;
  for (iter = 1; iter <= opts.max_iterations; ++iter) {
    // Isotropic + third-order shell potential.
    Vec V = J * qsh;
    for (Eigen::Index s = 0; s < V.size(); ++s) {
      V(s) += shells.third_order(s) * qsh(s) * qsh(s);
    }

    Mat H = H0;
    for (Eigen::Index mu = 0; mu < nbf; ++mu) {
      const int sh_mu = bf_to_shell[mu];
      for (Eigen::Index nu = 0; nu < nbf; ++nu) {
        const int sh_nu = bf_to_shell[nu];
        H(mu, nu) -= 0.5 * S(mu, nu) * (V(sh_mu) + V(sh_nu));
      }
    }

    Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(H, S);
    if (es.info() != Eigen::Success) {
      throw std::runtime_error(
          "run_charge_only_periodic_scc: eigensolver failed");
    }
    orbital_energies = es.eigenvalues();
    C = es.eigenvectors();

    orbital_occupations = Vec::Zero(nbf);
    for (int i = 0; i < n_occ; ++i) orbital_occupations(i) = 2.0;

    Mat Cocc = C.leftCols(n_occ);
    P = 2.0 * (Cocc * Cocc.transpose());

    Mat PS = P * S;
    Vec pop = shell_populations(PS, bf_to_shell, n_shells);
    Vec qsh_new = z_sh - pop;

    double e_es = 0.5 * qsh_new.dot(J * qsh_new);
    double e_third = 0.0;
    for (Eigen::Index s = 0; s < qsh_new.size(); ++s) {
      const double q = qsh_new(s);
      e_third += shells.third_order(s) * q * q * q / 3.0;
    }
    double e_h0 = (P.cwiseProduct(H0)).sum();
    double scc_energy = e_h0 + e_es + e_third;
    double total_energy = scc_energy + e_rep;

    double dq_max = (qsh_new - qsh).cwiseAbs().maxCoeff();
    double de = std::abs(total_energy - prev_energy);
    occ::log::info("{:>4d}  {:>20.12f}  {:>12.2e}  {:>12.2e}", iter,
                    total_energy, de, dq_max);

    bool e_ok = (iter > 1) && de < opts.energy_threshold;
    bool q_ok = dq_max < opts.charge_threshold;
    if (e_ok && q_ok) {
      converged = true;
      Vec atom_charges = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_charges(shells.atom[s]) += qsh_new(s);
      PeriodicSccResult r;
      r.scc_energy = scc_energy;
      r.repulsion_energy = e_rep;
      r.total_energy = total_energy;
      r.shell_charges = qsh_new;
      r.atomic_charges = atom_charges;
      r.orbital_energies = orbital_energies;
      r.orbital_occupations = orbital_occupations;
      r.density_matrix = P;
      r.overlap_matrix = S;
      r.orbital_coefficients = C;
      r.n_iterations = iter;
      r.converged = true;
      occ::log::info("Converged in {} iterations.", iter);
      return r;
    }

    Mat x = qsh_new;
    Mat err = qsh_new - qsh;
    diis.extrapolate(x, err);
    if (static_cast<std::size_t>(iter) > diis_start) {
      qsh = x.col(0);
    } else {
      qsh = (1.0 - opts.damping_factor) * qsh_new + opts.damping_factor * qsh;
    }
    prev_energy = total_energy;
  }

  occ::log::warn("Periodic GFN2 SCC did not converge in {} iterations",
                  opts.max_iterations);
  PeriodicSccResult r;
  r.scc_energy = prev_energy - e_rep;
  r.repulsion_energy = e_rep;
  r.total_energy = prev_energy;
  r.shell_charges = qsh;
  Vec atom_charges = Vec::Zero(sys.atoms.size());
  for (int s = 0; s < n_shells; ++s) atom_charges(shells.atom[s]) += qsh(s);
  r.atomic_charges = atom_charges;
  r.orbital_energies = orbital_energies;
  r.orbital_occupations = orbital_occupations;
  r.density_matrix = P;
  r.overlap_matrix = S;
  r.orbital_coefficients = C;
  r.n_iterations = iter;
  r.converged = false;
  return r;
}

} // namespace occ::xtb
