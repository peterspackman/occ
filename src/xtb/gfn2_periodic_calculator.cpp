#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/disp/d4.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/camm.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/gfn2_periodic_calculator.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/kpoint_grid.h>
#include <occ/xtb/multipole_damping.h>
#include <occ/xtb/multipole_ewald.h>
#include <occ/xtb/multipole_ints.h>
#include <occ/xtb/periodic_integrals.h>
#include <occ/xtb/repulsion.h>
#include <optional>
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

  // Multipole AO matrices — Bloch-summed at Γ with per-atom origin (Ket =
  // atom-of-row centered, Bra = atom-of-col-image centered). Mirrors
  // dftbplus's tblite bridge: each AO pair contributes to atom_of(row) via
  // Ket and atom_of(col) via Bra in the CAMM partition, no R*S correction
  // needed. Critical for dense crystals where image AO overlaps are large.
  qm::IntegralEngine engine(basis);
  std::optional<PeriodicMultipoleAO> mp_ao;
  Vec mp_radii;
  std::optional<MultipolePairTensors> mp_tensors;  // Ewald path
  if (opts.include_multipoles) {
    mp_ao = build_periodic_multipole_ao(sys, params, images);
    mp_radii = multipole_radii(sys.atoms, cn, params);
    if (opts.multipole_ewald) {
      mp_tensors = build_multipole_ewald_tensors(sys, mp_radii, params);
    }
  }

  // Native D4: Dispersion owns its own lattice sums internally.
  std::optional<occ::disp::D4Dispersion> native_d4;
  if (opts.include_dispersion) {
    native_d4.emplace(sys.atoms);
    const auto &g = params.globals();
    native_d4->set_damping(
        occ::disp::D4Damping{g.s6, g.s8, g.s9, g.a1, g.a2, 16});
    native_d4->set_cutoffs(opts.disp_cutoff, /*disp3=*/40.0,
                            /*cn=*/30.0);
  }
  double e_disp = 0.0;

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

  // EEQ-based initial guess. Uses the Ewald-summed periodic A matrix so the
  // starting charges respect the lattice (matters for ionic crystals where
  // the molecular EEQ would give qualitatively wrong charges).
  Vec qsh;
  try {
    qsh = eeq_initial_shell_charges_periodic(sys.atoms, shells,
                                              sys.lattice_bohr,
                                              opts.total_charge);
  } catch (const std::exception &) {
    qsh = Vec::Zero(n_shells);
  }
  double prev_energy = 0.0;
  Vec orbital_energies, orbital_occupations;
  Mat C, P;
  // Atomic multipole moments — mixed alongside qsh in DIIS so that the H1
  // multipole contribution stabilizes together with the shell charges. With
  // qsh-only DIIS the moments re-derive from the latest P each iteration
  // without smoothing, which can produce sustained ~mHa charge sloshing on
  // dense molecular crystals (multipole H1 feedback is significant).
  CammMoments mom;
  if (opts.include_multipoles) {
    mom.dipm = Mat3N::Zero(3, sys.atoms.size());
    mom.qp = Mat::Zero(6, sys.atoms.size());
  }
  const int n_atoms = static_cast<int>(sys.atoms.size());
  // DIIS state size: qsh (n_shells) + dipm (3 × n_atoms) + qp (6 × n_atoms).
  const int diis_size = opts.include_multipoles
                             ? (n_shells + 9 * n_atoms)
                             : n_shells;
  auto pack_state = [&](const Vec &qsh_v, const CammMoments &m, Vec &out) {
    out.head(n_shells) = qsh_v;
    if (opts.include_multipoles) {
      Eigen::Map<const Vec> dipm_flat(m.dipm.data(), 3 * n_atoms);
      Eigen::Map<const Vec> qp_flat(m.qp.data(), 6 * n_atoms);
      out.segment(n_shells, 3 * n_atoms) = dipm_flat;
      out.segment(n_shells + 3 * n_atoms, 6 * n_atoms) = qp_flat;
    }
  };
  auto unpack_state = [&](const Vec &state, Vec &qsh_v, CammMoments &m) {
    qsh_v = state.head(n_shells);
    if (opts.include_multipoles) {
      Eigen::Map<Vec>(m.dipm.data(), 3 * n_atoms) =
          state.segment(n_shells, 3 * n_atoms);
      Eigen::Map<Vec>(m.qp.data(), 6 * n_atoms) =
          state.segment(n_shells + 3 * n_atoms, 6 * n_atoms);
    }
  };

  const std::size_t diis_start = 3;
  const std::size_t diis_subspace = 8;
  occ::core::diis::DIIS diis(diis_start, diis_subspace);

  occ::log::info("{:=^72s}", "  GFN2-xTB periodic SCC (Γ-only)  ");
  occ::log::info("nbf = {}   n_shells = {}   n_electrons = {}   multipoles = {}",
                  nbf, n_shells, n_elec, opts.include_multipoles ? "on" : "off");
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

    AnisotropicEnergy e_aniso{0.0, 0.0};
    if (opts.include_multipoles && iter > 1) {
      // mom is the DIIS-mixed atomic multipole state from the previous
      // iteration's diagonalization. Don't recompute from current P here —
      // that would skip the DIIS mixing on the moments and reintroduce
      // the H1 feedback oscillation.
      Vec atom_q = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q(shells.atom[s]) += qsh(s);
      // Periodic anisotropic ES: Ewald-summed pair tensors + clean tensor
      // potentials (no `rai`-based gauge corrections — those are the molecular
      // partition's compensation for global-origin AO integrals; we now use
      // atom-centered Bra/Ket integrals and a clean tensor potential like
      // tblite's `get_potential` in coulomb/multipole.f90).
      AnisotropicPotentials pot;
      if (opts.multipole_ewald) {
        pot = anisotropic_potentials_ewald(sys.atoms, atom_q, mom,
                                             *mp_tensors, params);
        e_aniso = anisotropic_energy_ewald(sys.atoms, atom_q, mom,
                                            *mp_tensors, params);
      } else {
        pot = anisotropic_potentials_periodic(sys.atoms, images, atom_q,
                                                mp_radii, mom, params);
        e_aniso = anisotropic_energy_periodic(sys.atoms, images, atom_q,
                                                mp_radii, mom, params);
      }
      // Periodic H1 with Bra/Ket atom-centered AO matrices: matches the
      // tensor-only potential by using each side's atom-centered integrals.
      apply_anisotropic_h1_periodic(H, S, mp_ao->D_ket, mp_ao->D_bra,
                                     mp_ao->Q_ket, mp_ao->Q_bra,
                                     bf_to_atom, pot);
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

    if (native_d4) {
      Vec atom_q_new = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q_new(shells.atom[s]) += qsh_new(s);
      native_d4->set_charges(atom_q_new);
      e_disp = native_d4->energy_periodic(sys.lattice_bohr);
    }

    double e_es = 0.5 * qsh_new.dot(J * qsh_new);
    double e_third = 0.0;
    for (Eigen::Index s = 0; s < qsh_new.size(); ++s) {
      const double q = qsh_new(s);
      e_third += shells.third_order(s) * q * q * q / 3.0;
    }
    double e_h0 = (P.cwiseProduct(H0)).sum();
    double scc_energy = e_h0 + e_es + e_third + e_aniso.aes + e_aniso.polariz;
    double total_energy = scc_energy + e_rep + e_disp;

    double dq_max = (qsh_new - qsh).cwiseAbs().maxCoeff();
    double de = std::abs(total_energy - prev_energy);
    occ::log::info("{:>4d}  {:>20.12f}  {:>12.2e}  {:>12.2e}", iter,
                    total_energy, de, dq_max);
    occ::log::debug(
        "    breakdown: H0={:>14.6f}  ES={:>14.6f}  3rd={:>10.3e}  "
        "AES={:>10.3e}  pol={:>10.3e}  rep={:>10.3e}  disp={:>10.3e}",
        e_h0, e_es, e_third, e_aniso.aes, e_aniso.polariz, e_rep, e_disp);
    occ::log::debug(
        "    diagnostics: |q|_max={:>10.3e}  Σq={:>+.3e}  ΣP={:>10.3e}",
        qsh_new.cwiseAbs().maxCoeff(), qsh_new.sum(), P.sum());
    if (opts.include_multipoles && iter > 1) {
      Vec atom_q_dbg = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q_dbg(shells.atom[s]) += qsh(s);
      // Show the DIIS-mixed mom that was used in this iteration's H1.
      occ::log::debug(
          "    multipoles:  |atom q|_max={:>10.3e}  |dipm|_max={:>10.3e}  "
          "|qp|_max={:>10.3e}",
          atom_q_dbg.cwiseAbs().maxCoeff(),
          mom.dipm.cwiseAbs().maxCoeff(),
          mom.qp.cwiseAbs().maxCoeff());
    }

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
      r.dispersion_energy = e_disp;
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

    // Compute the new multipole moments from the new density matrix; these
    // will be DIIS-mixed alongside qsh and used in the next iteration's H1.
    CammMoments mom_new;
    if (opts.include_multipoles) {
      mom_new = compute_camm_moments_periodic(
          sys.atoms, bf_to_atom, P, mp_ao->D_ket, mp_ao->D_bra,
          mp_ao->Q_ket, mp_ao->Q_bra);
    }

    // Pack (qsh_new, mom_new) into a single DIIS state vector. Error is the
    // change since the previous (qsh, mom). Extrapolate, then unpack back.
    Vec state(diis_size);
    Vec state_prev(diis_size);
    pack_state(qsh_new, mom_new, state);
    pack_state(qsh, mom, state_prev);
    Mat x = state;
    Mat err = state - state_prev;
    diis.extrapolate(x, err);
    if (static_cast<std::size_t>(iter) > diis_start) {
      Vec extrapolated = x.col(0);
      unpack_state(extrapolated, qsh, mom);
    } else {
      // Linear damping on the full state (charges + moments).
      Vec mixed = (1.0 - opts.damping_factor) * state +
                  opts.damping_factor * state_prev;
      unpack_state(mixed, qsh, mom);
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

PeriodicSccResult
run_periodic_scc_kpoints(const PeriodicSystem &sys,
                          const Gfn2Parameters &params,
                          const std::vector<KPoint> &kpoints,
                          const PeriodicSccOptions &opts) {
  if (kpoints.empty()) {
    throw std::runtime_error("run_periodic_scc_kpoints: empty k-grid");
  }
  // Build basis, shells, integrals (central-cell).
  auto basis = build_aobasis(sys.atoms, params);
  auto shells = build_shell_table(sys.atoms, params);
  const int nbf = static_cast<int>(basis.nbf());
  const int n_shells = static_cast<int>(shells.atom.size());
  const auto bf_to_atom = basis.bf_to_atom();
  const auto bf_to_shell = basis.bf_to_shell();
  const Vec z_sh = shells.ref_occ;

  auto images = build_lattice_images(sys.lattice_bohr, opts.real_cutoff);
  Vec cn = gfn_coordination_numbers_periodic(sys.atoms, images);
  const double e_rep = repulsion_energy_periodic(sys.atoms, params, images);
  auto S_per_T = periodic_overlap_blocks(sys, params, images);
  auto H0_per_T = periodic_h0_blocks(sys, params, images, S_per_T, cn);

  auto ewald = build_ewald_data(sys, /*tol=*/1e-10, opts.ewald_alpha,
                                 opts.ewald_residual_cutoff);
  Mat J = periodic_klopman_ohno_gamma(sys, shells, params, ewald);

  std::optional<occ::disp::D4Dispersion> native_d4;
  if (opts.include_dispersion) {
    native_d4.emplace(sys.atoms);
    const auto &g = params.globals();
    native_d4->set_damping(
        occ::disp::D4Damping{g.s6, g.s8, g.s9, g.a1, g.a2, 16});
    native_d4->set_cutoffs(opts.disp_cutoff, /*disp3=*/40.0,
                            /*cn=*/30.0);
  }
  double e_disp = 0.0;

  // Pre-compute Bloch-summed S(k), H0(k) for each k (geometry-cached).
  const int n_k = static_cast<int>(kpoints.size());
  std::vector<CMat> S_k(n_k), H0_k(n_k);
  for (int ik = 0; ik < n_k; ++ik) {
    S_k[ik] = bloch_sum(S_per_T, images, kpoints[ik].k);
    H0_k[ik] = bloch_sum(H0_per_T, images, kpoints[ik].k);
  }

  // Electron count (per primitive cell).
  double n_elec_total = 0.0;
  for (Eigen::Index i = 0; i < z_sh.size(); ++i) n_elec_total += z_sh(i);
  n_elec_total -= opts.total_charge;
  if (std::abs(std::round(n_elec_total) - n_elec_total) > 1e-6) {
    throw std::runtime_error(
        "run_periodic_scc_kpoints: non-integer electron count");
  }
  const int n_elec = static_cast<int>(std::round(n_elec_total));
  if (n_elec % 2 != 0) {
    throw std::runtime_error(
        "run_periodic_scc_kpoints: open-shell n_elec=" +
        std::to_string(n_elec));
  }
  const double n_pairs_per_cell = 0.5 * n_elec;

  // EEQ initial guess (periodic Ewald A matrix).
  Vec qsh;
  try {
    qsh = eeq_initial_shell_charges_periodic(sys.atoms, shells,
                                              sys.lattice_bohr,
                                              opts.total_charge);
  } catch (const std::exception &) {
    qsh = Vec::Zero(n_shells);
  }
  double prev_energy = 0.0;
  std::vector<Vec> orbital_energies_k(n_k);
  std::vector<Mat> orbital_occupations_k(n_k, Mat::Zero(nbf, 1));
  std::vector<CMat> C_k(n_k);
  std::vector<CMat> P_k(n_k);

  const std::size_t diis_start = 3;
  const std::size_t diis_subspace = 8;
  occ::core::diis::DIIS diis(diis_start, diis_subspace);

  occ::log::info("{:=^72s}", "  GFN2-xTB periodic SCC (k-point)  ");
  occ::log::info(
      "nbf = {}   n_shells = {}   n_electrons = {}   n_kpts = {}", nbf,
      n_shells, n_elec, n_k);
  occ::log::info("real_cutoff = {:.1f} Bohr   ewald α = {:.4f}   #G = {}",
                  opts.real_cutoff, ewald.alpha, ewald.g_vectors.size());
  occ::log::info("{:>4s}  {:>20s}  {:>12s}  {:>12s}", "iter", "E (Hartree)",
                  "|ΔE|", "max|Δq|");

  bool converged = false;
  int iter = 0;
  for (iter = 1; iter <= opts.max_iterations; ++iter) {
    Vec V = J * qsh;
    for (Eigen::Index s = 0; s < V.size(); ++s) {
      V(s) += shells.third_order(s) * qsh(s) * qsh(s);
    }

    // Solve at every k.
    for (int ik = 0; ik < n_k; ++ik) {
      CMat H = H0_k[ik];
      for (Eigen::Index mu = 0; mu < nbf; ++mu) {
        const int sh_mu = bf_to_shell[mu];
        for (Eigen::Index nu = 0; nu < nbf; ++nu) {
          const int sh_nu = bf_to_shell[nu];
          H(mu, nu) -= 0.5 * S_k[ik](mu, nu) * (V(sh_mu) + V(sh_nu));
        }
      }
      auto sol = solve_generalized_hermitian(H, S_k[ik]);
      orbital_energies_k[ik] = sol.eigenvalues;
      C_k[ik] = sol.eigenvectors;
    }

    // Aufbau across all (k, band) pairs sorted by energy.
    struct StateRef { double energy; int ik; int band; double weight; };
    std::vector<StateRef> states;
    states.reserve(static_cast<size_t>(n_k * nbf));
    for (int ik = 0; ik < n_k; ++ik) {
      for (int b = 0; b < nbf; ++b) {
        states.push_back({orbital_energies_k[ik](b), ik, b, kpoints[ik].weight});
      }
    }
    std::sort(states.begin(), states.end(),
              [](const StateRef &a, const StateRef &b) {
                return a.energy < b.energy;
              });

    // Each state holds 2*w_k electrons (closed shell). Fill from below to
    // n_elec, with a fractional last state if needed.
    std::vector<Vec> occ_k(n_k, Vec::Zero(nbf));
    double remaining = static_cast<double>(n_elec);
    for (const auto &st : states) {
      const double cap = 2.0 * st.weight;
      if (remaining <= 0.0) break;
      const double take = std::min(cap, remaining);
      occ_k[st.ik](st.band) = take / st.weight;  // per-k occupation 0..2
      remaining -= take;
    }
    orbital_occupations_k = std::vector<Mat>(n_k);
    for (int ik = 0; ik < n_k; ++ik)
      orbital_occupations_k[ik] = occ_k[ik];

    // Density at each k and Mulliken populations summed over k.
    Vec pop = Vec::Zero(n_shells);
    double e_h0 = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
      // P(k) = C diag(n_i) C^H
      CMat W = C_k[ik];
      for (Eigen::Index i = 0; i < W.cols(); ++i) {
        W.col(i) *= occ_k[ik](i);
      }
      P_k[ik] = W * C_k[ik].adjoint();
      // Mulliken: n_μ = Re[(P S)_μμ], summed with weight.
      CMat PS = P_k[ik] * S_k[ik];
      const double w = kpoints[ik].weight;
      for (Eigen::Index mu = 0; mu < nbf; ++mu) {
        pop(bf_to_shell[mu]) += w * PS(mu, mu).real();
      }
      // Band-energy / e_h0 contribution: tr(P · H0) at k.
      const std::complex<double> tr = (P_k[ik].array() *
                                       H0_k[ik].transpose().array()).sum();
      e_h0 += w * tr.real();
    }

    Vec qsh_new = z_sh - pop;

    if (native_d4) {
      Vec atom_q_new = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q_new(shells.atom[s]) += qsh_new(s);
      native_d4->set_charges(atom_q_new);
      e_disp = native_d4->energy_periodic(sys.lattice_bohr);
    }

    double e_es = 0.5 * qsh_new.dot(J * qsh_new);
    double e_third = 0.0;
    for (Eigen::Index s = 0; s < qsh_new.size(); ++s) {
      const double q = qsh_new(s);
      e_third += shells.third_order(s) * q * q * q / 3.0;
    }
    double scc_energy = e_h0 + e_es + e_third;
    double total_energy = scc_energy + e_rep + e_disp;

    double dq_max = (qsh_new - qsh).cwiseAbs().maxCoeff();
    double de = std::abs(total_energy - prev_energy);
    occ::log::info("{:>4d}  {:>20.12f}  {:>12.2e}  {:>12.2e}", iter,
                    total_energy, de, dq_max);
    occ::log::debug(
        "    breakdown: H0={:>14.6f}  ES={:>14.6f}  3rd={:>10.3e}  "
        "rep={:>10.3e}  disp={:>10.3e}",
        e_h0, e_es, e_third, e_rep, e_disp);
    occ::log::debug(
        "    diagnostics: |q|_max={:>10.3e}  Σq={:>+.3e}",
        qsh_new.cwiseAbs().maxCoeff(), qsh_new.sum());

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
      r.dispersion_energy = e_disp;
      r.total_energy = total_energy;
      r.shell_charges = qsh_new;
      r.atomic_charges = atom_charges;
      // Report Γ-point quantities (first k) for convenience; full per-k data
      // could be returned via a richer struct in a future revision.
      r.orbital_energies = orbital_energies_k[0];
      r.orbital_occupations = orbital_occupations_k[0];
      r.density_matrix = P_k[0].real();
      r.overlap_matrix = S_k[0].real();
      r.orbital_coefficients = C_k[0].real();
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

  occ::log::warn("Periodic GFN2 SCC (k-point) did not converge in {} iterations",
                  opts.max_iterations);
  PeriodicSccResult r;
  r.scc_energy = prev_energy - e_rep;
  r.repulsion_energy = e_rep;
  r.total_energy = prev_energy;
  r.shell_charges = qsh;
  Vec atom_charges = Vec::Zero(sys.atoms.size());
  for (int s = 0; s < n_shells; ++s) atom_charges(shells.atom[s]) += qsh(s);
  r.atomic_charges = atom_charges;
  r.orbital_energies = orbital_energies_k.empty() ? Vec() : orbital_energies_k[0];
  r.orbital_occupations =
      orbital_occupations_k.empty() ? Mat() : orbital_occupations_k[0];
  r.density_matrix = P_k.empty() ? Mat() : P_k[0].real();
  r.overlap_matrix = P_k.empty() ? Mat() : S_k[0].real();
  r.orbital_coefficients = C_k.empty() ? Mat() : C_k[0].real();
  r.n_iterations = iter;
  r.converged = false;
  return r;
}

} // namespace occ::xtb
