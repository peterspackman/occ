#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
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
  occ::timing::start(occ::timing::xtb_setup);
  // Build basis, shells, and integrals from the central-cell atoms.
  auto basis = build_aobasis(sys.atoms, params);
  auto shells = build_shell_table(sys.atoms, params);
  const int nbf = static_cast<int>(basis.nbf());
  const int n_shells = static_cast<int>(shells.atom.size());
  const auto bf_to_atom = basis.bf_to_atom();
  const auto bf_to_shell = basis.bf_to_shell();
  const Vec z_sh = shells.ref_occ;

  // Per-quantity lattice translations: AO matrices decay as exp(-α·r²) —
  // a tight cutoff is safe and cheap. CN and repulsion need slightly wider
  // sums.
  auto cn_images = build_lattice_images(sys.lattice_bohr, opts.cn_cutoff);
  auto rep_images = build_lattice_images(sys.lattice_bohr, opts.rep_cutoff);
  auto ao_images = build_lattice_images(sys.lattice_bohr, opts.ao_cutoff);
  // Periodic CN, repulsion, AO blocks, Bloch sums at Γ.
  Vec cn = gfn_coordination_numbers_periodic(sys.atoms, cn_images);
  const double e_rep = repulsion_energy_periodic(sys.atoms, params, rep_images);
  occ::timing::start(occ::timing::xtb_overlap);
  auto S_per_T = periodic_overlap_blocks(sys, params, ao_images);
  occ::timing::stop(occ::timing::xtb_overlap);
  occ::timing::start(occ::timing::xtb_h0);
  auto H0_per_T = periodic_h0_blocks(sys, params, ao_images, S_per_T, cn);
  occ::timing::stop(occ::timing::xtb_h0);
  Mat S = bloch_sum_gamma(S_per_T);
  Mat H0 = bloch_sum_gamma(H0_per_T);

  // Periodic shell-resolved γ via Ewald.
  occ::timing::start(occ::timing::xtb_ewald_gamma);
  auto ewald = build_ewald_data(sys, /*tol=*/1e-10, opts.ewald_alpha,
                                 opts.ewald_residual_cutoff);
  Mat J = periodic_klopman_ohno_gamma(sys, shells, params, ewald);
  occ::timing::stop(occ::timing::xtb_ewald_gamma);

  // Multipole AO matrices — Bloch-summed at Γ with per-atom origin (Ket =
  // atom-of-row centered, Bra = atom-of-col-image centered). With this
  // convention each AO pair contributes to atom_of(row) via Ket and
  // atom_of(col) via Bra in the CAMM partition, with no R*S correction
  // needed — critical for dense crystals where image AO overlaps are large.
  qm::IntegralEngine engine(basis);
  std::optional<PeriodicMultipoleAO> mp_ao;
  Vec mp_radii;
  std::optional<MultipolePairTensors> mp_tensors;  // Ewald path
  if (opts.include_multipoles) {
    occ::timing::start(occ::timing::xtb_multipole_ao);
    mp_ao = build_periodic_multipole_ao(sys, params, ao_images);
    mp_radii = multipole_radii(sys.atoms, cn, params);
    mp_tensors = build_multipole_ewald_tensors(sys, mp_radii, params);
    occ::timing::stop(occ::timing::xtb_multipole_ao);
  }
  occ::timing::stop(occ::timing::xtb_setup);

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
  occ::log::info("cutoffs (Bohr): cn={:.1f}  rep={:.1f}  ao={:.1f}   "
                  "ewald α={:.4f}   #G={}",
                  opts.cn_cutoff, opts.rep_cutoff, opts.ao_cutoff,
                  ewald.alpha, ewald.g_vectors.size());
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

    if (opts.include_multipoles && iter > 1) {
      // H1 uses the DIIS-mixed multipole state from the previous iteration's
      // diagonalization (`mom`). At iter 1 `mom = 0`: the multipole H1 only
      // turns on once the first density has produced non-zero CAMM.
      occ::timing::start(occ::timing::xtb_aes);
      Vec atom_q = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q(shells.atom[s]) += qsh(s);
      AnisotropicPotentials pot = anisotropic_potentials_ewald(
          sys.atoms, atom_q, mom, *mp_tensors, params);
      apply_anisotropic_h1_periodic(H, S, mp_ao->D_ket, mp_ao->D_bra,
                                     mp_ao->Q_ket, mp_ao->Q_bra,
                                     bf_to_atom, pot);
      occ::timing::stop(occ::timing::xtb_aes);
    }

    occ::timing::start(occ::timing::xtb_eigensolve);
    Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(H, S);
    if (es.info() != Eigen::Success) {
      throw std::runtime_error(
          "run_charge_only_periodic_scc: eigensolver failed");
    }
    orbital_energies = es.eigenvalues();
    C = es.eigenvectors();
    occ::timing::stop(occ::timing::xtb_eigensolve);

    occ::timing::start(occ::timing::xtb_density);
    orbital_occupations = Vec::Zero(nbf);
    for (int i = 0; i < n_occ; ++i) orbital_occupations(i) = 2.0;

    Mat Cocc = C.leftCols(n_occ);
    P = 2.0 * (Cocc * Cocc.transpose());

    Mat PS = P * S;
    Vec pop = shell_populations(PS, bf_to_shell, n_shells);
    Vec qsh_new = z_sh - pop;
    occ::timing::stop(occ::timing::xtb_density);

    // Compute the new CAMM multipoles from the just-solved density. The
    // multipole energy is reported from these post-density values so the
    // per-iter energy is a self-consistent (P, q, μ) triple rather than a
    // mix of input H1 multipoles with output charges.
    CammMoments mom_new;
    AnisotropicEnergy e_aniso{0.0, 0.0};
    if (opts.include_multipoles) {
      occ::timing::start(occ::timing::xtb_camm);
      mom_new = compute_camm_moments_periodic(
          sys.atoms, bf_to_atom, P, mp_ao->D_ket, mp_ao->D_bra,
          mp_ao->Q_ket, mp_ao->Q_bra);
      occ::timing::stop(occ::timing::xtb_camm);
      Vec atom_q_new = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q_new(shells.atom[s]) += qsh_new(s);
      occ::timing::start(occ::timing::xtb_aes);
      e_aniso = anisotropic_energy_ewald(sys.atoms, atom_q_new, mom_new,
                                          *mp_tensors, params);
      occ::timing::stop(occ::timing::xtb_aes);
    }

    if (native_d4) {
      occ::timing::start(occ::timing::xtb_dispersion);
      Vec atom_q_new = Vec::Zero(sys.atoms.size());
      for (int s = 0; s < n_shells; ++s)
        atom_q_new(shells.atom[s]) += qsh_new(s);
      native_d4->set_charges(atom_q_new);
      e_disp = native_d4->energy_periodic(sys.lattice_bohr);
      occ::timing::stop(occ::timing::xtb_dispersion);
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
      // Per-atom dump for diagnostic comparison.
      for (int a = 0; a < static_cast<int>(sys.atoms.size()); ++a) {
        occ::log::debug(
            "    atom {:3d} (Z={:>2d})  q={:+.6f}  d=({:+.6f}, {:+.6f}, "
            "{:+.6f})  qp_xx={:+.6f} yy={:+.6f} zz={:+.6f} xy={:+.6f} "
            "xz={:+.6f} yz={:+.6f}",
            a + 1, sys.atoms[a].atomic_number, atom_q_dbg(a),
            mom.dipm(0, a), mom.dipm(1, a), mom.dipm(2, a),
            mom.qp(0, a), mom.qp(2, a), mom.qp(5, a),
            mom.qp(1, a), mom.qp(3, a), mom.qp(4, a));
      }
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

    // mom_new is the freshly computed CAMM from the new density (computed
    // above); reuse it as the input for next iteration's H1 after DIIS mixing.
    // Pack (qsh_new, mom_new) into a single DIIS state vector. Error is the
    // change since the previous (qsh, mom). Extrapolate, then unpack back.
    occ::timing::start(occ::timing::xtb_diis);
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
    occ::timing::stop(occ::timing::xtb_diis);
    prev_energy = total_energy;
  }

  occ::log::warn("Periodic GFN2 SCC did not converge in {} iterations",
                  opts.max_iterations);
  PeriodicSccResult r;
  r.scc_energy = prev_energy - e_rep - e_disp;
  r.repulsion_energy = e_rep;
  r.dispersion_energy = e_disp;
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

  auto cn_images = build_lattice_images(sys.lattice_bohr, opts.cn_cutoff);
  auto rep_images = build_lattice_images(sys.lattice_bohr, opts.rep_cutoff);
  auto ao_images = build_lattice_images(sys.lattice_bohr, opts.ao_cutoff);
  Vec cn = gfn_coordination_numbers_periodic(sys.atoms, cn_images);
  const double e_rep = repulsion_energy_periodic(sys.atoms, params, rep_images);
  auto S_per_T = periodic_overlap_blocks(sys, params, ao_images);
  auto H0_per_T = periodic_h0_blocks(sys, params, ao_images, S_per_T, cn);

  auto ewald = build_ewald_data(sys, /*tol=*/1e-10, opts.ewald_alpha,
                                 opts.ewald_residual_cutoff);
  Mat J = periodic_klopman_ohno_gamma(sys, shells, params, ewald);

  // Multipole machinery: per-T AO blocks (for k-point Bloch sums of D_ket /
  // D_bra / Q_ket / Q_bra) and the Ewald-summed multipole pair tensors
  // (independent of k — only the AO matrices carry the band-structure phase).
  std::optional<PeriodicMultipoleAOBlocks> mp_blocks;
  Vec mp_radii;
  std::optional<MultipolePairTensors> mp_tensors;
  if (opts.include_multipoles) {
    mp_blocks = build_periodic_multipole_ao_blocks(sys, params, ao_images);
    mp_radii = multipole_radii(sys.atoms, cn, params);
    mp_tensors = build_multipole_ewald_tensors(sys, mp_radii, params);
  }

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

  // Pre-compute Bloch-summed S(k), H0(k) for each k (geometry-cached). The
  // multipole AO Bloch sums are deferred to inside the SCC loop because they
  // change with the SCC potentials (vs/vd/vq), but actually no — the AO
  // matrices themselves are geometry-only. Cache them per-k too if multipoles
  // are on so we don't pay 18 Bloch sums per (iter × k).
  const int n_k = static_cast<int>(kpoints.size());
  std::vector<CMat> S_k(n_k), H0_k(n_k);
  std::vector<CMatTriple> D_ket_k, D_bra_k;
  std::vector<std::array<CMat, 6>> Q_ket_k, Q_bra_k;
  if (opts.include_multipoles) {
    D_ket_k.resize(n_k);
    D_bra_k.resize(n_k);
    Q_ket_k.resize(n_k);
    Q_bra_k.resize(n_k);
  }
  // Bloch sums are independent across k — thread them.
  occ::parallel::parallel_for(size_t{0}, static_cast<size_t>(n_k),
                                [&](size_t ik) {
    S_k[ik] = bloch_sum(S_per_T, ao_images, kpoints[ik].k);
    H0_k[ik] = bloch_sum(H0_per_T, ao_images, kpoints[ik].k);
    if (opts.include_multipoles) {
      D_ket_k[ik] = bloch_sum_triple(mp_blocks->D_ket, ao_images, kpoints[ik].k);
      D_bra_k[ik] = bloch_sum_triple(mp_blocks->D_bra, ao_images, kpoints[ik].k);
      Q_ket_k[ik] = bloch_sum_array6(mp_blocks->Q_ket, ao_images, kpoints[ik].k);
      Q_bra_k[ik] = bloch_sum_array6(mp_blocks->Q_bra, ao_images, kpoints[ik].k);
    }
  });

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

  // Atomic multipole state (DIIS-mixed alongside qsh, same as the Γ-only
  // path). Initial value zero — first iter has no multipole H1 contribution.
  const int n_atoms = static_cast<int>(sys.atoms.size());
  CammMoments mom;
  if (opts.include_multipoles) {
    mom.dipm = Mat3N::Zero(3, n_atoms);
    mom.qp = Mat::Zero(6, n_atoms);
  }
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

  occ::log::info("{:=^72s}", "  GFN2-xTB periodic SCC (k-point)  ");
  occ::log::info(
      "nbf = {}   n_shells = {}   n_electrons = {}   n_kpts = {}", nbf,
      n_shells, n_elec, n_k);
  occ::log::info("cutoffs (Bohr): cn={:.1f}  rep={:.1f}  ao={:.1f}   "
                  "ewald α={:.4f}   #G={}",
                  opts.cn_cutoff, opts.rep_cutoff, opts.ao_cutoff,
                  ewald.alpha, ewald.g_vectors.size());
  occ::log::info("{:>4s}  {:>20s}  {:>12s}  {:>12s}", "iter", "E (Hartree)",
                  "|ΔE|", "max|Δq|");

  bool converged = false;
  int iter = 0;
  for (iter = 1; iter <= opts.max_iterations; ++iter) {
    Vec V = J * qsh;
    for (Eigen::Index s = 0; s < V.size(); ++s) {
      V(s) += shells.third_order(s) * qsh(s) * qsh(s);
    }

    // Anisotropic potentials from the DIIS-mixed (qsh, mom). At iter 1
    // mom = 0, so vd / vq vanish and the multipole H1 contribution is
    // identically zero (same convention as the Γ-only path).
    AnisotropicPotentials pot;
    if (opts.include_multipoles) {
      Vec atom_q = Vec::Zero(n_atoms);
      for (int s = 0; s < n_shells; ++s) atom_q(shells.atom[s]) += qsh(s);
      pot = anisotropic_potentials_ewald(sys.atoms, atom_q, mom, *mp_tensors,
                                           params);
    }

    // Solve at every k. Each k-point is independent — H(k) construction +
    // generalized eigensolve thread cleanly.
    occ::parallel::parallel_for(size_t{0}, static_cast<size_t>(n_k),
                                  [&](size_t ik) {
      CMat H = H0_k[ik];
      for (Eigen::Index mu = 0; mu < nbf; ++mu) {
        const int sh_mu = bf_to_shell[mu];
        for (Eigen::Index nu = 0; nu < nbf; ++nu) {
          const int sh_nu = bf_to_shell[nu];
          H(mu, nu) -= 0.5 * S_k[ik](mu, nu) * (V(sh_mu) + V(sh_nu));
        }
      }
      if (opts.include_multipoles && iter > 1) {
        apply_anisotropic_h1_kpoint(H, S_k[ik], D_ket_k[ik], D_bra_k[ik],
                                      Q_ket_k[ik], Q_bra_k[ik], bf_to_atom,
                                      pot);
      }
      auto sol = solve_generalized_hermitian(H, S_k[ik]);
      orbital_energies_k[ik] = sol.eigenvalues;
      C_k[ik] = sol.eigenvectors;
    });

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

    // Density at each k. Mulliken populations + (if multipoles on) CAMM
    // partition are accumulated in the same k-loop.
    Vec pop = Vec::Zero(n_shells);
    double e_h0 = 0.0;
    CammMoments mom_new;
    if (opts.include_multipoles) {
      mom_new.dipm = Mat3N::Zero(3, n_atoms);
      mom_new.qp = Mat::Zero(6, n_atoms);
    }
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
      if (opts.include_multipoles) {
        accumulate_camm_kpoint(bf_to_atom, P_k[ik], w, D_bra_k[ik],
                                Q_bra_k[ik], mom_new);
      }
    }

    Vec qsh_new = z_sh - pop;

    AnisotropicEnergy e_aniso{0.0, 0.0};
    if (opts.include_multipoles) {
      Vec atom_q_new = Vec::Zero(n_atoms);
      for (int s = 0; s < n_shells; ++s) atom_q_new(shells.atom[s]) += qsh_new(s);
      e_aniso = anisotropic_energy_ewald(sys.atoms, atom_q_new, mom_new,
                                           *mp_tensors, params);
    }

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

    // DIIS over packed (qsh; dipm; qpat) — matches the Γ-only path so the
    // multipole H1 contribution stabilises together with the shell charges.
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
      Vec mixed = (1.0 - opts.damping_factor) * state +
                  opts.damping_factor * state_prev;
      unpack_state(mixed, qsh, mom);
    }
    prev_energy = total_energy;
  }

  occ::log::warn("Periodic GFN2 SCC (k-point) did not converge in {} iterations",
                  opts.max_iterations);
  PeriodicSccResult r;
  r.scc_energy = prev_energy - e_rep - e_disp;
  r.repulsion_energy = e_rep;
  r.dispersion_energy = e_disp;
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
