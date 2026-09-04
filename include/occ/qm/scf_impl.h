#pragma once
#include <ankerl/unordered_dense.h>
#include <limits>
#include <fmt/format.h>
#include <occ/qm/hf.h>

namespace occ::qm {

using gto::OCC_MINIMAL_BASIS;

template <SCFMethod P>
SCF<P>::SCF(P &procedure, SpinorbitalKind sk) : m_procedure(procedure) {
  ctx.n_electrons = m_procedure.active_electrons();
  ctx.n_frozen_electrons =
      m_procedure.total_electrons() - m_procedure.active_electrons();
  occ::log::debug("{} active electrons", ctx.n_electrons);
  occ::log::debug("{} frozen electrons", ctx.n_frozen_electrons);
  ctx.nbf = m_procedure.nbf();
  size_t rows, cols;
  std::tie(rows, cols) = occ::qm::matrix_dimensions(sk, ctx.nbf);
  ctx.S = Mat::Zero(rows, cols);
  ctx.T = Mat::Zero(rows, cols);
  ctx.V = Mat::Zero(rows, cols);
  ctx.H = Mat::Zero(rows, cols);
  ctx.F = Mat::Zero(rows, cols);
  ctx.Vecp = Mat::Zero(rows, cols);

  ctx.mo.kind = sk;
  ctx.mo.D = Mat::Zero(rows, cols);
  ctx.mo.C = Mat::Zero(rows, cols);
  ctx.mo.energies = Vec::Zero(rows);
  ctx.mo.n_ao = ctx.nbf;

  ctx.V_ext = Mat::Zero(rows, cols);
  ctx.energy["nuclear.repulsion"] = m_procedure.nuclear_repulsion_energy();
}

template <SCFMethod P> int SCF<P>::n_alpha() const { return ctx.n_occ; }

template <SCFMethod P> int SCF<P>::n_beta() const {
  return ctx.n_electrons - ctx.n_occ;
}

template <SCFMethod P> int SCF<P>::charge() const {
  double nuclear_charge = 0.0;
  for (const auto &atom : atoms()) {
    nuclear_charge += atom.atomic_number;
  }
  return nuclear_charge - ctx.n_electrons - ctx.n_frozen_electrons;
}

template <SCFMethod P> int SCF<P>::multiplicity() const {
  return ctx.n_unpaired_electrons + 1;
}

template <SCFMethod P> void SCF<P>::set_charge(int c) {
  set_charge_multiplicity(c, multiplicity());
}

template <SCFMethod P> void SCF<P>::set_multiplicity(int m) {
  set_charge_multiplicity(charge(), m);
}

template <SCFMethod P>
const MolecularOrbitals &SCF<P>::molecular_orbitals() const {
  return ctx.mo;
}

template <SCFMethod P> Wavefunction SCF<P>::wavefunction() const {
  Wavefunction wfn;
  wfn.atoms = m_procedure.atoms();
  wfn.basis = m_procedure.aobasis();
  wfn.nbf = wfn.basis.nbf();
  wfn.mo = ctx.mo;
  wfn.num_electrons = ctx.n_electrons;
  wfn.num_frozen_electrons = ctx.n_frozen_electrons;
  wfn.have_energies = true;
  wfn.energy.core = ctx.energy.at("electronic.1e");
  wfn.energy.kinetic = ctx.energy.at("electronic.kinetic");
  wfn.energy.nuclear_attraction = ctx.energy.at("electronic.nuclear");
  wfn.energy.nuclear_repulsion = ctx.energy.at("nuclear.repulsion");
  if (ctx.energy.contains("electronic.coulomb"))
    wfn.energy.coulomb = ctx.energy.at("electronic.coulomb");
  if (ctx.energy.contains("electronic.exchange"))
    wfn.energy.exchange = ctx.energy.at("electronic.exchange");
  wfn.energy.total = ctx.energy.at("total");
  wfn.T = ctx.T;
  wfn.V = ctx.V;
  wfn.method = m_procedure.name();
  wfn.converged = ctx.converged;
  return wfn;
}

template <SCFMethod P>
void SCF<P>::set_charge_multiplicity(int chg, unsigned int mult) {
  int current_charge = charge();
  bool state_changed = false;
  log::debug("Setting charge = {}, multiplicity = {} in scf", chg, mult);
  if (chg != current_charge) {
    ctx.n_electrons -= chg - current_charge;
    state_changed = true;
    if (ctx.n_electrons < 1) {
      throw std::runtime_error("Invalid charge: systems with no "
                               "electrons are not supported");
    }
  }
  if (mult != multiplicity() || state_changed) {
    state_changed = true;
    ctx.n_unpaired_electrons = mult - 1;
    if (is_odd(ctx.n_electrons + ctx.n_unpaired_electrons)) {
      throw std::runtime_error(
          fmt::format("Invalid spin state for {} electrons: number of unpaired "
                      "electrons ({}) must have the same parity",
                      ctx.n_electrons, ctx.n_unpaired_electrons));
    }
  }
  if (state_changed)
    update_occupied_orbital_count();
}

template <SCFMethod P> void SCF<P>::update_occupied_orbital_count() {
  switch (ctx.mo.kind) {
  case Restricted: {
    ctx.n_occ = ctx.n_electrons / 2;
    if (is_odd(ctx.n_electrons)) {
      throw std::runtime_error(
          fmt::format("Invalid num electrons ({}) for restricted SCF: not even",
                      ctx.n_electrons));
    }
    break;
  }
  case Unrestricted: {
    ctx.n_occ = (ctx.n_electrons - ctx.n_unpaired_electrons) / 2;
    ctx.n_unpaired_electrons = n_beta() - n_alpha();
    break;
  }
  case General: {
    ctx.n_occ = ctx.n_electrons;
    break;
  }
  }

  occ::log::debug("Setting MO n_alpha = {}, n_beta = {}", ctx.mo.n_alpha,
                  ctx.mo.n_beta);
  ctx.mo.n_alpha = n_alpha();
  ctx.mo.n_beta = n_beta();
}

template <SCFMethod P>
const std::vector<occ::core::Atom> &SCF<P>::atoms() const {
  return m_procedure.atoms();
}

template <SCFMethod P> void SCF<P>::set_conditioning_orthogonalizer() {
  if (ctx.mo.kind == Unrestricted) {
    ctx.orthogonalizer.build(block::a(ctx.S));
  } else {
    ctx.orthogonalizer.build(ctx.S);
  }
}

template <SCFMethod P> void SCF<P>::set_core_matrices() {

  bool calc_ecp = m_procedure.have_effective_core_potentials();
  switch (ctx.mo.kind) {
  case SpinorbitalKind::Restricted: {
    ctx.S = m_procedure.compute_overlap_matrix();
    ctx.T = m_procedure.compute_kinetic_matrix();
    ctx.V = m_procedure.compute_nuclear_attraction_matrix();
    if (calc_ecp) {
      ctx.Vecp = m_procedure.compute_effective_core_potential_matrix();
    }
    break;
  }
  case SpinorbitalKind::Unrestricted: {
    block::a(ctx.S) = m_procedure.compute_overlap_matrix();
    block::b(ctx.S) = block::a(ctx.S);
    block::a(ctx.T) = m_procedure.compute_kinetic_matrix();
    block::b(ctx.T) = block::a(ctx.T);
    block::a(ctx.V) = m_procedure.compute_nuclear_attraction_matrix();
    block::b(ctx.V) = block::a(ctx.V);
    if (calc_ecp) {
      block::a(ctx.Vecp) =
          m_procedure.compute_effective_core_potential_matrix();
      block::b(ctx.Vecp) = block::a(ctx.Vecp);
    }
    break;
  }
  case SpinorbitalKind::General: {
    block::aa(ctx.S) = m_procedure.compute_overlap_matrix();
    block::aa(ctx.T) = m_procedure.compute_kinetic_matrix();
    block::aa(ctx.V) = m_procedure.compute_nuclear_attraction_matrix();
    block::bb(ctx.S) = block::aa(ctx.S);
    block::bb(ctx.T) = block::aa(ctx.T);
    block::bb(ctx.V) = block::aa(ctx.V);
    if (calc_ecp) {
      block::aa(ctx.Vecp) =
          m_procedure.compute_effective_core_potential_matrix();
      block::bb(ctx.Vecp) = block::aa(ctx.Vecp);
    }
    break;
  }
  }
  // `ctx.V_ext` is populated by `set_external_potential(...)` and stays
  // zero otherwise — see `<occ/qm/external_potential.h>`.
  ctx.H = ctx.T + ctx.V + ctx.Vecp + ctx.V_ext;
}

template <SCFMethod P>
void SCF<P>::set_initial_guess_from_wfn(const Wavefunction &wfn) {
  log::info("Setting initial guess from existing wavefunction");
  m_have_initial_guess = true;
  ctx.mo = wfn.mo;
  update_occupied_orbital_count();
  set_core_matrices();
  set_conditioning_orthogonalizer();
}

template <SCFMethod P>
void SCF<P>::add_guess_potential(const Mat &potential) {
  // The guess builds one nbf x nbf operator; spread it over whichever spin
  // blocks this calculation carries, exactly as `set_core_matrices` does for
  // the nuclear attraction.
  switch (ctx.mo.kind) {
  case Restricted:
    ctx.F += potential;
    break;
  case Unrestricted:
    block::a(ctx.F) += potential;
    block::b(ctx.F) += potential;
    break;
  case General:
    block::aa(ctx.F) += potential;
    block::bb(ctx.F) += potential;
    break;
  }
}

template <SCFMethod P>
void SCF<P>::add_guess_density(const Guess &guess) {
  // Only a density is known here -- there are no orbitals yet -- so this has
  // to be the density-only Fock build, and the orbitals come out of
  // diagonalising the result. `compute_fock_mixed_basis` takes a single
  // square density normalised to half the electron count and replicates its
  // contribution across the spin blocks, which is all a guess needs: the
  // first real iteration resolves the spins.
  occ::qm::MolecularOrbitals mo_guess;
  mo_guess.kind = ctx.mo.kind;
  mo_guess.n_ao = guess.density.rows();
  mo_guess.n_alpha = n_alpha();
  mo_guess.n_beta = n_beta();
  mo_guess.D = guess.density;
  ctx.F += m_procedure.compute_fock_mixed_basis(
      mo_guess, guess.density_basis, guess.density_is_shell_diagonal);
}

template <SCFMethod P> void SCF<P>::compute_initial_guess() {
  if (m_have_initial_guess)
    return;

  // The guess needs the occupation counts to be settled -- they decide how
  // many orbitals come out of the diagonalisation, and density-fitted
  // exchange is built from the occupied coefficients. Idempotent, so calling
  // it here costs nothing when the SCF driver has already done it.
  update_occupied_orbital_count();

  log::info("Computing core hamiltonian");
  set_core_matrices();
  ctx.F = ctx.H;
  occ::timing::start(occ::timing::category::la);
  set_conditioning_orthogonalizer();
  occ::timing::stop(occ::timing::category::la);

  const GuessKind kind = select_guess(m_guess_kind, m_procedure.aobasis());

  // Time the guess only when there is one, which also keeps the category from
  // being entered twice over. The atomic guess runs a nested SCF per element,
  // each starting from the core Hamiltonian; `StopWatch::start` simply
  // overwrites the start point, so a nested region would leave the outer one
  // measuring from the wrong instant. Guarding on the kind means the nested
  // calculations never enter the category at all, and the core guess -- which
  // is free by definition -- is the only thing that goes unmeasured.
  const bool timed = kind != GuessKind::Core;
  if (timed)
    occ::timing::start(occ::timing::category::guess);

  const GuessRequest request{m_procedure.aobasis(), charge(), ctx.n_electrons};
  const Guess guess = build_guess(kind, request);
  // Report what was built, not what was asked for: `build_guess` falls back
  // to the core Hamiltonian when a guess cannot be built for this system.
  log::info("Initial guess: {}", guess_kind_name(guess.kind));

  if (guess.potential.size() > 0)
    add_guess_potential(guess.potential);
  else if (guess.density.size() > 0)
    add_guess_density(guess);

  // Every guess ends here: whatever it added to the core Hamiltonian, the
  // orbitals come from diagonalising the result. That also means every guess
  // leaves behind a full set of orbitals, which the first Fock build needs --
  // density-fitted exchange is built from the occupied coefficients, not the
  // density.
  ctx.orthogonalizer.orthogonalize_molecular_orbitals(ctx.mo, ctx.F);
  m_have_initial_guess = true;

  if (timed)
    occ::timing::stop(occ::timing::category::guess);
}

template <SCFMethod P>
void SCF<P>::set_external_potential(const Mat &V_ext_single,
                                    double nuclear_energy,
                                    std::string_view label) {
  if (label.empty()) {
    throw std::runtime_error(
        "External potential label must be non-empty");
  }
  const size_t nbf = m_procedure.nbf();
  if (static_cast<size_t>(V_ext_single.rows()) != nbf ||
      static_cast<size_t>(V_ext_single.cols()) != nbf) {
    throw std::runtime_error(fmt::format(
        "External potential matrix shape {}x{} does not match basis nbf={}",
        V_ext_single.rows(), V_ext_single.cols(), nbf));
  }
  switch (ctx.mo.kind) {
  case SpinorbitalKind::Restricted:
    ctx.V_ext = V_ext_single;
    break;
  case SpinorbitalKind::Unrestricted:
    block::a(ctx.V_ext) = V_ext_single;
    block::b(ctx.V_ext) = V_ext_single;
    break;
  case SpinorbitalKind::General:
    block::aa(ctx.V_ext) = V_ext_single;
    block::bb(ctx.V_ext) = V_ext_single;
    break;
  }
  ctx.external_potential_label.assign(label);
  ctx.energy["nuclear." + ctx.external_potential_label] = nuclear_energy;
  log::info("External potential '{}' set: nuclear–external energy = {:.8f} Ha",
            label, nuclear_energy);
}

template <SCFMethod P>
void SCF<P>::apply_level_shift(Mat &F, double shift) const {
  // Saunders-Hillier: F' = F + b (S - S P_occ S), with P_occ the occupied
  // density in the AO metric. S - S P_occ S is S times the virtual projector,
  // so the shift lands entirely on the virtual block and the occupied
  // eigenvalues are untouched.
  //
  // The spin-resolved kinds carry one Fock block per spin, each with its own
  // occupied space, so each is shifted against its own density.
  switch (ctx.mo.kind) {
  case SpinorbitalKind::Restricted: {
    // D is the total density, i.e. twice the occupied projector.
    const Mat SDS = ctx.S * (0.5 * ctx.mo.D) * ctx.S;
    F.noalias() += shift * (ctx.S - SDS);
    break;
  }
  case SpinorbitalKind::Unrestricted: {
    const Mat Sa = ctx.S * block::a(ctx.mo.D) * ctx.S;
    const Mat Sb = ctx.S * block::b(ctx.mo.D) * ctx.S;
    block::a(F).noalias() += shift * (ctx.S - Sa);
    block::b(F).noalias() += shift * (ctx.S - Sb);
    break;
  }
  case SpinorbitalKind::General: {
    // The overlap is block diagonal but the density is not: spin mixing puts
    // weight in the ab/ba blocks, and dropping them would shift against a
    // projector that is not idempotent.
    block::aa(F).noalias() +=
        shift * (ctx.S - ctx.S * block::aa(ctx.mo.D) * ctx.S);
    block::bb(F).noalias() +=
        shift * (ctx.S - ctx.S * block::bb(ctx.mo.D) * ctx.S);
    block::ab(F).noalias() -= shift * (ctx.S * block::ab(ctx.mo.D) * ctx.S);
    block::ba(F).noalias() -= shift * (ctx.S * block::ba(ctx.mo.D) * ctx.S);
    break;
  }
  }
}

template <SCFMethod P> void SCF<P>::update_scf_energy(bool incremental) {

  // One-electron terms are traces against the *current* density, so they must
  // be refreshed every cycle — only the two-electron build is incremental.
  {
    occ::timing::start(occ::timing::category::la);
    ctx.energy["electronic.kinetic"] =
        2 * expectation(ctx.mo.kind, ctx.mo.D, ctx.T);
    ctx.energy["electronic.nuclear"] =
        2 * expectation(ctx.mo.kind, ctx.mo.D, ctx.V);
    ctx.energy["electronic.1e"] = 2 * expectation(ctx.mo.kind, ctx.mo.D, ctx.H);
    occ::timing::stop(occ::timing::category::la);
  }
  if (m_procedure.usual_scf_energy()) {
    occ::timing::start(occ::timing::category::la);
    ctx.energy["electronic"] = 0.5 * ctx.energy["electronic.1e"];
    ctx.energy["electronic"] += expectation(ctx.mo.kind, ctx.mo.D, ctx.F);
    ctx.energy["electronic.2e"] =
        ctx.energy["electronic"] - ctx.energy["electronic.1e"];
    ctx.energy["total"] =
        ctx.energy["electronic"] + ctx.energy["nuclear.repulsion"];
    if (!ctx.external_potential_label.empty()) {
      ctx.energy["total"] +=
          ctx.energy["nuclear." + ctx.external_potential_label];
    }
    occ::timing::stop(occ::timing::category::la);
  }
  if (m_procedure.have_effective_core_potentials()) {
    ctx.energy["electronic.ecp"] = expectation(ctx.mo.kind, ctx.mo.D, ctx.Vecp);
  }
  if (!ctx.external_potential_label.empty()) {
    ctx.energy["electronic." + ctx.external_potential_label] =
        2 * expectation(ctx.mo.kind, ctx.mo.D, ctx.V_ext);
  }
  m_procedure.update_scf_energy(ctx.energy);
}

template <SCFMethod P> inline const char *SCF<P>::scf_kind() const {
  switch (ctx.mo.kind) {
  case Unrestricted:
    return "unrestricted";
  case General:
    return "general";
  default:
    return "restricted";
  }
}

template <SCFMethod P> double SCF<P>::compute_scf_energy() {
  if (ctx.converged)
    return ctx.energy["total"];
  // compute one-body integrals
  // count the number of electrons
  bool incremental{false};
  update_occupied_orbital_count();

  // Initialize convergence accelerator with settings
  convergence_accelerator.set_strategy(convergence_settings.diis_strategy);
  convergence_accelerator.set_switch_threshold(convergence_settings.diis_switch_threshold);

  compute_initial_guess();
  ctx.K = m_procedure.compute_schwarz_ints();
  Mat D_diff = ctx.mo.D;
  Mat D_last;
  Mat FD_comm = Mat::Zero(ctx.F.rows(), ctx.F.cols());
  update_scf_energy(incremental);
  log::info("starting {} scf iterations", scf_kind());
  log::debug("{} electrons total", ctx.n_electrons);
  log::debug("{} alpha electrons", n_alpha());
  log::debug("{} beta electrons", n_beta());
  log::debug("net charge {}", charge());
  total_time = 0.0;

  do {
    const auto tstart = std::chrono::high_resolution_clock::now();
    ++iter;
    // Last iteration's energy and density
    auto ehf_last = ctx.energy["electronic"];
    D_last = ctx.mo.D;
    ctx.H = ctx.T + ctx.V + ctx.Vecp + ctx.V_ext;
    m_procedure.update_core_hamiltonian(ctx.mo, ctx.H);
    incremental = true;

    if (incremental_fock_supported() && not incremental_Fbuild_started &&
        convergence_settings.start_incremental_fock(diis_error)) {
      incremental_Fbuild_started = true;
      reset_incremental_fock_formation = false;
      last_reset_iteration = iter - 1;
      next_reset_threshold = diis_error / 10;
      log::debug("starting incremental fock build");
    }
    if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
      ctx.F = ctx.H;
      D_diff = ctx.mo.D;
      incremental = false;
    }
    if (reset_incremental_fock_formation && incremental_Fbuild_started) {
      reset_incremental_fock_formation = false;
      last_reset_iteration = iter;
      next_reset_threshold = diis_error / 10;
      log::debug("resetting incremental fock build");
    }

    if (incremental)
      ++num_incremental_fock_builds;

    // build a new Fock matrix
    std::swap(ctx.mo.D, D_diff);
    ctx.F += m_procedure.compute_fock(ctx.mo, ctx.K);
    std::swap(ctx.mo.D, D_diff);

    // compute HF energy with the non-extrapolated Fock matrix
    update_scf_energy(incremental);
    ediff_rel = std::abs((ctx.energy["electronic"] - ehf_last) /
                         ctx.energy["electronic"]);

    // Apply convergence acceleration (DIIS extrapolation)
    Mat F_diis = convergence_accelerator.update(
        ctx.mo.kind, ctx.S, ctx.mo.D, ctx.F, ctx.energy["electronic"]);
    diis_error = convergence_accelerator.max_error();

    if (diis_error < next_reset_threshold || iter - last_reset_iteration >= 8)
      reset_incremental_fock_formation = true;

    // Level shift the virtuals while the density is still far from converged.
    //
    // Adding b * (S - S P_occ S) raises every virtual orbital by b Hartree
    // and leaves the occupied block alone, which widens the gap the next
    // diagonalisation sees and stops occupied and virtual orbitals trading
    // places between cycles. `effective_level_shift` returns zero once the
    // commutator drops below its threshold, so the converged solution is the
    // unshifted one and the energy is unaffected.
    const double shift =
        convergence_settings.effective_level_shift(diis_error);
    if (shift != 0.0)
      apply_level_shift(F_diis, shift);

    ctx.orthogonalizer.orthogonalize_molecular_orbitals(ctx.mo, F_diis);
    D_diff = ctx.mo.D - D_last;

    const auto tstop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed = tstop - tstart;

    if (iter == 1) {
      log::info("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}", "#",
                "E (Hartrees)", "|dE|/E", "max|FDS-SDF|", "T (s)");
    }
    log::info("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}", iter,
              ctx.energy["total"], ediff_rel, diis_error, time_elapsed.count());
    log::flush();
    total_time += time_elapsed.count();

    ctx.converged = convergence_settings.energy_and_commutator_converged(
        ediff_rel, diis_error);
  } while (!ctx.converged && (iter < maxiter));

  if (ctx.converged) {
    log::info("{} spinorbital SCF energy converged after {:.5f} seconds",
              scf_kind(), total_time);
    log::info("{}", ctx.energy.to_string());
  } else {
    log::error("{} spinorbital SCF did not converge after {} iterations "
               "({:.5f} seconds): |dE|/E={:.3e}, max|FDS-SDF|={:.3e}",
               scf_kind(), iter, total_time, ediff_rel, diis_error);
    log::info("last energies (not converged):\n{}", ctx.energy.to_string());
  }
  return ctx.energy["total"];
}
} // namespace occ::qm
