#pragma once

namespace occ::qm {

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

  ctx.Vpc = Mat::Zero(rows, cols);
  ctx.energy["nuclear.repulsion"] = m_procedure.nuclear_repulsion_energy();
  if (!m_procedure.supports_incremental_fock_build())
    convergence_settings.incremental_fock_threshold = 0.0;
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

template <SCFMethod P>
Mat SCF<P>::compute_soad(const Mat &overlap_minbs) const {
  // computes Superposition-Of-Atomic-Densities guess for the
  // molecular density matrix in minimal basis; occupies subshells by
  // smearing electrons evenly over the orbitals compute number of
  // atomic orbitals
  size_t nao = 0;
  bool spherical = m_procedure.aobasis().is_pure();
  for (const auto &atom : atoms()) {
    const auto Z = atom.atomic_number;
    nao += occ::qm::guess::minimal_basis_nao(Z, spherical);
  }

  // compute the minimal basis density
  Mat D_minbs = Mat::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  // const auto &frozen_electrons = m_procedure.frozen_electrons();
  for (const auto &atom : atoms()) {
    const auto Z = atom.atomic_number;
    // the following code might be useful for a minimal
    // basis guess with ECPs
    /*
    double remaining_frozen = frozen_electrons[atom_index];
    */

    auto occvec = occ::qm::guess::minimal_basis_occupation_vector(Z, spherical);

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

    occ::log::debug("Occupation vector for atom {} sum: {}",
    atom_index, std::accumulate(occvec.begin(), occvec.end(), 0.0));
    */
    int bf = 0;
    for (const auto &occ : occvec) {
      D_minbs(ao_offset + bf, ao_offset + bf) = occ;
      bf++;
    }
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

  for (int bf = 0; bf < D_minbs.rows(); bf++) {
    const double ovlp = overlap_minbs(bf, bf);
    if (std::abs(ovlp - 1.0) > 1e-6) {
      occ::log::debug("Normalising overlap min basis bf{} = {}", bf, ovlp);
    }
    D_minbs(bf, bf) /= ovlp;
  }
  double diagonal_sum = (D_minbs * overlap_minbs).diagonal().sum();
  ;
  double difference = diagonal_sum - ctx.n_electrons;
  occ::log::debug("Minimal basis guess diagonal sum: {}", diagonal_sum);
  if (std::abs(difference) > 1e-6)
    occ::log::warn(
        "Warning! Difference between diagonal sum and num electrons: {}",
        difference);
  return D_minbs * 0.5; // we use densities normalized to # of electrons/2
}

template <SCFMethod P> void SCF<P>::set_conditioning_orthogonalizer() {
  double S_condition_number_threshold =
      1.0 / std::numeric_limits<double>::epsilon();
  occ::core::ConditioningOrthogonalizerResult g;
  if (ctx.mo.kind == Unrestricted) {
    g = core::conditioning_orthogonalizer(block::a(ctx.S),
                                          S_condition_number_threshold);
  } else {
    g = core::conditioning_orthogonalizer(ctx.S, S_condition_number_threshold);
  }

  ctx.Xinv = g.result_inverse;
  ctx.XtX_condition_number = g.result_condition_number;
  ctx.X = g.result;
}

template <SCFMethod P> void SCF<P>::set_core_matrices() {

  bool calc_ecp = m_procedure.have_effective_core_potentials();
  bool calc_pc = ctx.point_charges.size() > 0;
  switch (ctx.mo.kind) {
  case SpinorbitalKind::Restricted: {
    ctx.S = m_procedure.compute_overlap_matrix();
    ctx.T = m_procedure.compute_kinetic_matrix();
    ctx.V = m_procedure.compute_nuclear_attraction_matrix();
    if (calc_ecp) {
      ctx.Vecp = m_procedure.compute_effective_core_potential_matrix();
    }
    if (calc_pc) {
      ctx.Vpc = m_procedure.compute_point_charge_interaction_matrix(
          ctx.point_charges);
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
    if (calc_pc) {
      block::a(ctx.Vpc) = m_procedure.compute_point_charge_interaction_matrix(
          ctx.point_charges);
      block::b(ctx.Vpc) = block::a(ctx.Vpc);
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
    if (calc_pc) {
      block::aa(ctx.Vpc) = m_procedure.compute_point_charge_interaction_matrix(
          ctx.point_charges);
      block::bb(ctx.Vpc) = block::aa(ctx.Vpc);
    }

    break;
  }
  }
  ctx.H = ctx.T + ctx.V + ctx.Vecp + ctx.Vpc;
}

template <SCFMethod P>
void SCF<P>::set_initial_guess_from_wfn(const Wavefunction &wfn) {
  log::info("Setting initial guess from existing wavefunction");
  m_have_initial_guess = true;
  ctx.mo = wfn.mo;
  update_occupied_orbital_count();
  set_core_matrices();
  // F = H;
  set_conditioning_orthogonalizer();
  // mo.update(X, F);
}

template <SCFMethod P> void SCF<P>::compute_initial_guess() {
  if (m_have_initial_guess)
    return;

  log::info("Computing core hamiltonian");
  set_core_matrices();
  ctx.F = ctx.H;
  occ::timing::start(occ::timing::category::la);
  set_conditioning_orthogonalizer();
  occ::timing::stop(occ::timing::category::la);

  occ::timing::start(occ::timing::category::guess);
  if (m_procedure.have_effective_core_potentials()) {
    // use core guess
    log::info("Computing initial guess using core hamiltonian with ECPs");
    ctx.mo.update(ctx.X, ctx.F);
    occ::timing::stop(occ::timing::category::guess);
    return;
  }

  log::info("Computing initial guess using SOAD in minimal basis");
  Mat D_minbs;
  if (m_procedure.aobasis().name() == OCC_MINIMAL_BASIS) {
    D_minbs = compute_soad(
        m_procedure.compute_overlap_matrix()); // compute guess in minimal basis
    switch (ctx.mo.kind) {
    case Restricted:
      ctx.mo.D = D_minbs;
      break;
    case Unrestricted:
      block::a(ctx.mo.D) =
          D_minbs * (static_cast<double>(n_alpha()) / ctx.n_electrons);
      block::b(ctx.mo.D) =
          D_minbs * (static_cast<double>(n_beta()) / ctx.n_electrons);
      break;
    case General:
      block::aa(ctx.mo.D) = D_minbs * 0.5;
      block::bb(ctx.mo.D) = D_minbs * 0.5;
      break;
    }
  } else {
    // if basis != minimal basis, map non-representable SOAD guess
    // into the AO basis
    // by diagonalizing a Fock matrix
    log::debug("Projecting minimal basis guess into atomic orbital "
               "basis...");
    const auto tstart = std::chrono::high_resolution_clock::now();
    auto minbs = occ::qm::AOBasis::load(m_procedure.atoms(), OCC_MINIMAL_BASIS);
    minbs.set_pure(m_procedure.aobasis().is_pure());
    D_minbs = compute_soad(m_procedure.compute_overlap_matrix_for_basis(
        minbs)); // compute guess in minimal basis
    occ::log::debug("Loaded minimal basis {}", OCC_MINIMAL_BASIS);
    occ::qm::MolecularOrbitals mo_minbs;
    mo_minbs.kind = ctx.mo.kind;
    mo_minbs.D = D_minbs;
    ctx.F += m_procedure.compute_fock_mixed_basis(mo_minbs, minbs, true);
    ctx.mo.update(ctx.X, ctx.F);

    const auto tstop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed = tstop - tstart;
    log::debug("SOAD projection into AO basis took {:.5f} s",
               time_elapsed.count());
  }
  m_have_initial_guess = true;
  occ::timing::stop(occ::timing::category::guess);
}

template <SCFMethod P>
void SCF<P>::set_point_charges(const PointChargeList &charges) {
  log::info("Including potential from {} point charges", charges.size());
  ctx.energy["nuclear.point_charge"] =
      m_procedure.nuclear_point_charge_interaction_energy(charges);
  ctx.energy["nuclear.total"] =
      ctx.energy["nuclear.point_charge"] + ctx.energy["nuclear.repulsion"];
  ctx.point_charges = charges;
}

template <SCFMethod P> void SCF<P>::update_scf_energy(bool incremental) {

  if (!incremental) {
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
    const auto pcloc = ctx.energy.find("nuclear.point_charge");
    if (pcloc != ctx.energy.end()) {
      ctx.energy["total"] += pcloc->second;
    }
    occ::timing::stop(occ::timing::category::la);
  }
  if (m_procedure.have_effective_core_potentials()) {
    ctx.energy["electronic.ecp"] = expectation(ctx.mo.kind, ctx.mo.D, ctx.Vecp);
  }
  if (ctx.point_charges.size() > 0) {
    ctx.energy["electronic.point_charge"] =
        2 * expectation(ctx.mo.kind, ctx.mo.D, ctx.Vpc);
  }
  m_procedure.update_scf_energy(ctx.energy, incremental);
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
    ctx.H = ctx.T + ctx.V + ctx.Vecp + ctx.Vpc;
    m_procedure.update_core_hamiltonian(ctx.mo, ctx.H);
    incremental = true;

    if (not incremental_Fbuild_started &&
        convergence_settings.start_incremental_fock(diis_error)) {
      incremental_Fbuild_started = true;
      reset_incremental_fock_formation = false;
      last_reset_iteration = iter - 1;
      next_reset_threshold = diis_error / 10;
      log::debug("starting incremental fock build");
    }
    if (true || reset_incremental_fock_formation ||
        not incremental_Fbuild_started) {
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

    // build a new Fock matrix
    std::swap(ctx.mo.D, D_diff);
    ctx.F += m_procedure.compute_fock(ctx.mo, ctx.K);
    std::swap(ctx.mo.D, D_diff);

    // compute HF energy with the non-extrapolated Fock matrix
    update_scf_energy(incremental);
    ediff_rel = std::abs((ctx.energy["electronic"] - ehf_last) /
                         ctx.energy["electronic"]);

    Mat F_diis = diis.update(ctx.S, ctx.mo.D, ctx.F);
    // double prev_error = diis_error;
    diis_error = diis.max_error();
    /*
    bool use_ediis = (diis_error > 1e-1) || (prev_error /
    diis.min_error() > 1.1);

    Mat F_ediis = ediis.update(D, F, energy["electronic"]);
    if(use_ediis) F_diis = F_ediis;
    else if(diis_error > 1e-4) {
        F_diis = (10 * diis_error) * F_ediis + (1 - 10 * diis_error)
    * F_diis;
    }
    */

    if (diis_error < next_reset_threshold || iter - last_reset_iteration >= 8)
      reset_incremental_fock_formation = true;

    ctx.mo.update(ctx.X, F_diis);
    D_diff = ctx.mo.D - D_last;

    const auto tstop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed = tstop - tstart;

    if (iter == 1) {
      log::info("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}", "#",
                "E (Hartrees)", "|dE|/E", "max|FDS-SDF|", "T (s)");
    }
    log::info("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}", iter,
              ctx.energy["total"], ediff_rel, diis_error, time_elapsed.count());
    std::cout << std::flush;
    total_time += time_elapsed.count();

  } while (!convergence_settings.energy_and_commutator_converged(ediff_rel,
                                                                 diis_error) &&
           (iter < maxiter));
  log::info("{} spinorbital SCF energy converged after {:.5f} seconds",
            scf_kind(), total_time);
  log::info("{}", ctx.energy.to_string());
  ctx.converged = true;
  return ctx.energy["total"];
}
} // namespace occ::qm
