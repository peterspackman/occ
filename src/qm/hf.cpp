#include <algorithm>
#include <occ/core/constants.h>
#include <occ/qm/fitting_basis.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/auto_aux_basis.h>
#include <occ/qm/cphf.h>
#include <occ/qm/hf.h>

namespace occ::qm {

void HartreeFock::set_density_fitting_basis(
    const std::string &density_fitting_basis, double auto_aux_threshold) {
  occ::gto::AOBasis dfbasis;

  // Check for "auto" option (case-insensitive)
  std::string name_lower = occ::util::to_lower_copy(density_fitting_basis);
  if (name_lower == "auto") {
    occ::log::info("Generating automatic auxiliary basis (threshold={:.1e})",
                   auto_aux_threshold);
    auto result = generate_auto_aux(m_engine.aobasis(), auto_aux_threshold);

    // Log statistics before moving
    size_t nshells = result.aux_basis.size();
    size_t naux = result.aux_basis.nbf();
    occ::log::info("Auto auxiliary basis: {} shells, {} functions generated in {:.2f}s",
                   nshells, naux, result.time_seconds);

    dfbasis = std::move(result.aux_basis);

    // Warn if auxiliary basis is very large (likely to cause conditioning issues)
    if (naux > 2000) {
      occ::log::warn("Auto auxiliary basis has {} functions - this may cause "
                     "ill-conditioning. Consider using a pre-defined auxiliary basis "
                     "(e.g., def2-universal-jkfit) or a looser threshold (--df-auto-threshold=1e-2).",
                     naux);
    }
  } else {
    dfbasis = occ::gto::AOBasis::load(atoms(), density_fitting_basis);
  }

  dfbasis.set_kind(m_engine.aobasis().kind());
  m_df_basis_name = density_fitting_basis;
  m_df_auto_threshold = auto_aux_threshold;
  m_df_aux_shells = dfbasis.shells();
  m_df_engine = std::make_unique<IntegralEngineDF>(
      atoms(), m_engine.aobasis().shells(), m_df_aux_shells);
  // A long-range twin built for a previous basis would now be stale.
  m_lr_df_engine.reset();
  m_lr_engine.reset();
}

void HartreeFock::set_density_fitting_policy(IntegralEngineDF::Policy policy) {
  if (m_df_engine) {
    m_df_engine->set_integral_policy(policy);
  }
}

void HartreeFock::set_density_fitting_memory_limit(size_t bytes) {
  if (m_df_engine) {
    m_df_engine->set_integral_store_memory_limit(bytes);
  }
}

void HartreeFock::set_coulomb_method(CoulombMethod method) {
  if (m_df_engine) {
    m_df_engine->set_coulomb_method(method);
    if (method == CoulombMethod::SplitRIJ) {
      occ::log::info("Using Split-RI-J for Coulomb matrix (Neese 2003)");
    }
  }
}

namespace {
std::unique_ptr<occ::qm::cosx::SemiNumericalExchange>
make_cosx_engine(const occ::gto::AOBasis &basis,
                 occ::numint::COSXGridLevel level,
                 const std::optional<occ::qm::cosx::Settings> &settings) {
  auto engine = std::make_unique<occ::qm::cosx::SemiNumericalExchange>(
      basis, occ::numint::GridSettings::for_cosx(level));
  engine->set_use_esp(true);               // Use ESP-based COSX
  engine->set_use_spatial_hierarchy(true); // Use TBB parallel path
  if (settings)
    engine->set_settings(*settings);
  return engine;
}
} // namespace

void HartreeFock::set_cosx_exchange(occ::numint::COSXGridLevel level) {
  m_cosx_schedule.reset();
  m_cosx_corrections.clear();
  m_cosx_last_approx = false;
  for (auto &e : m_cosx_stash)
    e.reset();
  m_cosx_level = level;
  m_cosx_engine =
      make_cosx_engine(m_engine.aobasis(), level, effective_cosx_settings());
}

void HartreeFock::set_cosx_grid_schedule(const COSXGridSchedule &schedule) {
  set_cosx_exchange(schedule.iterate);
  m_cosx_schedule = schedule;
  m_cosx_stage = 0;
}

void HartreeFock::set_cosx_settings(const occ::qm::cosx::Settings &settings) {
  m_cosx_settings = settings;
  apply_cosx_settings();
}

occ::qm::cosx::Settings HartreeFock::effective_cosx_settings() const {
  auto settings = m_cosx_settings.value_or(occ::qm::cosx::Settings{});
  settings.f_threshold /= m_cosx_exchange_scale;
  return settings;
}

void HartreeFock::apply_cosx_settings() {
  const auto settings = effective_cosx_settings();
  if (m_cosx_engine)
    m_cosx_engine->set_settings(settings);
  for (auto &e : m_cosx_stash)
    if (e)
      e->set_settings(settings);
}

void HartreeFock::set_cosx_exchange_scale(double scale) {
  scale = std::clamp(std::abs(scale), 1e-3, 1.0);
  if (scale == m_cosx_exchange_scale)
    return;
  m_cosx_exchange_scale = scale;
  apply_cosx_settings();
}

void HartreeFock::switch_cosx_grid(occ::numint::COSXGridLevel level) {
  if (!m_cosx_engine || level == m_cosx_level)
    return;
  m_cosx_stash[static_cast<int>(m_cosx_level)] = std::move(m_cosx_engine);
  auto &slot = m_cosx_stash[static_cast<int>(level)];
  m_cosx_engine = slot ? std::move(slot)
                       : make_cosx_engine(m_engine.aobasis(), level,
                                          effective_cosx_settings());
  m_cosx_level = level;
}

bool HartreeFock::advance_scf_grid(double diis_error, bool converged) {
  if (!m_cosx_schedule || !m_cosx_engine || m_cosx_stage != 0)
    return false;
  if (!converged && diis_error >= m_cosx_schedule->refine_threshold)
    return false;
  occ::log::info("COSX: switching from {} to {}",
                 occ::numint::cosx_grid_level_to_string(m_cosx_level),
                 occ::numint::cosx_grid_level_to_string(
                     m_cosx_schedule->refine));
  switch_cosx_grid(m_cosx_schedule->refine);
  m_cosx_stage = 1;
  m_cosx_corrections.clear();
  return true;
}

void HartreeFock::set_scf_fock_incremental(bool incremental) {
  m_cosx_incremental = incremental;
  // difference-density builds screen on the plain charge bound (see
  // SemiNumericalExchange::set_far_field_screening)
  if (m_cosx_engine)
    m_cosx_engine->set_far_field_screening(!incremental);
  for (auto &e : m_cosx_stash)
    if (e)
      e->set_far_field_screening(!incremental);
}

// The SCF's exchange on the current grid. On the refine stage of a schedule
// with a frozen correction, an exact build (first use, or after a refresh
// request) also builds the iterate grid's K at the same density and keeps the
// difference; other builds are the iterate grid's K plus that difference.
template <typename Build>
Mat HartreeFock::cosx_scf_exchange(const std::array<double, 4> &key,
                                   Build &&build) const {
  const bool frozen = m_cosx_schedule && m_cosx_stage == 1 &&
                      m_cosx_schedule->frozen_correction &&
                      m_cosx_schedule->iterate != m_cosx_level;
  const auto *coarse =
      frozen ? m_cosx_stash[static_cast<int>(m_cosx_schedule->iterate)].get()
             : nullptr;
  if (!coarse) {
    m_cosx_last_approx = false;
    return build(*m_cosx_engine);
  }
  auto it = std::find_if(m_cosx_corrections.begin(), m_cosx_corrections.end(),
                         [&](const auto &c) { return c.key == key; });
  if (it != m_cosx_corrections.end() && it->generation == m_cosx_generation) {
    m_cosx_last_approx = true;
    Mat K = build(*coarse);
    if (!m_cosx_incremental)
      K += it->dK;
    return K;
  }
  if (m_cosx_incremental)
    throw std::logic_error("COSX frozen grid correction refreshed on a "
                           "difference-density build");
  Mat K = build(*m_cosx_engine);
  Mat dK = K - build(*coarse);
  if (it == m_cosx_corrections.end())
    m_cosx_corrections.push_back({key, m_cosx_generation, std::move(dK)});
  else
    *it = {key, m_cosx_generation, std::move(dK)};
  m_cosx_last_approx = false;
  return K;
}

namespace {
double cosx_exchange_energy(
    occ::qm::cosx::SemiNumericalExchange &engine, const MolecularOrbitals &mo,
    const HartreeFock::ExchangeOperators &operators) {
  double e = 0.0;
  for (const auto &[c, omega] : operators) {
    if (c == 0.0)
      continue;
    const Mat K = engine.compute_K(mo, std::numeric_limits<double>::epsilon(),
                                   Mat(), omega);
    e -= c * expectation(mo.kind, mo.D, K);
  }
  return e;
}
} // namespace

double HartreeFock::finalize_cosx_grid(const MolecularOrbitals &mo,
                                       const ExchangeOperators &operators) {
  if (!m_cosx_schedule || !m_cosx_engine || m_cosx_stage != 1)
    return 0.0;
  if (m_cosx_schedule->final == m_cosx_level) {
    m_cosx_stage = 2; // converged on the final grid already
    return 0.0;
  }
  const occ::numint::COSXGridLevel refine = m_cosx_level;
  switch_cosx_grid(m_cosx_schedule->final);
  m_cosx_stage = 2;
  const double dE = cosx_energy_shift_from(refine, mo, operators);
  occ::log::info("COSX: final exchange energy on {}: correction {:.10f}",
                 occ::numint::cosx_grid_level_to_string(m_cosx_level), dE);
  return dE;
}

double HartreeFock::cosx_energy_shift_from(occ::numint::COSXGridLevel other,
                                           const MolecularOrbitals &mo,
                                           const ExchangeOperators &operators) {
  if (!m_cosx_engine || other == m_cosx_level)
    return 0.0;
  const occ::numint::COSXGridLevel current = m_cosx_level;
  const double e_current = cosx_exchange_energy(*m_cosx_engine, mo, operators);
  switch_cosx_grid(other);
  const double e_other = cosx_exchange_energy(*m_cosx_engine, mo, operators);
  switch_cosx_grid(current);
  return e_current - e_other;
}

double HartreeFock::final_energy_correction(const MolecularOrbitals &mo) {
  return finalize_cosx_grid(mo, {{1.0, 0.0}});
}

HartreeFock::HartreeFock(const AOBasis &basis)
    : SCFMethodBase(basis.atoms()), m_engine(basis) {

  update_electron_count();

  std::vector<int> frozen(basis.atoms().size(), 0);
  int num_frozen = basis.total_ecp_electrons();
  if (num_frozen > 0) {
    frozen = basis.ecp_electrons();
  }
  set_frozen_electrons(frozen);
  m_num_frozen = num_frozen;
}

HartreeFock HartreeFock::with_new_basis(const AOBasis &new_basis) const {
  HartreeFock new_hf(new_basis);
  new_hf.set_precision(this->integral_precision());
  copy_two_electron_setup_to(new_hf);
  return new_hf;
}

void HartreeFock::copy_two_electron_setup_to(HartreeFock &other) const {
  if (m_df_engine) {
    other.set_density_fitting_basis(m_df_basis_name, m_df_auto_threshold);
    other.m_df_engine->set_integral_policy(m_df_engine->integral_policy());
    other.set_coulomb_method(m_df_engine->coulomb_method());
  }
  // Settings first: the engines are built with them.
  if (m_cosx_settings)
    other.set_cosx_settings(*m_cosx_settings);
  other.m_cosx_exchange_scale = m_cosx_exchange_scale;
  if (m_cosx_engine) {
    if (m_cosx_schedule)
      other.set_cosx_grid_schedule(*m_cosx_schedule);
    else
      other.set_cosx_exchange(m_cosx_level);
  }
}

double HartreeFock::nuclear_point_charge_interaction_energy(
    const PointChargeList &point_charges) const {
  double etot = 0.0;

  int i = 0;
  for (const auto &atom : atoms()) {
    const auto pos_a = atom.position();
    double Z = atom.atomic_number - m_frozen_electrons[i];

    for (const auto &pc : point_charges) {
      const auto &pos_pc = pc.position();
      etot += Z * pc.charge() / (pos_pc - pos_a).norm();
    }
    i++;
  }
  return etot;
}

double HartreeFock::wolf_point_charge_interaction_energy(
    const PointChargeList &point_charges,
    const std::vector<double> &partial_charges, double alpha_angs,
    double rc_angs) const {
  if (point_charges.size() == 0)
    return 0.0;
  const double alpha = alpha_angs / occ::units::ANGSTROM_TO_BOHR;
  const double rc = rc_angs * occ::units::ANGSTROM_TO_BOHR; // cutoff radius

  // Pre-calculate Wolf self-terms
  const double t2 = std::erfc(alpha * rc) / rc;

  const auto &atom_list = atoms();
  const double t3 = (t2 + 2 * alpha / occ::constants::sqrt_pi<double>);

  double total = 0.0;
  double total_self = 0.0;
  double total_intra = 0.0;
  // Nuclear-point charge interactions
  int i = 0;
  for (const auto &atom : atom_list) {
    const auto pos_a = atom.position();
    double Z = atom.atomic_number - m_frozen_electrons[i];
    double q = partial_charges[i];
    double total_i = 0.0;
    for (const auto &pc : point_charges) {
      const auto &pos_pc = pc.position();
      const double rij = (pos_pc - pos_a).norm();
      if (rij > rc)
        continue;
      const double qj = pc.charge();
      total_i += qj * (std::erfc(alpha * rij) / rij - t2);
    }
    for (int j = 0; j < atom_list.size(); j++) {
      if (i == j)
        continue;
      const auto &pos_j = atom_list[j].position();
      const double rij = (pos_j - pos_a).norm();
      if (rij > rc)
        continue;
      const double qj = partial_charges[j];
      total_intra += Z * qj * std::erf(alpha * rij) / rij;
    }
    i++;
    total_self -= Z * q * t3;
    total += Z * total_i;
  }
  total -= total_intra;
  total += total_self;
  return total;
}

Mat HartreeFock::compute_fock(const MolecularOrbitals &mo,
                              const Mat &Schwarz) const {
  if (m_df_engine && m_cosx_engine) {
    // DF + COSX: use DF for J, COSX for K
    // Both have same scale convention, so just J - K
    auto jk = compute_JK(mo, Schwarz);
    return jk.J - jk.K;
  } else if (m_df_engine) {
    return m_df_engine->fock_operator(mo);
  } else if (m_cosx_engine) {
    // COSX-only: J from regular engine, K from COSX
    auto jk = compute_JK(mo, Schwarz);
    return jk.J - jk.K;
  } else {
    return m_engine.fock_operator(mo.kind, mo, Schwarz);
  }
}

MatTriple HartreeFock::compute_fock_gradient(const MolecularOrbitals &mo,
                                             const Mat &Schwarz) const {
  return m_engine.fock_operator_grad(mo.kind, mo, Schwarz);
}

Mat HartreeFock::compute_effective_core_potential_matrix() const {
  return m_engine.effective_core_potential();
}

Mat HartreeFock::compute_fock_mixed_basis(const MolecularOrbitals &mo_minbs,
                                          const gto::AOBasis &bs,
                                          bool is_shell_diagonal) {
  if (mo_minbs.kind == SpinorbitalKind::Restricted) {
    return m_engine.fock_operator_mixed_basis(mo_minbs.D, bs,
                                              is_shell_diagonal);
  } else if (mo_minbs.kind == SpinorbitalKind::Unrestricted) {
    const auto [rows, cols] =
        occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(
            m_engine.aobasis().nbf());
    Mat F = Mat::Zero(rows, cols);
    qm::block::a(F) =
        m_engine.fock_operator_mixed_basis(mo_minbs.D, bs, is_shell_diagonal);
    qm::block::b(F) = qm::block::a(F);
    return F;
  } else { // kind == SpinorbitalKind::General
    const auto [rows, cols] =
        occ::qm::matrix_dimensions<SpinorbitalKind::General>(
            m_engine.aobasis().nbf());
    Mat F = Mat::Zero(rows, cols);
    qm::block::aa(F) =
        m_engine.fock_operator_mixed_basis(mo_minbs.D, bs, is_shell_diagonal);
    qm::block::bb(F) = qm::block::aa(F);
    return F;
  }
}

JKPair HartreeFock::compute_JK(const MolecularOrbitals &mo,
                               const Mat &Schwarz) const {
  if (m_df_engine && m_cosx_engine) {
    // DF + COSX: use DF for J, COSX for K
    JKPair result;
    result.J = m_df_engine->coulomb(mo);
    result.K = cosx_scf_exchange({0, 1, 0, 0}, [&](const auto &e) {
      return e.compute_K(mo);
    });
    return result;
  } else if (m_df_engine) {
    return m_df_engine->coulomb_and_exchange(mo);
  } else if (m_cosx_engine) {
    // COSX: compute J normally, K via seminumerical integration
    JKPair result;
    result.J = m_engine.coulomb(mo.kind, mo, Schwarz);
    result.K = cosx_scf_exchange({0, 1, 0, 0}, [&](const auto &e) {
      return e.compute_K(mo);
    });
    return result;
  } else {
    return m_engine.coulomb_and_exchange(mo.kind, mo, Schwarz);
  }
}

Mat HartreeFock::compute_K(const MolecularOrbitals &mo,
                           const Mat &Schwarz) const {
  // Exchange-only build, routed per engine. Used by the range-separated path
  // to build the long-range K without also computing an unused J matrix.
  if (m_cosx_engine) {
    // COSX provides K directly (J is handled by the regular/DF engine).
    return cosx_scf_exchange({0, 1, 0, 0}, [&](const auto &e) {
      return e.compute_K(mo);
    });
  } else if (m_df_engine) {
    return m_df_engine->exchange(mo);
  } else {
    // The exact engine has no exchange-only entry point; take K from the
    // fused build (J is computed but discarded - the exact path is not the
    // target for large range-separated hybrids).
    return m_engine.coulomb_and_exchange(mo.kind, mo, Schwarz).K;
  }
}

JKPair HartreeFock::coulomb_and_range_separated_exchange(
    const MolecularOrbitals &mo, double omega, double alpha, double beta,
    const Mat &Schwarz) {
  const double w_full = alpha + beta; // weight on K[1/r]
  const double w_lr = -beta;          // weight on K[erf(omega*r)/r]

  JKPair result;

  // Fastest path: COSX evaluates both operators in a single grid sweep
  // (restricted and unrestricted; the general spinorbital case falls through).
  if (m_cosx_engine &&
      (mo.kind == SpinorbitalKind::Restricted ||
       mo.kind == SpinorbitalKind::Unrestricted) &&
      w_full != 0.0 && w_lr != 0.0) {
    result.J = compute_J(mo, Schwarz);
    result.K = cosx_scf_exchange({1, alpha, beta, omega}, [&](const auto &e) {
      return e.compute_K_range_separated(mo, alpha, beta, omega);
    });
    return result;
  }

  // General path: the Coulomb term always uses the full 1/r operator; build
  // only the exchange operators that carry a nonzero weight.
  if (w_full != 0.0) {
    JKPair full = compute_JK(mo, Schwarz); // J and K[1/r] together
    result.J = full.J;
    result.K = w_full * full.K;
  } else {
    result.J = compute_J(mo, Schwarz);
    result.K = Mat::Zero(result.J.rows(), result.J.cols());
  }

  if (w_lr != 0.0) {
    const Mat K_lr = compute_K_long_range(mo, omega, Schwarz);
    result.K.noalias() += w_lr * K_lr;
  }

  return result;
}

void HartreeFock::ensure_long_range_engines(double omega) const {
  if (m_lr_omega == omega && (m_lr_engine || m_lr_df_engine))
    return;

  m_lr_engine = std::make_unique<IntegralEngine>(m_engine.aobasis());
  m_lr_engine->set_precision(m_engine.precision());
  m_lr_engine->set_range_separated_omega(omega);

  if (m_df_engine) {
    m_lr_df_engine = std::make_unique<IntegralEngineDF>(
        atoms(), m_engine.aobasis().shells(), m_df_aux_shells);
    m_lr_df_engine->set_precision(m_df_engine->precision());
    // The store only ever serves the full Coulomb operator, so this engine
    // recomputes regardless; saying so is clearer than relying on the default.
    m_lr_df_engine->set_integral_policy(IntegralEngineDF::Policy::Direct);
    // Builds the attenuated fitting metric (P|erf(omega r)/r|Q) once, here,
    // instead of on every Fock build.
    m_lr_df_engine->set_range_separated_omega(omega);
  }

  m_lr_omega = omega;
}

Mat HartreeFock::compute_K_long_range(const MolecularOrbitals &mo, double omega,
                                      const Mat &Schwarz) const {
  if (m_cosx_engine)
    return cosx_scf_exchange({2, 1, 0, omega}, [&](const auto &e) {
      return e.compute_K(mo, std::numeric_limits<double>::epsilon(), Schwarz,
                         omega);
    });
  ensure_long_range_engines(omega);
  if (m_lr_df_engine)
    return m_lr_df_engine->exchange(mo);
  return m_lr_engine->coulomb_and_exchange(mo.kind, mo, Schwarz).K;
}

MatTriple
HartreeFock::compute_K_gradient_long_range(const MolecularOrbitals &mo,
                                           double omega,
                                           const Mat &Schwarz) const {
  // The gradient path is always the conventional four-centre engine, so the
  // long-range twin is exactly what it needs.
  ensure_long_range_engines(omega);
  return m_lr_engine->coulomb_exchange_grad(mo.kind, mo, Schwarz).K;
}

std::vector<JKPair>
HartreeFock::compute_JK_list(const std::vector<MolecularOrbitals> &mos,
                             const Mat &Schwarz) const {
  return m_engine.coulomb_and_exchange_list(mos[0].kind, mos, Schwarz);
}

std::vector<Mat>
HartreeFock::compute_J_list(const std::vector<MolecularOrbitals> &mos,
                            const Mat &Schwarz) const {
  return m_engine.coulomb_list(mos[0].kind, mos, Schwarz);
}

Mat HartreeFock::compute_J(const MolecularOrbitals &mo,
                           const Mat &Schwarz) const {
  if (m_df_engine) {
    return m_df_engine->coulomb(mo);
  } else {
    return m_engine.coulomb(mo.kind, mo, Schwarz);
  }
}

std::vector<Mat>
HartreeFock::compute_response_fock(const MolecularOrbitals &ref,
                                   const std::vector<Mat> &D1,
                                   const Mat &Schwarz) const {
  const auto kind = ref.kind;
  if (kind == SpinorbitalKind::Restricted)
    return compute_response_jk(D1, 1.0, 1.0, 0.0, Schwarz, &ref);
  std::vector<MolecularOrbitals> mos(D1.size());
  for (size_t k = 0; k < D1.size(); k++) {
    mos[k].kind = kind;
    mos[k].D = D1[k];
  }
  // The Fock build is F = H + J[D] - K[D] in these engines' own scaling, and
  // linear in D, so the response is the same build on D1.
  auto jk = m_engine.coulomb_and_exchange_list(kind, mos, Schwarz);
  std::vector<Mat> result;
  result.reserve(jk.size());
  for (auto &p : jk)
    result.push_back(p.J - p.K);
  return result;
}

IntegralEngineDF *HartreeFock::exchange_df_engine(double omega) const {
  if (omega == 0.0)
    return m_df_engine.get();
  ensure_long_range_engines(omega);
  return m_lr_df_engine.get();
}

namespace {
// Coefficient of the full-range exchange (the part a four-centre pass can
// share with the Coulomb).
double full_range_exchange(const HartreeFock::ExchangeOperators &K) {
  double c0 = 0.0;
  for (const auto &[c, omega] : K)
    if (omega == 0.0)
      c0 += c;
  return c0;
}
bool has_exchange(const HartreeFock::ExchangeOperators &K) {
  for (const auto &op : K)
    if (op.first != 0.0)
      return true;
  return false;
}
} // namespace

const occ::qm::cosx::SemiNumericalExchange::SecondDerivatives &
HartreeFock::cosx_second_derivatives(const MolecularOrbitals &mo, double c,
                                     double omega) const {
  auto it = std::find_if(
      m_cosx_second_derivatives.begin(), m_cosx_second_derivatives.end(),
      [&](const auto &e) {
        return e.engine == m_cosx_engine.get() && e.c == c &&
               e.omega == omega && e.D.rows() == mo.D.rows() && e.D == mo.D;
      });
  if (it != m_cosx_second_derivatives.end())
    return it->terms;
  // Screened as the gradient is, relative to the operator's coefficient.
  auto terms = m_cosx_engine->compute_K_second_derivatives(
      mo, omega, 1e-10 / std::clamp(std::abs(c), 1e-3, 1.0));
  std::erase_if(m_cosx_second_derivatives, [&](const auto &e) {
    return e.c == c && e.omega == omega;
  });
  m_cosx_second_derivatives.push_back(
      {m_cosx_engine.get(), c, omega, mo.D, std::move(terms)});
  return m_cosx_second_derivatives.back().terms;
}

Mat HartreeFock::compute_jk_hessian(const MolecularOrbitals &mo, double jf,
                                    const ExchangeOperators &K,
                                    const Mat &Schwarz) const {
  if (m_cosx_engine && has_exchange(K)) {
    Mat H = compute_jk_hessian(mo, jf, {}, Schwarz);
    for (const auto &[c, omega] : K)
      if (c != 0.0)
        H += c * cosx_second_derivatives(mo, c, omega).hessian;
    return H;
  }
  const Eigen::Index n3 = 3 * atoms().size();
  Mat H = Mat::Zero(n3, n3);
  if (m_df_engine) {
    // The engines differentiate +tr(D K), which enters the energy as -c.
    if (jf != 0.0)
      H += jf * m_df_engine->coulomb_hessian(mo);
    for (const auto &[c, omega] : K)
      if (c != 0.0)
        H -= c * exchange_df_engine(omega)->exchange_hessian(mo);
    return H;
  }
  const double c0 = full_range_exchange(K);
  if (jf != 0.0 || c0 != 0.0)
    H += m_engine.jk_hessian(mo, jf, c0, Schwarz);
  for (const auto &[c, omega] : K) {
    if (c == 0.0 || omega == 0.0)
      continue;
    ensure_long_range_engines(omega);
    H += m_lr_engine->jk_hessian(mo, 0.0, c, Schwarz);
  }
  return H;
}

std::vector<Mat> HartreeFock::compute_jk_derivative_fock(
    const MolecularOrbitals &mo, double jf, const ExchangeOperators &K,
    const Mat &Schwarz) const {
  if (m_cosx_engine && has_exchange(K))
    throw std::runtime_error("COSX Fock derivatives are only formed "
                             "contracted with the occupied orbitals "
                             "(compute_jk_derivative_fock_occ)");
  const size_t n3 = 3 * atoms().size();
  const auto [rows, cols] = matrix_dimensions(mo.kind, nbf());
  std::vector<Mat> F(n3, Mat::Zero(rows, cols));
  // G is per spin like F, or (Coulomb) one matrix for both spins.
  auto add = [&](const std::vector<Mat> &G, double scale) {
    for (size_t X = 0; X < n3; X++)
      for (Eigen::Index r = 0; r < rows; r += G[X].rows())
        F[X].middleRows(r, G[X].rows()) += scale * G[X];
  };
  if (m_df_engine) {
    if (jf != 0.0)
      add(m_df_engine->coulomb_derivative_fock(mo), jf);
    for (const auto &[c, omega] : K)
      if (c != 0.0)
        add(exchange_df_engine(omega)->exchange_derivative_fock(mo), -c);
    return F;
  }
  const double c0 = full_range_exchange(K);
  if (jf != 0.0 || c0 != 0.0)
    add(m_engine.jk_derivative_fock(mo, jf, c0, Schwarz), 1.0);
  for (const auto &[c, omega] : K) {
    if (c == 0.0 || omega == 0.0)
      continue;
    ensure_long_range_engines(omega);
    add(m_lr_engine->jk_derivative_fock(mo, 0.0, c, Schwarz), 1.0);
  }
  return F;
}

std::vector<Mat> HartreeFock::compute_jk_derivative_fock_occ(
    const MolecularOrbitals &mo, double jf, const ExchangeOperators &K,
    const Mat &Schwarz) const {
  if (m_cosx_engine && has_exchange(K)) {
    auto F = compute_jk_derivative_fock_occ(mo, jf, {}, Schwarz);
    for (const auto &[c, omega] : K) {
      if (c == 0.0)
        continue;
      const auto &Kx = cosx_second_derivatives(mo, c, omega).fock;
      for (size_t X = 0; X < F.size(); X++)
        F[X] -= c * Kx[X];
    }
    return F;
  }
  // RI-K contracts with C_o inside (no nbf x nbf products); everything
  // else is formed in full and contracted.
  if (m_df_engine) {
    auto F = compute_jk_derivative_fock(mo, jf, {}, Schwarz);
    for (auto &f : F)
      f = cphf::contract_occupied(f, mo);
    for (const auto &[c, omega] : K) {
      if (c == 0.0)
        continue;
      const auto Kx = exchange_df_engine(omega)->exchange_derivative_fock_occ(mo);
      for (size_t X = 0; X < F.size(); X++)
        F[X] -= c * Kx[X];
    }
    return F;
  }
  auto F = compute_jk_derivative_fock(mo, jf, K, Schwarz);
  for (auto &f : F)
    f = cphf::contract_occupied(f, mo);
  return F;
}

std::vector<Mat> HartreeFock::compute_exchange_nonvariational_fock_occ(
    const MolecularOrbitals &mo, const ExchangeOperators &K) const {
  if (!m_cosx_engine || !has_exchange(K))
    return {};
  std::vector<Mat> F(3 * atoms().size(), Mat::Zero(nbf(), mo.n_alpha));
  for (const auto &[c, omega] : K) {
    if (c == 0.0)
      continue;
    const auto &d = cosx_second_derivatives(mo, c, omega);
    for (size_t X = 0; X < F.size(); X++)
      F[X] -= c * (d.energy_fock[X] - d.fock[X]);
  }
  return F;
}

std::vector<Mat> HartreeFock::compute_response_jk(
    const std::vector<Mat> &D1, double jf, double kf, double omega,
    const Mat &Schwarz, const MolecularOrbitals *ref) const {
  if (omega != 0.0 && jf != 0.0)
    throw std::runtime_error("compute_response_jk: attenuated operator is "
                             "exchange only");
  if (jf == 0.0 && kf == 0.0)
    return std::vector<Mat>(D1.size(), Mat::Zero(nbf(), nbf()));
  // Four-centre for both: one fused pass.
  if (!m_df_engine && !m_cosx_engine) {
    if (omega == 0.0)
      return m_engine.coulomb_exchange_response(D1, jf, kf, Schwarz);
    ensure_long_range_engines(omega);
    return m_lr_engine->coulomb_exchange_response(D1, 0.0, kf, Schwarz);
  }
  // Otherwise each part follows the operator the SCF used: RI-J or
  // four-centre Coulomb; COSX (one shared grid sweep), RI-K or four-centre
  // exchange.
  std::vector<Mat> G;
  if (jf != 0.0) {
    if (m_df_engine) {
      G = m_df_engine->coulomb_response(D1);
      for (auto &g : G)
        g *= jf;
    } else {
      G = m_engine.coulomb_exchange_response(D1, jf, 0.0, Schwarz);
    }
  } else {
    G.assign(D1.size(), Mat::Zero(nbf(), nbf()));
  }
  if (kf == 0.0)
    return G;
  if (m_cosx_engine) {
    const auto K = m_cosx_engine->compute_K_response(D1, omega);
    for (size_t k = 0; k < G.size(); k++)
      G[k] -= kf * K[k];
    return G;
  }
  if (!ref || ref->kind != SpinorbitalKind::Restricted)
    throw std::runtime_error("RI-K response: needs the restricted reference "
                             "orbitals");
  const Mat &C = ref->Cocc;
  const auto Cx = factor_response_densities(D1, C);
  IntegralEngineDF *df = m_df_engine.get();
  if (omega != 0.0) {
    ensure_long_range_engines(omega);
    df = m_lr_df_engine.get();
  }
  const auto K = df->exchange_response(C, Cx);
  for (size_t k = 0; k < G.size(); k++)
    G[k] -= kf * K[k];
  return G;
}

// RI-K needs each density factorised as Cx C^T + C Cx^T, which holds for any
// density without a virtual-virtual block (all CPHF densities):
// Cx = D1 S C - C (C^T S D1 S C) / 2.
std::vector<Mat>
HartreeFock::factor_response_densities(const std::vector<Mat> &D1,
                                       const Mat &C) const {
  const Mat S = compute_overlap_matrix();
  const Mat SC = S * C;
  std::vector<Mat> Cx(D1.size());
  for (size_t k = 0; k < D1.size(); k++) {
    const Mat DSC = D1[k] * SC;
    Cx[k] = DSC - 0.5 * C * (SC.transpose() * DSC);
    const double resid =
        (Cx[k] * C.transpose() + C * Cx[k].transpose() - D1[k])
            .cwiseAbs()
            .maxCoeff();
    if (resid > 1e-8 * std::max(1.0, D1[k].cwiseAbs().maxCoeff()))
      throw std::runtime_error(
          "RI-K response: density has a virtual-virtual block");
  }
  return Cx;
}

IntegralEngineDF *HartreeFock::approximate_coulomb_engine() const {
  if (m_df_engine)
    return m_df_engine.get();
  if (!m_approx_df_tried) {
    m_approx_df_tried = true;
    try {
      const auto &basis = m_engine.aobasis();
      auto aux = occ::gto::AOBasis::load(
          basis.atoms(),
          resolve_fitting_basis(basis.name(), occ::qm::FittingKind::J));
      aux.set_kind(basis.kind());
      m_approx_df = std::make_unique<IntegralEngineDF>(
          basis.atoms(), basis.shells(), aux.shells());
    } catch (const std::exception &e) {
      occ::log::debug("Approximate response: four-centre Coulomb ({})",
                      e.what());
    }
  }
  return m_approx_df.get();
}

std::vector<Mat> HartreeFock::response_occ(const MolecularOrbitals &mo,
                                           const std::vector<Mat> &Cx,
                                           double jf,
                                           const ExchangeOperators &K,
                                           const Mat &Schwarz,
                                           ResponseMode mode) const {
  const bool approx = mode == ResponseMode::Approximate;
  const auto channels = cphf::spin_channels(mo);
  const size_t nspin = channels.size();
  std::vector<Eigen::Index> col(nspin, 0);
  for (size_t s = 1; s < nspin; s++)
    col[s] = col[s - 1] + channels[s - 1].Co.cols();
  const size_t n = Cx.size();
  // First-order densities D1^s = Cx^s Co^s^T + Co^s Cx^s^T. Unrestricted,
  // the batch holds Dt = (D1^a + D1^b)/2 followed by Ds = (D1^a - D1^b)/2:
  // G^a,b = (jf 2J - K)[Dt] -/+ K[Ds], so the Coulomb term only sees the
  // first n densities, and a closed shell (Ds = 0) is the restricted case.
  std::vector<Mat> D1(nspin * n);
  for (size_t k = 0; k < n; k++) {
    Mat Dspin[2];
    for (size_t s = 0; s < nspin; s++) {
      const auto &Co = channels[s].Co;
      const Mat T = Cx[k].middleCols(col[s], Co.cols()) * Co.transpose();
      Dspin[s] = T + T.transpose();
    }
    if (nspin == 1) {
      D1[k] = std::move(Dspin[0]);
    } else {
      D1[k] = 0.5 * (Dspin[0] + Dspin[1]);
      D1[n + k] = 0.5 * (Dspin[0] - Dspin[1]);
    }
  }
  const std::vector<Mat> Dt(D1.begin(), D1.begin() + n);
  const size_t exchange_only = D1.size() - n;
  // Full AO contributions (contracted with Co at the end) and those already
  // contracted (RI-K never forms the full matrix).
  std::vector<Mat> Gao(D1.size(), Mat::Zero(nbf(), nbf()));
  std::vector<Mat> Gocc(n, Mat::Zero(nbf(), Cx.empty() ? 0 : Cx[0].cols()));

  IntegralEngineDF *jdf =
      approx ? approximate_coulomb_engine() : m_df_engine.get();
  bool coulomb_pending = jf != 0.0;
  if (coulomb_pending && jdf) {
    const auto J = jdf->coulomb_response(Dt);
    for (size_t k = 0; k < n; k++)
      Gao[k] += jf * J[k];
    coulomb_pending = false;
  }
  for (const auto &[c, omega] : K) {
    if (c == 0.0)
      continue;
    if (!approx && m_cosx_engine) {
      const auto Kc = m_cosx_engine->compute_K_response(D1, omega);
      for (size_t k = 0; k < D1.size(); k++)
        Gao[k] -= c * Kc[k];
    } else if (!approx && m_df_engine) {
      for (size_t s = 0; s < nspin; s++) {
        const auto &Co = channels[s].Co;
        std::vector<Mat> Cxs(n);
        for (size_t k = 0; k < n; k++)
          Cxs[k] = Cx[k].middleCols(col[s], Co.cols());
        const auto Kc =
            exchange_df_engine(omega)->exchange_response_occ(Co, Cxs);
        for (size_t k = 0; k < n; k++)
          Gocc[k].middleCols(col[s], Co.cols()) -= c * Kc[k];
      }
    } else if (omega == 0.0) {
      // four-centre, sharing the pass with a four-centre Coulomb
      const auto G = m_engine.coulomb_exchange_response(
          D1, coulomb_pending ? jf : 0.0, c, Schwarz, exchange_only);
      coulomb_pending = false;
      for (size_t k = 0; k < D1.size(); k++)
        Gao[k] += G[k];
    } else {
      ensure_long_range_engines(omega);
      const auto G = m_lr_engine->coulomb_exchange_response(D1, 0.0, c, Schwarz);
      for (size_t k = 0; k < D1.size(); k++)
        Gao[k] += G[k];
    }
  }
  if (coulomb_pending) {
    const auto G = m_engine.coulomb_exchange_response(Dt, jf, 0.0, Schwarz);
    for (size_t k = 0; k < n; k++)
      Gao[k] += G[k];
  }
  std::vector<Mat> result(n);
  for (size_t k = 0; k < n; k++) {
    result[k] = std::move(Gocc[k]);
    for (size_t s = 0; s < nspin; s++) {
      const auto &Co = channels[s].Co;
      const Mat G = nspin == 1  ? Gao[k]
                    : s == 0    ? Mat(Gao[k] + Gao[n + k])
                                : Mat(Gao[k] - Gao[n + k]);
      result[k].middleCols(col[s], Co.cols()) += G * Co;
    }
  }
  return result;
}

Mat3N HartreeFock::compute_jk_gradient(const MolecularOrbitals &mo, double jf,
                                       double kf, const Mat &Schwarz) const {
  return m_engine.jk_gradient(mo, jf, kf, Schwarz);
}

Mat3N HartreeFock::compute_exchange_gradient(const MolecularOrbitals &mo,
                                             double kf, double omega,
                                             const Mat &Schwarz) const {
  // The gradient takes kf times this term, so screening it at 1e-10 / |kf|
  // keeps the absolute screening error of the gradient that of HF, far below
  // the COSX quadrature error.
  if (m_cosx_engine)
    return kf * m_cosx_engine->compute_K_gradient(
                    mo, omega, 1e-10 / std::clamp(std::abs(kf), 1e-3, 1.0));
  // RI-K, for the operator the SCF fitted (the attenuated twin for omega);
  // exchange_gradient differentiates +tr(D K).
  if (m_df_engine) {
    if (omega == 0.0)
      return -kf * m_df_engine->exchange_gradient(mo);
    ensure_long_range_engines(omega);
    return -kf * m_lr_df_engine->exchange_gradient(mo);
  }
  if (omega == 0.0)
    return compute_jk_gradient(mo, 0.0, kf, Schwarz);
  return compute_k_gradient_long_range(mo, omega, kf, Schwarz);
}

Mat3N HartreeFock::compute_k_gradient_long_range(const MolecularOrbitals &mo,
                                                 double omega, double kf,
                                                 const Mat &Schwarz) const {
  ensure_long_range_engines(omega);
  return m_lr_engine->jk_gradient(mo, 0.0, kf, Schwarz);
}

MatTriple HartreeFock::compute_J_gradient(const MolecularOrbitals &mo,
                                          const Mat &Schwarz) const {
  return m_engine.coulomb_grad(mo.kind, mo, Schwarz);
}

JKTriple HartreeFock::compute_JK_gradient(const MolecularOrbitals &mo,
                                          const Mat &Schwarz) const {
  return m_engine.coulomb_exchange_grad(mo.kind, mo, Schwarz);
}

Mat HartreeFock::compute_kinetic_matrix() const {
  using Op = occ::qm::cint::Operator;
  return m_engine.one_electron_operator(Op::kinetic);
}

Mat HartreeFock::compute_overlap_matrix() const {
  using Op = occ::qm::cint::Operator;
  return m_engine.one_electron_operator(Op::overlap);
}

Mat HartreeFock::compute_overlap_matrix_for_basis(
    const occ::gto::AOBasis &basis) const {
  using Op = occ::qm::cint::Operator;
  occ::qm::IntegralEngine temporary_engine(basis);
  return temporary_engine.one_electron_operator(Op::overlap);
}

Mat HartreeFock::compute_nuclear_attraction_matrix() const {
  using Op = occ::qm::cint::Operator;
  return m_engine.one_electron_operator(Op::nuclear);
}

Mat HartreeFock::compute_point_charge_interaction_matrix(
    const PointChargeList &point_charges, double alpha) const {
  return m_engine.point_charge_potential(point_charges, alpha);
}

Mat HartreeFock::compute_wolf_interaction_matrix(
    const PointChargeList &point_charges,
    const std::vector<double> &partial_charges, double alpha, double rc) const {
  return m_engine.wolf_point_charge_potential(
      point_charges, partial_charges, alpha / occ::units::ANGSTROM_TO_BOHR,
      rc * occ::units::ANGSTROM_TO_BOHR);
}

Mat3N HartreeFock::electronic_electric_field_contribution(
    const MolecularOrbitals &mo, const Mat3N &positions) const {
  double delta = 1e-8;
  occ::Mat3N efield_fd(positions.rows(), positions.cols());
  for (size_t i = 0; i < 3; i++) {
    auto pts_delta = positions;
    pts_delta.row(i).array() += delta;
    auto esp_f = electronic_electric_potential_contribution(mo, pts_delta);
    pts_delta.row(i).array() -= 2 * delta;
    auto esp_b = electronic_electric_potential_contribution(mo, pts_delta);
    efield_fd.row(i) = -(esp_f - esp_b) / (2 * delta);
  }
  return efield_fd;
}

Vec HartreeFock::electronic_electric_potential_contribution(
    const MolecularOrbitals &mo, const Mat3N &positions) const {
  return m_engine.electric_potential_mmd(mo, positions);
}

Mat HartreeFock::compute_schwarz_ints() const { return m_engine.schwarz(); }

MatTriple HartreeFock::compute_kinetic_gradient() const {
  using Op = occ::qm::cint::Operator;
  return m_engine.one_electron_operator_grad(Op::kinetic);
}

MatTriple HartreeFock::compute_overlap_gradient() const {
  using Op = occ::qm::cint::Operator;
  return m_engine.one_electron_operator_grad(Op::overlap);
}

MatTriple HartreeFock::compute_nuclear_attraction_gradient() const {
  using Op = occ::qm::cint::Operator;
  return m_engine.one_electron_operator_grad(Op::nuclear);
}

MatTriple HartreeFock::compute_rinv_gradient_for_atom(size_t atom_index) const {
  return m_engine.rinv_operator_grad_atom(atom_index);
}


} // namespace occ::qm
