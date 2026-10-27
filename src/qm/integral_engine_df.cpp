#include <Eigen/Cholesky>
#include "detail/df_kernels.h"

namespace occ::qm {

using ShellPairList = std::vector<std::vector<size_t>>;
using ShellList = std::vector<gto::Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellKind = gto::Shell::Kind;
using Op = cint::Operator;
using IntegralResult = IntegralEngine::IntegralResult<3>;

IntegralEngineDF::IntegralEngineDF(const AtomList &atoms, const ShellList &ao,
                                   const ShellList &df)
    : m_ao_engine(atoms, ao), m_aux_engine(atoms, df) {
  m_ao_engine.set_auxiliary_basis(df, false);
  occ::timing::start(occ::timing::category::df);
  // don't use the shellpair list for this, results in
  // issues with positive-definiteness
  occ::log::debug("Computing V = (P|Q) in df basis");
  Mat V = m_aux_engine.one_electron_operator(Op::coulomb,
                                             false); // V = (P|Q) in df basis
  occ::timing::stop(occ::timing::category::df);

  occ::timing::start(occ::timing::category::la);
  occ::log::debug("Factorizing the Coulomb metric V");
  V_LLt.compute(V);
  if (!V_LLt.uses_cholesky()) {
    // Reached the eigendecomposition fallback: the results are still correct,
    // but the auxiliary basis is a poor fit for this geometry and says so.
    const double min_diag = V.diagonal().minCoeff();
    const double max_diag = V.diagonal().maxCoeff();
    occ::log::warn(
        "Coulomb metric is not positive definite for this auxiliary basis; "
        "using an eigendecomposition\n"
        "  Auxiliary basis size: {} functions ({} vectors discarded)\n"
        "  Diagonal range: [{:.2e}, {:.2e}]\n"
        "Suggestions:\n"
        "  - Try a different auxiliary basis (e.g., def2-universal-jkfit)\n"
        "  - If using --df-basis=auto, try a looser threshold "
        "(--df-auto-threshold=1e-3)\n"
        "  - Check for near-linear dependencies in the molecular geometry",
        V.rows(), V_LLt.num_discarded(), min_diag, max_diag);
  }
  m_V_LLt_full = V_LLt;
  occ::timing::stop(occ::timing::category::la);
}

void IntegralEngineDF::set_coulomb_method(CoulombMethod method) {
  m_coulomb_method = method;
  // Reset SplitRIJ if switching away from it
  if (method != CoulombMethod::SplitRIJ) {
    m_split_rij.reset();
  }
}

Mat IntegralEngineDF::build_b_direct(Eigen::Ref<const Mat> C_left,
                                     Eigen::Ref<const Mat> C_right) const {
  // Integral-direct density-fitting B tensor:
  //   B(i*nR + a, P) = Σ_μν C_left(μ,i) C_right(ν,a) (μν|P)
  // computed by streaming the 3-center integrals (no dense (μν|P) store). The
  // kernel parallelizes over auxiliary shells, so each thread writes a disjoint
  // set of B columns — no thread-local accumulation needed.
  const Eigen::Index nL = C_left.cols();
  const Eigen::Index nR = C_right.cols();
  const Eigen::Index naux = static_cast<Eigen::Index>(m_aux_engine.nbf());

  Mat B = Mat::Zero(nL * nR, naux);

  auto lambda = [&](const IntegralResult &args) {
    const Eigen::Index nmu = args.dims[0];
    const Eigen::Index nnu = args.dims[1];
    const Eigen::Index nP = args.dims[2];
    const auto Cl_mu = C_left.middleRows(args.bf[0], nmu);  // nmu x nL
    const auto Cr_nu = C_right.middleRows(args.bf[1], nnu); // nnu x nR
    const bool offdiag = (args.bf[0] != args.bf[1]);
    const auto Cl_nu = C_left.middleRows(args.bf[1], nnu);
    const auto Cr_mu = C_right.middleRows(args.bf[0], nmu);

    for (Eigen::Index pp = 0; pp < nP; ++pp) {
      Eigen::Map<const Mat> M(args.buffer + pp * nmu * nnu, nmu, nnu); // μ x ν
      Eigen::Map<Mat> Bview(B.col(args.bf[2] + pp).data(), nR, nL);    // a x i
      // (μν|P) contribution
      Bview.noalias() += Cr_nu.transpose() * (M.transpose() * Cl_mu);
      // (νμ|P) contribution (canonical shellpairs only store μ≥ν)
      if (offdiag)
        Bview.noalias() += Cr_mu.transpose() * (M * Cl_nu);
    }
  };

  occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
  if (m_ao_engine.is_spherical()) {
    detail::compute_three_center_integrals_tbb<ShellKind::Spherical>(
        lambda, m_ao_engine.env(), m_ao_engine.aobasis(),
        m_ao_engine.auxbasis(), m_ao_engine.shellpairs(), opt);
  } else {
    detail::compute_three_center_integrals_tbb<ShellKind::Cartesian>(
        lambda, m_ao_engine.env(), m_ao_engine.aobasis(),
        m_ao_engine.auxbasis(), m_ao_engine.shellpairs(), opt);
  }
  return B;
}

Mat IntegralEngineDF::exchange(const MolecularOrbitals &mo) {
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  // Exchange contracts every pair with every occupied orbital, so a partial
  // store would need the remainder's integrals for all of them: it stays
  // direct then, and only J uses what is stored.
  if (use_stored_integrals() && ensure_packed_store() && m_packed.complete())
    return packed_exchange(mo);
  if (m_ao_engine.is_spherical()) {
    occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
    switch (mo.kind) {
    default: // Restricted
      return detail::direct_exchange_operator_kernel_r<ShellKind::Spherical>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case U:
      return detail::direct_exchange_operator_kernel_u<ShellKind::Spherical>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case G:
      return detail::direct_exchange_operator_kernel_g<ShellKind::Spherical>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    }
  } else {
    occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
    switch (mo.kind) {
    default: // Restricted
      return detail::direct_exchange_operator_kernel_r<ShellKind::Cartesian>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case U:
      return detail::direct_exchange_operator_kernel_u<ShellKind::Cartesian>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case G:
      return detail::direct_exchange_operator_kernel_g<ShellKind::Cartesian>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    }
  }
}

Mat IntegralEngineDF::coulomb(const MolecularOrbitals &mo) {
  // Check for Split-RI-J method first
  if (m_coulomb_method == CoulombMethod::SplitRIJ) {
    // Lazy-initialize SplitRIJ engine
    if (!m_split_rij) {
      occ::log::debug("Initializing Split-RI-J engine");
      m_split_rij = std::make_unique<SplitRIJ>(
          m_ao_engine.aobasis(), m_ao_engine.auxbasis(),
          m_ao_engine.shellpairs(), m_ao_engine.schwarz());
    }
    // Split-RI-J doesn't compute (μν|P) integrals - no stored/direct mode
    return m_split_rij->coulomb(mo);
  }

  // Traditional RI-J path
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  if (use_stored_integrals() && ensure_packed_store())
    return packed_coulomb(mo);
  if (m_ao_engine.is_spherical()) {
    occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
    switch (mo.kind) {
    default: // Restricted
      return detail::direct_coulomb_operator_kernel_r<ShellKind::Spherical>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case U:
      return detail::direct_coulomb_operator_kernel_u<ShellKind::Spherical>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case G:
      return detail::direct_coulomb_operator_kernel_g<ShellKind::Spherical>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    }
  } else {
    occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
    switch (mo.kind) {
    default: // Restricted
      return detail::direct_coulomb_operator_kernel_r<ShellKind::Cartesian>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case U:
      return detail::direct_coulomb_operator_kernel_u<ShellKind::Cartesian>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case G:
      return detail::direct_coulomb_operator_kernel_g<ShellKind::Cartesian>(
          m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    }
  }
}

JKPair IntegralEngineDF::coulomb_and_exchange(const MolecularOrbitals &mo) {
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  // A partial store only helps J; K would then recompute every integral
  // anyway, so the fused direct kernel, which computes each once for both, is
  // cheaper than semi-direct J plus direct K.
  if (use_stored_integrals() && ensure_packed_store() && m_packed.complete())
    return {coulomb(mo), exchange(mo)};
  if (m_ao_engine.is_spherical()) {
    occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
    switch (mo.kind) {
    default: // Restricted
      return detail::direct_coulomb_and_exchange_operator_kernel_r<
          ShellKind::Spherical>(m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case U:
      return detail::direct_coulomb_and_exchange_operator_kernel_u<
          ShellKind::Spherical>(m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case G:
      return detail::direct_coulomb_and_exchange_operator_kernel_g<
          ShellKind::Spherical>(m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    }
  } else {
    occ::qm::cint::Optimizer opt(m_ao_engine.env(), Op::coulomb, 3);
    switch (mo.kind) {
    default:
      return detail::direct_coulomb_and_exchange_operator_kernel_r<
          ShellKind::Cartesian>(m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case U:
      return detail::direct_coulomb_and_exchange_operator_kernel_u<
          ShellKind::Cartesian>(m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    case G:
      return detail::direct_coulomb_and_exchange_operator_kernel_g<
          ShellKind::Cartesian>(m_ao_engine, m_aux_engine, mo, V_LLt, opt);
    }
  }
}

Mat IntegralEngineDF::fock_operator(const MolecularOrbitals &mo) {
  auto [J, K] = coulomb_and_exchange(mo);
  J.noalias() -= K;
  return J;
}

namespace {

// The Coulomb contribution of the shell pairs the packed store left out:
// with x null it adds g(P,c) = sum D_c(μ,ν) (μν|P) over them, else it adds
// J_c(μ,ν) = sum_P (μν|P) x(P,c) into J.
template <ShellKind kind>
void unstored_coulomb(IntegralEngine &engine, const ShellPairList &pairs,
                      const std::vector<Mat> &D, Mat &g, const Mat *x,
                      std::vector<Mat> &J) {
  const size_t nc = D.size(), nbf = engine.nbf(),
               naux = engine.auxbasis().nbf();
  occ::parallel::thread_local_storage<Mat> g_local(
      [&]() { return Mat::Zero(naux, nc); });
  occ::parallel::thread_local_storage<std::vector<Mat>> J_local(
      [&]() { return std::vector<Mat>(x ? nc : 0, Mat::Zero(nbf, nbf)); });
  auto f = [&](const IntegralResult &args) {
    const bool offdiag = args.bf[0] != args.bf[1];
    const size_t n = size_t(args.dims[0]) * args.dims[1];
    for (int P = 0; P < args.dims[2]; P++) {
      Eigen::Map<const Mat> M(args.buffer + P * n, args.dims[0], args.dims[1]);
      const size_t r = args.bf[2] + P;
      for (size_t c = 0; c < nc; c++) {
        if (x) {
          auto &Jc = J_local.local()[c];
          Jc.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
              (*x)(r, c) * M;
          if (offdiag)
            Jc.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
                (*x)(r, c) * M.transpose();
          continue;
        }
        double v =
            (D[c].block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                 .array() *
             M.array())
                .sum();
        if (offdiag)
          v += (D[c].block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                    .array() *
                M.transpose().array())
                   .sum();
        g_local.local()(r, c) += v;
      }
    }
  };
  cint::Optimizer opt(engine.env(), Op::coulomb, 3);
  detail::compute_three_center_integrals_tbb<kind>(
      f, engine.env(), engine.aobasis(), engine.auxbasis(), pairs, opt);
  if (x) {
    for (const auto &local : J_local)
      for (size_t c = 0; c < nc; c++)
        J[c] += local[c];
  } else {
    for (const auto &local : g_local)
      g += local;
  }
}

} // namespace

void IntegralEngineDF::build_packed_store(size_t max_bytes) {
  // Only ever the full Coulomb operator (see `use_stored_integrals`), so the
  // metric folded in is the omega = 0 one whatever is active
  occ::timing::start(occ::timing::category::df);
  m_packed.build(m_ao_engine, m_V_LLt_full, max_bytes);
  occ::timing::stop(occ::timing::category::df);
}

bool IntegralEngineDF::ensure_packed_store() {
  if (!m_packed.built())
    build_packed_store(m_policy == Policy::Stored
                           ? PackedDFStore::unlimited
                           : m_integral_store_memory_limit);
  return !m_packed.empty();
}

bool IntegralEngineDF::ensure_complete_packed_store(size_t max_bytes) {
  if (m_packed.complete())
    return true;
  if (m_omega != 0.0 || packed_storage_bytes() > max_bytes)
    return false;
  build_packed_store(PackedDFStore::unlimited);
  return true;
}

Mat IntegralEngineDF::packed_coulomb(const MolecularOrbitals &mo) {
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  occ::timing::start(occ::timing::category::df);
  std::vector<Mat> D;
  switch (mo.kind) {
  default:
    D = {mo.D};
    break;
  case U:
    D = {block::a(mo.D) + block::b(mo.D)};
    break;
  case G:
    // both diagonal blocks see the field of the whole density
    D = {block::aa(mo.D) + block::bb(mo.D)};
    break;
  }
  Mat g = m_packed.contract(D);
  const bool partial = !m_packed.complete();
  std::vector<Mat> unused;
  const bool spherical = m_ao_engine.is_spherical();
  if (partial) {
    if (spherical)
      unstored_coulomb<ShellKind::Spherical>(m_ao_engine, m_packed.unstored(),
                                             D, g, nullptr, unused);
    else
      unstored_coulomb<ShellKind::Cartesian>(m_ao_engine, m_packed.unstored(),
                                             D, g, nullptr, unused);
  }
  // A complete store is folded and already carries L^-1: its g is
  // L^-1 (μν|P) D, and expanding with it again applies L^-T, so V^-1 is
  // complete
  const Mat x = partial ? Mat(V_LLt.solve(g)) : g;
  std::vector<Mat> J = m_packed.expand(x);
  if (partial) {
    if (spherical)
      unstored_coulomb<ShellKind::Spherical>(m_ao_engine, m_packed.unstored(),
                                             D, g, &x, J);
    else
      unstored_coulomb<ShellKind::Cartesian>(m_ao_engine, m_packed.unstored(),
                                             D, g, &x, J);
  }
  occ::timing::stop(occ::timing::category::df);
  const size_t nbf = m_ao_engine.nbf();
  switch (mo.kind) {
  default:
    return 2 * J[0];
  case U: {
    Mat result(2 * nbf, nbf);
    block::a(result) = 2 * J[0];
    block::b(result) = 2 * J[0];
    return result;
  }
  case G: {
    Mat result = Mat::Zero(2 * nbf, 2 * nbf);
    block::aa(result) = 2 * J[0];
    block::bb(result) = 2 * J[0];
    return result;
  }
  }
}

Mat IntegralEngineDF::packed_exchange(const MolecularOrbitals &mo) {
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  occ::timing::start(occ::timing::category::df);
  const size_t nbf = m_ao_engine.nbf();
  Mat result;
  switch (mo.kind) {
  default:
    result = m_packed.exchange({mo.Cocc}, {{0, 0}})[0];
    break;
  case U: {
    const auto K = m_packed.exchange(
        {Mat(block::a(mo.Cocc)), Mat(block::b(mo.Cocc))}, {{0, 0}, {1, 1}});
    result = Mat(2 * nbf, nbf);
    block::a(result) = K[0];
    block::b(result) = K[1];
    break;
  }
  case G: {
    const auto K =
        m_packed.exchange({Mat(block::a(mo.Cocc)), Mat(block::b(mo.Cocc))},
                          {{0, 0}, {1, 1}, {0, 1}});
    // K[2] = sum_P (B_P C_a)(B_P C_b)^T is the alpha-row, beta-column block,
    // which is block::ba; K_ab is its transpose
    result = Mat(2 * nbf, 2 * nbf);
    block::aa(result) = K[0];
    block::bb(result) = K[1];
    block::ba(result) = K[2];
    block::ab(result) = K[2].transpose();
    break;
  }
  }
  occ::timing::stop(occ::timing::category::df);
  return result;
}

void IntegralEngineDF::set_range_separated_omega(double omega) {
  if (omega == m_omega)
    return;
  m_ao_engine.set_range_separated_omega(omega);
  m_aux_engine.set_range_separated_omega(omega);
  // The fitting metric must use the same operator as the 3-centre integrals:
  // for the long-range exchange that is (P|erf(omega r)/r|Q), not (P|1/r|Q).
  // Keep the full-Coulomb factorization so switching omega back is free.
  if (omega == 0.0) {
    V_LLt = m_V_LLt_full;
  } else if (omega == m_lr_omega) {
    V_LLt = m_V_LLt_lr; // cached: the SCF toggles omega on every Fock build
  } else {
    occ::timing::start(occ::timing::category::df);
    Mat V = m_aux_engine.one_electron_operator(Op::coulomb, false);
    occ::timing::stop(occ::timing::category::df);
    occ::timing::start(occ::timing::category::la);
    // (P|erf(omega r)/r|Q) is far more ill-conditioned than (P|1/r|Q): the
    // attenuated kernel damps the high-exponent auxiliary functions, so a
    // JK-fitting basis is close to linearly dependent under it. CoulombMetric
    // drops the resulting near-null space.
    V_LLt.compute(V);
    occ::timing::stop(occ::timing::category::la);
    if (!V_LLt.uses_cholesky())
      occ::log::debug("Long-range Coulomb metric (omega = {}) is near-singular, "
                      "as expected: discarded {} of {} auxiliary vectors",
                      omega, V_LLt.num_discarded(), V.rows());
    m_V_LLt_lr = V_LLt;
    m_lr_omega = omega;
  }
  m_omega = omega;
}

void IntegralEngineDF::set_precision(double precision) {
  // the shell-pair screen, and with it the stored pairs, depend on it
  if (precision != m_precision)
    m_packed.clear();
  m_precision = precision;
  m_ao_engine.set_precision(precision);
  m_aux_engine.set_precision(precision);
}

} // namespace occ::qm
