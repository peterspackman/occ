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
    : m_ao_engine(atoms, ao), m_aux_engine(atoms, df), m_integral_store(0, 0) {
  m_ao_engine.set_auxiliary_basis(df, false);
  occ::timing::start(occ::timing::category::df);
  // don't use the shellpair list for this, results in
  // issues with positive-definiteness
  occ::log::debug("Computing V = (P|Q) in df basis");
  Mat V = m_aux_engine.one_electron_operator(Op::coulomb,
                                             false); // V = (P|Q) in df basis
  occ::timing::stop(occ::timing::category::df);

  occ::timing::start(occ::timing::category::la);
  occ::log::debug("Computing LLt decomposition of V");
  V_LLt = Eigen::LLT<Mat>(V);
  if (V_LLt.info() != Eigen::Success) {
    // Gather diagnostic information
    const size_t naux = V.rows();
    const double min_diag = V.diagonal().minCoeff();
    const double max_diag = V.diagonal().maxCoeff();
    const double cond_approx = max_diag / std::max(min_diag, 1e-16);

    occ::log::error(
        "LLT decomposition of Coulomb metric failed!\n"
        "  Auxiliary basis size: {} functions\n"
        "  Diagonal range: [{:.2e}, {:.2e}]\n"
        "  Approx condition number: {:.2e}",
        naux, min_diag, max_diag, cond_approx);

    if (min_diag < 0) {
      occ::log::error("  Matrix has negative diagonal elements - basis may have linear dependencies");
    } else if (cond_approx > 1e10) {
      occ::log::error("  Matrix is ill-conditioned - auxiliary basis may be poorly suited");
    }

    occ::log::error("Suggestions:\n"
                    "  - Try a different auxiliary basis (e.g., def2-universal-jkfit)\n"
                    "  - If using --df-basis=auto, try a looser threshold (--df-auto-threshold=1e-3)\n"
                    "  - Check for near-linear dependencies in the molecular geometry");
  }
  occ::timing::stop(occ::timing::category::la);
}

void IntegralEngineDF::set_coulomb_method(CoulombMethod method) {
  m_coulomb_method = method;
  // Reset SplitRIJ if switching away from it
  if (method != CoulombMethod::SplitRIJ) {
    m_split_rij.reset();
  }
}

void IntegralEngineDF::compute_stored_integrals() {
  occ::timing::start(occ::timing::category::df);
  if (m_integral_store.rows() == 0) {
    occ::log::info("Storing 3-center integrals, computing with {} threads",
                   occ::parallel::get_num_threads());
    size_t nbf = m_ao_engine.nbf();
    size_t ndf = m_aux_engine.nbf();
    m_integral_store = Mat::Zero(nbf * nbf, ndf);
    auto lambda = [&](const IntegralResult &args) {
      size_t offset = 0;
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        auto x = Eigen::Map<Mat>(m_integral_store.col(i).data(), nbf, nbf);
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        x.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = buf_mat;
        if (args.bf[0] != args.bf[1])
          x.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) =
              buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
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
  }
  occ::timing::stop(occ::timing::category::df);
}

Mat IntegralEngineDF::exchange(const MolecularOrbitals &mo) {
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  bool direct = !use_stored_integrals();
  if (!direct) {
    compute_stored_integrals();
    switch (mo.kind) {
    default: // Restricted
      return detail::stored_exchange_kernel_r(
          m_integral_store, m_ao_engine.aobasis(), m_ao_engine.auxbasis(), mo,
          V_LLt);
    case U:
      return detail::stored_exchange_kernel_u(
          m_integral_store, m_ao_engine.aobasis(), m_ao_engine.auxbasis(), mo,
          V_LLt);
    case G:
      return detail::stored_exchange_kernel_g(
          m_integral_store, m_ao_engine.aobasis(), m_ao_engine.auxbasis(), mo,
          V_LLt);
    }
  } else if (m_ao_engine.is_spherical()) {
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
  bool direct = !use_stored_integrals();
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  if (!direct) {
    compute_stored_integrals();
    switch (mo.kind) {
    default: // Restricted
      return detail::stored_coulomb_kernel_r(m_integral_store,
                                             m_ao_engine.aobasis(),
                                             m_ao_engine.auxbasis(), mo, V_LLt);
    case U:
      return detail::stored_coulomb_kernel_u(m_integral_store,
                                             m_ao_engine.aobasis(),
                                             m_ao_engine.auxbasis(), mo, V_LLt);
    case G:
      return detail::stored_coulomb_kernel_g(m_integral_store,
                                             m_ao_engine.aobasis(),
                                             m_ao_engine.auxbasis(), mo, V_LLt);
    }
  } else if (m_ao_engine.is_spherical()) {
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
  bool direct = !use_stored_integrals();
  if (!direct) {
    compute_stored_integrals();
    return {coulomb(mo), exchange(mo)};
  } else if (m_ao_engine.is_spherical()) {
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

void IntegralEngineDF::set_range_separated_omega(double omega) {
  m_ao_engine.set_range_separated_omega(omega);
  m_aux_engine.set_range_separated_omega(omega);
  set_integral_policy(Policy::Direct);
}

void IntegralEngineDF::set_precision(double precision) {
  m_precision = precision;
  m_ao_engine.set_precision(precision);
  m_aux_engine.set_precision(precision);
}

} // namespace occ::qm
