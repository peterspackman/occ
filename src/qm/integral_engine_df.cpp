#include "detail/df_kernels.h"
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/expectation.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mp2_components.h>

namespace occ::qm {

using ShellPairList = std::vector<std::vector<size_t>>;
using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellKind = Shell::Kind;
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
    occ::log::warn(
        "LLT decomposition of Coulomb metric in DF was not successful!");
  }
  occ::timing::stop(occ::timing::category::la);
}

void IntegralEngineDF::compute_stored_integrals() {
  occ::timing::start(occ::timing::category::df);
  if (m_integral_store.rows() == 0) {
    occ::log::info("Storing 3-center integrals");
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

Eigen::Tensor<double, 4>
IntegralEngineDF::four_center_integrals_tensor() const {
  occ::timing::start(occ::timing::category::df);

  // Ensure 3-center integrals are computed and stored
  const_cast<IntegralEngineDF *>(this)->compute_stored_integrals();

  const size_t nbf = m_ao_engine.aobasis().nbf();
  const size_t naux = m_aux_engine.nbf();
  const auto nthreads = occ::parallel::get_num_threads();

  occ::log::debug("Computing AO integral tensor using DF approximation");
  occ::log::debug("Basis functions: {} AO, {} auxiliary, {} threads", nbf, naux,
                  nthreads);

  // Create single shared result tensor like the conventional AO code
  Eigen::Tensor<double, 4> result(nbf, nbf, nbf, nbf);
  result.setZero();
  
  // Use TBB parallel_for to distribute work over μν pairs
  size_t total_pairs = nbf * nbf;
  occ::parallel::parallel_for(size_t(0), total_pairs, [&](size_t pair_idx) {
    size_t mu = pair_idx / nbf;
    size_t nu = pair_idx % nbf;

    // For each ρσ pair, compute (μν|ρσ) = Σ_P (μν|P) * V^(-1) * (ρσ|P)
    for (size_t rho = 0; rho < nbf; ++rho) {
      for (size_t sigma = 0; sigma < nbf; ++sigma) {
        // Extract (ρσ|P) vector
        Vec rhosigmaP = Vec::Zero(naux);
        for (size_t P = 0; P < naux; ++P) {
          const auto eri_P =
              Eigen::Map<const Mat>(m_integral_store.col(P).data(), nbf, nbf);
          rhosigmaP(P) = eri_P(rho, sigma);
        }

        // Solve V * x = (ρσ|P) to get x = V^(-1) * (ρσ|P)
        Vec x = V_LLt.solve(rhosigmaP);

        // Compute (μν|ρσ) = (μν|P) * V^(-1) * (ρσ|P)
        double integral_value = 0.0;
        for (size_t P = 0; P < naux; ++P) {
          const auto eri_P =
              Eigen::Map<const Mat>(m_integral_store.col(P).data(), nbf, nbf);
          integral_value += eri_P(mu, nu) * x(P);
        }

        result(mu, nu, rho, sigma) = integral_value;
      }
    }
  });

  occ::timing::stop(occ::timing::category::df);
  occ::log::debug("DF AO integral tensor computation completed");

  return result;
}

MP2Components IntegralEngineDF::compute_df_mp2_energy(
    const MolecularOrbitals &mo, const Vec &orbital_energies,
    const MP2OrbitalSpec &orbital_spec) const {
  occ::timing::start(occ::timing::category::df);

  const_cast<IntegralEngineDF *>(this)->compute_stored_integrals();

  const size_t nbf = m_ao_engine.aobasis().nbf();
  const size_t naux = m_aux_engine.nbf();

  double mp2_energy = 0.0;
  double same_spin_energy = 0.0;
  double opposite_spin_energy = 0.0;

  // Precompute virtual orbital coefficients
  Mat C_virt = Mat::Zero(nbf, orbital_spec.n_active_virt);
  for (size_t a = 0; a < orbital_spec.n_active_virt; ++a) {
    size_t a_full = orbital_spec.n_total_occ + a;
    C_virt.col(a) = mo.C.col(a_full);
  }

  // Main DF-MP2 computation loop
  for (size_t i = 0; i < orbital_spec.n_active_occ; ++i) {
    size_t i_full = i + orbital_spec.n_frozen_core;
    auto c_i = mo.C.col(i_full);

    // Transform (μν|P) -> (iν|P)
    Mat iuP = Mat::Zero(nbf, naux);
    for (size_t P = 0; P < naux; ++P) {
      const auto eri_P =
          Eigen::Map<const Mat>(m_integral_store.col(P).data(), nbf, nbf);
      iuP.col(P) = eri_P * c_i;
    }

    // Transform (iν|P) -> (ia|P)
    Mat iaP = C_virt.transpose() * iuP;

    // Apply Coulomb metric
    Mat X = V_LLt.solve(iaP.transpose());

    // Contract with other occupied orbitals
    for (size_t j = 0; j < orbital_spec.n_active_occ; ++j) {
      size_t j_full = j + orbital_spec.n_frozen_core;
      auto c_j = mo.C.col(j_full);

      // Transform (μν|P) -> (jν|P) -> (jb|P)
      Mat juP = Mat::Zero(nbf, naux);
      for (size_t P = 0; P < naux; ++P) {
        const auto eri_P =
            Eigen::Map<const Mat>(m_integral_store.col(P).data(), nbf, nbf);
        juP.col(P) = eri_P * c_j;
      }
      Mat jbP = C_virt.transpose() * juP;

      // Compute (ia|jb) integrals
      Mat integral_block = iaP * V_LLt.solve(jbP.transpose());

      // Accumulate MP2 energy contributions
      for (size_t a = 0; a < orbital_spec.n_active_virt; ++a) {
        for (size_t b = 0; b < orbital_spec.n_active_virt; ++b) {
          double integral_iajb = integral_block(a, b);
          double integral_ibja = integral_block(b, a);

          // Compute energy denominator
          size_t a_full = orbital_spec.n_total_occ + a;
          size_t b_full = orbital_spec.n_total_occ + b;
          double denominator =
              orbital_energies(i_full) + orbital_energies(j_full) -
              orbital_energies(a_full) - orbital_energies(b_full);

          constexpr double denominator_threshold = 1e-12;
          if (std::abs(denominator) > denominator_threshold) {
            double numerator =
                integral_iajb * (2.0 * integral_iajb - integral_ibja);
            mp2_energy += numerator / denominator;

            opposite_spin_energy +=
                2.0 * integral_iajb * integral_iajb / denominator;
            same_spin_energy += -integral_iajb * integral_ibja / denominator;
          }
        }
      }
    }
  }
  occ::timing::stop(occ::timing::category::df);

  MP2Components result;
  result.total_correlation = mp2_energy;
  result.same_spin_correlation = same_spin_energy;
  result.opposite_spin_correlation = opposite_spin_energy;
  result.orbital_info.n_frozen_core = orbital_spec.n_frozen_core;
  result.orbital_info.n_active_occ = orbital_spec.n_active_occ;
  result.orbital_info.n_active_virt = orbital_spec.n_active_virt;
  result.orbital_info.n_total_occ = orbital_spec.n_total_occ;
  result.orbital_info.n_total_virt = orbital_spec.n_total_virt;
  result.orbital_info.e_min_used = orbital_spec.e_min;
  result.orbital_info.e_max_used = orbital_spec.e_max;

  return result;
}

} // namespace occ::qm
