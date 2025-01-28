#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/dft/dft_method.h>
#include <occ/dft/functional.h>
#include <occ/dft/grid.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/dft/range_separated_parameters.h>
#include <occ/dft/xc_potential_matrix.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>
#include <string>
#include <vector>

namespace occ::dft {
using occ::qm::expectation;
using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using PointChargeList = std::vector<occ::core::PointCharge>;

using occ::IVec;
using occ::Mat3N;
using occ::MatN4;
using occ::Vec;

namespace block = occ::qm::block;

namespace impl {

template <SpinorbitalKind spinorbital_kind, int derivative_order>
void set_params(DensityFunctional::Params &params, const Mat &rho) {
  if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
    params.rho.col(0) = rho.col(0);
  } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
    // correct assignment
    params.rho.col(0) = block::a(rho.col(0));
    params.rho.col(1) = block::b(rho.col(0));
  }

  if constexpr (derivative_order > 0) {
    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
      params.sigma.col(0) = (rho.block(0, 1, rho.rows(), 3).array() *
                             rho.block(0, 1, rho.rows(), 3).array())
                                .rowwise()
                                .sum();
    } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
      const auto rho_a = block::a(rho.array());
      const auto rho_b = block::b(rho.array());
      const auto &dx_rho_a = rho_a.col(1);
      const auto &dy_rho_a = rho_a.col(2);
      const auto &dz_rho_a = rho_a.col(3);
      const auto &dx_rho_b = rho_b.col(1);
      const auto &dy_rho_b = rho_b.col(2);
      const auto &dz_rho_b = rho_b.col(3);
      params.sigma.col(0) =
          dx_rho_a * dx_rho_a + dy_rho_a * dy_rho_a + dz_rho_a * dz_rho_a;
      params.sigma.col(1) =
          dx_rho_a * dx_rho_b + dy_rho_a * dy_rho_b + dz_rho_a * dz_rho_b;
      params.sigma.col(2) =
          dx_rho_b * dx_rho_b + dy_rho_b * dy_rho_b + dz_rho_b * dz_rho_b;
    }
  }
  if constexpr (derivative_order > 1) {
    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
      params.laplacian.col(0) = rho.col(4);
      params.tau.col(0) = rho.col(5);
    } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
      params.laplacian.col(0) = block::a(rho.col(4));
      params.laplacian.col(1) = block::b(rho.col(4));
      params.tau.col(0) = block::a(rho.col(5));
      params.tau.col(1) = block::b(rho.col(5));
    }
  }
}

} // namespace impl

class DFT : public qm::SCFMethodBase {

public:
  DFT(const std::string &, const AOBasis &, const BeckeGridSettings & = {});
  inline const auto &aobasis() const { return m_hf.aobasis(); }
  inline auto nbf() const { return m_hf.nbf(); }

  void set_integration_grid(const BeckeGridSettings & = {});

  inline void
  set_density_fitting_basis(const std::string &density_fitting_basis) {
    m_hf.set_density_fitting_basis(density_fitting_basis);
  }

  inline void set_precision(double precision) { m_hf.set_precision(precision); }

  double exchange_correlation_energy() const { return m_exc_dft; }
  double exchange_energy_total() const { return m_exchange_energy; }

  bool usual_scf_energy() const { return false; }
  void update_scf_energy(occ::core::EnergyComponents &energy,
                         bool incremental) const {
    if (incremental) {
      energy["electronic.2e"] += m_two_electron_energy;
      energy["electronic"] += m_two_electron_energy;
      energy["electronic.dft_xc"] += exchange_correlation_energy();
    } else {
      energy["electronic"] = energy["electronic.1e"];
      energy["electronic.2e"] = m_two_electron_energy;
      energy["electronic"] += m_two_electron_energy;
      energy["electronic.dft_xc"] = exchange_correlation_energy();
    }

    if (m_nlc_energy != 0.0) {
      energy["electronic.nonlocal_correlation"] = m_nlc_energy;
    }

    energy["total"] = energy["electronic"] + energy["nuclear.repulsion"];

    const auto pcloc = energy.find("nuclear.point_charge");
    if (pcloc != energy.end()) {
      energy["total"] += pcloc->second;
    }
  }
  bool supports_incremental_fock_build() const { return false; }
  inline bool have_effective_core_potentials() const {
    return m_hf.have_effective_core_potentials();
  }

  int density_derivative() const;
  inline double exact_exchange_factor() const {
    return std::accumulate(
        m_funcs.polarized.begin(), m_funcs.polarized.end(), 0.0,
        [&](double a, const auto &v) { return a + v.exact_exchange_factor(); });
  }

  RangeSeparatedParameters range_separated_parameters() const;

  inline double
  nuclear_point_charge_interaction_energy(const PointChargeList &pc) const {
    return m_hf.nuclear_point_charge_interaction_energy(pc);
  }

  auto compute_kinetic_matrix() const { return m_hf.compute_kinetic_matrix(); }

  auto compute_overlap_matrix() const { return m_hf.compute_overlap_matrix(); }

  auto compute_overlap_matrix_for_basis(const occ::qm::AOBasis &bs) const {
    return m_hf.compute_overlap_matrix_for_basis(bs);
  }

  auto compute_nuclear_attraction_matrix() const {
    return m_hf.compute_nuclear_attraction_matrix();
  }

  auto compute_effective_core_potential_matrix() const {
    return m_hf.compute_effective_core_potential_matrix();
  }

  auto compute_point_charge_interaction_matrix(
      const PointChargeList &point_charges) const {
    return m_hf.compute_point_charge_interaction_matrix(point_charges);
  }

  auto compute_schwarz_ints() const { return m_hf.compute_schwarz_ints(); }

  template <unsigned int order = 1>
  inline auto compute_electronic_multipoles(const MolecularOrbitals &mo,
                                            const Vec3 &o = {0.0, 0.0,
                                                             0.0}) const {
    return m_hf.template compute_electronic_multipoles<order>(mo, o);
  }

  template <unsigned int order = 1>
  inline auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0,
                                                          0.0}) const {
    return m_hf.template compute_nuclear_multipoles<order>(o);
  }

  template <unsigned int order = 1>
  inline auto compute_multipoles(const MolecularOrbitals &mo,
                                 const Vec3 &o = {0.0, 0.0, 0.0}) const {
    return m_hf.template compute_multipoles<order>(mo, o);
  }

  template <int derivative_order,
            SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
  Mat compute_K_dft(const MolecularOrbitals &mo, const Mat &Schwarz) {
    using occ::parallel::nthreads;
    const auto &basis = m_hf.aobasis();
    size_t K_rows, K_cols;
    size_t nbf = basis.nbf();
    const auto &D = mo.D;
    std::tie(K_rows, K_cols) =
        occ::qm::matrix_dimensions<spinorbital_kind>(nbf);
    Mat K = Mat::Zero(K_rows, K_cols);
    m_two_electron_energy = 0.0;
    m_exc_dft = 0.0;

    constexpr size_t BLOCKSIZE = 64;
    size_t num_rows_factor = 1;
    if (spinorbital_kind == SpinorbitalKind::Unrestricted)
      num_rows_factor = 2;

    double total_density_a{0.0}, total_density_b{0.0};
    const Mat D2 = 2 * D;
    DensityFunctional::Family family{DensityFunctional::Family::LDA};
    if constexpr (derivative_order == 1) {
      family = DensityFunctional::Family::GGA;
    }
    if constexpr (derivative_order == 2) {
      family = DensityFunctional::Family::MGGA;
    }

    std::vector<Mat> Kt(occ::parallel::nthreads, Mat::Zero(D.rows(), D.cols()));
    std::vector<double> energies(occ::parallel::nthreads, 0.0);
    std::vector<double> alpha_densities(occ::parallel::nthreads, 0.0);
    std::vector<double> beta_densities(occ::parallel::nthreads, 0.0);

    const auto &funcs = (spinorbital_kind == SpinorbitalKind::Unrestricted)
                            ? m_funcs.polarized
                            : m_funcs.unpolarized;
    for (const auto &atom_grid : m_atom_grids) {
      const auto &atom_pts = atom_grid.points;
      const auto &atom_weights = atom_grid.weights;
      const size_t npt_total = atom_pts.cols();
      const size_t num_blocks = npt_total / BLOCKSIZE + 1;

      auto lambda = [&](int thread_id) {
        Mat rho_storage(num_rows_factor * BLOCKSIZE,
                        occ::density::num_components(derivative_order));
        for (size_t block = 0; block < num_blocks; block++) {
          if (block % nthreads != thread_id)
            continue;
          Eigen::Index l = block * BLOCKSIZE;
          Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
          Eigen::Index npt = u - l;
          if (npt <= 0)
            continue;
          Eigen::Ref<Mat> rho = rho_storage.block(0, 0, num_rows_factor * npt,
                                                  rho_storage.cols());

          auto &k = Kt[thread_id];
          const auto &pts_block = atom_pts.middleCols(l, npt);
          const auto &weights_block = atom_weights.segment(l, npt);
          auto gto_vals =
              occ::gto::evaluate_basis(basis, pts_block, derivative_order);
          occ::density::evaluate_density<derivative_order, spinorbital_kind>(
              D2, gto_vals, rho);

          double max_density_block = rho.col(0).maxCoeff();
          if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
            alpha_densities[thread_id] += rho.col(0).dot(weights_block);
          } else if constexpr (spinorbital_kind ==
                               SpinorbitalKind::Unrestricted) {
            Vec rho_a_tmp = block::a(rho.col(0));
            Vec rho_b_tmp = block::b(rho.col(0));
            double tot_density_a = rho_a_tmp.dot(weights_block);
            double tot_density_b = rho_b_tmp.dot(weights_block);
            alpha_densities[thread_id] += tot_density_a;
            beta_densities[thread_id] += tot_density_b;
          }
          if (max_density_block < m_density_threshold)
            continue;

          DensityFunctional::Params params(npt, family, spinorbital_kind);
          impl::set_params<spinorbital_kind, derivative_order>(params, rho);

          DensityFunctional::Result res(npt, family, spinorbital_kind);
          for (const auto &func : funcs) {
            res += func.evaluate(params);
          }

          Mat KK = Mat::Zero(k.rows(), k.cols());

          // Weight the arrays by the grid weights
          res.weight_by(weights_block);
          xc_potential_matrix<spinorbital_kind, derivative_order>(
              res, rho, gto_vals, KK, energies[thread_id]);
          k.noalias() += KK;
        }
      };
      occ::timing::start(occ::timing::category::dft_xc);
      occ::parallel::parallel_do(lambda);
      occ::timing::stop(occ::timing::category::dft_xc);
    }
    for (size_t i = 0; i < nthreads; i++) {
      K += Kt[i];
      m_exc_dft += energies[i];
      total_density_a += alpha_densities[i];
      total_density_b += beta_densities[i];
    }
    occ::log::debug("Total density: alpha = {} beta = {}", total_density_a,
                    total_density_b);

    return K;
  }

  inline Mat compute_J(const MolecularOrbitals &mo,
                       const Mat &Schwarz = Mat()) const {
    return m_hf.compute_J(mo, Schwarz);
  }

  inline Mat compute_vxc(const MolecularOrbitals &mo,
                         const Mat &Schwarz = Mat()) {
    int deriv = density_derivative();
    switch (mo.kind) {
    case SpinorbitalKind::Unrestricted: {
      occ::log::debug("Unrestricted vxc evaluation");
      switch (deriv) {
      case 0:
        return compute_K_dft<0, SpinorbitalKind::Unrestricted>(mo, Schwarz);
      case 1:
        return compute_K_dft<1, SpinorbitalKind::Unrestricted>(mo, Schwarz);
      case 2:
        return compute_K_dft<2, SpinorbitalKind::Unrestricted>(mo, Schwarz);
      default:
        throw std::runtime_error(
            "Not implemented: DFT for derivative order > 2");
      }
    }
    case SpinorbitalKind::Restricted: {
      switch (deriv) {
      case 0:
        return compute_K_dft<0, SpinorbitalKind::Restricted>(mo, Schwarz);
      case 1:
        return compute_K_dft<1, SpinorbitalKind::Restricted>(mo, Schwarz);
      case 2:
        return compute_K_dft<2, SpinorbitalKind::Restricted>(mo, Schwarz);
      default:
        throw std::runtime_error(
            "Not implemented: DFT for derivative order > 2");
      }
    }
    default:
      throw std::runtime_error("Not implemented: DFT for General spinorbitals");
    }
  }

  inline qm::JKPair compute_JK(const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) {

    qm::JKPair jk;
    jk.K = -compute_vxc(mo, Schwarz);
    double ecoul{0.0}, exc{0.0};
    double exchange_factor = exact_exchange_factor();
    RangeSeparatedParameters rs = range_separated_parameters();
    if (rs.omega != 0.0) {
      // range separated hybrid
      qm::JKPair jk_short_range = m_hf.compute_JK(mo, Schwarz);
      jk.J = jk_short_range.J;

      m_hf.set_range_separated_omega(rs.omega);
      qm::JKPair jk_long_range = m_hf.compute_JK(mo, Schwarz);
      Mat Khf = jk_short_range.K * (rs.alpha + rs.beta) +
                jk_long_range.K * (-rs.beta);
      exc = -expectation(mo.kind, mo.D, Khf);
      jk.K.noalias() += Khf;
      m_hf.set_range_separated_omega(0.0);
    } else if (exchange_factor != 0.0) {
      // global hybrid
      qm::JKPair jk_hf = m_hf.compute_JK(mo, Schwarz);
      jk.J = jk_hf.J;

      exc = -expectation(mo.kind, mo.D, jk_hf.K) * exchange_factor;
      jk.K.noalias() += jk_hf.K * exchange_factor;
    } else {
      jk.J = m_hf.compute_J(mo, Schwarz);
    }
    ecoul = expectation(mo.kind, mo.D, jk.J);
    occ::log::debug("E_xc (DFT): {:20.12f}  E_ex: {:20.12f} E_coul: {:20.12f}",
                    m_exc_dft, exc, ecoul);
    m_exchange_energy = m_exc_dft + exc;
    m_two_electron_energy += m_exchange_energy + ecoul;
    return jk;
  }

  inline bool have_nonlocal_correlation() const {
    for (const auto &func : m_funcs.unpolarized) {
      if (func.needs_nlc_correction()) {
        return true;
      }
    }

    return false;
  }

  inline double post_scf_nlc_correction(const MolecularOrbitals &mo) {
    if (have_nonlocal_correlation()) {
      auto nlc_result = m_nlc(m_hf.aobasis(), mo);
      m_nlc_energy = nlc_result.energy;
    }
    return m_nlc_energy;
  }

  Mat compute_fock(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) {
    auto [J, K] = compute_JK(mo, Schwarz);
    return J - K;
  }

  const auto &hf() const { return m_hf; }

  inline Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs,
                                      const qm::AOBasis &bs,
                                      bool is_shell_diagonal) {
    return m_hf.compute_fock_mixed_basis(mo_bs, bs, is_shell_diagonal);
  }

  inline Vec
  electronic_electric_potential_contribution(const MolecularOrbitals &mo,
                                             const Mat3N &pts) const {
    return m_hf.electronic_electric_potential_contribution(mo, pts);
  }

  inline Mat3N
  electronic_electric_field_contribution(const MolecularOrbitals &mo,
                                         const Mat3N &pts) const {
    return m_hf.electronic_electric_field_contribution(mo, pts);
  }

  void update_core_hamiltonian(const MolecularOrbitals &mo, Mat &H) { return; }

  void set_method(const std::string &method_string);

  inline const std::string &method_string() const { return m_method_string; }

  inline const auto &functionals() const { return m_funcs; }

  inline std::string name() const { return method_string(); }

private:
  std::string m_method_string{"svwn5"};
  occ::qm::HartreeFock m_hf;
  MolecularGrid m_grid;
  Functionals m_funcs;
  std::vector<AtomGrid> m_atom_grids;
  NonLocalCorrelationFunctional m_nlc;
  mutable double m_two_electron_energy{0.0};
  mutable double m_exc_dft{0.0};
  mutable double m_exchange_energy{0.0};
  mutable double m_nlc_energy{0.0};
  double m_density_threshold{1e-10};
  RangeSeparatedParameters m_rs_params;
};
} // namespace occ::dft
