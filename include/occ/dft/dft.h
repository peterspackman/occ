#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/dft/dft_gradient_kernels.h>
#include <occ/dft/dft_kernels.h>
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
    return m_method.exchange_factor();
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

    size_t num_rows_factor =
        (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;

    double total_density_a{0.0}, total_density_b{0.0};
    const Mat D2 = 2 * D;

    std::vector<Mat> Kt(occ::parallel::nthreads, Mat::Zero(D.rows(), D.cols()));
    std::vector<double> energies(occ::parallel::nthreads, 0.0);
    std::vector<double> alpha_densities(occ::parallel::nthreads, 0.0);
    std::vector<double> beta_densities(occ::parallel::nthreads, 0.0);

    const auto &funcs = (spinorbital_kind == SpinorbitalKind::Unrestricted)
                            ? m_method.functionals_polarized
                            : m_method.functionals;

    for (const auto &func : funcs) {
      occ::log::debug("vxc functional: {}, polarized: {}", func.name(),
                      func.polarized());
    }

    occ::timing::start(occ::timing::category::dft_xc);

    const auto &molecular_grid = m_grid.get_molecular_grid_points();
    const auto &all_points = molecular_grid.points();
    const auto &all_weights = molecular_grid.weights();
    const size_t npt_total = all_points.cols();

    const size_t num_blocks = (npt_total + m_blocksize - 1) / m_blocksize;

    occ::log::debug("Processing {} grid points in {} blocks", npt_total,
                    num_blocks);

    auto lambda = [&](int thread_id) {
      for (size_t block = 0; block < num_blocks; block++) {
        if (block % nthreads != thread_id)
          continue;

        Eigen::Index l = block * m_blocksize;
        Eigen::Index u =
            std::min<Eigen::Index>(npt_total, (block + 1) * m_blocksize);
        Eigen::Index npt = u - l;

        if (npt <= 0)
          continue;

        const auto &pts_block = all_points.middleCols(l, npt);
        const auto &weights_block = all_weights.segment(l, npt);

        auto gto_vals =
            occ::gto::evaluate_basis(basis, pts_block, derivative_order);

        kernels::process_grid_block<derivative_order, spinorbital_kind>(
            D2, gto_vals, pts_block, weights_block, funcs, Kt[thread_id],
            energies[thread_id], alpha_densities[thread_id],
            beta_densities[thread_id], m_density_threshold);
      }
    };

    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::dft_xc);

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
    for (const auto &func : m_method.functionals) {
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

  MatTriple compute_fock_gradient(const MolecularOrbitals &mo,
                                  const Mat &Schwarz = Mat()) const;

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

  /**
   * Calculate the DFT exchange-correlation contribution to the atomic gradient
   *
   * @param mo Molecular orbitals
   * @return Matrix of gradients with dimensions 3 x natoms
   */
  Mat3N compute_xc_gradient(const MolecularOrbitals &mo,
                            const Mat &Schwarz = Mat()) const;

  inline Mat3N additional_atomic_gradients(const MolecularOrbitals &mo) const {
    return compute_xc_gradient(mo);
  }

  /**
   * Calculate the nuclear repulsion contribution to the atomic gradient
   *
   * @return Matrix of gradients with dimensions 3 x natoms
   */
  Mat3N nuclear_repulsion_gradient() const {
    return m_hf.nuclear_repulsion_gradient();
  }

  /**
   * Calculate gradients for specific DFT components
   */
  MatTriple compute_overlap_gradient() const {
    return m_hf.compute_overlap_gradient();
  }

  MatTriple compute_kinetic_gradient() const {
    return m_hf.compute_kinetic_gradient();
  }

  MatTriple compute_nuclear_attraction_gradient() const {
    return m_hf.compute_nuclear_attraction_gradient();
  }

  MatTriple compute_rinv_gradient_for_atom(size_t atom) const {
    return m_hf.compute_rinv_gradient_for_atom(atom);
  }

  qm::JKTriple compute_JK_gradient(const MolecularOrbitals &mo,
                                   const Mat &Schwarz = Mat()) const;

  MatTriple compute_J_gradient(const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) const {
    return m_hf.compute_J_gradient(mo, Schwarz);
  }

  void update_core_hamiltonian(const MolecularOrbitals &mo, Mat &H) { return; }

  void set_method(const std::string &method_string);

  inline const std::string &method_string() const { return m_method_string; }

  inline const auto &functionals() const { return m_method.functionals; }

  inline std::string name() const { return method_string(); }

  inline void set_block_size(size_t blocksize) { m_blocksize = blocksize; }

private:
  template <int derivative_order, SpinorbitalKind spinorbital_kind>
  Mat3N compute_xc_gradient_impl(const MolecularOrbitals &mo,
                                 const Mat &Schwarz) const {
    using occ::parallel::nthreads;
    const auto &basis = m_hf.aobasis();
    const auto &atoms = m_hf.atoms();
    const size_t natoms = atoms.size();
    const size_t nbf = basis.nbf();
    const auto D = 2.0 * mo.D;

    Mat3N gradient = Mat3N::Zero(3, natoms);

    std::vector<Mat3N> gradients_t(nthreads, Mat3N::Zero(3, natoms));

    const auto &funcs = (spinorbital_kind == SpinorbitalKind::Unrestricted)
                            ? m_method.functionals_polarized
                            : m_method.functionals;

    for (const auto &func : funcs) {
      occ::log::debug("vxc functional: {}, polarized: {}", func.name(),
                      func.polarized());
    }

    occ::timing::start(occ::timing::category::dft_gradient);

    const auto &molecular_grid = m_grid.get_molecular_grid_points();
    const auto &all_points = molecular_grid.points();
    const auto &all_weights = molecular_grid.weights();
    const size_t npt_total = all_points.cols();

    const size_t num_blocks = (npt_total + m_blocksize - 1) / m_blocksize;

    occ::log::debug(
        "Processing {} grid points in {} blocks for gradient calculation",
        npt_total, num_blocks);

    auto lambda = [&](int thread_id) {
      for (size_t block = 0; block < num_blocks; block++) {
        if (block % nthreads != thread_id)
          continue;

        Eigen::Index l = block * m_blocksize;
        Eigen::Index u =
            std::min<Eigen::Index>(npt_total, (block + 1) * m_blocksize);
        Eigen::Index npt = u - l;

        if (npt <= 0)
          continue;

        const auto &pts_block = all_points.middleCols(l, npt);
        const auto &weights_block = all_weights.segment(l, npt);

        auto gto_vals =
            occ::gto::evaluate_basis(basis, pts_block, 1 + derivative_order);

        kernels::process_grid_block_gradient<derivative_order,
                                             spinorbital_kind>(
            D, gto_vals, pts_block, weights_block, funcs,
            gradients_t[thread_id], m_density_threshold, basis);
      }
    };

    occ::parallel::parallel_do(lambda);

    // Combine results from all threads
    for (size_t i = 0; i < nthreads; i++) {
      gradient += gradients_t[i];
    }

    occ::timing::stop(occ::timing::category::dft_gradient);
    return gradient;
  }

  std::string m_method_string{"svwn5"};
  occ::qm::HartreeFock m_hf;
  MolecularGrid m_grid;
  DFTMethod m_method;
  NonLocalCorrelationFunctional m_nlc;
  mutable double m_two_electron_energy{0.0};
  mutable double m_exc_dft{0.0};
  mutable double m_exchange_energy{0.0};
  mutable double m_nlc_energy{0.0};
  double m_density_threshold{1e-10};
  RangeSeparatedParameters m_rs_params;
  size_t m_blocksize{64};
};
} // namespace occ::dft
