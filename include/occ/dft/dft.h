#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/dft/dft_gradient_kernels.h>
#include <occ/dft/dft_kernels.h>
#include <occ/dft/dft_method.h>
#include <occ/dft/functional.h>
#include <occ/numint/grid_types.h>
#include <occ/numint/grid_utils.h>
#include <occ/numint/molecular_grid.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/dft/range_separated_parameters.h>
#include <occ/dft/xc_potential_matrix.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/qm/scf_method.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>
#include <cstdlib>
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
  DFT(const std::string &, const gto::AOBasis &, const occ::numint::GridSettings & = {});
  inline const auto &aobasis() const { return m_hf.aobasis(); }
  inline auto nbf() const { return m_hf.nbf(); }

  void set_integration_grid(const occ::numint::GridSettings & = {});
  void set_nlc_grid(const gto::AOBasis &basis, const occ::numint::GridSettings &settings = {110, 50, 50, 1e-7, false});

  inline void
  set_density_fitting_basis(const std::string &density_fitting_basis,
                            double auto_aux_threshold = 1e-4) {
    m_hf.set_density_fitting_basis(density_fitting_basis, auto_aux_threshold);
  }

  inline void set_density_fitting_policy(qm::IntegralEngineDF::Policy policy) {
    m_hf.set_density_fitting_policy(policy);
  }

  inline void set_density_fitting_memory_limit(size_t bytes) {
    m_hf.set_density_fitting_memory_limit(bytes);
  }

  inline void set_coulomb_method(qm::CoulombMethod method) {
    m_hf.set_coulomb_method(method);
  }

  // Seminumerical (COSX) exchange for the exact-exchange (hybrid) component.
  // DFT computes its exact exchange via m_hf.compute_JK, which routes K through
  // the COSX engine when one is set, so this just forwards to the HF object.
  inline void
  set_cosx_exchange(occ::numint::COSXGridLevel level = occ::numint::COSXGridLevel::Grid1) {
    m_hf.set_cosx_exchange(level);
  }

  inline void
  set_cosx_grid_schedule(const qm::COSXGridSchedule &s = {}) {
    m_hf.set_cosx_grid_schedule(s);
  }

  /// SCF hooks for the COSX grid schedule (see HartreeFock).
  inline bool advance_scf_grid(double diis_error, bool converged) {
    return m_hf.advance_scf_grid(diis_error, converged);
  }
  inline bool scf_fock_approximate() const {
    return m_hf.scf_fock_approximate();
  }
  inline void set_scf_fock_incremental(bool incremental) {
    m_hf.set_scf_fock_incremental(incremental);
  }
  inline void refresh_scf_fock_correction() {
    m_hf.refresh_scf_fock_correction();
  }
  inline double incremental_fock_floor() const {
    return m_hf.incremental_fock_floor();
  }
  inline double final_energy_correction(const MolecularOrbitals &mo) {
    return m_hf.finalize_cosx_grid(mo, exchange_operators());
  }
  inline double cosx_energy_shift_from(occ::numint::COSXGridLevel other,
                                       const MolecularOrbitals &mo) {
    return m_hf.cosx_energy_shift_from(other, mo, exchange_operators());
  }
  inline bool using_cosx() const { return m_hf.using_cosx(); }
  inline bool using_density_fitting() const {
    return m_hf.using_density_fitting();
  }
  /// Exact-exchange operators (c_t, omega_t) in the energy.
  qm::HartreeFock::ExchangeOperators exchange_operators() const {
    const RangeSeparatedParameters rs = range_separated_parameters();
    if (rs.omega != 0.0)
      return {{rs.alpha + rs.beta, 0.0}, {-rs.beta, rs.omega}};
    return {{exact_exchange_factor(), 0.0}};
  }

  inline void set_cosx_settings(const occ::qm::cosx::Settings &settings) {
    m_hf.set_cosx_settings(settings);
  }

  inline void set_precision(double precision) { m_hf.set_precision(precision); }
  
  inline double integral_precision() const { return m_hf.integral_precision(); }
  
  /**
   * @brief Create a new DFT instance with the same settings but different basis
   * @param new_basis The new basis set to use  
   * @return New DFT instance
   */
  DFT with_new_basis(const gto::AOBasis &new_basis) const;

  double exchange_correlation_energy() const { return m_exc_dft; }
  double exchange_energy_total() const { return m_exchange_energy; }

  bool usual_scf_energy() const { return false; }
  void update_scf_energy(occ::core::EnergyComponents &energy) const {
    energy["electronic"] = energy["electronic.1e"];
    energy["electronic.2e"] = m_two_electron_energy;
    energy["electronic"] += m_two_electron_energy;
    energy["electronic.dft_xc"] = exchange_correlation_energy();

    if (m_nlc_energy != 0.0) {
      energy["electronic.nonlocal_correlation"] = m_nlc_energy;
    }

    double total = energy["electronic"] + energy["nuclear.repulsion"];
    // Fold in any external-potential nuclear contribution recorded by SCF
    // (`nuclear.<label>` — see `SCF::set_external_potential`). Skip
    // `nuclear.repulsion`, which is already counted above.
    for (const auto &[key, value] : energy) {
      if (key.size() > 8 && key.compare(0, 8, "nuclear.") == 0 &&
          key != "nuclear.repulsion") {
        total += value;
      }
    }
    energy["total"] = total;
  }
  qm::FockBuildProperties fock_build_properties() const {
    // The XC contribution is a quadrature over the density itself, so the
    // two-electron build is not a linear function of D.
    return {.linear_in_density = false};
  }
  inline bool have_effective_core_potentials() const {
    return m_hf.have_effective_core_potentials();
  }

  int density_derivative() const;
  /// Whether any component depends on the density Laplacian.
  bool needs_laplacian() const;
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

  auto compute_overlap_matrix_for_basis(const occ::gto::AOBasis &bs) const {
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

  double wolf_point_charge_interaction_energy(
      const PointChargeList &point_charges,
      const std::vector<double> &molecular_charges, double alpha,
      double cutoff) const {
    return m_hf.wolf_point_charge_interaction_energy(point_charges,
                                                      molecular_charges, alpha,
                                                      cutoff);
  }

  auto compute_wolf_interaction_matrix(
      const PointChargeList &point_charges,
      const std::vector<double> &molecular_charges, double alpha,
      double cutoff) const {
    return m_hf.compute_wolf_interaction_matrix(point_charges,
                                                 molecular_charges, alpha,
                                                 cutoff);
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

    double total_density_a{0.0}, total_density_b{0.0};
    const Mat D2 = 2 * D;
    constexpr int nspin_d =
        (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;
    // D2 = F F^T per spin from the occupied orbitals (empty if mo.D is not
    // that product): batches with more functions than twice the occupied
    // count form their products with D through F.
    const Mat F2 = kernels::density_factor(mo, D2, nspin_d);

    const auto &funcs = (spinorbital_kind == SpinorbitalKind::Unrestricted)
                            ? m_method.functionals_polarized
                            : m_method.functionals;

    for (const auto &func : funcs) {
      occ::log::debug("vxc functional: {}, polarized: {}", func.name(),
                      func.polarized());
    }

    occ::timing::start(occ::timing::category::dft_xc);

    const auto &molecular_grid = m_grid.get_molecular_grid_points();

    // Consolidated thread-local storage with better alignment
    struct ThreadLocalData {
      Mat K_local;
      double energy_local{0.0};
      double alpha_density_local{0.0};
      double beta_density_local{0.0};

      ThreadLocalData(size_t rows, size_t cols) : K_local(Mat::Zero(rows, cols)) {}
    };

    occ::parallel::thread_local_storage<ThreadLocalData> thread_data(
        [K_rows, K_cols]() { return ThreadLocalData(K_rows, K_cols); }
    );

    // Dense (full-nbf) path: used for unrestricted, and as a screening-off
    // fallback (set OCC_XC_NO_SCREEN=1) for benchmarking / regression checks.
    auto run_dense = [&]() {
      const auto &all_points = molecular_grid.points();
      const auto &all_weights = molecular_grid.weights();
      const size_t npt_total = all_points.cols();

      occ::log::debug("Processing {} grid points with adaptive blocking",
                      npt_total);

      // Let TBB determine optimal grain size based on work complexity
      const size_t min_points_per_chunk = std::max(size_t(32), nbf / 4);
      const size_t max_chunks = nthreads * 8; // Allow good work stealing
      size_t adaptive_grain =
          std::max(min_points_per_chunk, npt_total / max_chunks);

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, npt_total, adaptive_grain),
          [&](const tbb::blocked_range<size_t> &range) {
            const size_t l = range.begin();
            const size_t u = range.end();
            const size_t npt = u - l;

            if (npt == 0)
              return;

            const auto &pts_block = all_points.middleCols(l, npt);
            const auto &weights_block = all_weights.segment(l, npt);

            auto gto_vals =
                occ::gto::evaluate_basis(basis, pts_block, derivative_order);
            auto &data = thread_data.local();

            Mat H(K_rows, K_cols);
            const kernels::BatchDensity dens{&D2, nullptr,
                                             static_cast<Eigen::Index>(nbf)};
            if (kernels::process_grid_block<derivative_order,
                                            spinorbital_kind>(
                    dens, gto_vals, pts_block, weights_block, funcs, H,
                    data.energy_local, data.alpha_density_local,
                    data.beta_density_local, m_density_threshold)) {
              for (size_t r0 = 0; r0 < K_rows; r0 += nbf) {
                auto h = H.middleRows(r0, nbf);
                data.K_local.middleRows(r0, nbf) += h + h.transpose();
              }
            }
          });
    };

    const bool use_screen = (m_xc_screening_threshold > 0.0) &&
                            (std::getenv("OCC_XC_NO_SCREEN") == nullptr);
    if (!use_screen) {
      run_dense();
    } else {
      // ---- block-sparse screened path (restricted & unrestricted) ---------
      // Iterate spatially-compact grid batches (Morton-ordered leaves with a
      // precomputed bounding sphere). For each batch keep only the shells whose
      // decay radius (|phi| > screening threshold) reaches the batch sphere,
      // then run collocation + density + Vxc on the compact (nbf_local)
      // matrices and scatter back into K. Shrinks the AMX-bound GEMMs from nbf
      // to nbf_local (-> constant for large systems): the lever that helps an
      // AMX-bound kernel, and it restores thread-scaling by shrinking the
      // non-scaling GEMM fraction.
      constexpr int nspin =
          (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;
      const auto &shells = basis.shells();
      const size_t nsh = basis.nsh();
      const Vec extents =
          occ::gto::evaluate_decay_cutoff(basis, m_xc_screening_threshold);
      const auto &hier = molecular_grid.get_hierarchy();
      const auto &spts = hier.sorted_points();
      const auto &swts = hier.sorted_weights();
      const size_t nleaves = hier.num_leaves();
      occ::log::debug("Processing {} grid points in {} spatial leaves "
                      "(screening tol {:.0e})",
                      spts.cols(), nleaves, m_xc_screening_threshold);

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, nleaves),
          [&](const tbb::blocked_range<size_t> &range) {
            auto &data = thread_data.local();
            for (size_t li = range.begin(); li != range.end(); ++li) {
              const auto &leaf = hier.leaves()[li];
              if (leaf.count == 0)
                continue;
              auto pts_block = spts.middleCols(leaf.offset, leaf.count);

              // significant shells for this batch (ascending shell order)
              std::vector<int> sig;
              sig.reserve(nsh);
              for (size_t s = 0; s < nsh; ++s) {
                double dist = (shells[s].origin - leaf.bounds.center).norm();
                if (dist - extents(s) < leaf.bounds.radius)
                  sig.push_back(static_cast<int>(s));
              }
              if (sig.empty())
                continue;

              const auto segs = basis_runs(basis, sig);
              const int nbf_local = segs.back().l0 + segs.back().len;
              Mat Dloc, Floc;
              const auto dens =
                  gather_batch_density(D2, F2, segs, nspin, Dloc, Floc);

              auto gto_vals = occ::gto::evaluate_basis_subset(
                  basis, pts_block, derivative_order, sig);

              Mat Kblk(nspin * nbf_local, nbf_local);
              if (!kernels::process_grid_block<derivative_order,
                                               spinorbital_kind>(
                      dens, gto_vals, pts_block,
                      swts.segment(leaf.offset, leaf.count), funcs, Kblk,
                      data.energy_local, data.alpha_density_local,
                      data.beta_density_local, m_density_threshold))
                continue;

              // Scatter the lower triangle of Kblk + Kblk^T into the
              // thread-local K, per spin block: half the traffic into K,
              // which for large systems does not stay in cache.
              for (int sp = 0; sp < nspin; ++sp) {
                const int goff = sp * static_cast<int>(nbf);
                const int loff = sp * nbf_local;
                for (size_t ia = 0; ia < segs.size(); ia++) {
                  const auto &a = segs[ia];
                  for (size_t ib = 0; ib <= ia; ib++) {
                    const auto &b = segs[ib];
                    data.K_local.block(goff + a.g0, b.g0, a.len, b.len)
                        .noalias() +=
                        Kblk.block(loff + a.l0, b.l0, a.len, b.len) +
                        Kblk.block(loff + b.l0, a.l0, b.len, a.len)
                            .transpose();
                  }
                }
              }
            }
          });
    }

    for (const auto &data : thread_data) {
      K.noalias() += data.K_local;
      m_exc_dft += data.energy_local;
      total_density_a += data.alpha_density_local;
      total_density_b += data.beta_density_local;
    }
    // The batches accumulate the lower triangle of V_xc (per spin block).
    for (size_t r0 = 0; r0 < K_rows; r0 += nbf) {
      auto blk = K.middleRows(r0, nbf);
      const Mat full = blk.template selfadjointView<Eigen::Lower>();
      blk = full;
    }

    occ::timing::stop(occ::timing::category::dft_xc);

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
      // Range-separated hybrid: the DFT (libxc) exchange already covers the
      // range-complement part; HF supplies the exact-exchange part
      //   K_HF = (alpha+beta)*K[1/r] - beta*K[erf(omega*r)/r],
      // and owns the choice of the most efficient build for it.
      qm::JKPair jk_hf = m_hf.coulomb_and_range_separated_exchange(
          mo, rs.omega, rs.alpha, rs.beta, Schwarz);
      jk.J = jk_hf.J;
      exc = -expectation(mo.kind, mo.D, jk_hf.K);
      jk.K.noalias() += jk_hf.K;
    } else if (exchange_factor != 0.0) {
      // global hybrid: K enters F scaled by the exchange fraction, and COSX
      // screens it accordingly
      m_hf.set_cosx_exchange_scale(exchange_factor);
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

  // WARNING: VV10/NLC gradients are not fully tested and may not be correct.
  // Current implementation is post-SCF only (not self-consistent) and
  // does not include SCF response. Use with caution.
  Mat3N compute_nlc_gradient(const MolecularOrbitals &mo) const {
    if (!have_nonlocal_correlation()) {
      const size_t natoms = m_hf.atoms().size();
      return Mat3N::Zero(3, natoms);
    }
    auto nlc_result = m_nlc.compute_gradient(m_hf.aobasis(), mo);
    return nlc_result.gradient;
  }

  Mat compute_fock(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) {
    auto [J, K] = compute_JK(mo, Schwarz);
    return J - K;
  }

  MatTriple compute_fock_gradient(const MolecularOrbitals &mo,
                                  const Mat &Schwarz = Mat());

  /// XC potential response V1[D1] for each first-order (per-spin) density,
  /// the exact derivative of this functional's V_xc at mo. Restricted LDA,
  /// GGA and meta-GGA (tau only), including hybrids' DFT part.
  std::vector<Mat> compute_xc_response(const MolecularOrbitals &mo,
                                       const std::vector<Mat> &D1) const;

  // --- Analytic Hessian building blocks -----------------------------------

  /// Empty if the analytic Hessian supports this calculation, otherwise why
  /// not (the caller falls back to finite differences).
  std::string analytic_hessian_unsupported(SpinorbitalKind kind) const {
    if (auto why = m_hf.analytic_hessian_unsupported(kind); !why.empty())
      return why;
    if (needs_laplacian())
      return "Laplacian-dependent meta-GGA functionals are not supported";
    if (kind != SpinorbitalKind::Restricted)
      for (const auto &f : m_method.functionals)
        if (f.family() == DensityFunctional::MGGA ||
            f.family() == DensityFunctional::HMGGA)
          return "unrestricted meta-GGA functionals are not supported";
    for (const auto &f : m_method.functionals)
      if (f.needs_nlc_correction())
        return "VV10 nonlocal correlation is not supported";
    return {};
  }

  Mat compute_one_electron_hessian(const Mat &P, const Mat &Q) const {
    return m_hf.compute_one_electron_hessian(P, Q);
  }
  void compute_one_electron_derivative_matrices(std::vector<Mat> &S1,
                                                std::vector<Mat> &H1) const {
    m_hf.compute_one_electron_derivative_matrices(S1, H1);
  }

  /// Skeleton XC Hessian: second derivatives of E_xc at fixed AO density,
  /// grid response included (points moving with their atoms, first and
  /// second partition-weight derivatives). LDA and GGA, restricted or
  /// unrestricted; meta-GGA (tau only) restricted.
  Mat compute_xc_hessian(const MolecularOrbitals &mo) const;

  /// Skeleton two-electron + XC Hessian (fixed density): Coulomb, the
  /// functional's exact exchange (global or range-separated) and XC.
  Mat two_electron_hessian(const MolecularOrbitals &mo,
                           const Mat &Schwarz = Mat()) const;

  /// Skeleton two-electron + XC Fock derivatives for every coordinate.
  std::vector<Mat> two_electron_derivative_fock(const MolecularOrbitals &mo,
                                                const Mat &Schwarz = Mat()) const;

  /// dV_xc/dR for every nuclear coordinate (index 3 * atom + xyz) at the
  /// fixed AO density of mo, including the grid response (points moving with
  /// their atoms, partition-weight derivatives). Restricted LDA, GGA and
  /// meta-GGA (tau only).
  std::vector<Mat> compute_xc_derivative_fock(const MolecularOrbitals &mo) const;
  /// The same contracted with the occupied coefficients, (dV_xc/dR) C_o
  /// (nbf x nocc each), on screened basis subsets: what the Hessian uses.
  /// Restricted or unrestricted (layout of cphf::occupied_orbitals).
  std::vector<Mat>
  compute_xc_derivative_fock_occ(const MolecularOrbitals &mo) const;
  /// Skeleton two-electron + XC Fock derivatives contracted with C_o.
  std::vector<Mat>
  two_electron_derivative_fock_occ(const MolecularOrbitals &mo,
                                   const Mat &Schwarz = Mat()) const;
  /// The functional's exact exchange part of d/dR (1/2 dE/dD - F) C_o: see
  /// HartreeFock::compute_exchange_nonvariational_fock_occ.
  std::vector<Mat>
  nonvariational_fock_derivative_occ(const MolecularOrbitals &mo) const {
    return m_hf.compute_exchange_nonvariational_fock_occ(mo,
                                                         exchange_operators());
  }

  /// Linear response of the full two-electron Kohn-Sham Fock matrix at mo:
  /// Coulomb, the functional's exact exchange (global or range-separated)
  /// and the XC kernel. The operation a CPKS solver needs.
  std::vector<Mat> compute_response_fock(const MolecularOrbitals &mo,
                                         const std::vector<Mat> &D1,
                                         const Mat &Schwarz = Mat()) const;

  /// The CPHF response, G[D1] C_o for D1 = Cx C_o^T + C_o Cx^T (see
  /// HartreeFock::response_occ): Coulomb and the functional's exact exchange
  /// with the SCF's operators, plus the XC kernel, contracted with C_o
  /// point by point (O(npt nbf nocc) per density).
  std::vector<Mat> compute_response_fock_occ(const MolecularOrbitals &mo,
                                             const std::vector<Mat> &Cx,
                                             const Mat &Schwarz = Mat()) const;
  /// The approximate response (RI-J, four-centre exchange, XC on a coarse
  /// grid) that CPHF preconditions with; always cheaper for DFT.
  std::vector<Mat>
  compute_response_fock_occ_approx(const MolecularOrbitals &mo,
                                   const std::vector<Mat> &Cx,
                                   const Mat &Schwarz = Mat()) const;
  bool has_approximate_response() const { return true; }
  /// The XC kernel's part of the CPHF response; `grid` selects another
  /// integration grid (default: the SCF's).
  std::vector<Mat>
  compute_xc_response_occ(const MolecularOrbitals &mo,
                          const std::vector<Mat> &Cx,
                          const occ::numint::MolecularGrid *grid = nullptr) const;

  /// Two-electron part of the nuclear gradient: Coulomb plus the functional's
  /// exact exchange (global or range-separated), contracted in the integral
  /// loop. 3 x natom.
  Mat3N two_electron_gradient(const MolecularOrbitals &mo,
                              const Mat &Schwarz = Mat()) const;

  const auto &hf() const { return m_hf; }
  inline const qm::IntegralEngine &integral_engine() const {
    return m_hf.integral_engine();
  }

  /// Forwarded to Hartree-Fock: this builds a starting density's Fock matrix,
  /// where exact exchange stands in for the functional perfectly well and
  /// saves constructing an XC grid before the first real iteration.
  inline Mat compute_fock_from_density(const MolecularOrbitals &mo,
                                       const Mat &Schwarz = Mat()) const {
    return m_hf.compute_fock_from_density(mo, Schwarz);
  }

  inline Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs,
                                      const gto::AOBasis &bs,
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

  inline Mat3N compute_ecp_gradient(const MolecularOrbitals &mo) const {
    return m_hf.compute_ecp_gradient(mo);
  }

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
                                   const Mat &Schwarz = Mat());

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

  /// Per-grid-batch shell screening tolerance for the XC build: the |phi|
  /// decay cutoff used to drop negligible basis functions over a spatial batch.
  /// Larger = more aggressive screening / faster, less accurate. <=0 disables
  /// screening (dense XC build).
  inline void set_xc_screening_threshold(double t) {
    m_xc_screening_threshold = t;
  }
  inline double xc_screening_threshold() const {
    return m_xc_screening_threshold;
  }

  /// Include the grid response in XC gradients (on by default): grid points
  /// move with their parent atom, and the Becke partition weights are
  /// differentiated. Without it the gradient is not translationally invariant.
  inline void set_grid_response(bool on) { m_grid_response = on; }
  inline bool grid_response() const { return m_grid_response; }

private:
  // A screened batch's basis functions as runs of consecutive functions:
  // consecutive significant shells are merged, so gathers and scatters copy
  // whole runs (typically whole atoms) rather than shell-pair blocks. g0 is
  // a run's first global function, l0 its first local one.
  struct BasisRun {
    int g0, l0, len;
  };
  static std::vector<BasisRun> basis_runs(const occ::gto::AOBasis &basis,
                                          const std::vector<int> &shells) {
    const auto &shell_list = basis.shells();
    const auto &first_bf = basis.first_bf();
    std::vector<BasisRun> runs;
    int nbf_local = 0;
    for (int s : shells) {
      const int len = static_cast<int>(shell_list[s].size());
      if (!runs.empty() && runs.back().g0 + runs.back().len == first_bf[s])
        runs.back().len += len;
      else
        runs.push_back({first_bf[s], nbf_local, len});
      nbf_local += len;
    }
    return runs;
  }

  // The batch's density matrix on its runs of functions: the blocks of D2,
  // or the rows of its factor F2 (D2_s = F2_s F2_s^T) when that has fewer
  // than half as many columns as the batch has functions. Spin block s
  // occupies rows [s nbf, ...) of D2 and F2, [s nbf_local, ...) locally.
  static kernels::BatchDensity
  gather_batch_density(const Mat &D2, const Mat &F2,
                       const std::vector<BasisRun> &runs, int nspin, Mat &Dloc,
                       Mat &Floc) {
    const int nbf = static_cast<int>(D2.cols());
    const int nbf_local = runs.back().l0 + runs.back().len;
    const bool factor = F2.size() > 0 && 2 * F2.cols() < nbf_local;
    if (factor)
      Floc.resize(nspin * nbf_local, F2.cols());
    else
      Dloc.resize(nspin * nbf_local, nbf_local);
    for (int sp = 0; sp < nspin; ++sp) {
      const int goff = sp * nbf, loff = sp * nbf_local;
      for (const auto &a : runs) {
        if (factor) {
          Floc.middleRows(loff + a.l0, a.len) =
              F2.middleRows(goff + a.g0, a.len);
          continue;
        }
        for (const auto &b : runs)
          Dloc.block(loff + a.l0, b.l0, a.len, b.len) =
              D2.block(goff + a.g0, b.g0, a.len, b.len);
      }
    }
    return {factor ? nullptr : &Dloc, factor ? &Floc : nullptr, nbf_local};
  }

  occ::numint::BeckeWeightDerivative becke_weight_derivative() const;
  Mat3N atom_positions() const;

  template <int derivative_order, SpinorbitalKind spinorbital_kind>
  Mat3N compute_xc_gradient_impl(const MolecularOrbitals &mo,
                                 const Mat &Schwarz) const {
    using occ::parallel::nthreads;
    const auto &basis = m_hf.aobasis();
    const auto &atoms = m_hf.atoms();
    const size_t natoms = atoms.size();
    const size_t nbf = basis.nbf();
    const Mat D = 2.0 * mo.D;
    constexpr int nspin_d =
        (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;
    const Mat F2 = kernels::density_factor(mo, D, nspin_d);

    Mat3N gradient = Mat3N::Zero(3, natoms);

    const auto &funcs = (spinorbital_kind == SpinorbitalKind::Unrestricted)
                            ? m_method.functionals_polarized
                            : m_method.functionals;

    for (const auto &func : funcs) {
      occ::log::debug("vxc functional: {}, polarized: {}", func.name(),
                      func.polarized());
    }

    const auto &molecular_grid = m_grid.get_molecular_grid_points();

    // Thread-local storage for gradients
    occ::parallel::thread_local_storage<Mat3N> gradients_local(
        [natoms]() { return Mat3N::Zero(3, natoms); }
    );

    // Gradients need one more derivative than the functional order:
    // LDA -> 1st derivs, GGA/MGGA -> 2nd derivs.
    const int max_deriv = (derivative_order == 0) ? 1 : 2;

    // Grid response. Every point moves with its parent atom: by translational
    // invariance that costs minus the point's orbital term, which the kernel
    // adds to the parent given per point. The per-point energy density is
    // kept for the partition-weight derivative.
    const auto &atom_blocks = molecular_grid.atom_blocks();
    const size_t npt_total = molecular_grid.num_points();
    std::vector<int> point_atom(npt_total);
    for (size_t a = 0; a < atom_blocks.size(); a++)
      for (size_t i = 0; i < atom_blocks[a].second; i++)
        point_atom[atom_blocks[a].first + i] = static_cast<int>(a);
    Vec energy_density = Vec::Zero(npt_total);
    const bool grid_response = m_grid_response;

    auto run_dense = [&]() {
      const auto &all_points = molecular_grid.points();
      const auto &all_weights = molecular_grid.weights();

      occ::log::debug(
          "Processing {} grid points for gradient with adaptive blocking",
          npt_total);

      const size_t min_points_per_chunk = std::max(size_t(64), nbf / 2);
      const size_t max_chunks = nthreads * 6;
      size_t adaptive_grain =
          std::max(min_points_per_chunk, npt_total / max_chunks);
      const size_t nchunks = (npt_total + adaptive_grain - 1) / adaptive_grain;

      occ::parallel::parallel_for(size_t(0), nchunks, [&](size_t c) {
        const size_t start = c * adaptive_grain;
        const size_t count = std::min(adaptive_grain, npt_total - start);
        if (count == 0)
          return;
        const auto pts_block = all_points.middleCols(start, count);
        const Vec weights_block = all_weights.segment(start, count);
        auto gto_vals = occ::gto::evaluate_basis(basis, pts_block, max_deriv);
        const kernels::BatchDensity dens{&D, nullptr,
                                         static_cast<Eigen::Index>(nbf)};
        kernels::process_grid_block_gradient<derivative_order,
                                             spinorbital_kind>(
            dens, gto_vals, pts_block, weights_block, funcs,
            gradients_local.local(), m_density_threshold, basis.bf_to_atom(),
            grid_response ? energy_density.data() + start : nullptr,
            grid_response ? point_atom.data() + start : nullptr);
      });
    };

    const bool use_screen = (m_xc_screening_threshold > 0.0) &&
                            (std::getenv("OCC_XC_NO_SCREEN") == nullptr);
    if (!use_screen) {
      run_dense();
    } else {
      // ---- block-sparse screened gradient (restricted & unrestricted) -----
      constexpr int nspin =
          (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;
      const auto &shells = basis.shells();
      const auto &shell_to_atom = basis.shell_to_atom();
      const size_t nsh = basis.nsh();
      const Vec extents =
          occ::gto::evaluate_decay_cutoff(basis, m_xc_screening_threshold);
      const auto &hier = molecular_grid.get_hierarchy();
      const auto &spts = hier.sorted_points();
      const auto &swts = hier.sorted_weights();
      const auto &perm = hier.permutation();
      const size_t nleaves = hier.num_leaves();

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, nleaves),
          [&](const tbb::blocked_range<size_t> &range) {
            auto &local_gradient = gradients_local.local();
            for (size_t li = range.begin(); li != range.end(); ++li) {
              const auto &leaf = hier.leaves()[li];
              if (leaf.count == 0)
                continue;

              std::vector<int> sig;
              sig.reserve(nsh);
              for (size_t s = 0; s < nsh; ++s) {
                double dist = (shells[s].origin - leaf.bounds.center).norm();
                if (dist - extents(s) < leaf.bounds.radius)
                  sig.push_back(static_cast<int>(s));
              }
              if (sig.empty())
                continue;

              std::vector<int> local_bf_to_atom;
              for (int s : sig)
                local_bf_to_atom.insert(local_bf_to_atom.end(),
                                        shells[s].size(), shell_to_atom[s]);
              const auto segs = basis_runs(basis, sig);
              Mat Dloc, Floc;
              const auto dens =
                  gather_batch_density(D, F2, segs, nspin, Dloc, Floc);

              const size_t n = leaf.count;
              auto pts = spts.middleCols(leaf.offset, n);
              const Vec wts = swts.segment(leaf.offset, n);
              std::vector<int> parent(n);
              for (size_t i = 0; i < n; i++)
                parent[i] = point_atom[perm[leaf.offset + i]];
              Vec edens = Vec::Zero(n);
              auto gto_vals =
                  occ::gto::evaluate_basis_subset(basis, pts, max_deriv, sig);
              kernels::process_grid_block_gradient<derivative_order,
                                                   spinorbital_kind>(
                  dens, gto_vals, pts, wts, funcs, local_gradient,
                  m_density_threshold, local_bf_to_atom,
                  grid_response ? edens.data() : nullptr,
                  grid_response ? parent.data() : nullptr);
              for (size_t i = 0; i < n; i++)
                energy_density(perm[leaf.offset + i]) = edens(i);
            }
          });
    }

    // Efficient gradient reduction
    for (const auto &local_gradient : gradients_local) {
        gradient.noalias() += local_gradient;
    }

    // Partition-weight response: sum_g e_g dw_g/dR.
    if (grid_response)
      gradient += occ::numint::becke_weight_gradient(
          molecular_grid.points(), molecular_grid.weights(), atom_blocks,
          energy_density, atom_positions(), m_grid.atomic_radii(),
          m_grid.settings().treutler_alrichs_adjustment);

    return gradient;
  }

  std::string m_method_string{"svwn5"};
  occ::qm::HartreeFock m_hf;
  occ::numint::MolecularGrid m_grid;
  /// Coarse grid for the approximate (preconditioning) CPHF response.
  mutable std::unique_ptr<occ::numint::MolecularGrid> m_coarse_grid;
  std::vector<Mat> response_occ(const MolecularOrbitals &mo,
                                const std::vector<Mat> &Cx, const Mat &Schwarz,
                                qm::HartreeFock::ResponseMode mode) const;
  bool m_grid_response{true};
  DFTMethod m_method;
  NonLocalCorrelationFunctional m_nlc;
  mutable double m_two_electron_energy{0.0};
  mutable double m_exc_dft{0.0};
  mutable double m_exchange_energy{0.0};
  mutable double m_nlc_energy{0.0};
  double m_density_threshold{1e-10};
  double m_xc_screening_threshold{1e-10};
  RangeSeparatedParameters m_rs_params;
  size_t m_blocksize{64};
};
} // namespace occ::dft
