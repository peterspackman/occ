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
#include <occ/numint/molecular_grid.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/dft/range_separated_parameters.h>
#include <occ/dft/xc_potential_matrix.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
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

            kernels::process_grid_block<derivative_order, spinorbital_kind>(
                D2, gto_vals, pts_block, weights_block, funcs, data.K_local,
                data.energy_local, data.alpha_density_local,
                data.beta_density_local, m_density_threshold);
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
      const auto &first_bf = basis.first_bf();
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

              // packed local bf segments (g0 = global first bf, l0 = local)
              struct Seg {
                int g0, l0, len;
              };
              int nbf_local = 0;
              std::vector<Seg> segs;
              segs.reserve(sig.size());
              for (int s : sig) {
                int len = static_cast<int>(shells[s].size());
                segs.push_back({first_bf[s], nbf_local, len});
                nbf_local += len;
              }

              // gather D_local from D2 per spin block. D2 has nbf columns for
              // both R and U; spin block sp occupies rows [sp*nbf, ...) in D2
              // and [sp*nbf_local, ...) in the packed local matrix.
              Mat Dloc(nspin * nbf_local, nbf_local);
              for (int sp = 0; sp < nspin; ++sp) {
                const int goff = sp * static_cast<int>(nbf);
                const int loff = sp * nbf_local;
                for (const auto &a : segs)
                  for (const auto &b : segs)
                    Dloc.block(loff + a.l0, b.l0, a.len, b.len) =
                        D2.block(goff + a.g0, b.g0, a.len, b.len);
              }

              auto gto_vals = occ::gto::evaluate_basis_subset(
                  basis, pts_block, derivative_order, sig);

              Mat Kblk = Mat::Zero(nspin * nbf_local, nbf_local);
              kernels::process_grid_block<derivative_order, spinorbital_kind>(
                  Dloc, gto_vals, pts_block,
                  swts.segment(leaf.offset, leaf.count), funcs, Kblk,
                  data.energy_local, data.alpha_density_local,
                  data.beta_density_local, m_density_threshold);

              // scatter Kblk into the thread-local K, per spin block
              for (int sp = 0; sp < nspin; ++sp) {
                const int goff = sp * static_cast<int>(nbf);
                const int loff = sp * nbf_local;
                for (const auto &a : segs)
                  for (const auto &b : segs)
                    data.K_local.block(goff + a.g0, b.g0, a.len, b.len)
                        .noalias() +=
                        Kblk.block(loff + a.l0, b.l0, a.len, b.len);
              }
            }
          });
    }

    // Efficient reduction with vectorized operations where possible
    for (const auto &data : thread_data) {
        K.noalias() += data.K_local;
        m_exc_dft += data.energy_local;
        total_density_a += data.alpha_density_local;
        total_density_b += data.beta_density_local;
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

  const auto &hf() const { return m_hf; }

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

    const auto &funcs = (spinorbital_kind == SpinorbitalKind::Unrestricted)
                            ? m_method.functionals_polarized
                            : m_method.functionals;

    for (const auto &func : funcs) {
      occ::log::debug("vxc functional: {}, polarized: {}", func.name(),
                      func.polarized());
    }

    occ::timing::start(occ::timing::category::dft_gradient);

    const auto &molecular_grid = m_grid.get_molecular_grid_points();

    // Thread-local storage for gradients
    occ::parallel::thread_local_storage<Mat3N> gradients_local(
        [natoms]() { return Mat3N::Zero(3, natoms); }
    );

    // Gradients need one more derivative than the functional order:
    // LDA -> 1st derivs, GGA/MGGA -> 2nd derivs.
    const int max_deriv = (derivative_order == 0) ? 1 : 2;

    auto run_dense = [&]() {
      const auto &all_points = molecular_grid.points();
      const auto &all_weights = molecular_grid.weights();
      const size_t npt_total = all_points.cols();

      occ::log::debug(
          "Processing {} grid points for gradient with adaptive blocking",
          npt_total);

      const size_t min_points_per_chunk = std::max(size_t(64), nbf / 2);
      const size_t max_chunks = nthreads * 6;
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
                occ::gto::evaluate_basis(basis, pts_block, max_deriv);
            auto &local_gradient = gradients_local.local();
            kernels::process_grid_block_gradient<derivative_order,
                                                 spinorbital_kind>(
                D, gto_vals, pts_block, weights_block, funcs, local_gradient,
                m_density_threshold, basis.bf_to_atom());
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
      const auto &first_bf = basis.first_bf();
      const auto &shell_to_atom = basis.shell_to_atom();
      const size_t nsh = basis.nsh();
      const Vec extents =
          occ::gto::evaluate_decay_cutoff(basis, m_xc_screening_threshold);
      const auto &hier = molecular_grid.get_hierarchy();
      const auto &spts = hier.sorted_points();
      const auto &swts = hier.sorted_weights();
      const size_t nleaves = hier.num_leaves();

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, nleaves),
          [&](const tbb::blocked_range<size_t> &range) {
            auto &local_gradient = gradients_local.local();
            for (size_t li = range.begin(); li != range.end(); ++li) {
              const auto &leaf = hier.leaves()[li];
              if (leaf.count == 0)
                continue;
              auto pts_block = spts.middleCols(leaf.offset, leaf.count);

              std::vector<int> sig;
              sig.reserve(nsh);
              for (size_t s = 0; s < nsh; ++s) {
                double dist = (shells[s].origin - leaf.bounds.center).norm();
                if (dist - extents(s) < leaf.bounds.radius)
                  sig.push_back(static_cast<int>(s));
              }
              if (sig.empty())
                continue;

              struct Seg {
                int g0, l0, len;
              };
              int nbf_local = 0;
              std::vector<Seg> segs;
              segs.reserve(sig.size());
              std::vector<int> local_bf_to_atom;
              for (int s : sig) {
                int len = static_cast<int>(shells[s].size());
                segs.push_back({first_bf[s], nbf_local, len});
                for (int k = 0; k < len; ++k)
                  local_bf_to_atom.push_back(shell_to_atom[s]);
                nbf_local += len;
              }

              Mat Dloc(nspin * nbf_local, nbf_local);
              for (int sp = 0; sp < nspin; ++sp) {
                const int goff = sp * static_cast<int>(nbf);
                const int loff = sp * nbf_local;
                for (const auto &a : segs)
                  for (const auto &b : segs)
                    Dloc.block(loff + a.l0, b.l0, a.len, b.len) =
                        D.block(goff + a.g0, b.g0, a.len, b.len);
              }

              auto gto_vals = occ::gto::evaluate_basis_subset(
                  basis, pts_block, max_deriv, sig);

              kernels::process_grid_block_gradient<derivative_order,
                                                   spinorbital_kind>(
                  Dloc, gto_vals, pts_block,
                  swts.segment(leaf.offset, leaf.count), funcs, local_gradient,
                  m_density_threshold, local_bf_to_atom);
            }
          });
    }

    // Efficient gradient reduction
    for (const auto &local_gradient : gradients_local) {
        gradient.noalias() += local_gradient;
    }

    occ::timing::stop(occ::timing::category::dft_gradient);
    return gradient;
  }

  std::string m_method_string{"svwn5"};
  occ::qm::HartreeFock m_hf;
  occ::numint::MolecularGrid m_grid;
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
