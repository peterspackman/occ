#pragma once
#include <chrono>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/cphf.h>
#include <occ/qm/gradients.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf.h>
#include <occ/qm/scf_convergence_settings.h>
#include <occ/gto/shell.h>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

// 3N x 3N Hessian matrix for storing d2E/dR_AdR_B
using HessianMatrix = Mat;

/// A procedure that provides every term the analytic Hessian needs.
template <typename Proc>
concept AnalyticHessianProcedure =
    requires(const Proc &p, const MolecularOrbitals &mo, const Mat &m,
             std::vector<Mat> &v, const std::vector<Mat> &cv) {
      { p.compute_one_electron_hessian(m, m) } -> std::convertible_to<Mat>;
      { p.two_electron_hessian(mo, m) } -> std::convertible_to<Mat>;
      p.compute_one_electron_derivative_matrices(v, v);
      { p.two_electron_derivative_fock_occ(mo, m) };
      { p.compute_response_fock_occ(mo, cv, m) };
      { p.compute_response_fock_occ_approx(mo, cv, m) };
      { p.has_approximate_response() } -> std::convertible_to<bool>;
      { p.nonvariational_fock_derivative_occ(mo) };
      { p.analytic_hessian_unsupported(mo.kind) } -> std::convertible_to<std::string>;
    };

/**
 * @brief Evaluates molecular Hessian matrices (second derivatives of energy)
 *
 * The HessianEvaluator class computes the Hessian matrix, which contains
 * second derivatives of the energy with respect to nuclear coordinates:
 * H_ij = d2E/dR_i dR_j
 *
 * Methods:
 * - Finite differences of analytic gradients, with configurable step size
 *   and the acoustic sum rule to skip one atom's displacements
 * - Analytic second derivatives via coupled-perturbed SCF, for procedures
 *   satisfying AnalyticHessianProcedure
 *
 * @tparam Proc The quantum chemical procedure type (e.g., HartreeFock, DFT)
 *
 * Usage example:
 * @code
 * HessianEvaluator<HartreeFock> hess_eval(hf);
 * hess_eval.set_step_size(0.005);  // 0.005 Bohr
 * hess_eval.set_use_acoustic_sum_rule(true);  // Use optimization
 * auto hessian = hess_eval(mo);
 * @endcode
 */
template <typename Proc> class HessianEvaluator {

public:
  /**
   * @brief Construct a HessianEvaluator for the given procedure
   * @param p The quantum chemical procedure (HF, DFT, etc.)
   */
  explicit HessianEvaluator(Proc &p)
      : m_proc(p), m_hessian(HessianMatrix::Zero(3 * p.atoms().size(),
                                                 3 * p.atoms().size())) {
    // Set tighter SCF convergence defaults for Hessian calculations
    m_scf_convergence_settings.energy_threshold = 1e-10;  // Default is 1e-6
    m_scf_convergence_settings.commutator_threshold = 1e-8;  // Default is 1e-5
  }

  /**
   * @brief Method for Hessian calculation
   */
  enum class Method {
    FiniteDifferences, ///< Central finite differences of gradients
    Analytical         ///< Analytic second derivatives via CPHF
  };

  /**
   * @brief Set the method for Hessian calculation
   * @param method The method to use
   */
  void set_method(Method method) { m_method = method; }

  /// CPHF settings used by the analytical method.
  void set_cphf_settings(const cphf::Settings &settings) {
    m_cphf_settings = settings;
  }

  /**
   * @brief Set the finite differences step size
   * @param h Step size in Bohr (typical: 0.001 to 0.01)
   *
   * Smaller step sizes reduce truncation error but increase numerical error.
   * Default: 0.005 Bohr (matches ORCA default)
   */
  void set_step_size(double h) {
    if (h <= 0) {
      throw std::invalid_argument("Step size must be positive");
    }
    m_step_size = h;
  }

  /**
   * @brief Enable/disable acoustic sum rule optimization
   * @param use If true, use acoustic sum rule to reduce computations
   *
   * The acoustic sum rule (translational invariance) reduces the number
   * of required displacements from 3N to 3(N-1), saving some computation
   * for a 3-atom system. Based on: d2E/dR_i dR_j = -Σ_k≠j d2E/dR_i dR_k
   */
  inline void set_use_acoustic_sum_rule(bool use) { m_use_acoustic_sum_rule = use; }

  /**
   * @brief Get the current step size for finite differences
   * @return Step size in Bohr
   */
  inline double step_size() const { return m_step_size; }

  /**
   * @brief Check if acoustic sum rule is enabled
   * @return True if acoustic sum rule optimization is enabled
   */
  inline bool use_acoustic_sum_rule() const { return m_use_acoustic_sum_rule; }

  /**
   * @brief Get the current calculation method
   * @return The Hessian calculation method
   */
  inline Method method() const { return m_method; }

  /**
   * @brief Set the SCF convergence settings for displaced calculations
   * @param settings SCF convergence settings to use
   *
   * By default, Hessian calculations use tighter convergence criteria
   * (energy_threshold=1e-10, commutator_threshold=1e-8) for improved accuracy.
   */
  void set_scf_convergence_settings(const SCFConvergenceSettings &settings) {
    m_scf_convergence_settings = settings;
  }

  /**
   * @brief Get the current SCF convergence settings
   * @return The SCF convergence settings used for displaced calculations
   */
  const SCFConvergenceSettings &scf_convergence_settings() const {
    return m_scf_convergence_settings;
  }

  /**
   * @brief Compute the nuclear repulsion contribution to the Hessian
   * @return Nuclear repulsion Hessian matrix (3N×3N)
   *
   * Computes d²V_nn/dR_A dR_B where V_nn is the nuclear-nuclear repulsion.
   * This is the only part of the Hessian that can be computed analytically
   * without significant computational cost.
   */
  HessianMatrix nuclear_repulsion() const {
    const auto &atoms = m_proc.atoms();
    size_t natom = atoms.size();
    HessianMatrix result = HessianMatrix::Zero(3 * natom, 3 * natom);

    for (size_t A = 0; A < natom; A++) {
      for (size_t B = 0; B < natom; B++) {
        if (A == B)
          continue;

        const auto &pos_A = atoms[A].position();
        const auto &pos_B = atoms[B].position();
        // ECP-reduced charges, as in the nuclear repulsion energy.
        const auto &frozen = m_proc.frozen_electrons();
        const double Z_A = atoms[A].atomic_number -
                           (A < frozen.size() ? frozen[A] : 0);
        const double Z_B = atoms[B].atomic_number -
                           (B < frozen.size() ? frozen[B] : 0);

        Vec3 R_AB = pos_A - pos_B;
        double r_AB = R_AB.norm();
        double r3 = r_AB * r_AB * r_AB;
        double r5 = r3 * r_AB * r_AB;

        // V = Z_A Z_B / r with r = |R_A - R_B|, so d/dR_B = -d/dR_A and
        // d²V/dR_A[i]dR_B[j] = -Z_A*Z_B * [3*R_AB[i]*R_AB[j]/r⁵ - δ_ij/r³]
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            result(3 * A + i, 3 * B + j) =
                -Z_A * Z_B * (3.0 * R_AB[i] * R_AB[j] / r5 - delta_ij / r3);
          }
        }
      }
    }

    // Diagonal terms: d²V_nn/dR_A[i]dR_A[j] = -sum_B≠A d²V_nn/dR_A[i]dR_B[j]
    for (size_t A = 0; A < natom; A++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          double sum = 0.0;
          for (size_t B = 0; B < natom; B++) {
            if (B != A)
              sum += result(3 * A + i, 3 * B + j);
          }
          result(3 * A + i, 3 * A + j) = -sum;
        }
      }
    }

    return result;
  }

  /**
   * @brief Compute the full molecular Hessian
   * @param wfn Wavefunction from converged SCF calculation
   * @return Complete Hessian matrix including nuclear and electronic
   * contributions
   *
   * This is the main interface for computing the Hessian. The method used
   * depends on the configuration (set_method, set_step_size, etc.).
   * The wavefunction provides both the molecular orbitals for the reference
   * calculation and the charge/multiplicity for displaced calculations.
   */
  const HessianMatrix &operator()(const Wavefunction &wfn) {
    occ::timing::start(occ::timing::hessian);

    // Compute Hessian based on selected method
    if (m_method == Method::FiniteDifferences) {
      m_hessian = compute_finite_differences(wfn);
    } else {
      if constexpr (AnalyticHessianProcedure<Proc>) {
        m_hessian = compute_analytical(wfn.mo);
      } else {
        throw std::runtime_error(
            "Analytical Hessian not available for this method");
      }
    }

    occ::timing::stop(occ::timing::hessian);

    // Log Hessian diagonal elements for each atom
    const auto &atoms = m_proc.atoms();
    occ::log::info("Hessian diagonal elements:");
    for (size_t atom = 0; atom < atoms.size(); atom++) {
      occ::log::info("{:2s}{:3d}: xx={:12.8f} yy={:12.8f} zz={:12.8f}",
                     core::Element(atoms[atom].atomic_number).symbol(), atom,
                     m_hessian(3 * atom + 0, 3 * atom + 0),
                     m_hessian(3 * atom + 1, 3 * atom + 1),
                     m_hessian(3 * atom + 2, 3 * atom + 2));
    }

    return m_hessian;
  }

  /**
   * @brief Analytic Hessian at converged orbitals.
   *
   *   H = H_nn + sum P h^{xy} + E2^{xy}[Gamma] - sum Q S^{xy}       (skeleton)
   *     + n sum_s [ 2 sum_{p,i} F^{(x)}_{pi} U^y_{pi}
   *               - 2 sum_{p,i} S^{(x)}_{pi} U^y_{pi} e_i
   *               -   sum_{ij}  S^{(x)}_{ij} e1^y_{ij} ]          (response)
   *
   * summed over the spin channels s (one restricted, two unrestricted) with
   * n electrons per orbital (2 restricted, 1 unrestricted). P, Q total
   * density and energy-weighted density; F^{(x)} = h^{(x)} + skeleton
   * two-electron Fock derivative of the channel; U^y_ij = -S^{(y)}_ij / 2 on
   * the occupied block and the CPHF solution on the virtual-occupied block;
   * e1^y = F^{(y)}_oo - S^{(y)}_oo (e_i + e_j)/2 + G[D^y]_oo.
   */
  HessianMatrix compute_analytical(const MolecularOrbitals &mo)
    requires AnalyticHessianProcedure<Proc>
  {
    if (const auto why = m_proc.analytic_hessian_unsupported(mo.kind);
        !why.empty())
      throw std::runtime_error("Analytical Hessian: " + why);
    if (mo.smearing.kind != OrbitalSmearing::Kind::None)
      throw std::runtime_error(
          "Analytical Hessian: smeared (fractional) occupations are not "
          "supported");
    const auto &atoms = m_proc.atoms();
    const Eigen::Index n3 = 3 * static_cast<Eigen::Index>(atoms.size());
    const auto channels = cphf::spin_channels(mo);
    const size_t nspin = channels.size();
    const double occ = nspin == 1 ? 2.0 : 1.0;
    const Eigen::Index nbf = channels[0].Co.rows();
    // Column of channel s in the Cx layout.
    std::vector<Eigen::Index> col(nspin, 0);
    for (size_t s = 1; s < nspin; s++)
      col[s] = col[s - 1] + channels[s - 1].Co.cols();
    const Eigen::Index ncol = col.back() + channels.back().Co.cols();

    Mat P = Mat::Zero(nbf, nbf), Q = Mat::Zero(nbf, nbf);
    for (const auto &c : channels) {
      P.noalias() += occ * c.Co * c.Co.transpose();
      Q.noalias() += occ * c.Co * c.eo.asDiagonal() * c.Co.transpose();
    }
    const Mat schwarz = m_proc.compute_schwarz_ints();

    using Clock = std::chrono::steady_clock;
    auto since = [](Clock::time_point t0) {
      return std::chrono::duration<double>(Clock::now() - t0).count();
    };
    auto t0 = Clock::now();
    occ::log::info("Analytic Hessian ({} coordinates): skeleton terms", n3);
    HessianMatrix H = nuclear_repulsion();
    H += m_proc.compute_one_electron_hessian(P, Q);
    H += m_proc.two_electron_hessian(mo, schwarz);
    occ::log::info("  skeleton terms done ({:.2f} s)", since(t0));

    t0 = Clock::now();
    occ::log::info("Analytic Hessian: derivative Fock matrices");
    std::vector<Mat> S1, H1;
    m_proc.compute_one_electron_derivative_matrices(S1, H1);
    const auto G1o = m_proc.two_electron_derivative_fock_occ(mo, schwarz);
    occ::log::info("  derivative Fock matrices done ({:.2f} s)", since(t0));
    t0 = Clock::now();

    auto response = [&](const std::vector<Mat> &Cx) {
      return m_proc.compute_response_fock_occ(mo, Cx, schwarz);
    };
    cphf::Solver solver(mo, response, m_cphf_settings);

    // MO-basis first-derivative quantities per coordinate and channel,
    // columns occupied; the occupied-occupied density change
    // D_oo = -Co S1_oo Co^T as perturbed occupied coefficients,
    // Cx = -Co S1_oo / 2.
    std::vector<std::vector<Mat>> F1_po(n3), S1_po(n3);
    std::vector<Mat> Cx_oo(n3, Mat(nbf, ncol));
    for (Eigen::Index k = 0; k < n3; k++) {
      for (size_t s = 0; s < nspin; s++) {
        const auto &c = channels[s];
        const Eigen::Index no = c.Co.cols();
        Mat C(nbf, no + c.Cv.cols());
        C << c.Co, c.Cv;
        F1_po[k].push_back(C.transpose() *
                           (H1[k] * c.Co + G1o[k].middleCols(col[s], no)));
        S1_po[k].push_back(C.transpose() * S1[k] * c.Co);
        Cx_oo[k].middleCols(col[s], no) =
            -0.5 * c.Co * S1_po[k][s].topRows(no);
      }
    }

    // CPHF right-hand sides: -(F^{(x)} - e_i S^{(x)})_ai - G[D_oo]_ai
    const auto Goo = response(Cx_oo);
    std::vector<Mat> B(n3, solver.vo_zero());
    for (Eigen::Index k = 0; k < n3; k++) {
      for (size_t s = 0; s < nspin; s++) {
        const auto &c = channels[s];
        const Eigen::Index no = c.Co.cols(), nv = c.Cv.cols();
        solver.block(B[k], s) =
            -(F1_po[k][s].bottomRows(nv) -
              S1_po[k][s].bottomRows(nv) * c.eo.asDiagonal()) -
            c.Cv.transpose() * Goo[k].middleCols(col[s], no);
      }
    }
    occ::log::info("Analytic Hessian: solving CPHF for {} perturbations", n3);
    // With a cheaper approximate response (DFT: the XC kernel on a coarse
    // grid), solve with it first and finish with the exact one from there:
    // the exact iterations only remove the difference between the two.
    std::vector<Mat> U0;
    if (m_proc.has_approximate_response()) {
      auto approx = [&](const std::vector<Mat> &Cx) {
        return m_proc.compute_response_fock_occ_approx(mo, Cx, schwarz);
      };
      cphf::Solver coarse(mo, approx, m_cphf_settings);
      U0 = coarse.solve(B).U;
      occ::log::info("  approximate-response CPHF done ({:.2f} s)", since(t0));
    }
    auto res = solver.solve(B, U0.empty() ? nullptr : &U0);
    occ::log::info("CPHF: {} iterations, {} response Fock builds, {} "
                   "({:.2f} s)",
                   res.iterations, res.response_builds,
                   res.converged ? "converged" : "NOT converged", since(t0));
    if (!res.converged)
      occ::log::warn("CPHF did not converge: the Hessian is unreliable");

    // Per channel: the full U (occupied rows -S1_oo / 2) and e1, with the
    // response to the full first-order densities by linearity from the
    // solver's own G[D(U)] and G[D_oo].
    std::vector<std::vector<Mat>> U(n3), e1(n3);
    for (Eigen::Index k = 0; k < n3; k++) {
      const Mat G = res.G[k] + Goo[k];
      for (size_t s = 0; s < nspin; s++) {
        const auto &c = channels[s];
        const Eigen::Index no = c.Co.cols();
        const Mat S1oo = S1_po[k][s].topRows(no);
        const Mat esum =
            c.eo.replicate(1, no) + c.eo.transpose().replicate(no, 1);
        Mat u(no + c.Cv.cols(), no);
        u.topRows(no) = -0.5 * S1oo;
        u.bottomRows(c.Cv.cols()) = solver.block(res.U[k], s);
        U[k].push_back(std::move(u));
        e1[k].push_back(F1_po[k][s].topRows(no) -
                        0.5 * S1oo.cwiseProduct(esum) +
                        c.Co.transpose() * G.middleCols(col[s], no));
      }
    }

    HessianMatrix Hr = HessianMatrix::Zero(n3, n3);
    for (size_t s = 0; s < nspin; s++) {
      const auto &c = channels[s];
      const Eigen::Index no = c.Co.cols();
      for (Eigen::Index x = 0; x < n3; x++) {
        const Mat S1e = S1_po[x][s] * c.eo.asDiagonal();
        for (Eigen::Index y = 0; y < n3; y++) {
          const auto S1oo = S1_po[x][s].topRows(no);
          Hr(x, y) += occ * (2.0 * F1_po[x][s].cwiseProduct(U[y][s]).sum() -
                             2.0 * S1e.cwiseProduct(U[y][s]).sum() -
                             S1oo.cwiseProduct(e1[y][s]).sum());
        }
      }
    }
    // The density response above is that of the SCF, but what it multiplies
    // is the derivative of the gradient's skeleton term in the density,
    // which is F^{(x)} only when F = 1/2 dE/dD. Where it is not (COSX), the
    // procedure supplies the difference, so that this is the derivative of
    // the analytic gradient.
    {
      const auto dF = m_proc.nonvariational_fock_derivative_occ(mo);
      for (size_t s = 0; s < nspin; s++) {
        const auto &c = channels[s];
        const Eigen::Index no = c.Co.cols();
        Mat C(nbf, no + c.Cv.cols());
        C << c.Co, c.Cv;
        for (size_t x = 0; x < dF.size(); x++) {
          const Mat dFpo = C.transpose() * dF[x].middleCols(col[s], no);
          for (Eigen::Index y = 0; y < n3; y++)
            Hr(x, y) += occ * 2.0 * dFpo.cwiseProduct(U[y][s]).sum();
        }
      }
    }
    // The response part is -2n B_x^T U_y with A U_y = B_y. Its Hylleraas
    // form, -2n [B_x^T U_y + U_x^T (B_y - A U_y)], is stationary in U, so
    // adding -2n U_x^T R_y with the solver's final residuals R makes the
    // Hessian's error quadratic in the CPHF residual rather than linear
    // (no extra response builds), which is what lets the CPHF tolerance be
    // loose.
    for (Eigen::Index x = 0; x < n3; x++)
      for (Eigen::Index y = 0; y < n3; y++)
        Hr(x, y) -= 2.0 * occ * res.U[x].cwiseProduct(res.R[y]).sum();
    H += Hr;
    return 0.5 * (H + H.transpose());
  }

private:
  /**
   * @brief Compute Hessian using finite differences of gradients
   * @param wfn Wavefunction containing molecular orbitals and system information
   * @return Hessian matrix computed via finite differences
   *
   * Uses central finite differences:
   * H_ij = [G_i(R_j + h) - G_i(R_j - h)] / (2h)
   *
   * When acoustic sum rule is enabled, only 3(N-1) displacements are computed
   * and the remaining elements are derived from translational invariance.
   * The wavefunction is used as initial guess for displaced calculations.
   */
  HessianMatrix compute_finite_differences(const Wavefunction &wfn) {

    const auto &atoms = m_proc.atoms();
    const auto &basis = m_proc.aobasis();
    size_t natom = atoms.size();
    size_t ndof = 3 * natom;
    HessianMatrix result = HessianMatrix::Zero(ndof, ndof);

    // First, perform a reference SCF calculation with the same tight convergence
    // settings that will be used for displaced geometries
    occ::log::debug("Computing reference SCF with tight convergence settings");
    occ::qm::SCF<Proc> scf_reference(m_proc, wfn.mo.kind);
    scf_reference.set_charge_multiplicity(wfn.charge(), wfn.multiplicity());
    scf_reference.set_initial_guess_from_wfn(wfn);
    scf_reference.convergence_settings = m_scf_convergence_settings;
    scf_reference.compute_scf_energy();
    
    // Use this tightly converged wavefunction as the reference
    Wavefunction wfn_reference = scf_reference.wavefunction();

    size_t num_displacements = m_use_acoustic_sum_rule ? 3 * (natom - 1) : ndof;

    occ::log::info("Computing finite differences Hessian with h = {:.2e} Bohr",
                   m_step_size);
    if (m_use_acoustic_sum_rule) {
      occ::log::info("Translation invariance used");
      occ::log::info("Number of displacements: {} - {}", ndof,
                     ndof - num_displacements);
    } else {
      occ::log::info("Computing all {} Cartesian displacements", ndof);
    }
    occ::log::info("This requires {} SCF calculations + gradients",
                   2 * num_displacements);

    // Loop over independent degrees of freedom (possibly reduced by acoustic
    // sum rule)
    for (size_t dof_B = 0; dof_B < num_displacements; dof_B++) {
      size_t B = dof_B / 3; // atom index
      int j = dof_B % 3;    // coordinate index (0=x, 1=y, 2=z)

      // Forward step: create new atoms with R_B[j] + h
      auto atoms_forward = atoms;
      if (j == 0)
        atoms_forward[B].x += m_step_size;
      else if (j == 1)
        atoms_forward[B].y += m_step_size;
      else
        atoms_forward[B].z += m_step_size;

      // Create new basis and HF object, run SCF, then compute gradient
      auto basis_forward = occ::gto::AOBasis::load(atoms_forward, basis.name());
      basis_forward.set_pure(
          basis.is_pure()); // Preserve spherical/Cartesian setting
      Proc hf_forward = m_proc.with_new_basis(basis_forward);
      occ::qm::SCF<Proc> scf_forward(hf_forward, wfn_reference.mo.kind);
      
      // Set charge and multiplicity from reference wavefunction
      scf_forward.set_charge_multiplicity(wfn_reference.charge(), wfn_reference.multiplicity());
      
      // Use reference wavefunction as initial guess for faster convergence
      scf_forward.set_initial_guess_from_wfn(wfn_reference);
      
      // Apply configured SCF convergence settings for Hessian calculations
      scf_forward.convergence_settings = m_scf_convergence_settings;
      
      scf_forward.compute_scf_energy();
      occ::qm::MolecularOrbitals mo_forward = scf_forward.ctx.mo;

      GradientEvaluator<Proc> grad_eval_forward(hf_forward);
      Mat3N grad_forward = grad_eval_forward(mo_forward);

      // Backward step: create new atoms with R_B[j] - h
      auto atoms_backward = atoms;
      if (j == 0)
        atoms_backward[B].x -= m_step_size;
      else if (j == 1)
        atoms_backward[B].y -= m_step_size;
      else
        atoms_backward[B].z -= m_step_size;

      // Create new basis and HF object, run SCF, then compute gradient
      auto basis_backward =
          occ::gto::AOBasis::load(atoms_backward, basis.name());
      basis_backward.set_pure(
          basis.is_pure()); // Preserve spherical/Cartesian setting
      Proc hf_backward = m_proc.with_new_basis(basis_backward);
      occ::qm::SCF<Proc> scf_backward(hf_backward, wfn_reference.mo.kind);
      
      // Set charge and multiplicity from reference wavefunction
      scf_backward.set_charge_multiplicity(wfn_reference.charge(), wfn_reference.multiplicity());
      
      // Use reference wavefunction as initial guess for faster convergence
      scf_backward.set_initial_guess_from_wfn(wfn_reference);
      
      // Apply configured SCF convergence settings for Hessian calculations
      scf_backward.convergence_settings = m_scf_convergence_settings;
      
      scf_backward.compute_scf_energy();
      occ::qm::MolecularOrbitals mo_backward = scf_backward.ctx.mo;

      GradientEvaluator<Proc> grad_eval_backward(hf_backward);
      Mat3N grad_backward = grad_eval_backward(mo_backward);

      // Compute finite difference: (grad_forward - grad_backward) / (2h)
      Mat3N finite_diff = (grad_forward - grad_backward) / (2.0 * m_step_size);

      // Store in Hessian matrix column: convert Mat3N (3 x natom) to column
      // format
      for (size_t A = 0; A < natom; A++) {
        for (int i = 0; i < 3; i++) {
          size_t dof_A = 3 * A + i;
          result(dof_A, dof_B) = finite_diff(i, A);
        }
      }
    }

    // Apply acoustic sum rule if enabled
    if (m_use_acoustic_sum_rule && natom > 1) {
      // For the last atom: d²E/dR_A[i]dR_last[j] = -∑_{K=0}^{natom-2}
      // d²E/dR_A[i]dR_K[j]
      size_t last_atom = natom - 1;
      for (int coord = 0; coord < 3; coord++) {
        size_t last_dof = 3 * last_atom + coord;

        for (size_t dof_A = 0; dof_A < ndof; dof_A++) {
          double sum = 0.0;
          // Sum over all atoms except the last one
          for (size_t atom = 0; atom < natom - 1; atom++) {
            size_t dof_K = 3 * atom + coord;
            sum += result(dof_A, dof_K);
          }
          result(dof_A, last_dof) = -sum;
        }
      }
    }

    // The symmetric part, as the analytic Hessian: the Jacobian of a
    // gradient that is not exactly an energy's (COSX) is not symmetric.
    return 0.5 * (result + result.transpose());
  }

  Proc &m_proc;
  HessianMatrix m_hessian;
  double m_step_size{0.005};                  // Default: 0.005 Bohr
  bool m_use_acoustic_sum_rule{true};         // Default: use optimization
  Method m_method{Method::FiniteDifferences}; // Default: finite differences
  SCFConvergenceSettings m_scf_convergence_settings; // SCF settings for displaced calculations
  // Loose, because the Hylleraas form makes the Hessian's error quadratic in
  // the CPHF residual: 1e-5 is within 1e-9 of a 1e-11 solve (water PBE).
  // After an approximate-response solve the exact residual usually starts
  // below it, so the exact stage is a single response build.
  cphf::Settings m_cphf_settings{.tolerance = 1e-5};
};

} // namespace occ::qm
