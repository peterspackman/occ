#pragma once
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/gradients.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf.h>
#include <occ/qm/scf_convergence_settings.h>
#include <occ/gto/shell.h>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

// 3N x 3N Hessian matrix for storing d2E/dR_AdR_B
using HessianMatrix = Mat;

/**
 * @brief Evaluates molecular Hessian matrices (second derivatives of energy)
 *
 * The HessianEvaluator class computes the Hessian matrix, which contains
 * second derivatives of the energy with respect to nuclear coordinates:
 * H_ij = d2E/dR_i dR_j
 *
 * Currently supports:
 * - Finite differences method with configurable step size
 * - Acoustic sum rule optimization (reduces computational cost by ~33%)
 * - Nuclear repulsion Hessian computation
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
    FiniteDifferences, ///< Central finite differences (currently only supported
                       ///< method)
    Analytical         ///< Analytical second derivatives (not yet implemented)
  };

  /**
   * @brief Set the method for Hessian calculation
   * @param method The method to use (currently only FiniteDifferences
   * supported)
   */
  void set_method(Method method) {
    if (method == Method::Analytical) {
      throw std::runtime_error("Analytical Hessian not yet implemented");
    }
    m_method = method;
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
        const double Z_A = atoms[A].atomic_number;
        const double Z_B = atoms[B].atomic_number;

        Vec3 R_AB = pos_A - pos_B;
        double r_AB = R_AB.norm();
        double r3 = r_AB * r_AB * r_AB;
        double r5 = r3 * r_AB * r_AB;

        // d²V_nn/dR_A[i]dR_B[j] = Z_A*Z_B * [3*R_AB[i]*R_AB[j]/r⁵ - δ_ij/r³]
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            result(3 * A + i, 3 * B + j) =
                Z_A * Z_B * (3.0 * R_AB[i] * R_AB[j] / r5 - delta_ij / r3);
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
      throw std::runtime_error("Selected Hessian method not implemented");
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

    // Symmetrize the matrix (copy upper triangle to lower triangle)
    for (size_t i = 0; i < ndof; i++) {
      for (size_t j = i + 1; j < ndof; j++) {
        result(j, i) = result(i, j);
      }
    }

    return result;
  }

  Proc &m_proc;
  HessianMatrix m_hessian;
  double m_step_size{0.005};                  // Default: 0.005 Bohr
  bool m_use_acoustic_sum_rule{true};         // Default: use optimization
  Method m_method{Method::FiniteDifferences}; // Default: finite differences
  SCFConvergenceSettings m_scf_convergence_settings; // SCF settings for displaced calculations
};

} // namespace occ::qm
