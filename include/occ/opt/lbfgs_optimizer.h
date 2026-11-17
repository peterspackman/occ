#pragma once
#include <occ/core/molecule.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/optimization_state.h>
#include <Eigen/Cholesky>
#include <memory>

namespace occ::opt {

struct LbfgsSettings {
  int memory{20};       // correction pairs kept
  // Trust radius: the largest atomic displacement per step (Bohr). It
  // starts at max_step and moves within [min_step, max_step] by the ratio of
  // actual to predicted energy change; a step that raises the energy is
  // rejected and retaken from the previous point with a smaller radius.
  double max_step{0.3};
  double min_step{1e-3};
  bool reject_uphill{true};
  double initial_hessian{1.0}; // H0 for the first step (Hartree/Bohr^2)
  // Use the Berny model Hessian in Cartesians, B^T H B + shift I, as H0
  // (preconditioned L-BFGS) instead of (s.y / y.y) I.
  bool precondition{true};
  double precondition_shift{0.01}; // Hartree/Bohr^2, keeps H0 positive definite
};

/**
 * @brief L-BFGS geometry optimizer in Cartesian coordinates
 *
 * Ported from klasp: the two-loop recursion with H0 = (s.y / y.y) I from the
 * latest pair (or a model Hessian, see LbfgsSettings::precondition), pairs
 * with y.s <= 0 skipped and a fall back to steepest descent (history
 * cleared) when the direction goes uphill. In place of a line search the step
 * is scaled to a trust radius on the largest atomic displacement. Convergence
 * is checked on the Cartesian gradient and step with the same criteria as the
 * Berny optimizer.
 */
class LbfgsOptimizer {
public:
  LbfgsOptimizer(const core::Molecule &mol,
                 const ConvergenceCriteria &criteria = ConvergenceCriteria{},
                 const LbfgsSettings &settings = LbfgsSettings{});

  /// Energy (Hartree) and Cartesian gradient (Hartree/Bohr, 3 x N) at the
  /// geometry last returned by get_next_geometry().
  void update(double energy, const Mat3N &gradient);

  /// Check convergence at the current point, else take a step. Returns true
  /// when converged (the geometry is then left where it is).
  bool step();

  core::Molecule get_next_geometry() const;

  /// Energies are not used by the step, so this only keeps the reported
  /// energy change on one surface.
  void shift_energy_reference(double shift);

  bool is_converged() const { return m_converged; }
  double current_energy() const { return m_energy; }
  int current_step() const { return m_step; }
  double trust_radius() const { return m_trust; }
  int num_stored() const { return m_num_stored; }

private:
  void push_pair(const Vec &s, const Vec &y);
  Vec direction(const Vec &g) const;
  Vec apply_h0(const Vec &q) const;
  void build_preconditioner();

  core::Molecule m_molecule;
  ConvergenceCriteria m_criteria;
  LbfgsSettings m_settings;

  Vec m_x;      // current positions (Bohr), flattened 3N
  Vec m_g;      // gradient at m_x
  Vec m_x_prev; // previous point, empty before the first step
  Vec m_g_prev;
  double m_energy{0.0};
  double m_energy_prev{0.0};

  std::unique_ptr<InternalCoordinates> m_coords;
  Mat m_model_hessian;                 // internal coordinates
  Eigen::LDLT<Mat> m_precondition;     // Cartesian H0 at m_x

  Mat m_s, m_y; // (3N, memory) ring buffer columns
  Vec m_rho;
  int m_num_stored{0};
  int m_cursor{0};

  double m_trust{0.0};
  double m_predicted_change{0.0}; // quadratic model for the last step
  double m_last_disp{0.0};        // largest atomic displacement of it

  int m_step{0};
  bool m_converged{false};
};

} // namespace occ::opt
