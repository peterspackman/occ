#pragma once
#include <memory>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/opt/coordinates.h>
#include <occ/opt/optimization_state.h>
#include <optional>

namespace occ::opt {

struct OptimizerSettings {
  enum class Coordinates { TRIC, Cartesian };
  Coordinates coordinates{Coordinates::TRIC};

  // Trust radius on the RMS atomic displacement after optimal alignment
  // (Bohr); geomeTRIC's 0.1, 1e-4 and 0.3 Angstrom.
  double trust{0.1 * occ::units::ANGSTROM_TO_BOHR};
  double trust_min{1e-4 * occ::units::ANGSTROM_TO_BOHR};
  double trust_max{0.3 * occ::units::ANGSTROM_TO_BOHR};
  // Minima: a step whose energy change is below this multiple of the
  // predicted (negative) change is rejected.
  double reject_quality{-1.0};
  // Remove net force and torque from the gradient: right for an isolated
  // system, where they are numerical noise (e.g. from integration grids),
  // wrong with an external field.
  bool project_net_force_torque{true};

  // Search for a first-order saddle point instead of a minimum: maximise
  // along one Hessian mode, minimise along the rest. Start from a computed
  // Hessian (set_initial_hessian); the model has no negative curvature.
  bool transition_state{false};
  // The mode maximised at the first step, counting internal modes from the
  // lowest eigenvalue (0); later steps follow it by overlap.
  int ts_mode{0};

  /// Settings for a saddle-point search: a trust radius of 0.05 Angstrom,
  /// at most 0.15 (geomeTRIC's 0.01 and 0.03 took about 50% more gradients
  /// on GFN2-xTB test reactions, 0.1 and 0.3 slightly more).
  static OptimizerSettings saddle_point(int mode = 0) {
    OptimizerSettings s;
    s.transition_state = true;
    s.ts_mode = mode;
    s.trust = 0.05 * occ::units::ANGSTROM_TO_BOHR;
    s.trust_max = 0.15 * occ::units::ANGSTROM_TO_BOHR;
    return s;
  }
};

/// Geometry optimizer for minima and first-order saddle points (L.-P. Wang
/// and C. Song, J. Chem. Phys. 144, 214108 (2016), with the step control of
/// geomeTRIC's current releases).
///
/// For a minimum, the step is the Newton step on a positive definite
/// Hessian, shifted (Levenberg) to the length that moves the atoms by the
/// trust radius (RMS, after alignment), in TRIC or Cartesian coordinates;
/// the Hessian starts from Lindh's model and is updated by BFGS. For a saddle
/// point, the step is the restricted-step partitioned RFO step (maximising
/// along the followed mode, minimising along the rest), and the Hessian,
/// best computed, is updated by Bofill's formula. A step whose energy change
/// departs from the model too far is rejected and retaken from the last
/// accepted geometry with a smaller trust radius. Coordinates are rebuilt
/// where they become degenerate, the Hessian carried through Cartesians.
/// Convergence is judged on per-atom Cartesian gradients and aligned
/// displacements.
///
/// Use: update() with the energy and gradient at get_next_geometry(), then
/// step(), until step() returns true.
class GeometryOptimizer {
public:
  GeometryOptimizer(const core::Molecule &molecule,
                    const ConvergenceCriteria &criteria = {},
                    const OptimizerSettings &settings = {});

  /// Initial Hessian in Cartesians (Hartree/Bohr^2, 3N x 3N, atom-major
  /// xyz), e.g. a computed one, in place of the model.
  void set_initial_hessian(const Mat &cartesian_hessian);

  /// Energy (Hartree) and gradient (Hartree/Bohr, 3 x N) at the geometry
  /// last returned by get_next_geometry().
  void update(double energy, const Mat3N &gradient);

  /// Accept or reject the latest point, check convergence at the accepted
  /// one, and if not converged choose the next geometry. Returns true when
  /// converged; get_next_geometry() is then the converged point.
  bool step();

  core::Molecule get_next_geometry() const;

  /// Shift every stored energy, when the energy surface moves by a constant
  /// (e.g. a change of integration grid).
  void shift_energy_reference(double shift);

  bool is_converged() const { return m_converged; }
  /// Energy at the last accepted point (Hartree).
  double current_energy() const {
    return m_accepted ? m_accepted->energy : 0.0;
  }
  int current_step() const { return m_step; }
  double trust_radius() const { return m_trust; }
  const CoordinateSystem &coordinates() const { return *m_coordinates; }
  /// Saddle-point search: the followed mode as a unit Cartesian displacement
  /// (3N, atom-major), and the number of internal modes with negative
  /// curvature, both at the last accepted point.
  const Vec &followed_mode() const { return m_mode; }
  int negative_modes() const { return m_negative_modes; }

private:
  struct Point {
    Mat3N x;        // Bohr
    double energy;  // Hartree
    Mat3N gradient; // Hartree/Bohr
  };
  struct Step {
    Mat3N x;          // the new geometry
    double predicted; // model energy change
    double rmsd;      // aligned RMS displacement
  };

  void build_coordinates(const Mat3N &x, const Mat &cartesian_hessian,
                         bool model);
  void update_hessian(const Point &from, const Point &to);
  Step next_step(const Point &p);
  int follow_mode(const Mat3N &x, const Mat &modes, const Vec &curvature);

  IVec m_atomic_numbers;
  ConvergenceCriteria m_criteria;
  OptimizerSettings m_settings;
  std::unique_ptr<CoordinateSystem> m_coordinates;
  Mat m_hessian; // in the current coordinates

  Mat3N m_next;                  // geometry to evaluate (Bohr)
  std::optional<Point> m_latest; // evaluated at m_next
  std::optional<Point> m_accepted;
  double m_previous_accepted_energy{0.0};
  std::optional<Step> m_last_step;
  std::optional<Mat3N> m_last_rejected;
  double m_trust;
  Vec m_mode;
  int m_negative_modes{0};
  int m_step{0};
  bool m_converged{false};
};

/// Per-atom displacements from x0 to x after optimal superposition (Bohr).
Vec aligned_displacements(const Mat3N &x0, const Mat3N &x);

/// Gradient without net force and torque about the centroid.
Mat3N project_net_force_torque(const Mat3N &x, const Mat3N &gradient);

} // namespace occ::opt
