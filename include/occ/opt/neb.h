#pragma once
#include <deque>
#include <occ/core/molecule.h>
#include <vector>

namespace occ::opt {

struct NebSettings {
  int images{8};      // between the endpoints
  // Hartree/Bohr^2: springs stiffen from `spring` at or below the higher
  // endpoint's energy to spring_max at the highest image (energy-weighted,
  // Henkelman, Uberuaga and Jonsson 2000), drawing images to the barrier;
  // ORCA's two constants. spring_max = spring for uniform springs.
  double spring{0.01};
  double spring_max{0.1};
  // Image-dependent pair potential interpolation (Smidstrup et al., J. Chem.
  // Phys. 140, 214106 (2014)) in place of a straight line
  bool idpp{true};
  // Turn the highest image into a climbing image once the largest NEB force
  // on any atom is below climb_threshold (Hartree/Bohr; ORCA's)
  bool climbing_image{true};
  double climb_threshold{2e-2};
  // Converged when no atom of the climbing image feels a force above
  // climbing_force_max, and none of the other images, which only shape the
  // path, a NEB force above force_max (Hartree/Bohr)
  double climbing_force_max{1e-3};
  double force_max{5e-3};
  double max_step{0.1};    // Bohr, largest atomic displacement per step
  int memory{10};          // L-BFGS correction pairs
  // Remove net force and torque from each image's gradient (isolated systems)
  bool project_net_force_torque{true};
};

/// Nudged elastic band (Henkelman and Jonsson, J. Chem. Phys. 113, 9978
/// (2000)) between two structures with the same atoms in the same order, in
/// Cartesian coordinates: the potential force perpendicular to the improved
/// tangent plus the spring force along it (tangents and spring lengths from
/// neighbours superposed on each image, free of rigid rotation), with an
/// optional climbing image
/// (Henkelman, Uberuaga and Jonsson, J. Chem. Phys. 113, 9901 (2000)). The
/// band moves by L-BFGS on the NEB force, its steps capped, without energy
/// checks (the NEB force is not a gradient).
///
/// Use: set_endpoint_energies(), then update() with the energies and
/// gradients at images(), then step(), until step() returns true.
class NudgedElasticBand {
public:
  NudgedElasticBand(const core::Molecule &reactant,
                    const core::Molecule &product,
                    const NebSettings &settings = {});

  /// The images between the endpoints, to evaluate.
  std::vector<core::Molecule> images() const;
  /// All images, endpoints included.
  std::vector<core::Molecule> path() const;

  /// Energies of the endpoints (Hartree), which the tangents need.
  void set_endpoint_energies(double reactant, double product);

  /// Energies (Hartree) and gradients (Hartree/Bohr, 3 x N) at images().
  void update(const std::vector<double> &energies,
              const std::vector<Mat3N> &gradients);

  /// Check convergence, else move the band. Returns true when converged.
  bool step();

  bool is_converged() const { return m_converged; }
  bool climbing() const { return m_climbing; }
  int current_step() const { return m_step; }
  /// Path energies (Hartree), endpoints included.
  const Vec &energies() const { return m_energy; }
  /// Index in path() of the highest interior image.
  int highest_image() const;
  /// Unit tangent at path() image i (interior), 3N atom-major.
  Vec tangent(int i) const;
  /// Largest NEB force on any atom at the last update (Hartree/Bohr).
  double max_force() const { return m_max_force; }

private:
  Vec neb_force() const;
  std::pair<Vec, Vec> neighbour_differences(int i) const;
  void interpolate_idpp();
  void reset_history();

  IVec m_atomic_numbers;
  NebSettings m_settings;
  std::vector<Mat3N> m_x; // path, Bohr
  Vec m_energy;
  std::vector<Mat3N> m_gradient; // interior images
  bool m_have_endpoints{false}, m_have_update{false};

  std::deque<Vec> m_s, m_y;
  // the last accepted band
  Vec m_accepted_x, m_accepted_force, m_accepted_energy;
  std::vector<Mat3N> m_accepted_gradient;
  double m_accepted_max_force{0.0};
  double m_trust{0.0}; // Bohr, largest atomic displacement
  bool m_climbing{false}, m_converged{false};
  int m_climber{-1};
  double m_max_force{0.0};
  int m_step{0};
};

} // namespace occ::opt
