#pragma once
#include <memory>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/scc.h>

namespace occ::xtb {

class Gfn2Calculator;
class Gfn2Parameters;

// Native in-tree GFN2-xTB backend. Mirrors the public surface of
// `TbliteCalculator` so callers can swap. Supports only molecular systems
// (no PBC) for now — Phase 4d will add Crystal.
class NativeCalculator {
public:
  enum class Method { GFN2 };

  explicit NativeCalculator(const occ::core::Molecule &mol);
  explicit NativeCalculator(const occ::core::Dimer &dimer);
  ~NativeCalculator();

  double single_point_energy();

  // Per-atom Mulliken charges (xtb convention: positive when electron-deficient).
  Vec charges() const;

  // Wiberg bond orders.
  Mat bond_orders() const;

  // Atomic coordinates in Bohr (3 × N).
  inline const Mat3N &positions() const { return m_positions_bohr; }

  inline int num_atoms() const { return m_atomic_numbers.rows(); }

  // SCC controls.
  void set_charge(double c);
  void set_max_iterations(int iterations);
  void set_temperature(double temp);
  void set_mixer_damping(double damping_factor);

  // Geometry update — keeps the basis layout (atomic numbers must match).
  void update_structure(const Mat3N &positions);

  // Decomposed energies from the most recent single_point_energy() call.
  inline double scc_energy() const { return m_last_result.scc_energy; }
  inline double repulsion_energy() const { return m_last_result.repulsion_energy; }
  inline double dispersion_energy() const { return m_last_result.dispersion_energy; }
  inline const SccResult &last_result() const { return m_last_result; }

  occ::core::Molecule to_molecule() const;

  // Convert the converged SCC state into a `qm::Wavefunction` so that the
  // rest of occ (FCHK output, Mulliken/Hirshfeld analysis, ESP, etc.) can
  // consume the GFN2 result. Requires `single_point_energy()` to have run.
  occ::qm::Wavefunction to_wavefunction() const;

  // Print a GFN2-native results summary (energy decomposition, atomic
  // charges, HOMO/LUMO/gap) at INFO log level. Useful as the post-SCF
  // print step in `occ scf -m gfn2 ...`, where the standard wavefunction
  // properties printer is not the right thing.
  void print_summary() const;

  // Numerical nuclear gradient via central differences in Bohr. Returns a
  // 3 × N matrix in Hartree/Bohr. Each call costs `6N` SCC evaluations,
  // so this is fine for tens of atoms but not hundreds. `step_bohr`
  // controls the displacement (default 1e-3 Bohr).
  Mat3N compute_gradient_numerical(double step_bohr = 1e-3);

  // Energy + gradient pair, suitable for plugging into `BernyOptimizer`
  // or any other optimizer that wants both at once.
  std::pair<double, Mat3N>
  compute_energy_and_gradient(double step_bohr = 1e-3);

private:
  void initialize_calculator();

  Mat3N m_positions_bohr;
  IVec m_atomic_numbers;
  double m_charge{0};
  Method m_method{Method::GFN2};
  SccOptions m_opts;

  std::shared_ptr<Gfn2Parameters> m_params;
  std::unique_ptr<Gfn2Calculator> m_calc;
  SccResult m_last_result;
};

} // namespace occ::xtb
