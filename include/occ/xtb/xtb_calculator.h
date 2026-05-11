#pragma once
#include <array>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/core/vibration.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/gfn2_engine.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/gfn2_periodic_calculator.h>
#include <occ/xtb/periodic.h>
#include <occ/xtb/scc.h>
#include <occ/xtb/solvation_interface.h>
#include <occ/xtb/xtb_result.h>
#include <memory>
#include <optional>

namespace occ::crystal {
class Crystal;
}

namespace occ::xtb {

/// In-tree GFN2-xTB calculator. Mirrors the public surface of
/// `TbliteCalculator` so callers can swap backends. Handles isolated
/// molecules, dimers, and 3D periodic crystals (Γ-only or k-point sampled).
///
/// Typical usage:
/// \code
///   XtbCalculator calc(molecule);
///   calc.set_charge(0);
///   const XtbResult &r = calc.single_point();
///   double E = r.total_energy;
///   Mat3N grad = calc.gradient();
/// \endcode
class XtbCalculator {
public:
  /// Tight-binding method. Only GFN2 is implemented; the enum exists so the
  /// API can extend to GFN1 / GFN0 later without breaking callers.
  enum class Method { GFN2 };

  /// Construct from an isolated molecule (positions in Å — converted to Bohr
  /// internally). Inherits the molecule's charge.
  explicit XtbCalculator(const occ::core::Molecule &mol);
  /// Construct from a dimer; equivalent to a molecule built from the union of
  /// monomer atoms.
  explicit XtbCalculator(const occ::core::Dimer &dimer);
  /// Construct from a 3D periodic crystal. Defaults to Γ-only sampling; call
  /// `set_kpoints` to enable a Monkhorst-Pack mesh.
  explicit XtbCalculator(const occ::crystal::Crystal &crystal);

  // Move-only — IntegralEngine (held inside Gfn2Engine) owns a
  // unique_ptr<ESPEvaluator> so default copy is implicitly deleted.
  XtbCalculator(XtbCalculator &&) = default;
  XtbCalculator &operator=(XtbCalculator &&) = default;
  XtbCalculator(const XtbCalculator &) = delete;
  XtbCalculator &operator=(const XtbCalculator &) = delete;

  // ---------------------------------------------------------------------
  // Identity / topology
  // ---------------------------------------------------------------------

  /// Tight-binding method enum (currently always `Method::GFN2`).
  Method method() const { return m_method; }
  /// Method name as a string ("GFN2"). Convenience for bindings.
  std::string method_name() const { return "GFN2"; }
  /// Backend name as a string ("Native"). Distinguishes from `TbliteCalculator`.
  std::string backend_name() const { return "Native"; }
  /// True if the calculator was constructed from a `Crystal`.
  bool is_periodic() const { return m_periodic; }
  /// Number of atoms in the (central) cell.
  int num_atoms() const { return m_atomic_numbers.rows(); }
  /// Atomic numbers (length = `num_atoms()`).
  const IVec &atomic_numbers() const { return m_atomic_numbers; }
  /// Cartesian positions in Bohr (3 × N).
  const Mat3N &positions() const { return m_positions_bohr; }
  /// Lattice vectors as columns in Bohr (periodic only — throws otherwise).
  Mat3 lattice() const;

  // ---------------------------------------------------------------------
  // Configuration (setter / getter pairs)
  // ---------------------------------------------------------------------

  /// Net molecular / unit-cell charge in electrons.
  void set_charge(double c);
  double charge() const { return m_charge; }

  /// Number of unpaired electrons (open-shell). Only `n == 0` is currently
  /// supported; non-zero values throw `std::runtime_error`. The setter is
  /// kept for API parity with `TbliteCalculator`.
  void set_num_unpaired_electrons(int n);
  int num_unpaired_electrons() const { return 0; }

  /// Maximum SCC iterations.
  void set_max_iterations(int iterations);
  int max_iterations() const;

  /// Electronic temperature (K) for Fermi smearing.
  void set_temperature(double temp);
  double temperature() const;

  /// SCC mixer damping factor (weight on the previous iteration; ∈ [0, 1)).
  void set_mixer_damping(double damping_factor);
  double mixer_damping() const;

  /// Toggle CAMM multipole AES + on-site polarization (`true` for full GFN2).
  /// For the periodic k-point path this also enables the per-k Bloch-summed
  /// multipole AO matrices.
  void set_include_multipoles(bool on);
  bool include_multipoles() const;

  /// Toggle native D4 dispersion.
  void set_include_dispersion(bool on);
  bool include_dispersion() const;

  /// Monkhorst-Pack k-grid for the periodic SCC. `(1, 1, 1)` (default) uses
  /// the real-arithmetic Γ-only path; larger grids use the complex-eigensolve
  /// k-point path. No-op for molecular calculators.
  void set_kpoints(int n1, int n2, int n3);
  /// Read back the configured k-mesh.
  std::array<int, 3> kpoints() const;

  /// Enable an implicit-solvent model by name. Not yet implemented in the
  /// native backend — currently always returns `false`. Kept for API parity
  /// with `TbliteCalculator::set_solvent`. Real built-in models will land in
  /// Phase 7B (CPCM-X) / 7C (SMD); until then, use `set_solvation_model`
  /// directly to inject a custom `XtbSolvationModel`.
  bool set_solvent(const std::string &name);

  /// Attach an `XtbSolvationModel` (or clear it by passing `nullptr`). The
  /// model is consumed by the molecular SCC each iteration — gas-phase for
  /// `nullptr` or a `NullSolvationModel`. No-op on the periodic path until
  /// Phase 7B's periodic CPCM-X lands. The calculator keeps a shared
  /// reference; the same model can be re-attached after
  /// `update_structure(...)` and will rebuild its cavity on the next SCC.
  void set_solvation_model(std::shared_ptr<XtbSolvationModel> model);
  const std::shared_ptr<XtbSolvationModel> &solvation_model() const;

  // ---------------------------------------------------------------------
  // Geometry update — atomic numbers / shell layout are kept fixed.
  // ---------------------------------------------------------------------

  /// Update Cartesian positions (Bohr). Recomputes geometry-dependent caches
  /// (S, H0, γ, multipole AO blocks for periodic).
  void update_structure(const Mat3N &positions);
  /// Update positions and lattice vectors simultaneously (periodic only).
  void update_structure(const Mat3N &positions, const Mat3 &lattice_bohr);

  // ---------------------------------------------------------------------
  // Run + result access
  // ---------------------------------------------------------------------

  /// Run an SCC at the current geometry / configuration. Returns the cached
  /// result; subsequent calls re-run.
  const XtbResult &single_point();
  /// Convenience wrapper around `single_point()` returning just the total
  /// energy (Hartree).
  double single_point_energy();
  /// Read-only access to the most recent SCC result (call after
  /// `single_point()`).
  const XtbResult &last_result() const { return m_last_result; }

  // ---------------------------------------------------------------------
  // Derived quantities (post-SCC)
  // ---------------------------------------------------------------------

  /// Per-atom Mulliken charges (xtb convention: positive = electron-deficient).
  Vec charges() const;
  /// Wiberg bond-order matrix (N × N).
  Mat bond_orders() const;

  /// Total energy from the most recent SCC (Hartree).
  double total_energy() const { return m_last_result.total_energy; }
  /// Electronic + isotropic-Coulomb + (multipole AES if on) energy (Hartree).
  double scc_energy() const { return m_last_result.scc_energy; }
  /// Closed-form repulsion energy (Hartree).
  double repulsion_energy() const { return m_last_result.repulsion_energy; }
  /// Dispersion (D4) energy (Hartree); zero if dispersion is disabled.
  double dispersion_energy() const { return m_last_result.dispersion_energy; }

  // ---------------------------------------------------------------------
  // Gradient
  // ---------------------------------------------------------------------

  /// Analytical nuclear gradient (Hartree/Bohr, 3 × N). Runs a charge-only
  /// SCC internally (no anisotropic multipole contribution to the gradient
  /// energy) so the returned (energy, gradient) pair is self-consistent;
  /// see `compute_gradient_analytical` body for the breakdown.
  Mat3N gradient();
  /// Numerical 5-point central-difference gradient. Slow (6N SCC evals);
  /// useful as an oracle to validate the analytical version.
  Mat3N gradient_numerical(double step_bohr = 1e-3);
  /// (energy, gradient) pair. Uses analytical by default; pass
  /// `numerical = true` for the FD oracle.
  std::pair<double, Mat3N>
  compute_energy_and_gradient(bool numerical = false,
                              double step_bohr = 1e-3);

  // ---------------------------------------------------------------------
  // Hessian / vibrations
  // ---------------------------------------------------------------------

  /// Numerical 3N×3N Hessian (Hartree/Bohr²) via 5-point central
  /// differences of `gradient()`. Costs `6N` analytical-gradient evaluations
  /// (each is one multipole-on SCC + the analytical gradient assembly).
  /// The returned matrix is symmetrised. Inherits `gradient()`'s
  /// convention — full multipole-on (CAMM AES + on-site polarization)
  /// since Phase 5d-rest landed.
  Mat compute_hessian_numerical(double step_bohr = 0.005);

  /// Convenience: build the Hessian and run a normal-mode analysis,
  /// returning frequencies (cm⁻¹), normal modes, and the mass-weighted
  /// Hessian. Calls `compute_hessian_numerical` then
  /// `core::compute_vibrational_modes` with the calculator's atomic masses
  /// + positions. Set `project_tr_rot = true` to project translations and
  /// rotations out of the mass-weighted Hessian (ORCA-style PROJECTTR).
  occ::core::VibrationalModes
  compute_vibrational_modes(double step_bohr = 0.005,
                             bool project_tr_rot = false);

  // ---------------------------------------------------------------------
  // Conversion
  // ---------------------------------------------------------------------

  /// Snapshot the current atoms / positions as an `occ::core::Molecule`
  /// (positions returned in Å).
  occ::core::Molecule to_molecule() const;
  /// Snapshot the current periodic system as an `occ::crystal::Crystal`
  /// (periodic only — throws otherwise).
  occ::crystal::Crystal to_crystal() const;
  /// Convert the converged SCC state into a `qm::Wavefunction` so the rest
  /// of occ (FCHK output, Mulliken / Hirshfeld analysis, ESP, ...) can
  /// consume the GFN2 result. Requires `single_point()` to have run.
  occ::qm::Wavefunction to_wavefunction() const;

  /// Print a summary (energy decomposition, charges, HOMO/LUMO/gap) at
  /// INFO log level. Useful as the post-SCF print step in `occ scf`.
  void print_summary() const;

  // ---------------------------------------------------------------------
  // Deprecated: kept for backwards compatibility — prefer the names above.
  // ---------------------------------------------------------------------

  /// \deprecated Use `gradient_numerical` instead.
  Mat3N compute_gradient_numerical(double step_bohr = 1e-3) {
    return gradient_numerical(step_bohr);
  }
  /// \deprecated Use `gradient` instead.
  Mat3N compute_gradient_analytical() { return gradient(); }

private:
  void initialize_calculator();

  Mat3N m_positions_bohr;
  IVec m_atomic_numbers;
  double m_charge{0};
  Method m_method{Method::GFN2};
  SccOptions m_opts;

  Gfn2Parameters m_params;
  // Engine owns the basis / integrals / H0 / γ for the molecular SCC. Built
  // lazily — molecular ctors emplace it; the periodic ctor leaves it empty
  // since the periodic SCC drivers don't go through Gfn2Engine.
  std::optional<Gfn2Engine> m_calc;
  XtbResult m_last_result;

  // Periodic state — only populated when built from a Crystal.
  bool m_periodic{false};
  PeriodicSystem m_periodic_sys;
  PeriodicSccOptions m_periodic_opts;
  int m_kpoints[3]{1, 1, 1};
};

} // namespace occ::xtb
