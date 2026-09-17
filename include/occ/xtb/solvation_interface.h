#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/scrf/surfaces.h>
#include <optional>
#include <string>

namespace occ::xtb {

/// Per-element solvation surface data, consumed by the crystal-growth
/// (`occ::cg`) energy decomposition. Phase 2 unification: these are now
/// aliases for the shared `occ::scrf` types — the xTB and HF/DFT solvation
/// pipelines speak the same per-element shape.
using SolvationSurface = occ::scrf::SolvationSurface;
using SolvationSurfaces = occ::scrf::SolvationSurfaces;

/// Abstract interface for an implicit-solvent contribution to a GFN-xTB SCC.
///
/// Concrete models (CPCM-X in Phase 7B, SMD in Phase 7C) implement an
/// atom-resolved potential shift folded into the per-shell isotropic V plus
/// a scalar energy term added to the SCC breakdown. The contract is:
///
///   1. `initialize(positions, Z)` is called once at the top of an SCC, after
///      the engine has built its geometry caches. The model sizes its
///      internal buffers and (re)builds the cavity here.
///   2. `update(atom_charges)` is called at the start of every SCC iteration
///      with the input-iter atomic Mulliken charges (length = N_atoms). The
///      model solves its surface response and caches energy + potential.
///   3. `atom_potential()` returns the cached per-atom V_solv (Hartree),
///      length = N_atoms. The engine adds V_solv[atom_of(s)] to the per-shell
///      iso V before forming H.
///   4. `energy()` returns the cached scalar contribution to scc_energy,
///      Hartree, evaluated against the same `atom_charges` passed to
///      `update()`.
///
/// `NullSolvationModel` below is a no-op implementation used as a gate during
/// Phase 7A — it must not perturb gas-phase numbers.
class XtbSolvationModel {
public:
  virtual ~XtbSolvationModel() = default;

  virtual void initialize(const Mat3N &positions_bohr,
                          const IVec &atomic_numbers) = 0;

  virtual void update(const Vec &atomic_charges) = 0;

  /// Update from the atomic charges together with the CAMM atomic dipoles
  /// (3 × N) and traceless quadrupoles (6 × N, `CammMoments::qp` layout), so
  /// the anisotropic part of the density polarises the continuum too. The
  /// default drops them, which leaves charge-only models unchanged.
  virtual void update(const Vec &atomic_charges, const Mat3N & /*dipoles*/,
                      const Mat & /*quadrupoles*/) {
    update(atomic_charges);
  }

  /// Hand the model the solute's own short-range damping radii for the
  /// atom→cavity dipole and quadrupole kernels, so a multipolar reaction field
  /// treats the CAMM moments the way the rest of the method does. Called once
  /// per geometry, before the first `update`. The default ignores them.
  virtual void set_multipole_damping(const Vec & /*rco_bohr*/,
                                     double /*kdmp3*/, double /*kdmp5*/) {}

  virtual const Vec &atom_potential() const = 0;

  /// Conjugates of the atomic dipoles and quadrupoles at the last update:
  /// ∂E_solv/∂μ (3 × N) and ∂E_solv/∂Θ (6 × N, same layout as the input). The
  /// SCC folds these into its anisotropic potentials so the reaction field
  /// reaches the Fock matrix through the multipole channels as well as the
  /// charge one. Both are empty unless the model took the multipole update.
  virtual const Mat3N &dipole_potential() const {
    static const Mat3N none;
    return none;
  }
  virtual const Mat &quadrupole_potential() const {
    static const Mat none;
    return none;
  }

  /// ∂E_solv/∂R_co per atom, for damping radii that depend on the geometry.
  /// The caller closes the chain through its own ∂R_co/∂CN and ∂CN/∂R. Empty
  /// when the model applies no damping.
  virtual const Vec &damping_radius_gradient() const {
    static const Vec none;
    return none;
  }

  virtual double energy() const = 0;

  virtual std::string name() const = 0;

  /// Optional per-element decomposition of the latest solvation contribution.
  /// Concrete models (CPCM-X, SMD) override; the default returns
  /// `std::nullopt`. Reflects the state at the most recent `update(q)`.
  virtual std::optional<SolvationSurfaces> surfaces() const {
    return std::nullopt;
  }

  /// Analytical gradient of the solvation energy with respect to atomic
  /// positions (Hartree/Bohr, 3 × N_atoms). Frozen-cavity convention — the
  /// cavity points move rigidly with their parent atoms and per-element areas
  /// are constant. Concrete models override; the default returns an empty
  /// matrix so callers can detect "no gradient available" without surprise.
  virtual Mat3N gradient() const { return Mat3N(); }
};

/// No-op solvation model — preserves gas-phase numbers. Used as the Phase 7A
/// correctness gate and as a sentinel for "solvation off" in code paths that
/// already hold a model pointer.
class NullSolvationModel final : public XtbSolvationModel {
public:
  void initialize(const Mat3N & /*positions_bohr*/,
                  const IVec &atomic_numbers) override {
    m_potential = Vec::Zero(atomic_numbers.size());
  }

  using XtbSolvationModel::update;
  void update(const Vec & /*atomic_charges*/) override {}

  const Vec &atom_potential() const override { return m_potential; }

  double energy() const override { return 0.0; }

  std::string name() const override { return "null"; }

private:
  Vec m_potential;
};

} // namespace occ::xtb
