#pragma once
#include <occ/core/linear_algebra.h>
#include <optional>
#include <string>

namespace occ::xtb {

/// Per-element solvation surface data, the Phase 7D shape consumed by the
/// crystal-growth (`occ::cg`) energy decomposition.
///
/// All quantities are in atomic units: positions in Bohr, areas in Bohr²,
/// energies in Hartree. `atom_index(i)` is the atomic index this element was
/// generated on (0-based, < num_atoms).
struct SolvationSurface {
  Mat3N positions;
  Vec areas;
  IVec atom_index;
  Vec energies;

  size_t size() const { return static_cast<size_t>(areas.size()); }
  double total_energy() const { return energies.sum(); }
  double total_area() const { return areas.sum(); }
};

/// Bundle of optional surfaces — `coulomb` is the electrostatic cavity, `cds`
/// is SMD's cavitation–dispersion–solvent-rearrangement cavity. CPCM-X
/// populates `coulomb` only; SMD populates both. `NullSolvationModel`
/// returns no surfaces at all (the parent optional in `XtbResult` is empty).
struct SolvationSurfaces {
  std::optional<SolvationSurface> coulomb;
  std::optional<SolvationSurface> cds;

  double total_energy() const {
    double e = 0.0;
    if (coulomb)
      e += coulomb->total_energy();
    if (cds)
      e += cds->total_energy();
    return e;
  }
};

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

  virtual const Vec &atom_potential() const = 0;

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

  void update(const Vec & /*atomic_charges*/) override {}

  const Vec &atom_potential() const override { return m_potential; }

  double energy() const override { return 0.0; }

  std::string name() const override { return "null"; }

private:
  Vec m_potential;
};

} // namespace occ::xtb
