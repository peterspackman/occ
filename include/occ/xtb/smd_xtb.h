#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/smd_parameters.h>
#include <occ/solvent/surface.h>
#include <occ/xtb/solvation_interface.h>
#include <string>

namespace occ::xtb {

/// SMD ("Solvation Model based on Density") for GFN-xTB.
///
/// Two cavities:
///   • Electrostatic (ES) surface with SMD intrinsic Coulomb radii — feeds a
///     classical-COSMO ASC solve (same machinery as `CpcmXSolvationModel`,
///     just with different radii).
///   • CDS surface with SMD CDS radii — used purely geometrically to evaluate
///     the cavitation–dispersion–solvent rearrangement (CDS) energy
///     `E_cds = (Σ_a σ_a(geom)·A_a + γ·A_total) / (1000·E_h→kcal)`. The CDS
///     piece does not depend on the SCC charges; it is fixed once the
///     geometry is known and just rides along inside `energy()`.
///
/// Phase 2: this class is a thin adapter over `occ::scrf::ReactionFieldEngine`
/// configured with `Radii::SmdIntrinsicCoulomb + include_cds = true`. The xTB
/// SCC consumes solvation via the `XtbSolvationModel` virtual contract —
/// keeping the class makes that hookup transparent and gives callers an
/// "SMD-shaped" handle for inspection.
class SmdSolvationModel final : public XtbSolvationModel {
public:
  explicit SmdSolvationModel(std::string solvent = "water");

  void initialize(const Mat3N &positions_bohr,
                  const IVec &atomic_numbers) override;
  void update(const Vec &atomic_charges) override;
  const Vec &atom_potential() const override { return m_engine.atom_potential(); }
  double energy() const override { return m_engine.energy(); }
  std::string name() const override;
  std::optional<SolvationSurfaces> surfaces() const override;
  Mat3N gradient() const override { return m_engine.gradient(); }

  // ---------------------------------------------------------------------
  // Inspection / Phase 7D hooks
  // ---------------------------------------------------------------------

  const occ::solvent::SMDSolventParameters &parameters() const {
    return m_engine.smd_parameters();
  }
  double dielectric() const { return m_engine.dielectric(); }

  // ES (Coulomb) surface — feeds the SCC.
  const occ::solvent::surface::Surface &es_surface() const {
    return m_engine.es_cavity();
  }
  size_t num_es_surface_points() const {
    return m_engine.num_es_surface_points();
  }
  /// Apparent surface charge σ at the latest `update(q)`. Empty until then.
  const Vec &surface_charges() const { return m_engine.surface_charges(); }

  // CDS surface — geometry-only.
  const occ::solvent::surface::Surface &cds_surface() const {
    return m_engine.cds_cavity();
  }
  size_t num_cds_surface_points() const {
    return m_engine.num_cds_surface_points();
  }
  /// Per-element CDS energy contribution (Hartree). Length = number of CDS
  /// surface points. Stable across `update()` calls (geometry only).
  const Vec &cds_energy_elements() const {
    return m_engine.cds_energy_elements();
  }

  // Energy split (last `update`).
  double e_es() const { return m_engine.energy_es(); }
  double e_cds() const { return m_engine.energy_cds(); }

  /// Access the underlying engine (for inspection / tests).
  const occ::scrf::ReactionFieldEngine &engine() const { return m_engine; }

private:
  std::string m_solvent;
  occ::scrf::ReactionFieldEngine m_engine;
};

} // namespace occ::xtb
