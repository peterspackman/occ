#pragma once
#include <occ/core/linear_algebra.h>
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
/// Per-element decomposition (Phase 7D handle):
///   • `cds_energy_elements()[i] = (σ_atom_of(i) + γ)·area_i / scale` — fixed.
///   • ES per-element energy `½·σ_i·φ_i` is reconstructible on demand from
///     `surface_charges()` and the cavity geometry; the SCC tracks the
///     atom-resolved view (`atom_potential()`) for Fock-shift purposes.
class SmdSolvationModel final : public XtbSolvationModel {
public:
  explicit SmdSolvationModel(std::string solvent = "water");

  void initialize(const Mat3N &positions_bohr,
                  const IVec &atomic_numbers) override;
  void update(const Vec &atomic_charges) override;
  const Vec &atom_potential() const override { return m_v_solv; }
  double energy() const override { return m_energy; }
  std::string name() const override;
  std::optional<SolvationSurfaces> surfaces() const override;
  Mat3N gradient() const override;

  // ---------------------------------------------------------------------
  // Inspection / Phase 7D hooks
  // ---------------------------------------------------------------------

  const occ::solvent::SMDSolventParameters &parameters() const {
    return m_params;
  }
  double dielectric() const { return m_epsilon; }

  // ES (Coulomb) surface — feeds the SCC.
  const occ::solvent::surface::Surface &es_surface() const {
    return m_es_surface;
  }
  size_t num_es_surface_points() const { return m_es_surface.areas.size(); }
  /// Apparent surface charge σ at the latest `update(q)`. Empty until then.
  const Vec &surface_charges() const { return m_sigma; }

  // CDS surface — geometry-only.
  const occ::solvent::surface::Surface &cds_surface() const {
    return m_cds_surface;
  }
  size_t num_cds_surface_points() const { return m_cds_surface.areas.size(); }
  /// Per-element CDS energy contribution (Hartree). Length = number of CDS
  /// surface points. Stable across `update()` calls (geometry only).
  const Vec &cds_energy_elements() const { return m_cds_energy_elements; }

  // Energy split (last `update`).
  double e_es() const { return m_e_es; }
  double e_cds() const { return m_e_cds; }

private:
  std::string m_solvent;
  occ::solvent::SMDSolventParameters m_params;
  double m_epsilon{1.0};

  // Geometry cached at initialize() so the gradient method can recompute
  // atomic_surface_tension at displaced positions without needing the caller
  // to thread state through.
  Mat3N m_atom_positions;   // Bohr
  IVec m_atomic_numbers;
  // ES cavity radii (Bohr) — handed to the gradient so the smooth-cavity
  // diagonal term can reconstruct weights consistent with the cavity build.
  Vec m_es_radii;
  // Smoothing width used for both cavities (Bohr). See CpcmXOptions for the
  // rationale behind the 0.1 default.
  double m_smoothing_width_bohr{0.1};

  // ES branch
  occ::solvent::surface::Surface m_es_surface;
  Mat m_B;       // ncav_es × natom (kept for per-element ES decomposition)
  Mat m_G;       // ncav_es × natom
  Mat m_J_solv;  // natom × natom (symmetric, neg-def)

  // CDS branch (geometry-only)
  occ::solvent::surface::Surface m_cds_surface;
  Vec m_cds_energy_elements;
  // Per-atom sum of CDS surface area, converted to Å² and weighted by the
  // per-element scale factor — precomputed so the numerical CDS gradient
  // only needs to re-evaluate `atomic_surface_tension(R)` per FD step.
  Vec m_cds_area_per_atom_angs;
  double m_cds_total_area_angs{0.0};
  double m_e_cds{0.0};

  // Refreshed per update
  Vec m_atomic_charges;  // last q (needed alongside σ for the ES gradient)
  Vec m_sigma;
  Vec m_phi;       // ncav_es: per-element source potential φ = B·q
  Vec m_v_solv;
  double m_e_es{0.0};
  double m_energy{0.0};
};

} // namespace occ::xtb
