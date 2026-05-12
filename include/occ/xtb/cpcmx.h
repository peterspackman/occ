#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/surface.h>
#include <occ/xtb/solvation_interface.h>
#include <string>

namespace occ::xtb {

/// Construction options for `CpcmXSolvationModel`. Kept as a thin user-facing
/// surface; internally translated to `occ::scrf::Options` before being handed
/// to the underlying `ReactionFieldEngine`.
struct CpcmXOptions {
  /// Solvent name; looked up via `occ::solvent::get_dielectric` unless
  /// `dielectric_override > 0`.
  std::string solvent{"water"};
  /// Override the dielectric directly (useful for sweeps / regression tests).
  /// Any value `> 0` takes precedence over `solvent`.
  double dielectric_override{-1.0};
  /// COSMO/CPCM scaling factor in `f(ε) = (ε − 1) / (ε + x)`. The default
  /// `x = 0` is the ideal-conductor convention used by tblite's CPCM. The
  /// classical Klamt COSMO value is `x = 0.5`.
  double x{0.0};
  /// Solvent probe radius (Å) handed to the cavity builder. Note: the current
  /// `occ::solvent::surface::solvent_surface` implementation clamps this to
  /// 0.001 Å — left as-is to match the rest of the OCC pipeline.
  double probe_radius_angs{0.4};
  /// Smoothing width (Bohr) used by `solvent_surface` to soften the boolean
  /// cavity mask into a continuous erf-based weight. Zero falls back to the
  /// legacy hard-mask cavity and disables the smooth-cavity diagonal term
  /// in the analytical gradient. 0.1 Bohr is the validated default.
  double smoothing_width_bohr{0.1};
};

/// CPCM-X (xtb-flavoured CPCM) implicit-solvent model.
///
/// Treats each atom as a point charge of magnitude `q_atom` (the SCC's
/// converged Mulliken charge) producing an electrostatic potential
/// `φ_i = Σ_a q_atom_a / |r_i − R_a|` at cavity point `i`. Solves the COSMO
/// linear system `A σ = −f(ε) φ` for surface charges σ, then folds back the
/// atom-resolved screening potential `V_solv_a = Σ_i σ_i / |R_a − r_i|`. The
/// variational solvation energy is `E_solv = ½ q · V_solv = ½ σ · φ`.
///
/// Implementation is a thin adapter over `occ::scrf::ReactionFieldEngine`
/// (Phase 2). Kept as its own class because the SCC interacts with solvation
/// via the `XtbSolvationModel` virtual contract; the engine itself is just
/// the shared CPCM/COSMO math.
class CpcmXSolvationModel final : public XtbSolvationModel {
public:
  explicit CpcmXSolvationModel(CpcmXOptions opts = {});

  void initialize(const Mat3N &positions_bohr,
                  const IVec &atomic_numbers) override;
  void update(const Vec &atomic_charges) override;
  const Vec &atom_potential() const override { return m_engine.atom_potential(); }
  double energy() const override { return m_engine.energy(); }
  std::string name() const override;
  std::optional<SolvationSurfaces> surfaces() const override;
  Mat3N gradient() const override { return m_engine.gradient(); }

  /// Number of cavity surface elements (post-masking). May be zero for an
  /// empty / pathological input.
  size_t num_surface_points() const {
    return m_engine.num_es_surface_points();
  }
  const occ::solvent::surface::Surface &surface() const {
    return m_engine.es_cavity();
  }
  /// Surface apparent charge σ at the most recent `update()`. Empty until
  /// `update()` is called.
  const Vec &surface_charges() const { return m_engine.surface_charges(); }
  double dielectric() const { return m_engine.dielectric(); }
  double f_epsilon() const { return m_engine.f_epsilon(); }

  /// Access the underlying engine (for inspection / tests).
  const occ::scrf::ReactionFieldEngine &engine() const { return m_engine; }

private:
  CpcmXOptions m_opts;
  occ::scrf::ReactionFieldEngine m_engine;
};

} // namespace occ::xtb
