#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/solvent/surface.h>
#include <occ/xtb/solvation_interface.h>
#include <string>

namespace occ::xtb {

/// Construction options for `CpcmXSolvationModel`.
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
/// Heavy lifting happens once per `initialize()` (LU of `A`; assembly of the
/// dense atom-resolved response operator `J_solv = −f(ε) B^T A^{-1} B`).
/// `update(q)` is then two matrix-vector products and a dot product — cheap
/// inside the SCC iteration loop.
class CpcmXSolvationModel final : public XtbSolvationModel {
public:
  explicit CpcmXSolvationModel(CpcmXOptions opts = {});

  void initialize(const Mat3N &positions_bohr,
                  const IVec &atomic_numbers) override;
  void update(const Vec &atomic_charges) override;
  const Vec &atom_potential() const override { return m_v_solv; }
  double energy() const override { return m_energy; }
  std::string name() const override;

  /// Number of cavity surface elements (post-masking). May be zero for an
  /// empty / pathological input.
  size_t num_surface_points() const { return m_surface.areas.size(); }
  const occ::solvent::surface::Surface &surface() const { return m_surface; }
  /// Surface apparent charge σ at the most recent `update()`. Empty until
  /// `update()` is called.
  const Vec &surface_charges() const { return m_sigma; }
  double dielectric() const { return m_epsilon; }
  double f_epsilon() const { return m_f_eps; }

private:
  CpcmXOptions m_opts;
  double m_epsilon{1.0};
  double m_f_eps{0.0};

  occ::solvent::surface::Surface m_surface;
  // Pre-solved `−f(ε) · A^{-1} · B` (used to reconstruct σ for inspection).
  Mat m_G;
  // Atom-resolved response: `J_solv = B^T · G`. Symmetric, negative-definite.
  Mat m_J_solv;

  // State refreshed by `update(q)`.
  Vec m_sigma;
  Vec m_v_solv;
  double m_energy{0.0};
};

} // namespace occ::xtb
