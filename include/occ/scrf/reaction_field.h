#pragma once
#include <Eigen/LU>
#include <occ/core/linear_algebra.h>
#include <occ/scrf/cosmo_kernel.h>
#include <occ/scrf/surfaces.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/surface.h>
#include <string>

namespace occ::scrf {

/// Construction options for `ReactionFieldEngine`. One struct covers both
/// CPCM-X-style (vdW radii, no CDS) and SMD-style (intrinsic Coulomb radii +
/// CDS) configurations; concrete preset factories live below.
struct Options {
  /// Electrostatic kernel. Only CPCM is wired today; the slots are placeholders
  /// for future ALPB/GBSA Born-radii backends sharing the same engine API.
  enum class Backend { CPCM };

  /// Atomic-radii set used to build the ES cavity.
  ///   • CosmoVdW            : `solvent::cosmo::solvation_radii(Z)` (CPCM-X).
  ///   • SmdIntrinsicCoulomb : `solvent::smd::intrinsic_coulomb_radii(Z)` (SMD).
  ///   • Custom              : caller supplies `custom_es_radii_bohr` (e.g.
  ///                           DRACO-scaled charge-dependent radii).
  enum class Radii { CosmoVdW, SmdIntrinsicCoulomb, Custom };

  Backend backend{Backend::CPCM};
  Radii radii{Radii::CosmoVdW};

  /// Solvent name; looked up via `occ::solvent::get_dielectric` (and, when
  /// `include_cds` or `Radii::SmdIntrinsicCoulomb`, `get_smd_parameters`).
  std::string solvent{"water"};

  /// Override the dielectric directly. Any value `> 0` takes precedence over
  /// the solvent-name lookup. Useful for sweeps / regression tests.
  double dielectric_override{-1.0};

  /// COSMO/CPCM convention parameter in `f(ε) = (ε − 1)/(ε + x)`. `x = 0` is
  /// the CPCM ideal-conductor convention (used by tblite); `x = 0.5` is the
  /// classical Klamt COSMO.
  double f_eps_x{0.0};

  /// Solvent probe radius (Å) handed to the cavity builder.
  double probe_radius_angs{0.4};

  /// Smoothing width (Bohr) for the cavity weight. `0.0` falls back to the
  /// legacy boolean mask and disables the smooth-cavity diagonal gradient
  /// term. `0.1 Bohr` is the validated default (see `CpcmXOptions` notes).
  double smoothing_width_bohr{0.1};

  /// Build a second cavity for the SMD cavitation-dispersion-solvent term and
  /// pre-compute its per-element energy contributions. Only meaningful when
  /// a full SMD parameter set is available for `solvent`.
  bool include_cds{false};

  /// Custom ES-cavity radii (Bohr, length = N_atoms). Consumed when
  /// `radii == Radii::Custom`; ignored otherwise. Lets callers (DRACO,
  /// regression scripts) bypass the built-in radii tables.
  Vec custom_es_radii_bohr;

  // DRACO and CPCM-X σ-profile addons are not yet wired up; placeholders for
  // future work. The flags exist so callers can opt-in once support lands.
  bool draco_scaling{false};
  bool include_sigma_profile{false};
};

/// Self-consistent reaction-field engine — owns the cavity, the COSMO/CPCM
/// pre-factored response, and (optionally) the SMD CDS cavity. Provides both
/// an atom-resolved API (q → V_atom, used by xTB) and an Eulerian API
/// (φ at cavity → σ, used by HF/DFT).
///
/// Lifecycle:
///   1. construct with `Options`,
///   2. `initialize(atoms, Z)` to build the cavity and factor A,
///   3. per-iteration: `update_from_atom_charges(q)` OR `solve_asc(phi)`,
///   4. `energy()`, `surfaces()`, `gradient()` reflect the latest update.
///
/// `gradient()` is the frozen-cavity analytical gradient (plus FD CDS when
/// `include_cds`). It requires an atom-resolved update (it goes through the
/// SCC's Mulliken charges); calling it after a pure Eulerian solve returns an
/// empty matrix.
class ReactionFieldEngine {
public:
  explicit ReactionFieldEngine(Options opts = {});

  /// Build the cavity, pre-factor the COSMO A matrix, and (if
  /// `include_cds`) pre-evaluate the CDS branch. Idempotent on geometry
  /// change — call again whenever atoms move.
  void initialize(const Mat3N &atom_positions_bohr,
                  const IVec &atomic_numbers);

  /// Eulerian: solve `A σ = −f(ε)·φ_cavity`. Caches `σ` and `φ` so
  /// `energy_es()` / `surfaces()` reflect the result. Does NOT compute
  /// `V_atom` — use the atom-resolved path for that.
  void solve_asc(const Vec &phi_at_cavity);

  /// Atom-resolved: from input atomic charges, compute
  ///   φ = B·q,  σ = G·q,  V_atom = J_solv·q,  E_es = ½ q·V_atom.
  /// Updates all cached state; subsequent `energy()`, `surfaces()`, and
  /// `gradient()` reflect this update.
  void update_from_atom_charges(const Vec &atom_charges);

  /// Total cached solvation energy: E_es + E_cds. (E_cds = 0 when
  /// `include_cds == false`.)
  double energy() const { return m_e_es + m_e_cds; }
  double energy_es() const { return m_e_es; }
  double energy_cds() const { return m_e_cds; }

  /// Cached per-atom screening potential V_atom (Hartree, length = natom).
  /// Empty until the first atom-resolved update.
  const Vec &atom_potential() const { return m_v_solv; }

  /// Cached apparent surface charges σ on the ES cavity. Length = ncav_es.
  /// Empty before the first update.
  const Vec &surface_charges() const { return m_sigma; }

  /// Per-element decomposition of the latest update. Coulomb branch is
  /// populated whenever there has been an update; CDS branch is populated
  /// when `include_cds` (and reflects the cavity at `initialize()`, i.e.
  /// geometry-only).
  SolvationSurfaces surfaces() const;

  /// Frozen-cavity analytical gradient (Hartree/Bohr, 3 × natom). Requires
  /// a prior `update_from_atom_charges(q)` — returns an empty matrix
  /// otherwise. When `include_cds`, adds an FD CDS contribution on top.
  Mat3N gradient() const;

  // ---------------------------------------------------------------------
  // Inspection
  // ---------------------------------------------------------------------
  const occ::solvent::surface::Surface &es_cavity() const {
    return m_es_surface;
  }
  const occ::solvent::surface::Surface &cds_cavity() const {
    return m_cds_surface;
  }
  size_t num_es_surface_points() const {
    return static_cast<size_t>(m_es_surface.areas.size());
  }
  size_t num_cds_surface_points() const {
    return static_cast<size_t>(m_cds_surface.areas.size());
  }

  const Mat &B() const { return m_response.B; }
  const Mat &G() const { return m_response.G; }
  const Mat &J_solv() const { return m_response.J_solv; }

  double dielectric() const { return m_epsilon; }
  double f_epsilon() const { return m_f_eps; }
  const Options &options() const { return m_opts; }
  const occ::solvent::SMDSolventParameters &smd_parameters() const {
    return m_smd_params;
  }
  /// ES cavity radii (Bohr) used at initialize(). Empty before initialize().
  const Vec &es_atom_radii() const { return m_es_radii; }

  /// Per-element CDS energy (Hartree). Length = ncav_cds; empty if
  /// `include_cds == false`. Stable across `update*` calls (geometry-only).
  const Vec &cds_energy_elements() const { return m_cds_energy_elements; }

private:
  void rebuild_cds_branch();

  Options m_opts;

  // Filled at initialize()
  Mat3N m_atom_positions;       // Bohr
  IVec m_atomic_numbers;
  Vec m_es_radii;               // ES cavity radii (Bohr) used to build cavity
  double m_epsilon{1.0};
  double m_f_eps{0.0};
  occ::solvent::SMDSolventParameters m_smd_params;  // populated when needed

  occ::solvent::surface::Surface m_es_surface;
  // Pre-factored COSMO A on the ES cavity. Shared by atom-resolved and
  // Eulerian solves.
  Eigen::PartialPivLU<Mat> m_es_lu;
  detail::CosmoResponse m_response;

  // CDS branch (only populated when m_opts.include_cds)
  occ::solvent::surface::Surface m_cds_surface;
  Vec m_cds_energy_elements;
  double m_e_cds{0.0};

  // Refreshed on each update*()
  bool m_have_atom_charges{false};
  Vec m_atomic_charges;
  Vec m_phi;
  Vec m_sigma;
  Vec m_v_solv;
  double m_e_es{0.0};
};

} // namespace occ::scrf
