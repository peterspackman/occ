#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/numint/molecular_grid.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/opmatrix.h>
#include <occ/ints/esp.h>
#include <ankerl/unordered_dense.h>
#include <vector>
#include <memory>
#include <mutex>

namespace occ::qm::cosx {

/// Configuration settings for COSX seminumerical exchange
struct Settings {
  double screen_threshold{1e-4};      // Shell extent screening threshold (looser = smaller extents = more screening)
  double margin{1.0};                 // Geometric margin (Bohr)
  // Shell-pair cutoff: a pair is skipped for a grid batch when its ESP
  // estimate times max|F| falls below this, scaled by sqrt of the batch's
  // largest quadrature weight.
  //
  // The binding constraint is not the grid error but SCF convergence. This
  // error is density dependent, so unlike the systematic grid error it acts
  // as noise and sets a floor on how far DIIS can drive |FDS-SDF|: 1e-5 costs
  // extra iterations on compact molecules and 1e-4 stalls the SCF outright.
  // 1e-7 keeps two decades of margin on that while still leaving the
  // screening error some 1000x under the grid error at every grid level.
  // See the [cosx][screenbench] scan.
  double f_threshold{1e-7};
};

// Import grid types from occ::dft namespace
using occ::numint::MolecularGrid;
using occ::numint::AtomGrid;
using occ::numint::GridSettings;

/// The exchange grid in point batches: every atomic grid's points and
/// weights concatenated, each batch a contiguous (start, count) range of one
/// atom's points.
struct GridBatches {
    occ::Mat3N points;
    occ::Vec weights;
    std::vector<std::pair<Eigen::Index, Eigen::Index>> ranges;
};

struct GridBatchInfo {
    occ::Vec3 center;      // Centroid of grid points in batch
    double radius;         // Max distance from center to any point in batch
};

// Compute bounding sphere for a batch of grid points
GridBatchInfo compute_batch_info(Eigen::Ref<const occ::Mat3N> pts_block);

struct ScreenedShellPairs {
    std::vector<size_t> list1;  // Shell pair indices where both shells are near
    std::vector<size_t> list2;  // Shell pair indices where one shell is near
    // Note: list3 pairs are implicitly those not in list1 or list2 (skipped)
};

// Screen shell pairs based on distance from grid batch
// Returns indices of shell pairs that should be evaluated
// screening_extents: shell extents calculated with looser threshold (1e-6) for screening
ScreenedShellPairs screen_shell_pairs(
    const GridBatchInfo &batch,
    const std::vector<occ::gto::Shell> &shells,
    const Vec &screening_extents,
    double margin = 1.0  // Bohr
);

/// ORCA-style SHARK shell lists (shell-based, not pair-based)
/// See: Helmich-Paris et al. J. Chem. Phys. 155, 104109 (2021)
struct ShellLists {
    std::vector<size_t> list1;  // Shells geometrically close to batch
    std::vector<size_t> list2;  // Shells with significant F intermediate (subset of list1 based on density)
    std::vector<size_t> list3;  // Shells with significant differential overlap for ESP (subset of list2)
};

/// Screen shells based on geometric proximity to grid batch
/// Returns list-1: shells where distance - extent < batch_radius + margin
std::vector<size_t> screen_shells_geometric(
    const GridBatchInfo &batch,
    const std::vector<occ::gto::Shell> &shells,
    const Vec &screening_extents,
    double margin = 1.0  // Bohr
);

/// Screen shells based on F intermediate magnitude (density-weighted basis values)
/// Returns list-2: shells from list1 where max|F_shell| > threshold
/// F = sum_kappa P_kappa,lambda * X_kappa,g  (Eq. 8 in paper)
std::vector<size_t> screen_shells_density(
    const std::vector<size_t> &list1,
    const Mat &Fg,  // F intermediate: (npts x nbf)
    const std::vector<occ::gto::Shell> &shells,
    const std::vector<int> &first_bf,
    double threshold = 1e-10
);

/// Screen shells based on differential overlap for ESP integrals
/// Returns list-3: shells from list2 with non-negligible overlap at grid points
/// This is based on the "chain of spheres" - each basis function has a sphere
/// outside of which it's negligible
std::vector<size_t> screen_shells_overlap(
    const std::vector<size_t> &list2,
    const GridBatchInfo &batch,
    const std::vector<occ::gto::Shell> &shells,
    const Vec &screening_extents,
    double threshold = 1e-12
);

/// Build all three shell lists for a batch
ShellLists build_shell_lists(
    const GridBatchInfo &batch,
    const Mat &Fg,
    const std::vector<occ::gto::Shell> &shells,
    const std::vector<int> &first_bf,
    const Vec &screening_extents,
    double f_threshold = 1e-10,
    double overlap_threshold = 1e-12,
    double margin = 1.0
);

class SemiNumericalExchange {

public:
  SemiNumericalExchange(const gto::AOBasis &, const occ::numint::GridSettings & = {});
  /// Seminumerical exchange for one operator.
  ///
  /// `omega` selects it: zero for the full Coulomb `1/r`, nonzero for the
  /// attenuated `erf(omega r)/r`. Passed rather than held, because nothing
  /// here is cached per operator -- the grid, the shell-pair map and the
  /// density contraction are shared, and only the analytic ESP integral looks
  /// at omega at all.
  Mat compute_K(const qm::MolecularOrbitals &mo,
                double precision = std::numeric_limits<double>::epsilon(),
                const occ::Mat &Schwarz = occ::Mat(),
                double omega = 0.0) const;

  /// Fused range-separated exchange: builds the full Hartree-Fock exchange
  /// matrix of a range-separated hybrid,
  ///     K = (alpha+beta)*K[1/r] - beta*K[erf(omega*r)/r],
  /// in a SINGLE grid sweep. The grid-point basis values, density contraction,
  /// shell-pair screening and final back-contraction are shared between the two
  /// operators; only the analytic ESP integral is evaluated twice per shell
  /// pair. Equivalent to compute_K(full) and compute_K(long-range) combined,
  /// but without duplicating the per-batch overhead. (Restricted only.)
  Mat compute_K_range_separated(
      const qm::MolecularOrbitals &mo, double alpha, double beta, double omega,
      double precision = std::numeric_limits<double>::epsilon()) const;

  /// K[D_k] for several densities in shared grid sweeps, in the convention
  /// of compute_K for a restricted mo.D = D_k (CPHF response densities).
  std::vector<Mat> compute_K_response(const std::vector<Mat> &D,
                                      double omega = 0.0) const;

  Mat compute_overlap_matrix() const;

  /// Nuclear gradient (3 x natoms) of the COSX exchange energy, -tr(D K)
  /// restricted or -1/2 sum_s tr(P^s K[P^s]) unrestricted, for the operator
  /// selected by `omega` as in compute_K.
  /// Includes the basis-function, ESP-integral, grid-point, partition-weight
  /// and overlap-fitting (Q = S_num^{-1} S) derivatives. `threshold` screens
  /// shell pairs on charge x |u| x |v|.
  Mat3N compute_K_gradient(const qm::MolecularOrbitals &mo, double omega = 0.0,
                           double threshold = 1e-12) const;

  /// Second-derivative terms of the COSX exchange energy E_x = -tr(D K)
  /// (restricted, D = C_o C_o^T) at fixed AO density, with every dependence
  /// compute_K_gradient differentiates, for the operator selected by `omega`.
  struct SecondDerivatives {
    Mat hessian;                  ///< d2 E_x / dR dR (3N x 3N)
    std::vector<Mat> fock;        ///< dK/dR C_o, K the SCF's exchange matrix
    /// d/dR (-1/2 dE_x/dD) C_o: the COSX Fock is not exactly dE_x/dD, and
    /// it is this, not `fock`, whose density response the gradient sees.
    std::vector<Mat> energy_fock;
  };
  SecondDerivatives compute_K_second_derivatives(const qm::MolecularOrbitals &mo,
                                                 double omega = 0.0,
                                                 double threshold = 1e-12) const;

  const auto &engine() const { return m_engine; }

  /// Enable or disable the ESP-based approach (default: disabled for backward compatibility)
  void set_use_esp(bool use_esp) { m_use_esp = use_esp; }
  bool use_esp() const { return m_use_esp; }

  /// Enable or disable spatial hierarchy (default: enabled)
  void set_use_spatial_hierarchy(bool use_hierarchy) { m_use_spatial_hierarchy = use_hierarchy; }
  bool use_spatial_hierarchy() const { return m_use_spatial_hierarchy; }

  /// Configuration settings
  void set_settings(const Settings &settings);
  const Settings &settings() const { return m_settings; }

  /// Whether the exchange build's pair screen uses the far-field form of the
  /// charge bound (on by default). A difference-density SCF build turns it
  /// off: there the looser screen's errors accumulate over the incremental
  /// builds instead of being redone each time.
  void set_far_field_screening(bool on) { m_far_field_screening = on; }

  /// Range-separation parameter for the exchange operator.
  /// Get grid information
  size_t num_grid_points() const;
  size_t num_batches() const;
  size_t num_atoms() const;

  /// Get the underlying molecular grid
  const occ::numint::MolecularGrid& grid() const { return m_grid; }

private:
  // Spinorbital-specific compute_K implementations
  Mat compute_K_restricted(const qm::MolecularOrbitals &mo, double precision,
                           double omega) const;
  Mat compute_K_unrestricted(const qm::MolecularOrbitals &mo, double precision,
                             double omega) const;
  Mat compute_K_general(const qm::MolecularOrbitals &mo, double precision,
                        double omega) const;

  // Core K computation for a projected density matrix (nbf x nbf)
  // Returns (nbf x nbf) exchange matrix
  Mat compute_K_for_density(const Mat &D2q, double precision,
                            double omega) const;

  // Fused full + long-range K for a projected density matrix: accumulates
  // w_full * K[1/r] + w_lr * K[erf(omega*r)/r] in a single grid sweep.
  Mat compute_K_fused_rs_for_density(const Mat &D2q, double precision,
                                     double w_full, double w_lr,
                                     double omega) const;

  // sum_g w phi (A Z phi)^T for each Z = D2q[k]^T, the screen applied to
  // each density scaled to unit size (shared multi-density sweeps).
  std::vector<Mat> exchange_sweep(const std::vector<Mat> &D2q,
                                  double omega) const;

  // S_num^X Y for every nuclear coordinate X (basis functions, points moving
  // with their parent atoms, partition weights).
  std::vector<Mat> numerical_overlap_derivatives(const Mat &Y) const;

  // Precompute the significant shell-pair list and the per-pair geometry used
  // for per-batch geometric screening. Idempotent (only runs once).
  void prepare_screening() const;

  // Spatial batches of the grid, built once on first use.
  const GridBatches &grid_batches() const;

  std::vector<occ::core::Atom> m_atoms;
  gto::AOBasis m_basis;
  occ::numint::MolecularGrid m_grid;
  mutable occ::qm::IntegralEngine m_engine;

  std::vector<occ::numint::AtomGrid> m_atom_grids;
  Mat m_overlap;
  Mat m_numerical_overlap, m_overlap_projector;

  // ESP-based approach members
  bool m_use_esp = false;
  bool m_use_spatial_hierarchy = true;
  mutable std::unique_ptr<occ::ints::ESPEvaluator<double>> m_esp_evaluator;
  mutable ankerl::unordered_dense::map<size_t, size_t> m_shell_pair_map;  // flat_idx -> ESP idx
  mutable std::vector<std::pair<size_t, size_t>> m_significant_pairs;  // (p, q) list

  // Per-pair screening data, indexed by ESP shell-pair index.
  //
  // m_pair_charge is max|S_pq|, the magnitude of the pair's charge
  // distribution. The ESP integral of a pair falls off as 1/R, so a pair can
  // only be dropped on the strength of its charge and of the F intermediate it
  // meets - never on its distance from the grid batch (see the screening note
  // in seminumerical_exchange.cpp). Pairs sharing an origin are pinned to 1:
  // their overlap can vanish by symmetry while their ESP does not.
  mutable Vec m_pair_charge;
  // far-field bound {q1, spread} (ESPEvaluator::pair_far_estimate): the
  // charge bound tightened to q1 / (R - spread) at distance R, tail kept
  mutable Mat m_pair_far;
  bool m_far_field_screening{true};
  mutable Mat3N m_pair_centers;
  mutable Vec m_pair_extents;

  mutable GridBatches m_batches;
  mutable std::once_flag m_batches_once;

  // Shell extents for screening (calculated with screen_threshold, not shell extent threshold)
  Vec m_screening_extents;

  // Configuration settings
  Settings m_settings;
};
} // namespace occ::qm::cosx
