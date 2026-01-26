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

namespace occ::qm::cosx {

/// Configuration settings for COSX seminumerical exchange
struct Settings {
  double screen_threshold{1e-4};      // Shell extent screening threshold (looser = smaller extents = more screening)
  double margin{1.0};                 // Geometric margin (Bohr)
  double f_threshold{1e-10};          // F-intermediate threshold
};

// Import grid types from occ::dft namespace
using occ::dft::MolecularGrid;
using occ::dft::AtomGrid;
using occ::dft::GridSettings;

struct GridBatchInfo {
    occ::Vec3 center;      // Centroid of grid points in batch
    double radius;         // Max distance from center to any point in batch
};

// Compute bounding sphere for a batch of grid points
GridBatchInfo compute_batch_info(const occ::Mat3N &pts_block);

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
  SemiNumericalExchange(const gto::AOBasis &, const GridSettings & = {});
  Mat compute_K(const qm::MolecularOrbitals &mo,
                double precision = std::numeric_limits<double>::epsilon(),
                const occ::Mat &Schwarz = occ::Mat()) const;

  Mat compute_overlap_matrix() const;

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

  /// Get grid information
  size_t num_grid_points() const;
  size_t num_batches() const;
  size_t num_atoms() const;

  /// Get the underlying molecular grid
  const MolecularGrid& grid() const { return m_grid; }

private:
  // Spinorbital-specific compute_K implementations
  Mat compute_K_restricted(const qm::MolecularOrbitals &mo, double precision) const;
  Mat compute_K_unrestricted(const qm::MolecularOrbitals &mo, double precision) const;
  Mat compute_K_general(const qm::MolecularOrbitals &mo, double precision) const;

  // Core K computation for a projected density matrix (nbf x nbf)
  // Returns (nbf x nbf) exchange matrix
  Mat compute_K_for_density(const Mat &D2q, double precision) const;

  std::vector<occ::core::Atom> m_atoms;
  gto::AOBasis m_basis;
  MolecularGrid m_grid;
  mutable occ::qm::IntegralEngine m_engine;

  std::vector<AtomGrid> m_atom_grids;
  Mat m_overlap;
  Mat m_numerical_overlap, m_overlap_projector;

  // ESP-based approach members
  bool m_use_esp = false;
  bool m_use_spatial_hierarchy = true;
  mutable std::unique_ptr<occ::ints::ESPEvaluator<double>> m_esp_evaluator;
  mutable ankerl::unordered_dense::map<size_t, size_t> m_shell_pair_map;  // flat_idx -> ESP idx
  mutable std::vector<std::pair<size_t, size_t>> m_significant_pairs;  // (p, q) list

  // Shell extents for screening (calculated with screen_threshold, not shell extent threshold)
  Vec m_screening_extents;

  // Configuration settings
  Settings m_settings;
};
} // namespace occ::qm::cosx
