#include <algorithm>
#include <cctype>
#include <cmath>
#include <fmt/format.h>
#include <occ/io/grid_settings.h>
#include <stdexcept>

namespace occ::io {

// ORCA IntAcc values for COSX grids (from Table 2.47)
constexpr double COSX_INT_ACC_GRID1 = 3.816;
constexpr double COSX_INT_ACC_GRID2 = 4.020;
constexpr double COSX_INT_ACC_GRID3 = 4.338;

// ORCA 5-region angular grids (from Table 2.46)
// Format: {Region1, Region2, Region3, Region4, Region5}
constexpr std::array<size_t, 5> ORCA_ANGULAR_GRID1 = {14, 26, 50, 50, 26};
constexpr std::array<size_t, 5> ORCA_ANGULAR_GRID2 = {14, 26, 50, 110, 50};
constexpr std::array<size_t, 5> ORCA_ANGULAR_GRID3 = {26, 50, 110, 194, 110};

// Helper: get periodic table row for an element
static size_t get_periodic_row(size_t atomic_number) {
    if (atomic_number <= 2) return 1;        // H, He
    if (atomic_number <= 10) return 2;       // Li-Ne
    if (atomic_number <= 18) return 3;       // Na-Ar
    if (atomic_number <= 36) return 4;       // K-Kr
    if (atomic_number <= 54) return 5;       // Rb-Xe
    if (atomic_number <= 86) return 6;       // Cs-Rn
    return 7;                                 // Fr+
}

GridSettings GridSettings::from_grid_quality(GridQuality quality) {
  GridSettings settings;
  switch (quality) {
  case GridQuality::Coarse:
    settings.radial_points = 23;
    settings.max_angular_points = 170;
    settings.min_angular_points = 110;
    break;
  case GridQuality::Standard:
    settings.radial_points = 50;
    settings.max_angular_points = 194;
    settings.min_angular_points = 110;
    break;
  case GridQuality::Fine:
    settings.radial_points = 75;
    settings.max_angular_points = 302;
    settings.min_angular_points = 194;
    break;
  case GridQuality::VeryFine:
    settings.radial_points = 99;
    settings.max_angular_points = 590;
    settings.min_angular_points = 302;
    break;
  }
  return settings;
}

GridSettings GridSettings::for_sgx(size_t angular_points) {
  GridSettings settings;
  // SGX/COSX uses coarser, uniform grids
  // ORCA defaults: 50 angular points, ~25 radial points
  settings.min_angular_points = angular_points;
  settings.max_angular_points = angular_points;
  settings.radial_precision = 1e-5;  // Looser than DFT default (1e-12)
  settings.pruning_scheme = PruningScheme::None;  // Uniform angular grid
  settings.reduced_first_row_element_grid = false;  // Same grid for all elements
  return settings;
}

GridSettings GridSettings::for_cosx(COSXGridLevel level) {
  GridSettings settings;

  switch (level) {
  case COSXGridLevel::Grid1:
    settings.int_acc = COSX_INT_ACC_GRID1;
    settings.angular_regions = ORCA_ANGULAR_GRID1;
    settings.max_angular_points = 50;
    settings.min_angular_points = 14;
    break;
  case COSXGridLevel::Grid2:
    settings.int_acc = COSX_INT_ACC_GRID2;
    settings.angular_regions = ORCA_ANGULAR_GRID2;
    settings.max_angular_points = 110;
    settings.min_angular_points = 14;
    break;
  case COSXGridLevel::Grid3:
    settings.int_acc = COSX_INT_ACC_GRID3;
    settings.angular_regions = ORCA_ANGULAR_GRID3;
    settings.max_angular_points = 194;
    settings.min_angular_points = 26;
    break;
  }

  // COSX uses Treutler-Alrichs M3 mapping with Gauss-Chebyshev
  settings.treutler_alrichs_adjustment = true;
  // Reduced grid for H/He (one angular level lower)
  settings.reduced_first_row_element_grid = true;
  // Radial points calculated per-atom using IntAcc formula
  settings.radial_points = 0; // Will be calculated per-atom

  return settings;
}

std::string GridSettings::int_acc_string() const {
  if (int_acc > 0) {
    return fmt::format("{:.3f}", int_acc);
  }
  return "N/A";
}

GridSettings get_grid_settings(GridQuality quality) {
  return GridSettings::from_grid_quality(quality);
}

std::string grid_quality_to_string(GridQuality quality) {
  switch (quality) {
  case GridQuality::Coarse:
    return "Coarse";
  case GridQuality::Standard:
    return "Standard";
  case GridQuality::Fine:
    return "Fine";
  case GridQuality::VeryFine:
    return "VeryFine";
  default:
    return "Unknown";
  }
}

GridQuality grid_quality_from_string(const std::string &str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "coarse")
    return GridQuality::Coarse;
  if (lower_str == "standard")
    return GridQuality::Standard;
  if (lower_str == "fine")
    return GridQuality::Fine;
  if (lower_str == "veryfine")
    return GridQuality::VeryFine;
  if (lower_str == "very_fine")
    return GridQuality::VeryFine;
  if (lower_str == "very-fine")
    return GridQuality::VeryFine;

  throw std::invalid_argument("Unknown grid quality: " + str);
}

std::string cosx_grid_level_to_string(COSXGridLevel level) {
  switch (level) {
  case COSXGridLevel::Grid1:
    return "GRIDX 1";
  case COSXGridLevel::Grid2:
    return "GRIDX 2";
  case COSXGridLevel::Grid3:
    return "GRIDX 3";
  default:
    return "Unknown";
  }
}

double cosx_grid_int_acc(COSXGridLevel level) {
  switch (level) {
  case COSXGridLevel::Grid1:
    return COSX_INT_ACC_GRID1;
  case COSXGridLevel::Grid2:
    return COSX_INT_ACC_GRID2;
  case COSXGridLevel::Grid3:
    return COSX_INT_ACC_GRID3;
  default:
    return 4.0;
  }
}

size_t calculate_radial_points_orca(double int_acc, size_t atomic_number, double b) {
  // ORCA formula: nr = (15 * IntAcc - 40) + b * ROW
  size_t row = get_periodic_row(atomic_number);
  double nr = (15.0 * int_acc - 40.0) + b * static_cast<double>(row);
  return static_cast<size_t>(std::max(1.0, std::round(nr)));
}

void print_cosx_grid_summary(const GridSettings& settings, COSXGridLevel level,
                             size_t num_points, size_t num_batches, size_t num_atoms) {
  fmt::print("\n--------------------\n");
  fmt::print("COSX GRID GENERATION\n");
  fmt::print("--------------------\n\n");

  fmt::print("{}\n", cosx_grid_level_to_string(level));
  fmt::print("{:-<{}}\n", "", cosx_grid_level_to_string(level).length());

  fmt::print("{:<44} {:>12}\n", "General Integration Accuracy", fmt::format("IntAcc      ... {:.3f}", settings.int_acc));
  fmt::print("{:<44} {:>12}\n", "Radial Grid Type", "RadialGrid  ... M3 (Treutler-Alrichs)");

  // Angular grid info
  int angular_level = 0;
  if (level == COSXGridLevel::Grid1) angular_level = 1;
  else if (level == COSXGridLevel::Grid2) angular_level = 2;
  else if (level == COSXGridLevel::Grid3) angular_level = 3;
  fmt::print("{:<44} {:>12}\n", "Angular Grid (max. ang.)",
             fmt::format("AngularGrid ... {} (Lebedev-{})", angular_level, settings.max_angular_points));
  fmt::print("{:<44} {:>12}\n", "Angular grid pruning method", "GridPruning ... 5-region (ORCA-style)");
  fmt::print("{:<44} {:>12}\n", "Weight generation scheme", "WeightScheme... Becke");

  if (settings.reduced_first_row_element_grid) {
    fmt::print("Angular grids for H and He will be reduced by one unit\n");
  }

  fmt::print("\n");
  fmt::print("{:<44} {:>12}\n", "Total number of grid points", fmt::format("... {:>8}", num_points));
  fmt::print("{:<44} {:>12}\n", "Total number of batches", fmt::format("... {:>8}", num_batches));
  if (num_batches > 0) {
    double avg_per_batch = static_cast<double>(num_points) / num_batches;
    fmt::print("{:<44} {:>12}\n", "Average number of points per batch", fmt::format("... {:>8.0f}", avg_per_batch));
  }
  if (num_atoms > 0) {
    double avg_per_atom = static_cast<double>(num_points) / num_atoms;
    fmt::print("{:<44} {:>12}\n", "Average number of grid points per atom", fmt::format("... {:>8.0f}", avg_per_atom));
  }

  fmt::print("\n");
}

} // namespace occ::io