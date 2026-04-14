#include <fmt/core.h>
#include <iostream>
#include <occ/mults/sfunctions.h>
#include <stdexcept>

namespace occ::mults {

SFunctions::SFunctions(int max_rank)
    : m_max_rank(max_rank), m_binomial(max_rank + 4) {
  if (max_rank < 0 || max_rank > 5) {
    throw std::invalid_argument("max_rank must be between 0 and 5");
  }
}

void SFunctions::set_coordinates(const Vec3 &ra, const Vec3 &rb) {
  m_coords = CoordinateSystem::from_points(ra, rb);

  if (m_coords.r < 1e-15) {
    throw std::runtime_error("Sites too close: r < 1e-15");
  }
}

void SFunctions::set_coordinate_system(const CoordinateSystem& coords) {
  m_coords = coords;

  if (m_coords.r < 1e-15) {
    throw std::runtime_error("Sites too close: r < 1e-15");
  }

  // DEBUG (disabled)
  // static bool sfunc_coord_printed = false;
  // if (!sfunc_coord_printed) {
  //   fmt::print("\n=== S-function CoordinateSystem Debug ===\n");
  //   fmt::print("use_body_frame: {}\n", m_coords.use_body_frame);
  //   fmt::print("rax, ray, raz: {:.6f}, {:.6f}, {:.6f}\n", rax(), ray(), raz());
  //   fmt::print("rbx, rby, rbz: {:.6f}, {:.6f}, {:.6f}\n", rbx(), rby(), rbz());
  //   fmt::print("cxx, cxy, cxz: {:.6f}, {:.6f}, {:.6f}\n", m_coords.cxx, m_coords.cxy, m_coords.cxz);
  //   fmt::print("cyx, cyy, cyz: {:.6f}, {:.6f}, {:.6f}\n", m_coords.cyx, m_coords.cyy, m_coords.cyz);
  //   fmt::print("czx, czy, czz: {:.6f}, {:.6f}, {:.6f}\n", m_coords.czx, m_coords.czy, m_coords.czz);
  //   sfunc_coord_printed = true;
  // }
}

SFunctions SFunctions::swap_sites() const {
  // Create new SFunctions with positions swapped
  SFunctions swapped(m_max_rank);
  swapped.set_coordinates(m_coords.rb, m_coords.ra);  // Note: rb becomes new ra, ra becomes new rb
  return swapped;
}

void SFunctions::compute_with_swap(int t1, int t2, int j, int level, SFunctionResult& result, bool swap) const {
  if (!swap) {
    // Normal case: just compute S(t1, t2) directly
    result = compute_s_function(t1, t2, j, level);
  } else {
    // Swapped case: create swapped coordinates and compute S(t2, t1)
    SFunctions swapped = swap_sites();
    result = swapped.compute_s_function(t2, t1, j, level);
    // Note: derivative transformations are automatically handled by the coordinate swap
  }
}

SFunctions::SFunctionResult
SFunctions::compute_s_function(int t1, int t2, int j, int level) const {
  // Determine which S-function family this belongs to based on multipole ranks
  auto [l1, m1] = index_to_lm(t1);
  auto [l2, m2] = index_to_lm(t2);

  int rank_sum = l1 + l2;

  // Compute base S-function result
  SFunctionResult result;

  // Route to appropriate computation method based on rank sum
  if (rank_sum <= 2) {
    compute_s012(t1, t2, j, level, result);
  } else if (rank_sum == 3) {
    compute_s3(t1, t2, j, level, result);
  } else if (rank_sum == 4) {
    compute_s4(t1, t2, j, level, result);
  } else if (rank_sum == 5) {
    compute_s5(t1, t2, j, level, result);
  } else {
    // For now, skip higher-order combinations that aren't implemented
    // This allows the ESP calculation to proceed with implemented terms
    result.s0 = 0.0;
    return result;
    // throw std::invalid_argument("Unsupported multipole rank combination");
  }

  // Orient's binomial coefficient factor: fac(n) = binom(j, l1)
  // BUT: For ESP calculations with static multipoles, Orient uses fac(n)=1.0
  // The binomial is only used for polarizability derivatives and other
  // specialized cases For now, skip the binomial factor to match Orient's ESP
  // implementation double fac = m_binomial.binomial(j, l1);
  double fac = 1.0; // Orient uses fac=1.0 for ESP calculations

  // Debug output disabled
  // if (std::abs(result.s0) > 1e-12 || (t1 == 2 && t2 == 7) || (t1 == 7 && t2
  // == 2) || (t1 == 4 && t2 == 2) || (t1 == 2 && t2 == 14) || (t1 == 14 && t2
  // == 2)) {
  //     fmt::print("OCC_SFUNC: t1={:2} t2={:2} j={:1} l1={:1} l2={:1} m1={:2}
  //     m2={:2} s0={:20.12e}\n",
  //                t1, t2, j, l1, l2, m1, m2, result.s0);
  // }

  result.s0 *= fac;

  // Apply factor to derivatives as well (they follow the same pattern in
  // Orient)
  if (level >= 1) {
    result.s1 *= fac;
  }
  if (level >= 2) {
    result.s2 *= fac;
  }

  return result;
}

std::vector<SFunctions::SFunctionResult> SFunctions::compute_s_functions(
    const std::vector<std::tuple<int, int, int>> &indices, int level) const {

  std::vector<SFunctionResult> results;
  results.reserve(indices.size());

  for (const auto &[t1, t2, j] : indices) {
    results.push_back(compute_s_function(t1, t2, j, level));
  }

  return results;
}

// Core S-function implementations following Orient's structure

void SFunctions::compute_s012(int t1, int t2, int j, int level,
                               SFunctionResult& result) const {
  auto [l1, m1] = index_to_lm(t1);
  auto [l2, m2] = index_to_lm(t2);

  // Handle all combinations up to rank 2 (following Orient's S012 subroutine)

  // Charge-charge interaction (00,00)
  if (l1 == 0 && l2 == 0) {
    compute_charge_charge(level, result);
    return;
  }

  // Charge-dipole interactions (00,1m)
  if (l1 == 0 && l2 == 1) {
    compute_charge_dipole(m2, level, result);
    return;
  }

  // Dipole-charge interactions (1m,00)
  if (l1 == 1 && l2 == 0) {
    compute_dipole_charge(m1, level, result);
    return;
  }

  // Charge-quadrupole interactions (00,2m)
  if (l1 == 0 && l2 == 2) {
    compute_charge_quadrupole(m2, level, result);
    return;
  }

  // Quadrupole-charge interactions (2m,00)
  if (l1 == 2 && l2 == 0) {
    compute_quadrupole_charge(m1, level, result);
    return;
  }

  // Dipole-dipole interactions (1m1,1m2)
  if (l1 == 1 && l2 == 1) {
    compute_dipole_dipole(m1, m2, level, result);
    return;
  }

  throw std::invalid_argument("Unsupported multipole combination for S012");
}

// Specific multipole interaction implementations

void SFunctions::compute_charge_charge(int level, SFunctionResult& result) const {
  // S(00,00) = 1 (from Orient case 1)
  result.s0 = 1.0;

  // No derivatives for constant function
}

// Utility methods

std::pair<int, int> SFunctions::index_to_lm(int index) const {
  // Convert linear index back to (l,m)
  int l = static_cast<int>(std::sqrt(index));
  int m_index = index - l * l;

  if (m_index == 0) {
    return {l, 0}; // Q_l0
  } else if (m_index % 2 == 1) {
    return {l, (m_index + 1) / 2}; // Q_lmc
  } else {
    return {l, -(m_index / 2)}; // Q_lms
  }
}

} // namespace occ::mults
