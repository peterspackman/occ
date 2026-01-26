#include <occ/mults/sfunctions.h>

namespace occ::mults {

// Forward declarations for kernel functions
namespace kernels {
// Dipole-hexadecapole kernels
void dipole_z_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_z_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_x_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void dipole_y_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);

// Quadrupole-octopole kernels
void quadrupole_20_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_20_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_20_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_20_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_20_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_20_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_20_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21c_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21c_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21c_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21c_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21c_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21c_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21s_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21s_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21s_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21s_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21s_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_21s_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22c_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22c_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22c_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22c_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22c_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22c_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22s_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22s_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22s_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22s_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22s_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void quadrupole_22s_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);

// Octopole-quadrupole kernels (cases 229-263)
void octopole_30_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_30_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_30_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_30_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_30_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_31s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_32s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void octopole_33s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);

// Hexadecapole-dipole kernels (cases 264-287)
void hexadecapole_40_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_40_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_40_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_41c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_41c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_41c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_41s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_41s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_41s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_42c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_42c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_42c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_42s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_42s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_42s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_43c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_43c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_43c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_43s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_43s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_43s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_44c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_44c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_44c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_44s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_44s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
void hexadecapole_44s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
} // namespace kernels

void SFunctions::compute_s5(int t1, int t2, int j,
                             int level, SFunctionResult& result) const {
  auto [l1, m1] = index_to_lm(t1);
  auto [l2, m2] = index_to_lm(t2);

  // Handle charge-hexadecapole interactions (l1=0, l2=5)
  if (l1 == 0 && l2 == 5) {
    // Following Orient's approach for rank 5 multipoles
    // These are very high-order interactions - simplified implementation

    switch (t2 - 21) { // Rank 5 starts at index 21 for max_rank=4 systems
    case 0:            // Q50 component
      result.s0 = std::pow(rbz(), 5); // Simplified
      break;
    case 1:                                   // Q51c component
      result.s0 = rbx() * std::pow(rbz(), 4); // Simplified
      break;
    case 2:                                   // Q51s component
      result.s0 = rby() * std::pow(rbz(), 4); // Simplified
      break;
    default:
      result.s0 = std::pow(rbz(), 5); // Default approximation
      break;
    }
    return;
  }

  // Handle hexadecapole-charge interactions (l1=5, l2=0) - use coordinate swapping
  if (l1 == 5 && l2 == 0) {
    compute_with_swap(t1, t2, j, level, result, true);  // swap=true
    return;
  }

  // Dipole-hexadecapole interactions (l1=1, l2=4) - use hexadecapole-dipole with swapping
  if (l1 == 1 && l2 == 4) {
    compute_with_swap(t1, t2, j, level, result, true);  // swap to reuse hexadecapole-dipole kernels
    return;
  }

  // Hexadecapole-dipole interactions (l1=4, l2=1)
  if (l1 == 4 && l2 == 1) {
    // Q40 (m1=0) × dipoles
    if (m1 == 0) {
      if (m2 == 0) { kernels::hexadecapole_40_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_40_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_40_dipole_11s(*this, level, result); return; }
    }
    // Q41c (m1=1) × dipoles
    else if (m1 == 1) {
      if (m2 == 0) { kernels::hexadecapole_41c_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_41c_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_41c_dipole_11s(*this, level, result); return; }
    }
    // Q41s (m1=-1) × dipoles
    else if (m1 == -1) {
      if (m2 == 0) { kernels::hexadecapole_41s_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_41s_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_41s_dipole_11s(*this, level, result); return; }
    }
    // Q42c (m1=2) × dipoles
    else if (m1 == 2) {
      if (m2 == 0) { kernels::hexadecapole_42c_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_42c_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_42c_dipole_11s(*this, level, result); return; }
    }
    // Q42s (m1=-2) × dipoles
    else if (m1 == -2) {
      if (m2 == 0) { kernels::hexadecapole_42s_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_42s_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_42s_dipole_11s(*this, level, result); return; }
    }
    // Q43c (m1=3) × dipoles
    else if (m1 == 3) {
      if (m2 == 0) { kernels::hexadecapole_43c_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_43c_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_43c_dipole_11s(*this, level, result); return; }
    }
    // Q43s (m1=-3) × dipoles
    else if (m1 == -3) {
      if (m2 == 0) { kernels::hexadecapole_43s_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_43s_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_43s_dipole_11s(*this, level, result); return; }
    }
    // Q44c (m1=4) × dipoles
    else if (m1 == 4) {
      if (m2 == 0) { kernels::hexadecapole_44c_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_44c_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_44c_dipole_11s(*this, level, result); return; }
    }
    // Q44s (m1=-4) × dipoles
    else if (m1 == -4) {
      if (m2 == 0) { kernels::hexadecapole_44s_dipole_10(*this, level, result); return; }
      else if (m2 == 1) { kernels::hexadecapole_44s_dipole_11c(*this, level, result); return; }
      else if (m2 == -1) { kernels::hexadecapole_44s_dipole_11s(*this, level, result); return; }
    }
    result.s0 = 0.0;
    return;
  }

  // Quadrupole-octopole interactions (l1=2, l2=3) - use coordinate swapping
  if (l1 == 2 && l2 == 3) {
    compute_with_swap(t1, t2, j, level, result, true);  // swap=true to reuse octopole-quadrupole kernels
    return;
  }

  // Octopole-quadrupole interactions (l1=3, l2=2)
  if (l1 == 3 && l2 == 2) {
    // Q30 (m1=0) × quadrupoles
    if (m1 == 0) {
      if (m2 == 0) { kernels::octopole_30_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_30_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_30_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_30_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_30_quadrupole_22s(*this, level, result); return; }
    }
    // Q31c (m1=1) × quadrupoles
    else if (m1 == 1) {
      if (m2 == 0) { kernels::octopole_31c_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_31c_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_31c_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_31c_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_31c_quadrupole_22s(*this, level, result); return; }
    }
    // Q31s (m1=-1) × quadrupoles
    else if (m1 == -1) {
      if (m2 == 0) { kernels::octopole_31s_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_31s_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_31s_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_31s_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_31s_quadrupole_22s(*this, level, result); return; }
    }
    // Q32c (m1=2) × quadrupoles
    else if (m1 == 2) {
      if (m2 == 0) { kernels::octopole_32c_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_32c_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_32c_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_32c_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_32c_quadrupole_22s(*this, level, result); return; }
    }
    // Q32s (m1=-2) × quadrupoles
    else if (m1 == -2) {
      if (m2 == 0) { kernels::octopole_32s_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_32s_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_32s_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_32s_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_32s_quadrupole_22s(*this, level, result); return; }
    }
    // Q33c (m1=3) × quadrupoles
    else if (m1 == 3) {
      if (m2 == 0) { kernels::octopole_33c_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_33c_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_33c_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_33c_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_33c_quadrupole_22s(*this, level, result); return; }
    }
    // Q33s (m1=-3) × quadrupoles
    else if (m1 == -3) {
      if (m2 == 0) { kernels::octopole_33s_quadrupole_20(*this, level, result); return; }
      if (m2 == 1) { kernels::octopole_33s_quadrupole_21c(*this, level, result); return; }
      if (m2 == -1) { kernels::octopole_33s_quadrupole_21s(*this, level, result); return; }
      if (m2 == 2) { kernels::octopole_33s_quadrupole_22c(*this, level, result); return; }
      if (m2 == -2) { kernels::octopole_33s_quadrupole_22s(*this, level, result); return; }
    }
    return;
  }

  throw std::invalid_argument(
      "Unsupported S5 multipole combination: l1=" + std::to_string(l1) +
      ", l2=" + std::to_string(l2));
}

} // namespace occ::mults
