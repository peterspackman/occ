#include <occ/core/log.h>
#include <occ/mults/sfunctions.h>

namespace occ::mults {

// Forward declarations for kernel functions defined in sfunctions_s3_kernels.cpp
namespace kernels {
    void dipole_z_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
}

void
SFunctions::compute_dipole_quadrupole(int dip_comp, int quad_comp,
                                      int level, SFunctionResult& result) const {
  using namespace kernels;

  // Dipole-quadrupole S-functions from Orient's S3 subroutine (cases 34-48)
  // Dipole @ A (uses rax, ray, raz), Quadrupole @ B (uses rbx, rby, rbz)
  // Dipole component: 0=Q10 (z), 1=Q11c (x), -1=Q11s (y)
  // Quadrupole component: 0=Q20, 1=Q21c, -1=Q21s, 2=Q22c, -2=Q22s

  // Q10 (dipole z @ A) × quadrupole @ B
  if (dip_comp == 0) {
    if (quad_comp == 0)  { dipole_z_quadrupole_20(*this, level, result); return; }   // Orient case 34
    if (quad_comp == 1)  { dipole_z_quadrupole_21c(*this, level, result); return; }  // Orient case 35
    if (quad_comp == -1) { dipole_z_quadrupole_21s(*this, level, result); return; }  // Orient case 36
    if (quad_comp == 2)  { dipole_z_quadrupole_22c(*this, level, result); return; }  // Orient case 37
    if (quad_comp == -2) { dipole_z_quadrupole_22s(*this, level, result); return; }  // Orient case 38
  }
  // Q11c (dipole x @ A) × quadrupole @ B
  else if (dip_comp == 1) {
    if (quad_comp == 0)  { dipole_x_quadrupole_20(*this, level, result); return; }   // Orient case 39
    if (quad_comp == 1)  { dipole_x_quadrupole_21c(*this, level, result); return; }  // Orient case 40
    if (quad_comp == -1) { dipole_x_quadrupole_21s(*this, level, result); return; }  // Orient case 41
    if (quad_comp == 2)  { dipole_x_quadrupole_22c(*this, level, result); return; }  // Orient case 42
    if (quad_comp == -2) { dipole_x_quadrupole_22s(*this, level, result); return; }  // Orient case 43
  }
  // Q11s (dipole y @ A) × quadrupole @ B
  else if (dip_comp == -1) {
    if (quad_comp == 0)  { dipole_y_quadrupole_20(*this, level, result); return; }   // Orient case 44
    if (quad_comp == 1)  { dipole_y_quadrupole_21c(*this, level, result); return; }  // Orient case 45
    if (quad_comp == -1) { dipole_y_quadrupole_21s(*this, level, result); return; }  // Orient case 46
    if (quad_comp == 2)  { dipole_y_quadrupole_22c(*this, level, result); return; }  // Orient case 47
    if (quad_comp == -2) { dipole_y_quadrupole_22s(*this, level, result); return; }  // Orient case 48
  }

  result.s0 = 0.0;
}

void
SFunctions::compute_quadrupole_dipole(int quad_comp, int dip_comp,
                                      int level, SFunctionResult& result) const {
  using namespace kernels;

  // Quadrupole-dipole S-functions from Orient's S3 subroutine (cases 49-63)
  // Quadrupole @ A (uses rax, ray, raz), Dipole @ B (uses rbx, rby, rbz)
  // Quadrupole component: 0=Q20, 1=Q21c, -1=Q21s, 2=Q22c, -2=Q22s
  // Dipole component: 0=Q10 (z), 1=Q11c (x), -1=Q11s (y)

  // Q20 (quadrupole @ A) × dipole @ B
  if (quad_comp == 0) {
    if (dip_comp == 0)  { quadrupole_20_dipole_z(*this, level, result); return; }   // Orient case 49
    if (dip_comp == 1)  { quadrupole_20_dipole_x(*this, level, result); return; }   // Orient case 50
    if (dip_comp == -1) { quadrupole_20_dipole_y(*this, level, result); return; }   // Orient case 51
  }
  // Q21c (quadrupole @ A) × dipole @ B
  else if (quad_comp == 1) {
    if (dip_comp == 0)  { quadrupole_21c_dipole_z(*this, level, result); return; }  // Orient case 52
    if (dip_comp == 1)  { quadrupole_21c_dipole_x(*this, level, result); return; }  // Orient case 53
    if (dip_comp == -1) { quadrupole_21c_dipole_y(*this, level, result); return; }  // Orient case 54
  }
  // Q21s (quadrupole @ A) × dipole @ B
  else if (quad_comp == -1) {
    if (dip_comp == 0)  { quadrupole_21s_dipole_z(*this, level, result); return; }  // Orient case 55
    if (dip_comp == 1)  { quadrupole_21s_dipole_x(*this, level, result); return; }  // Orient case 56
    if (dip_comp == -1) { quadrupole_21s_dipole_y(*this, level, result); return; }  // Orient case 57
  }
  // Q22c (quadrupole @ A) × dipole @ B
  else if (quad_comp == 2) {
    if (dip_comp == 0)  { quadrupole_22c_dipole_z(*this, level, result); return; }  // Orient case 58
    if (dip_comp == 1)  { quadrupole_22c_dipole_x(*this, level, result); return; }  // Orient case 59
    if (dip_comp == -1) { quadrupole_22c_dipole_y(*this, level, result); return; }  // Orient case 60
  }
  // Q22s (quadrupole @ A) × dipole @ B
  else if (quad_comp == -2) {
    if (dip_comp == 0)  { quadrupole_22s_dipole_z(*this, level, result); return; }  // Orient case 61
    if (dip_comp == 1)  { quadrupole_22s_dipole_x(*this, level, result); return; }  // Orient case 62
    if (dip_comp == -1) { quadrupole_22s_dipole_y(*this, level, result); return; }  // Orient case 63
  }

  result.s0 = 0.0;
}

void SFunctions::compute_s3(int t1, int t2, int j, int level,
                            SFunctionResult& result) const {
  auto [l1, m1] = index_to_lm(t1);
  auto [l2, m2] = index_to_lm(t2);

  if (l1 + l2 != 3) {
    occ::log::error(
        "compute_s3 called with l1+l2 != 3: l1={}, l2={}, t1={}, t2={}", l1, l2,
        t1, t2);
  }

  // Handle rank sum = 3 cases following Orient's S3 subroutine

  // Charge-octopole interactions (00,3m)
  if (l1 == 0 && l2 == 3) {
    switch (m2) {
      case 0:  kernels::charge_octopole_30(*this, level, result); return;
      case 1:  kernels::charge_octopole_31c(*this, level, result); return;
      case -1: kernels::charge_octopole_31s(*this, level, result); return;
      case 2:  kernels::charge_octopole_32c(*this, level, result); return;
      case -2: kernels::charge_octopole_32s(*this, level, result); return;
      case 3:  kernels::charge_octopole_33c(*this, level, result); return;
      case -3: kernels::charge_octopole_33s(*this, level, result); return;
    }
    return;
  }

  // Handle symmetric cases using swapped coordinates and phase factors
  if (l1 == 3 && l2 == 0) {
    SFunctions swapped_sf(m_max_rank);
    swapped_sf.set_coordinates(m_coords.rb, m_coords.ra); // Swap A and B
    swapped_sf.compute_s3(0, t1, j, level, result);

    // Apply phase factor: For octopole-charge symmetry, use (-1)^(l1+1)
    // Orient convention: even m gives phase +1, odd m gives phase +1 too (no
    // flip needed) Actually, checking Orient output: we DON'T apply a phase for
    // these symmetric swaps The coordinate swap (ra<->rb) handles the sign
    // change via rbx->-rbx etc. So just return the swapped result as-is
    return;
  }

  // Note: Dipole-octopole (l1=1, l2=3) belongs in compute_s4, not here
  // since l1 + l2 = 4, not 3

  // Dipole-quadrupole interactions (l1=1, l2=2)
  if (l1 == 1 && l2 == 2) {
    // Use helper function that calls kernel functions
    // m1: dipole component (0=z, 1=x, -1=y)
    // m2: quadrupole component (0=Q20, 1=Q21c, -1=Q21s, 2=Q22c, -2=Q22s)
    compute_dipole_quadrupole(m1, m2, level, result);
    return;
  }

  // Quadrupole-dipole interactions (l1=2, l2=1)
  // Orient cases 49-63: These have DIFFERENT formulas than dipole-quadrupole!
  if (l1 == 2 && l2 == 1) {
    // Use helper function that calls kernel functions
    // m1: quadrupole component (0=Q20, 1=Q21c, -1=Q21s, 2=Q22c, -2=Q22s)
    // m2: dipole component (0=z, 1=x, -1=y)
    compute_quadrupole_dipole(m1, m2, level, result);
    return;
  }

  throw std::invalid_argument(
      "Unsupported S3 multipole combination: l1=" + std::to_string(l1) +
      ", l2=" + std::to_string(l2));
}

} // namespace occ::mults
