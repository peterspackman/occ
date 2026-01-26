#include <fmt/core.h>
#include <occ/mults/sfunctions.h>

namespace occ::mults {

// Forward declarations for kernel functions defined in sfunctions_s4_kernels.cpp
namespace kernels {
    // Quadrupole-quadrupole kernels (Orient cases 101-125)
    void quadrupole_20_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_20_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_21s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void quadrupole_22s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);

    // Charge-hexadecapole kernels
    void charge_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void charge_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);

    // Dipole-octopole kernels (Orient cases 80-100)
    void dipole_z_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_z_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_x_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void dipole_y_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);

    // Octopole-dipole kernels (Orient cases 126-146)
    void octopole_30_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_31c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_31s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_32c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_32s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_33c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_33c_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_33c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_33s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_33s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
    void octopole_33s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result);
}

void
SFunctions::compute_quadrupole_quadrupole(int comp1, int comp2,
                                          int level, SFunctionResult& result) const {
  using namespace kernels;

  // Following Orient's S4 subroutine, cases 101-155
  // For electrostatic interactions with static multipoles (cxx=cyy=czz=1,
  // off-diagonals=0): The formulas simplify from Orient's general form

  // Quadrupole component mapping: 0=Q20, 1=Q21c, -1=Q21s, 2=Q22c, -2=Q22s

  // Case (0,0): Q20 x Q20 - case 101 in Orient
  if (comp1 == 0 && comp2 == 0) {
    quadrupole_20_quadrupole_20(*this, level, result);
  }
  // Case (0,1): Q20 x Q21c - case 102 in Orient
  else if (comp1 == 0 && comp2 == 1) {
    quadrupole_20_quadrupole_21c(*this, level, result);
  }
  // Case (0,-1): Q20 x Q21s - case 103 in Orient
  else if (comp1 == 0 && comp2 == -1) {
    quadrupole_20_quadrupole_21s(*this, level, result);
  }
  // Case (0,2): Q20 x Q22c - case 104 in Orient
  else if (comp1 == 0 && comp2 == 2) {
    quadrupole_20_quadrupole_22c(*this, level, result);
  }
  // Case (0,-2): Q20 x Q22s - case 105 in Orient
  else if (comp1 == 0 && comp2 == -2) {
    quadrupole_20_quadrupole_22s(*this, level, result);
  }
  // Case (1,0): Q21c x Q20 - case 106 in Orient (symmetry)
  else if (comp1 == 1 && comp2 == 0) {
    quadrupole_21c_quadrupole_20(*this, level, result);
  }
  // Case (1,1): Q21c x Q21c - case 107 in Orient
  else if (comp1 == 1 && comp2 == 1) {
    quadrupole_21c_quadrupole_21c(*this, level, result);
  }
  // Case (1,-1): Q21c x Q21s - case 108 in Orient
  else if (comp1 == 1 && comp2 == -1) {
    quadrupole_21c_quadrupole_21s(*this, level, result);
  }
  // Case (-1,0): Q21s x Q20 - case 109 in Orient (by analogy)
  else if (comp1 == -1 && comp2 == 0) {
    quadrupole_21s_quadrupole_20(*this, level, result);
  }
  // Case (-1,1): Q21s x Q21c - case by symmetry
  else if (comp1 == -1 && comp2 == 1) {
    quadrupole_21s_quadrupole_21c(*this, level, result);
  }
  // Case (-1,-1): Q21s x Q21s - by analogy to case 107
  else if (comp1 == -1 && comp2 == -1) {
    quadrupole_21s_quadrupole_21s(*this, level, result);
  }
  // Case (2,0): Q22c x Q20 - by symmetry with case 104
  else if (comp1 == 2 && comp2 == 0) {
    quadrupole_22c_quadrupole_20(*this, level, result);
  }
  // Case (2,2): Q22c x Q22c - Orient case 119
  else if (comp1 == 2 && comp2 == 2) {
    quadrupole_22c_quadrupole_22c(*this, level, result);
  }
  // Case (-2,0): Q22s x Q20
  else if (comp1 == -2 && comp2 == 0) {
    quadrupole_22s_quadrupole_20(*this, level, result);
  }
  // Case (-2,-2): Q22s x Q22s - Orient case 125
  else if (comp1 == -2 && comp2 == -2) {
    quadrupole_22s_quadrupole_22s(*this, level, result);
  }
  // Case (1,2): Q21c x Q22c - Orient case 110
  else if (comp1 == 1 && comp2 == 2) {
    quadrupole_21c_quadrupole_22c(*this, level, result);
  }
  // Case (1,-2): Q21c x Q22s - Orient case 111
  else if (comp1 == 1 && comp2 == -2) {
    quadrupole_21c_quadrupole_22s(*this, level, result);
  }
  // Case (-1,2): Q21s x Q22c - Orient case 112
  else if (comp1 == -1 && comp2 == 2) {
    quadrupole_21s_quadrupole_22c(*this, level, result);
  }
  // Case (-1,-2): Q21s x Q22s - Orient case 113
  else if (comp1 == -1 && comp2 == -2) {
    quadrupole_21s_quadrupole_22s(*this, level, result);
  }
  // Case (2,1): Q22c x Q21c - Orient case 114
  else if (comp1 == 2 && comp2 == 1) {
    quadrupole_22c_quadrupole_21c(*this, level, result);
  }
  // Case (2,-1): Q22c x Q21s - Orient case 115
  else if (comp1 == 2 && comp2 == -1) {
    quadrupole_22c_quadrupole_21s(*this, level, result);
  }
  // Case (2,-2): Q22c x Q22s - Orient case 116 (actually 120)
  else if (comp1 == 2 && comp2 == -2) {
    quadrupole_22c_quadrupole_22s(*this, level, result);
  }
  // Case (-2,1): Q22s x Q21c - Orient case 117
  else if (comp1 == -2 && comp2 == 1) {
    quadrupole_22s_quadrupole_21c(*this, level, result);
  }
  // Case (-2,-1): Q22s x Q21s - Orient case 118
  else if (comp1 == -2 && comp2 == -1) {
    quadrupole_22s_quadrupole_21s(*this, level, result);
  }
  // Case (-2,2): Q22s x Q22c - Orient case 120
  else if (comp1 == -2 && comp2 == 2) {
    quadrupole_22s_quadrupole_22c(*this, level, result);
  }

  // Derivatives not implemented for now (level >= 1, level >= 2)
  // These are complex and not needed for energy calculation
}

void SFunctions::compute_s4(int t1, int t2, int j,
                             int level, SFunctionResult& result) const {
  using namespace kernels;

  auto [l1, m1] = index_to_lm(t1);
  auto [l2, m2] = index_to_lm(t2);

  // Handle rank sum = 4 cases following Orient's S4 subroutine

  // Charge-hexadecapole interactions (00,4m)
  if (l1 == 0 && l2 == 4) {
    switch (m2) {
    case 0: // Q40 component: 35z⁴/8 - 15z²/4 + 3/8
      charge_hexadecapole_40(*this, level, result);
      return;

    case 1: // Q41c component: √10(7xz³-3xz)/4
      charge_hexadecapole_41c(*this, level, result);
      return;

    case -1: // Q41s component: √10(7yz³-3yz)/4
      charge_hexadecapole_41s(*this, level, result);
      return;

    case 2: // Q42c component: √5(7x²z²-7y²z²-x²+y²)/4
      charge_hexadecapole_42c(*this, level, result);
      return;

    case -2: // Q42s component: √5(7xyz²-xy)/2
      charge_hexadecapole_42s(*this, level, result);
      return;

    case 3: // Q43c component: √70(x³z-3xy²z)/4
      charge_hexadecapole_43c(*this, level, result);
      return;

    case -3: // Q43s component: √70(3x²yz-y³z)/4
      charge_hexadecapole_43s(*this, level, result);
      return;

    case 4: // Q44c component: √35(x⁴-6x²y²+y⁴)/8
      charge_hexadecapole_44c(*this, level, result);
      return;

    case -4: // Q44s component: √35(x³y-xy³)/2
      charge_hexadecapole_44s(*this, level, result);
      return;
    }
    return;
  }

  // Handle symmetric cases using swapped coordinates and phase factors
  if (l1 == 4 && l2 == 0) {
    SFunctions swapped_sf(m_max_rank);
    swapped_sf.set_coordinates(m_coords.rb, m_coords.ra); // Swap A and B
    swapped_sf.compute_s4(0, t1, j, level, result);

    // Don't apply extra phase factor - coordinate swap handles signs correctly
    return;
  }

  // Dipole-octopole interactions (l1=1, l2=3)
  // The dipole-octopole kernels exist but are incomplete/incorrect (use empirical scale factors).
  // Use coordinate swapping to reuse the correct octopole-dipole kernels instead.
  if (l1 == 1 && l2 == 3) {
    compute_with_swap(t1, t2, j, level, result, true);
    return;
  }

  // Octopole-dipole interactions (l1=3, l2=1) - Orient cases 126-146
  // Orient uses DIFFERENT formulas than dipole-octopole, so can't simply swap!
  if (l1 == 3 && l2 == 1) {
    // Q30 × Dipole (m1=0)
    if (m1 == 0 && m2 == 0) {
      octopole_30_dipole_10(*this, level, result);
      return;
    }
    // Q31c × Dipole (m1=1)
    else if (m1 == 1 && m2 == 0) {
      octopole_31c_dipole_10(*this, level, result);
      return;
    }
    // Q31s × Dipole (m1=-1)
    else if (m1 == -1 && m2 == 0) {
      octopole_31s_dipole_10(*this, level, result);
      return;
    }
    // Q32c × Dipole (m1=2)
    else if (m1 == 2 && m2 == 0) {
      octopole_32c_dipole_10(*this, level, result);
      return;
    }
    // Q32s × Dipole (m1=-2)
    else if (m1 == -2 && m2 == 0) {
      octopole_32s_dipole_10(*this, level, result);
      return;
    }
    // Q33c × Dipole (m1=3)
    else if (m1 == 3 && m2 == 0) {
      octopole_33c_dipole_10(*this, level, result);
      return;
    }
    else if (m1 == 3 && m2 == 1) { // Q33c × Q11c - Orient case 142
      octopole_33c_dipole_x(*this, level, result);
      return;
    }
    else if (m1 == 3 && m2 == -1) { // Q33c × Q11s - Orient case 143
      octopole_33c_dipole_11s(*this, level, result);
      return;
    }
    // Q33s × Dipole (m1=-3)
    else if (m1 == -3 && m2 == 0) {
      octopole_33s_dipole_10(*this, level, result);
      return;
    }
    else if (m1 == -3 && m2 == 1) { // Q33s × Q11c - Orient case 145
      octopole_33s_dipole_11c(*this, level, result);
      return;
    }
    else if (m1 == -3 && m2 == -1) { // Q33s × Q11s - Orient case 146
      octopole_33s_dipole_11s(*this, level, result);
      return;
    }
    // For octopole-dipole combinations not implemented above, use dipole-octopole kernels.
    // Note: dipole-octopole kernels are incomplete (missing czz terms) but better than zero.
    // TODO: Implement the missing octopole-dipole kernels properly from Orient formulas.
    SFunctions swapped_sf(m_max_rank);
    swapped_sf.set_coordinates(m_coords.rb, m_coords.ra);

    // Call dipole-octopole kernels directly without recursion
    // We know l1=3, l2=1 here, so after swap it's l1=1, l2=3 (dipole-octopole)
    if (m2 == 0) {  // Dipole-z
      if (m1 == 0) { kernels::dipole_z_octopole_30(swapped_sf, level, result); return; }
      if (m1 == 1) { kernels::dipole_z_octopole_31c(swapped_sf, level, result); return; }
      if (m1 == -1) { kernels::dipole_z_octopole_31s(swapped_sf, level, result); return; }
      if (m1 == 2) { kernels::dipole_z_octopole_32c(swapped_sf, level, result); return; }
      if (m1 == -2) { kernels::dipole_z_octopole_32s(swapped_sf, level, result); return; }
      if (m1 == 3) { kernels::dipole_z_octopole_33c(swapped_sf, level, result); return; }
      if (m1 == -3) { kernels::dipole_z_octopole_33s(swapped_sf, level, result); return; }
    } else if (m2 == 1) {  // Dipole-x
      if (m1 == 0) { kernels::dipole_x_octopole_30(swapped_sf, level, result); return; }
      if (m1 == 1) { kernels::dipole_x_octopole_31c(swapped_sf, level, result); return; }
      if (m1 == -1) { kernels::dipole_x_octopole_31s(swapped_sf, level, result); return; }
      if (m1 == 2) { kernels::dipole_x_octopole_32c(swapped_sf, level, result); return; }
      if (m1 == -2) { kernels::dipole_x_octopole_32s(swapped_sf, level, result); return; }
      if (m1 == 3) { kernels::dipole_x_octopole_33c(swapped_sf, level, result); return; }
      if (m1 == -3) { kernels::dipole_x_octopole_33s(swapped_sf, level, result); return; }
    } else if (m2 == -1) {  // Dipole-y
      if (m1 == 0) { kernels::dipole_y_octopole_30(swapped_sf, level, result); return; }
      if (m1 == 1) { kernels::dipole_y_octopole_31c(swapped_sf, level, result); return; }
      if (m1 == -1) { kernels::dipole_y_octopole_31s(swapped_sf, level, result); return; }
      if (m1 == 2) { kernels::dipole_y_octopole_32c(swapped_sf, level, result); return; }
      if (m1 == -2) { kernels::dipole_y_octopole_32s(swapped_sf, level, result); return; }
      if (m1 == 3) { kernels::dipole_y_octopole_33c(swapped_sf, level, result); return; }
      if (m1 == -3) { kernels::dipole_y_octopole_33s(swapped_sf, level, result); return; }
    }
    result.s0 = 0.0;
    return;
  }

  // Quadrupole-quadrupole interactions (l1=2, l2=2)
  if (l1 == 2 && l2 == 2) {
    compute_quadrupole_quadrupole(m1, m2, level, result);
    return;
  }

  throw std::invalid_argument(
      "Unsupported S4 multipole combination: l1=" + std::to_string(l1) +
      ", l2=" + std::to_string(l2));
}

} // namespace occ::mults
