#include <occ/mults/sfunctions.h>
#include <fmt/core.h>

namespace occ::mults {

// Dipole-dipole interactions
void
SFunctions::compute_dipole_dipole(int comp1, int comp2, int level, SFunctionResult& result) const {
  // Following Orient's S012 subroutine, cases 13-21
  // For electrostatic interactions (j = l1+l2 = 2), the orientation matrix
  // terms (cxx, cyy, czz, etc.) are zero for static multipoles in the same
  // frame. Orient formulas: S0 = 3/2 * ra_i * rb_j + c_ij/2 For electrostatics
  // with no molecular rotation: c_ii = 1, c_ij = 0 (i≠j) But actually, for pure
  // electrostatic terms, these cancel out in the j-loop summation So we use the
  // simplified coordinate product terms only

  double rax_val = rax(), ray_val = ray(), raz_val = raz();
  double rbx_val = rbx(), rby_val = rby(), rbz_val = rbz();

  // Map component indices: 0 = z, 1 = x, -1 = y
  // to array indices for easier handling
  auto get_ra = [&](int m) -> double {
    if (m == 0)
      return raz_val; // Q10 (z)
    else if (m == 1)
      return rax_val; // Q11c (x)
    else
      return ray_val; // Q11s (y)
  };

  auto get_rb = [&](int m) -> double {
    if (m == 0)
      return rbz_val; // Q10 (z)
    else if (m == 1)
      return rbx_val; // Q11c (x)
    else
      return rby_val; // Q11s (y)
  };

  double ra_i = get_ra(comp1);
  double rb_j = get_rb(comp2);

  // Orient formula: S0 = 3/2 * ra_i * rb_j + c_ij/2
  // Use exact Orient formula with orientation matrix
  // Map component pairs to orientation matrix elements
  double c_ij = 0.0;
  if (comp1 == 0 && comp2 == 0)
    c_ij = czz(); // z,z
  else if (comp1 == 0 && comp2 == 1)
    c_ij = czx(); // z,x
  else if (comp1 == 0 && comp2 == -1)
    c_ij = czy(); // z,y
  else if (comp1 == 1 && comp2 == 0)
    c_ij = cxz(); // x,z
  else if (comp1 == 1 && comp2 == 1)
    c_ij = cxx(); // x,x
  else if (comp1 == 1 && comp2 == -1)
    c_ij = cxy(); // x,y
  else if (comp1 == -1 && comp2 == 0)
    c_ij = cyz(); // y,z
  else if (comp1 == -1 && comp2 == 1)
    c_ij = cyx(); // y,x
  else if (comp1 == -1 && comp2 == -1)
    c_ij = cyy(); // y,y

  result.s0 = 1.5 * ra_i * rb_j + 0.5 * c_ij;

  if (level >= 1) {
    // First derivatives
    // d/d(ra_i) = 1.5 * rb_j
    // d/d(rb_j) = 1.5 * ra_i
    // The orientation term 1/2 only contributes if comp1 == comp2 and involves
    // rotation derivatives

    // Map components to derivative indices
    // S1 indices: 0=rax, 1=ray, 2=raz, 3=rbx, 4=rby, 5=rbz
    int ra_idx = (comp1 == 0) ? 2 : (comp1 == 1) ? 0 : 1; // raz, rax, ray
    int rb_idx = (comp2 == 0) ? 5 : (comp2 == 1) ? 3 : 4; // rbz, rbx, rby

    result.s1[ra_idx] = 1.5 * rb_j;
    result.s1[rb_idx] = 1.5 * ra_i;

    // Orientation derivative: d(S0)/d(c_ij) = 0.5 for ALL component combinations
    // Map component pairs (comp1, comp2) to orientation matrix element indices
    // Orientation matrix: c_ij where i is from site A, j is from site B
    // s1 indices 6-14 use ROW-MAJOR ordering to match the test framework
    //   index:              6    7    8    9    10   11   12   13   14
    //   matrix (i,j):      (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)(2,0)(2,1)(2,2)
    //   names:             cxx  cxy  cxz  cyx  cyy  cyz  czx  czy  czz
    //
    // Component mapping: comp = 0 (z) → matrix row/col 2
    //                    comp = 1 (x) → matrix row/col 0
    //                    comp = -1 (y) → matrix row/col 1
    int i = (comp1 == 0) ? 2 : (comp1 == 1) ? 0 : 1;  // matrix row (site A)
    int j = (comp2 == 0) ? 2 : (comp2 == 1) ? 0 : 1;  // matrix col (site B)
    int c_idx = 6 + i * 3 + j;  // ROW-MAJOR: row * ncols + col (matches test framework)

    result.s1[c_idx] = 0.5;  // d(S0)/d(c_ij) = 0.5

    // DEBUG (disabled)
    // static int dipole_debug_count = 0;
    // if (dipole_debug_count < 5) {
    //   fmt::print("DEBUG dipole-dipole call {}: comp1={}, comp2={}, i={}, j={}, c_idx={}, s1[{}]={}\n",
    //              dipole_debug_count, comp1, comp2, i, j, c_idx, c_idx, result.s1[c_idx]);
    //   dipole_debug_count++;
    // }

    if (level >= 2) {
      // Second derivatives
      // d²/d(ra_i)d(rb_j) = 1.5
      // Orient packs second derivatives in a specific order
      // S2 index for d²/d(ra_i)d(rb_j) depends on the component pair

      // Orient S2 indices (from cases 13-21):
      // Case 13 (z,z): S2(18) = 3/2
      // Case 14 (z,x): S2(9) = 3/2
      // Case 15 (z,y): S2(13) = 3/2
      // Case 16 (x,z): S2(16) = 3/2
      // Case 17 (x,x): S2(7) = 3/2
      // Case 18 (x,y): S2(11) = 3/2
      // Case 19 (y,z): S2(17) = 3/2
      // Case 20 (y,x): S2(8) = 3/2
      // Case 21 (y,y): S2(12) = 3/2

      // Map component pairs to S2 index (1-based in Orient, 0-based here)
      int s2_idx = -1;
      if (comp1 == 0 && comp2 == 0)
        s2_idx = 17; // z,z -> 18-1
      else if (comp1 == 0 && comp2 == 1)
        s2_idx = 8; // z,x -> 9-1
      else if (comp1 == 0 && comp2 == -1)
        s2_idx = 12; // z,y -> 13-1
      else if (comp1 == 1 && comp2 == 0)
        s2_idx = 15; // x,z -> 16-1
      else if (comp1 == 1 && comp2 == 1)
        s2_idx = 6; // x,x -> 7-1
      else if (comp1 == 1 && comp2 == -1)
        s2_idx = 10; // x,y -> 11-1
      else if (comp1 == -1 && comp2 == 0)
        s2_idx = 16; // y,z -> 17-1
      else if (comp1 == -1 && comp2 == 1)
        s2_idx = 7; // y,x -> 8-1
      else if (comp1 == -1 && comp2 == -1)
        s2_idx = 11; // y,y -> 12-1

      if (s2_idx >= 0) {
        result.s2[s2_idx] = 1.5;
      }
    }
  }
}

void
SFunctions::compute_charge_quadrupole(int m_component, int level, SFunctionResult& result) const {
  // Following Orient's convention for quadrupole components:
  // m=0  -> Q20 = 3z²/2 - 1/2
  // m=1  -> Q21c = √3 xz
  // m=-1 -> Q21s = √3 yz
  // m=2  -> Q22c = √3(x²-y²)/2
  // m=-2 -> Q22s = √3 xy

  double e2rx_val = rbx(), e2ry_val = rby(), e2rz_val = rbz();

  switch (m_component) {
  case 0: // Q20 component
    result.s0 = 1.5 * e2rz_val * e2rz_val - 0.5;
    if (level >= 1) {
      result.s1[5] = 3.0 * e2rz_val; // d/d(rbz), normalized
      if (level >= 2) {
        result.s2[20] = 3.0; // d²/d(rbz)², normalized
      }
    }
    break;

  case 1: // Q21c component
    result.s0 = rt3 * e2rx_val * e2rz_val;
    if (level >= 1) {
      result.s1[3] = rt3 * e2rz_val; // d/d(rbx), normalized
      result.s1[5] = rt3 * e2rx_val; // d/d(rbz), normalized
      if (level >= 2) {
        result.s2[18] = rt3; // d²/d(rbx)d(rbz), normalized
      }
    }
    break;

  case -1: // Q21s component
    result.s0 = rt3 * e2ry_val * e2rz_val;
    if (level >= 1) {
      result.s1[4] = rt3 * e2rz_val; // d/d(rby), normalized
      result.s1[5] = rt3 * e2ry_val; // d/d(rbz), normalized
      if (level >= 2) {
        result.s2[19] = rt3; // d²/d(rby)d(rbz), normalized
      }
    }
    break;

  case 2: // Q22c component
    result.s0 = rt3 * (e2rx_val * e2rx_val - e2ry_val * e2ry_val) / 2.0;
    if (level >= 1) {
      result.s1[3] = rt3 * e2rx_val;  // d/d(rbx), normalized
      result.s1[4] = -rt3 * e2ry_val; // d/d(rby), normalized
      if (level >= 2) {
        result.s2[9] = rt3;   // d²/d(rbx)², normalized
        result.s2[14] = -rt3; // d²/d(rby)², normalized
      }
    }
    break;

  case -2: // Q22s component
    result.s0 = rt3 * e2rx_val * e2ry_val;
    if (level >= 1) {
      result.s1[3] = rt3 * e2ry_val; // d/d(rbx), normalized
      result.s1[4] = rt3 * e2rx_val; // d/d(rby), normalized
      if (level >= 2) {
        result.s2[13] = rt3; // d²/d(rbx)d(rby), normalized
      }
    }
    break;

  default:
    throw std::invalid_argument("Invalid quadrupole component");
  }
}

void
SFunctions::compute_quadrupole_charge(int m_component, int level, SFunctionResult& result) const {
  // For quadrupole @ A, charge @ B: use rax, ray, raz (same as
  // charge_quadrupole)

  double rax_val = rax(), ray_val = ray(), raz_val = raz();

  switch (m_component) {
  case 0: // Q20 component
    result.s0 = 1.5 * raz_val * raz_val - 0.5;
    if (level >= 1) {
      result.s1[2] = 3.0 * raz_val; // d/d(raz)
      if (level >= 2) {
        result.s2[8] = 3.0; // d²/d(raz)²
      }
    }
    break;

  case 1: // Q21c component
    result.s0 = rt3 * rax_val * raz_val;
    if (level >= 1) {
      result.s1[0] = rt3 * raz_val; // d/d(rax)
      result.s1[2] = rt3 * rax_val; // d/d(raz)
      if (level >= 2) {
        result.s2[6] = rt3; // d²/d(rax)d(raz)
      }
    }
    break;

  case -1: // Q21s component
    result.s0 = rt3 * ray_val * raz_val;
    if (level >= 1) {
      result.s1[1] = rt3 * raz_val; // d/d(ray)
      result.s1[2] = rt3 * ray_val; // d/d(raz)
      if (level >= 2) {
        result.s2[7] = rt3; // d²/d(ray)d(raz)
      }
    }
    break;

  case 2: // Q22c component
    result.s0 = rt3 * (rax_val * rax_val - ray_val * ray_val) / 2.0;
    if (level >= 1) {
      result.s1[0] = rt3 * rax_val;  // d/d(rax)
      result.s1[1] = -rt3 * ray_val; // d/d(ray)
      if (level >= 2) {
        result.s2[0] = rt3;  // d²/d(rax)²
        result.s2[4] = -rt3; // d²/d(ray)²
      }
    }
    break;

  case -2: // Q22s component
    result.s0 = rt3 * rax_val * ray_val;
    if (level >= 1) {
      result.s1[0] = rt3 * ray_val; // d/d(rax)
      result.s1[1] = rt3 * rax_val; // d/d(ray)
      if (level >= 2) {
        result.s2[3] = rt3; // d²/d(rax)d(ray)
      }
    }
    break;

  default:
    throw std::invalid_argument("Invalid quadrupole component");
  }
}

} // namespace occ::mults
