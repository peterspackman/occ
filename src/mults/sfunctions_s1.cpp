#include <occ/mults/sfunctions.h>

namespace occ::mults {

void SFunctions::compute_charge_dipole(int m_component,
                                       int level, SFunctionResult& result) const {
  // Following Orient's convention for dipole components:
  // m=0 -> z component (Q10)
  // m=1 -> x component (Q11c)
  // m=-1 -> y component (Q11s)
  //
  // CRITICAL: For charge@A interacting with dipole@B, Orient uses rax/ray/raz
  // (site A coordinates) NOT rbx/rby/rbz (site B coordinates). This gives the
  // correct sign! Orient result for charge-dipole-x: s0 = rax = -0.577... (not
  // rbx = +0.577...)

  switch (m_component) {
  case 0:              // Q10 component (z) - dipole is at site B, use rbz
    result.s0 = rbz(); // Use site B coordinate (dipole is at B)
    if (level >= 1) {
      result.s1[5] = 1.0; // d/d(rbz), normalized
    }
    break;

  case 1:              // Q11c component (x) - dipole is at site B, use rbx
    result.s0 = rbx(); // Use site B coordinate (dipole is at B)
    if (level >= 1) {
      result.s1[3] = 1.0; // d/d(rbx), normalized
    }
    break;

  case -1:             // Q11s component (y) - dipole is at site B, use rby
    result.s0 = rby(); // Use site B coordinate (dipole is at B)
    if (level >= 1) {
      result.s1[4] = 1.0; // d/d(rby), normalized
    }
    break;

  default:
    throw std::invalid_argument("Invalid dipole component");
  }
}

// Reversed multipole-charge interactions
void SFunctions::compute_dipole_charge(int m_component,
                                       int level, SFunctionResult& result) const {
  // For dipole @ A, charge @ B: the S-function depends on the dipole direction
  // relative to the A→B vector. Since the dipole is at A, we use rax, ray, raz
  // which are the components of the unit vector from A to B.

  switch (m_component) {
  case 0: // Q10 component (z)
    result.s0 = raz();
    if (level >= 1) {
      result.s1[2] = 1.0; // d/d(raz)
    }
    break;

  case 1: // Q11c component (x)
    result.s0 = rax();
    if (level >= 1) {
      result.s1[0] = 1.0; // d/d(rax)
    }
    break;

  case -1: // Q11s component (y)
    result.s0 = ray();
    if (level >= 1) {
      result.s1[1] = 1.0; // d/d(ray)
    }
    break;

  default:
    throw std::invalid_argument("Invalid dipole component");
  }
}

} // namespace occ::mults
