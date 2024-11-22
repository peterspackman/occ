#pragma once
#include <string>

namespace occ::core::rotor {

enum Rotor { Spherical, Linear, Oblate, Prolate, Asymmetric };

inline std::string to_string(Rotor r) {
  switch (r) {
  case Spherical:
    return "spherical top";
  case Linear:
    return "linear top";
  case Oblate:
    return "oblate symmetric top";
  case Prolate:
    return "prolate symmetric top";
  default:
    return "asymmetric top";
  }
}

inline Rotor classify(double IA, double IB, double IC, double epsilon = 1e-12) {
  bool ab_same = std::abs(IA - IB) < epsilon;
  // bool ac_same = std::abs(IA - IC) < epsilon;
  bool bc_same = std::abs(IB - IC) < epsilon;
  if (bc_same) {
    if (ab_same)
      return Spherical;
    if (std::abs(IA) < epsilon)
      return Linear;
    if (IA < IB)
      return Prolate;
  }
  if (ab_same && IB < IC)
    return Oblate;
  return Asymmetric;
}

} // namespace occ::core::rotor
