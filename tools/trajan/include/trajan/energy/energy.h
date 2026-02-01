#pragma once
#include <trajan/energy/xtb.h>

namespace trajan::energy {

enum class EnergyModel { GFN1xTB, GFN2xTB, GFNFF };

struct SinglePoint {
  double energy{0};
  occ::Mat3N forces;
  occ::Mat3 virial;
};

} // namespace trajan::energy
