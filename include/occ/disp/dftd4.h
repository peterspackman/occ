#pragma once
#include <occ/core/molecule.h>

namespace occ::disp {

struct DFTD4Params {
  double s6{0.0};
  double s8{0.0};
  double s10{0.0};
  double s9{0.0};
  double a1{0.0};
  double a2{0.0};
  int alp{0};
};

} // namespace occ::disp
