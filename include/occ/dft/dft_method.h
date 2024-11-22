#pragma once
#include <occ/dft/functional.h>
#include <vector>

namespace occ::dft {

struct Functionals {
  std::vector<DensityFunctional> unpolarized;
  std::vector<DensityFunctional> polarized;
};

Functionals parse_dft_method(const std::string &method_string);

} // namespace occ::dft
