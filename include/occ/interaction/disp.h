#pragma once
#include <occ/core/atom.h>
#include <vector>

namespace occ::disp {
using occ::core::Atom;

double ce_model_dispersion_energy(std::vector<Atom> &atoms_a,
                                  std::vector<Atom> &atoms_b);

} // namespace occ::disp
