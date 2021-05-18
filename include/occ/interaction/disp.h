#pragma once
#include <libint2/atom.h>
#include <vector>

namespace occ::disp
{
using libint2::Atom;

double ce_model_dispersion_energy(std::vector<Atom> &atoms_a, std::vector<Atom> &atoms_b);

}
