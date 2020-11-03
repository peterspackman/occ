#pragma once
#include <libint2/atom.h>
#include <vector>

namespace tonto::disp
{
using libint2::Atom;

double d2_interaction_energy(std::vector<Atom> &atoms_a, std::vector<Atom> &atoms_b);

}
