#pragma once
#include <occ/solvent/smd_parameters.h>
#include <occ/core/atom.h>
#include <vector>

namespace occ::solvent::draco {


Vec coordination_numbers(const std::vector<core::Atom> &atoms);

Vec smd_coulomb_radii(const Vec &charges, 
		      const std::vector<core::Atom> &atoms, const SMDSolventParameters &);

}
