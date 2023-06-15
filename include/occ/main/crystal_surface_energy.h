#pragma once
#include <occ/crystal/crystal.h>
#include <string>

namespace occ::main {

void calculate_crystal_surface_energies(
    const std::string &filename, const occ::crystal::Crystal &crystal,
    const occ::crystal::CrystalDimers &uc_dimers, int max_number_of_surfaces,
    int sign = -1);

}
