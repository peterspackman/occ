#pragma once
#include <occ/crystal/crystal.h>
#include <occ/core/molecule.h>

namespace occ::io {

occ::crystal::Crystal load_crystal(const std::string &filename);
occ::core::Molecule load_molecule(const std::string &filename);


}
