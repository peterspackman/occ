#pragma once
#include <occ/qm/wavefunction.h>
#include <occ/io/occ_input.h>

namespace occ::main {

using occ::io::OccInput;
using occ::qm::Wavefunction;

void calculate_properties(const OccInput&, const Wavefunction&);

}
