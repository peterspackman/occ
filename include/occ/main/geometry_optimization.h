#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>

namespace occ::main {

qm::Wavefunction optimization_calculation(const io::OccInput &);

} // namespace occ::main
