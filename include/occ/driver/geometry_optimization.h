#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>

namespace occ::driver {

qm::Wavefunction geometry_optimization(const io::OccInput &);

} // namespace occ::driver
