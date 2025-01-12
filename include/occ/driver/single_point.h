#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>

namespace occ::driver {

qm::Wavefunction single_point(const io::OccInput &);
qm::Wavefunction single_point(const io::OccInput &, const qm::Wavefunction &);

} // namespace occ::driver
