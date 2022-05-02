#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>
#include <optional>

namespace occ::main {

using occ::io::OccInput;
using occ::qm::Wavefunction;

Wavefunction single_point_calculation(const OccInput &);
// with initial guess
Wavefunction single_point_calculation(const OccInput &, const Wavefunction &);

} // namespace occ::main
