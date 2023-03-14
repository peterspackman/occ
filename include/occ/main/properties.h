#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>

namespace occ::main {

using occ::io::OccInput;
using occ::qm::Wavefunction;

void calculate_dispersion(const OccInput &, const Wavefunction &);
void calculate_properties(const OccInput &, const Wavefunction &);

} // namespace occ::main
