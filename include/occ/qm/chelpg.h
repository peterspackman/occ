#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::qm {
class Wavefunction;

Vec chelpg_charges(const Wavefunction &wfn);

} // namespace occ::qm
