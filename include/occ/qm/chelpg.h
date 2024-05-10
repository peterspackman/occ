#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::qm {

struct Wavefunction;

Vec chelpg_charges(const Wavefunction &wfn);

} // namespace occ::qm
