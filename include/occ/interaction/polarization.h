#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::pol {

double ce_model_polarization_energy(const occ::IVec &atomic_numbers,
                                    const occ::Mat3N &efield,
                                    bool charged = false);

}
