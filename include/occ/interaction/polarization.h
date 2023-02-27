#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::interaction {

double ce_model_polarization_energy(const occ::IVec &atomic_numbers,
                                    const occ::Mat3N &efield,
                                    bool charged = false);

double polarization_energy(const occ::Vec &polarizabilities,
                           const occ::Mat3N &efield);

} // namespace occ::interaction
