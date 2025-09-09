#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::interaction {

extern const std::array<double, 110> Thakkar_atomic_polarizability;
extern const std::array<double, 110> Charged_atomic_polarizibility;

double ce_model_polarization_energy(const occ::IVec &atomic_numbers,
                                    const occ::Mat3N &efield,
                                    bool charged = false);

double polarization_energy(const occ::Vec &polarizabilities,
                           const occ::Mat3N &efield);

} // namespace occ::interaction
