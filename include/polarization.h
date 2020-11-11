#pragma once
#include "linear_algebra.h"

namespace tonto::pol {

double ce_model_polarization_energy(const tonto::IVec& atomic_numbers, const tonto::Mat3N& efield);

}
