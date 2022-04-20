#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::sht {

std::pair<Vec, Vec> gauss_legendre_quadrature(int N);
void gauss_legendre_quadrature(Vec &roots, Vec &weights, int N);

} // namespace occ::sht
