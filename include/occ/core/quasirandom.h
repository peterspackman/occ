#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

/**
 * Generate an D dimensional Korobov type quasi-random vector based on the
 * generalized Fibonacci sequence.
 *
 * Based on the R_1, R_2 sequences available here:
 *     `https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/`
 *
 * Parameters:
 * \param ndims number of dimensions to sample
 * \param count the number of points in space to generate
 * \param seed the seed (offset) into the sequence, default = 0
 *
 * \returns a ndims dimensional sampling point
 */

Mat quasirandom_kgf(size_t ndims, size_t count, size_t seed = 0);

} // namespace occ::core
