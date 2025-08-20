/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#pragma once
#include <utility>

namespace occ::opt {

/**
 * @brief Perform cubic polynomial fitting for linear search
 *
 * Fits a cubic polynomial through two points with known function values
 * and derivatives, then finds the minimum of the polynomial.
 *
 * @param y0 Function value at x=0
 * @param y1 Function value at x=1
 * @param g0 Derivative at x=0
 * @param g1 Derivative at x=1
 * @return Pair of (x_min, y_min) where the cubic has its minimum,
 *         or (NaN, NaN) if no valid minimum exists in [0,1]
 */
std::pair<double, double> fit_cubic(double y0, double y1, double g0, double g1);

/**
 * @brief Perform quartic polynomial fitting for linear search
 *
 * Fits a quartic polynomial through two points with known function values
 * and derivatives, then finds the minimum of the polynomial.
 *
 * @param y0 Function value at x=0
 * @param y1 Function value at x=1
 * @param g0 Derivative at x=0
 * @param g1 Derivative at x=1
 * @return Pair of (x_min, y_min) where the quartic has its minimum,
 *         or (NaN, NaN) if no valid minimum exists
 */
std::pair<double, double> fit_quartic(double y0, double y1, double g0, double g1);

/**
 * @brief Perform linear search between two points using polynomial interpolation
 *
 * This implements a linear search algorithm, which attempts quartic
 * interpolation first, falls back to cubic if quartic fails, and finally
 * uses simple selection if both polynomial fits fail.
 *
 * @param E0 Energy at current point
 * @param E1 Energy at best point
 * @param g0 Directional derivative at current point
 * @param g1 Directional derivative at best point
 * @return Pair of (t, E) where t is the interpolation parameter [0,1]
 *         and E is the interpolated energy
 */
std::pair<double, double> linear_search(double E0, double E1, double g0, double g1);

}  // namespace occ::opt