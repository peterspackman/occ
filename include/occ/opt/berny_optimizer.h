/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#pragma once
#include <occ/core/molecule.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/optimization_state.h>

namespace occ::opt {

/**
 * @brief Berny geometry optimizer implementing the Berny algorithm
 *
 * This class provides a rational function optimization (RFO) method for
 * molecular geometry optimization using internal coordinates, closely
 * following the Berny reference implementation.
 */
struct BernyOptimizer {
  /**
   * @brief Construct a new Berny Optimizer
   *
   * @param mol Initial molecular geometry
   * @param criteria Convergence criteria (uses default values if not specified)
   */
  BernyOptimizer(const core::Molecule &mol,
                 const ConvergenceCriteria &criteria = ConvergenceCriteria{});

  /**
   * @brief Perform one optimization step
   *
   * @return true if optimization has converged
   */
  bool step();

  /**
   * @brief Get the next geometry to evaluate
   *
   * @return Molecule with updated coordinates for energy/gradient calculation
   */
  core::Molecule get_next_geometry() const;

  /**
   * @brief Update optimizer with energy and gradient from calculation
   *
   * @param energy Total energy at current geometry
   * @param gradient Cartesian gradient matrix (3 x N_atoms)
   */
  void update(double energy, const Mat3N &gradient);

  /**
   * @brief Check if optimization has converged
   *
   * @return true if convergence criteria are satisfied
   */
  bool is_converged() const { return state.converged; }

  /**
   * @brief Get current energy
   *
   * @return Most recent energy value
   */
  double current_energy() const { return state.energy; }

  /**
   * @brief Get current optimization step number
   *
   * @return Step number (0-indexed)
   */
  int current_step() const { return state.step_number; }

  /**
   * @brief Get current trust radius
   *
   * @return Current trust radius value
   */
  double current_trust_radius() const { return state.trust_radius; }

private:
  OptimizationState state;
  InternalCoordinates coords;
  core::Molecule molecule;

  // Step logging
  bool step_logging_enabled_ = false;
  std::string step_log_prefix_ = "opt_step_";

  void log_optimization_step(const Mat3N &input_coords,
                             const Mat3N &input_gradients,
                             const Mat3N &output_coords) const;
};

} // namespace occ::opt
