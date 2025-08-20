/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#pragma once
#include <occ/core/graph.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/opt/angle_coordinate.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/opt/dihedral_coordinate.h>
#include <vector>

namespace occ::opt {

class InternalCoordinates {
public:
  struct Options {
    bool include_dihedrals{true};
    bool superweak_dihedrals{false};
  };

  InternalCoordinates(const Mat3N &, const IVec &,
                      const Options & = {true, false});
  InternalCoordinates(const core::Molecule &, const Options & = {true, false});

  inline size_t size() const {
    return m_bonds.size() + m_angles.size() + m_dihedrals.size();
  }

  void build_covalent_bonds();
  void build_connectivity_matrix();
  void add_vdw_bonds();
  void build_bond_coordinates();
  void build_angle_coordinates();
  void build_dihedral_coordinates();

  // Coordinate vector methods
  Vec to_vector(const Mat3N &positions) const;
  Vec to_vector_with_template(const Mat3N &positions,
                              const Vec &template_q) const;

  // Wilson B-matrix methods
  Mat wilson_b_matrix(const Mat3N &positions) const;
  Mat hessian_guess();

  inline auto &bonds() const { return m_bonds; }
  inline auto &angles() const { return m_angles; }
  inline auto &dihedrals() const { return m_dihedrals; }
  inline auto &weights() const { return m_weights; }

private:
  void build(const Mat3N &, const IVec &, const Options &);

  std::vector<BondCoordinate> m_bonds;
  std::vector<AngleCoordinate> m_angles;
  std::vector<DihedralCoordinate> m_dihedrals;

  Vec m_weights; // Weights for each internal coordinate
  MaskMat m_connectivity;

  Mat3N m_positions;
  IVec m_atomic_numbers;
  std::vector<int> m_fragments; // Fragment assignment for each atom

  MaskMat m_covalent_bonds;
  MaskMat m_all_bonds;

  Options m_options{};

  // Helper methods
  std::vector<int> get_connected_components(const MaskMat &bond_matrix);
  bool is_fully_connected(const MaskMat &bond_matrix);
  bool same_fragment(int i, int j);
};

// Transform step from internal to Cartesian coordinates (returns new positions)
Mat3N transform_step_to_cartesian(const Vec &internal_step,
                                  const InternalCoordinates &coords,
                                  const Mat3N &positions, const Mat &B_inv);

// Coordinate transformation: returns (new_q, new_positions)
std::pair<Vec, Mat3N> update_geometry(const Vec &current_q,
                                      const Vec &internal_step,
                                      const InternalCoordinates &coords,
                                      const Mat3N &positions, const Mat &B_inv);

} // namespace occ::opt
