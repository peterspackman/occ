/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <ankerl/unordered_dense.h>
#include <cmath>
#include <fmt/core.h>
#include <occ/core/bondgraph.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/opt/angle_coordinate.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/opt/dihedral_coordinate.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/pseudoinverse.h>
#include <occ/opt/species_data.h>
#include <stack>
#include <tuple>

namespace occ::opt {

InternalCoordinates::InternalCoordinates(
    const occ::core::Molecule &mol, const InternalCoordinates::Options &opts) {
  build(mol.positions(), mol.atomic_numbers(), opts);
}

InternalCoordinates::InternalCoordinates(
    const Mat3N &pos, const IVec &nums,
    const InternalCoordinates::Options &opts) {
  build(pos, nums, opts);
}

namespace {

inline double distance(const Vec3 &a, const Vec3 &b) { return (a - b).norm(); }

std::vector<DihedralCoordinate> get_dihedrals_for_bond(
    const std::pair<int, int> center, const Mat3N &positions,
    const MaskMat &connectivity_matrix, // C matrix
    const MaskMat &bond_matrix,         // Full bond matrix (covalent + VdW)
    bool superweak = false) {

  const double linearity_threshold = 5.0 * M_PI / 180.0; // linearity threshold
  std::vector<DihedralCoordinate> dihedrals;

  int center_0, center_1;
  std::tie(center_0, center_1) = center;
  int n_atoms = positions.cols();

  // Get neighbors using bond matrix
  std::vector<int> neigh_l, neigh_r;

  for (int i = 0; i < n_atoms; i++) {
    if (i != center_0 && i != center_1) {
      if (bond_matrix(center_0, i)) {
        neigh_l.push_back(i);
      }
      if (bond_matrix(center_1, i)) {
        neigh_r.push_back(i);
      }
    }
  }

  // Evaluate angles and filter for nonlinearity
  std::vector<int> nonlinear_l, nonlinear_r;

  for (int nl : neigh_l) {
    AngleCoordinate angle_l(nl, center_0, center_1);
    double ang = angle_l(positions);
    if (ang >= linearity_threshold && ang < M_PI - linearity_threshold) {
      nonlinear_l.push_back(nl);
    }
  }

  for (int nr : neigh_r) {
    AngleCoordinate angle_r(center_0, center_1, nr);
    double ang = angle_r(positions);
    if (ang >= linearity_threshold && ang < M_PI - linearity_threshold) {
      nonlinear_r.push_back(nr);
    }
  }

  // Only process if center[0] < center[1]
  if (center_0 < center_1) {
    // Count weak bonds in the center
    int nweak = 0;
    // Check if center bond is weak (not in connectivity matrix)
    if (!connectivity_matrix(center_0, center_1)) {
      nweak = 1;
    }

    // Generate dihedrals
    for (int nl : nonlinear_l) {
      for (int nr : nonlinear_r) {
        if (nl == nr)
          continue; // skip this case

        // Count weak bonds:
        // weak = nweak + (0 if C[nl, center[0]] else 1) + (0 if C[center[0],
        // nr] else 1)
        int weak = nweak;
        bool nl_center0_covalent = connectivity_matrix(nl, center_0);
        bool center0_nr_covalent = connectivity_matrix(center_0, nr);

        if (!nl_center0_covalent)
          weak++; // (0 if C[nl, center[0]] else 1)
        if (!center0_nr_covalent)
          weak++; // (0 if C[center[0], nr] else 1)

        // Skip if more than 1 weak bond when superweak=False
        if (!superweak && weak > 1) {
          continue; // skip this case
        }

        // Create dihedral
        dihedrals.emplace_back(nl, center_0, center_1, nr);
      }
    }
  }

  return dihedrals;
}

std::vector<DihedralCoordinate>
get_dihedrals(const std::vector<BondCoordinate> &bonds, const Mat3N &positions,
              const MaskMat &connectivity_matrix, const MaskMat &bond_matrix,
              bool superweak = false) {

  std::vector<DihedralCoordinate> all_dihedrals;

  for (const auto &bond : bonds) {
    auto bond_dihedrals =
        get_dihedrals_for_bond(std::make_pair(bond.i, bond.j), positions,
                               connectivity_matrix, bond_matrix, superweak);
    all_dihedrals.insert(all_dihedrals.end(), bond_dihedrals.begin(),
                         bond_dihedrals.end());
  }

  // Remove duplicates
  ankerl::unordered_dense::set<std::tuple<int, int, int, int>> seen_dihedrals;
  std::vector<DihedralCoordinate> unique_dihedrals;

  for (const auto &dih : all_dihedrals) {
    auto key = std::make_tuple(dih.i, dih.j, dih.k, dih.l);
    if (seen_dihedrals.find(key) == seen_dihedrals.end()) {
      seen_dihedrals.insert(key);
      unique_dihedrals.push_back(dih);
    }
  }
  return unique_dihedrals;
}

} // anonymous namespace

// Helper method to get connected components from a bond matrix
std::vector<int>
InternalCoordinates::get_connected_components(const MaskMat &bond_matrix) {
  int n = bond_matrix.rows();
  std::vector<int> component(n, -1);
  int current_component = 0;

  for (int i = 0; i < n; i++) {
    if (component[i] == -1) {
      // Start DFS from unvisited node
      std::stack<int> stack;
      stack.push(i);

      while (!stack.empty()) {
        int node = stack.top();
        stack.pop();

        if (component[node] != -1)
          continue;

        component[node] = current_component;

        // Add all connected neighbors
        for (int j = 0; j < n; j++) {
          if (j != node && bond_matrix(node, j) && component[j] == -1) {
            stack.push(j);
          }
        }
      }
      current_component++;
    }
  }

  return component;
}

// Check if all atoms are in the same connected component
bool InternalCoordinates::is_fully_connected(const MaskMat &bond_matrix) {
  auto components = get_connected_components(bond_matrix);

  // All atoms should be in component 0 if fully connected
  for (int comp : components) {
    if (comp != 0)
      return false;
  }
  return true;
}

// Check if two atoms are in the same fragment (covalently connected)
bool InternalCoordinates::same_fragment(int i, int j) {
  return m_fragments[i] == m_fragments[j];
}

void InternalCoordinates::build_covalent_bonds() {
  // Build initial covalent bond matrix
  int n = m_atomic_numbers.rows();
  m_covalent_bonds = MaskMat::Zero(n, n);
  constexpr double tolerance = 1.3;

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist = (m_positions.col(i) - m_positions.col(j)).norm();
      double cov_sum = get_covalent_radius(m_atomic_numbers(i)) +
                       get_covalent_radius(m_atomic_numbers(j));

      if (dist < cov_sum * tolerance) {
        m_covalent_bonds(i, j) = m_covalent_bonds(j, i) = true;
      }
    }
  }
}

void InternalCoordinates::build_connectivity_matrix() {
  // Get connected components from covalent bonds only
  m_fragments = get_connected_components(m_covalent_bonds);

  // Build connectivity matrix (true if atoms are covalently connected)
  int n = m_atomic_numbers.rows();
  m_connectivity = MaskMat::Zero(n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m_connectivity(i, j) = (m_fragments[i] == m_fragments[j]);
    }
  }
}

void InternalCoordinates::add_vdw_bonds() {
  // Add VdW bonds until everything is connected
  m_all_bonds = m_covalent_bonds; // Start with covalent bonds
  double shift = 0.0;

  while (!is_fully_connected(m_all_bonds)) {
    for (int i = 0; i < m_atomic_numbers.rows(); i++) {
      for (int j = i + 1; j < m_atomic_numbers.rows(); j++) {
        if (m_all_bonds(i, j))
          continue; // Already connected

        // Only add VdW bonds between different fragments
        if (same_fragment(i, j))
          continue;

        double dist = (m_positions.col(i) - m_positions.col(j)).norm();
        double vdw_sum = get_vdw_radius(m_atomic_numbers(i)) +
                         get_vdw_radius(m_atomic_numbers(j));

        if (dist < vdw_sum + shift) {
          m_all_bonds(i, j) = m_all_bonds(j, i) = true;
        }
      }
    }
    shift += 1.0;
    if (shift > 10.0)
      break; // Safety
  }
}

void InternalCoordinates::build_bond_coordinates() {
  // Create bond coordinates from ALL bonds
  m_bonds.clear();

  for (int i = 0; i < m_atomic_numbers.rows(); i++) {
    for (int j = i + 1; j < m_atomic_numbers.rows(); j++) {
      if (m_all_bonds(i, j)) {
        BondCoordinate::Type type = m_covalent_bonds(i, j)
                                        ? BondCoordinate::Type::COVALENT
                                        : BondCoordinate::Type::VDW;
        m_bonds.emplace_back(i, j, type);
      }
    }
  }
}

void InternalCoordinates::build_angle_coordinates() {
  // Create angles from bond connectivity
  m_angles.clear();

  for (int j = 0; j < m_atomic_numbers.rows(); j++) {
    std::vector<int> neighbors;
    for (int i = 0; i < m_atomic_numbers.rows(); i++) {
      if (i != j && m_all_bonds(i, j)) {
        neighbors.push_back(i);
      }
    }

    // Create angles for each pair of neighbors
    for (size_t i1 = 0; i1 < neighbors.size(); i1++) {
      for (size_t i2 = i1 + 1; i2 < neighbors.size(); i2++) {
        AngleCoordinate angle(neighbors[i1], j, neighbors[i2]);
        // condition: angle > pi/4
        if (angle(m_positions) > M_PI / 4) {
          m_angles.push_back(angle);
        }
      }
    }
  }
}

void InternalCoordinates::build_dihedral_coordinates() {
  // Use the get_dihedrals algorithm
  // Key: Uses connectivity matrix (C) for weak detection, all_bonds for
  // neighbors
  m_dihedrals = get_dihedrals(m_bonds, m_positions, m_connectivity, m_all_bonds,
                              m_options.superweak_dihedrals);
}

void InternalCoordinates::build(const Mat3N &positions, const IVec &numbers,
                                const InternalCoordinates::Options &options) {

  m_positions = positions;
  m_atomic_numbers = numbers;
  m_options = options;

  // Follow coordinate ordering
  build_covalent_bonds();      // Step 1
  build_connectivity_matrix(); // Step 2
  add_vdw_bonds();             // Step 3
  build_bond_coordinates();    // Step 4
  build_angle_coordinates();   // Step 5

  if (m_options.include_dihedrals) {
    build_dihedral_coordinates(); // Step 6
  }

  m_weights = Vec::Ones(size());
}

// Class methods for coordinate vectors
Vec InternalCoordinates::to_vector(const Mat3N &positions) const {
  Vec result(size());
  int idx = 0;

  // Evaluate coordinates using coordinate classes
  for (const auto &bond : m_bonds) {
    result(idx++) = bond(positions);
  }
  for (const auto &angle : m_angles) {
    result(idx++) = angle(positions);
  }
  for (const auto &dihedral : m_dihedrals) {
    result(idx++) = dihedral(positions);
  }

  return result;
}

Vec InternalCoordinates::to_vector_with_template(const Mat3N &positions,
                                                 const Vec &template_q) const {
  // Get raw coordinates first
  Vec q = to_vector(positions);

  // Log detailed coordinate breakdown
  occ::log::trace("=== to_vector_with_template DEBUG ===");
  occ::log::trace(
      "positions shape: ({}, {}), first atom: [{:.8f}, {:.8f}, {:.8f}]",
      positions.rows(), positions.cols(), positions(0, 0), positions(1, 0),
      positions(2, 0));

  // Log bonds
  occ::log::trace("Number of bonds: {}", m_bonds.size());
  if (m_bonds.size() > 0) {
    double bond_norm = 0.0;
    for (size_t i = 0; i < std::min(size_t(5), m_bonds.size()); i++) {
      occ::log::trace("  Bond {}: {:.8f}", i, q(i));
      bond_norm += q(i) * q(i);
    }
    occ::log::trace("  First 5 bonds RMS: {:.8f}",
                    std::sqrt(bond_norm / std::min(size_t(5), m_bonds.size())));
  }

  // Log angles
  occ::log::trace("Number of angles: {}", m_angles.size());
  if (m_angles.size() > 0) {
    int angle_start = m_bonds.size();
    double angle_norm = 0.0;
    for (size_t i = 0; i < std::min(size_t(5), m_angles.size()); i++) {
      occ::log::trace("  Angle {}: {:.8f}", i, q(angle_start + i));
      angle_norm += q(angle_start + i) * q(angle_start + i);
    }
    occ::log::trace(
        "  First 5 angles RMS: {:.8f}",
        std::sqrt(angle_norm / std::min(size_t(5), m_angles.size())));
  }

  // Log dihedrals
  occ::log::trace("Number of dihedrals: {}", m_dihedrals.size());
  if (m_dihedrals.size() > 0) {
    int dih_start = m_bonds.size() + m_angles.size();
    double dih_norm = 0.0;
    for (size_t i = 0; i < std::min(size_t(5), m_dihedrals.size()); i++) {
      occ::log::trace("  Dihedral {}: {:.8f}", i, q(dih_start + i));
      dih_norm += q(dih_start + i) * q(dih_start + i);
    }
    occ::log::trace(
        "  First 5 dihedrals RMS: {:.8f}",
        std::sqrt(dih_norm / std::min(size_t(5), m_dihedrals.size())));
  }

  occ::log::trace("Total q norm: {:.8f}, RMS: {:.8f}", q.norm(),
                  std::sqrt(q.squaredNorm() / q.size()));

  if (template_q.size() != q.size()) {
    occ::log::trace("No template or size mismatch - returning raw coordinates");
    return q;
  }

  // Apply discontinuity handling
  occ::log::trace("=== DISCONTINUITY HANDLING DEBUG ===");
  occ::log::trace("Input q norm: {:.8f}, template_q norm: {:.8f}", q.norm(),
                  template_q.norm());
  occ::log::trace("First 5 diffs: [{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}]",
                  q(0) - template_q(0), q(1) - template_q(1),
                  q(2) - template_q(2), q(3) - template_q(3),
                  q(4) - template_q(4));

  // Approach: iterate through all coordinates and handle each type
  // Track swapped dihedrals
  std::vector<int> swapped_dihedrals;
  ankerl::unordered_dense::set<int> candidate_angles;

  // Process ALL coordinates
  for (size_t i = 0; i < q.size(); i++) {
    double diff = q(i) - template_q(i);

    // Check if this is a dihedral coordinate
    if (i >= m_bonds.size() + m_angles.size()) {
      // This is a dihedral coordinate
      int dihedral_idx = i - m_bonds.size() - m_angles.size();

      // Check for 2π discontinuity: if abs(abs(diff) - 2π) < π/2
      if (std::abs(std::abs(diff) - 2.0 * M_PI) < M_PI / 2.0) {
        // Unwrap 2π jump
        double old_val = q(i);
        q(i) -= 2.0 * M_PI * std::copysign(1.0, diff);
        occ::log::trace("  Dihedral {} unwrapped 2π: {:.8f} -> {:.8f}",
                        dihedral_idx, old_val, q(i));
      }
      // Check for π discontinuity: elif abs(abs(diff) - π) < π/2
      else if (std::abs(std::abs(diff) - M_PI) < M_PI / 2.0) {
        // Dihedral flipped by π
        double old_val = q(i);
        q(i) -= M_PI * std::copysign(1.0, diff);
        swapped_dihedrals.push_back(dihedral_idx);
        occ::log::trace("  Dihedral {} flipped by π: {:.8f} -> {:.8f}",
                        dihedral_idx, old_val, q(i));

        // Add angles as candidates
        // Since we don't have dih.angles, we need to find which angles are
        // associated with this dihedral
        const auto &dih = m_dihedrals[dihedral_idx];
        for (size_t a = 0; a < m_angles.size(); a++) {
          const auto &ang = m_angles[a];
          // Check if angle is formed by consecutive atoms in the dihedral
          // i-j-k-l
          if ((ang.i == dih.i && ang.j == dih.j &&
               ang.k == dih.k) || // angle i-j-k
              (ang.i == dih.j && ang.j == dih.k &&
               ang.k == dih.l)) { // angle j-k-l
            candidate_angles.insert(a);
            occ::log::trace(
                "    Added angle {} as candidate (dihedral atoms {}-{}-{}-{})",
                a, dih.i, dih.j, dih.k, dih.l);
          }
        }
      }
    }
  }

  // Process candidate angles
  for (int angle_idx : candidate_angles) {
    int coord_idx = m_bonds.size() + angle_idx;
    const auto &ang = m_angles[angle_idx];

    // Logic: "candidate angle was swapped if each dihedral that contains it was
    // either swapped or all its angles are candidates"
    bool should_swap = true;

    // Check all dihedrals that contain this angle
    for (size_t d = 0; d < m_dihedrals.size(); d++) {
      const auto &dih = m_dihedrals[d];

      // Check if this angle is part of this dihedral
      bool angle_in_dihedral =
          (ang.i == dih.i && ang.j == dih.j && ang.k == dih.k) || // angle i-j-k
          (ang.i == dih.j && ang.j == dih.k && ang.k == dih.l);   // angle j-k-l

      if (angle_in_dihedral) {
        // Check if this dihedral was swapped
        bool dihedral_swapped =
            std::find(swapped_dihedrals.begin(), swapped_dihedrals.end(), d) !=
            swapped_dihedrals.end();

        if (!dihedral_swapped) {
          // Dihedral was not swapped - check if all its angles are candidates
          bool all_angles_candidates = true;

          // Find all angles associated with this dihedral
          for (size_t a = 0; a < m_angles.size(); a++) {
            const auto &other_ang = m_angles[a];
            if ((other_ang.i == dih.i && other_ang.j == dih.j &&
                 other_ang.k == dih.k) ||
                (other_ang.i == dih.j && other_ang.j == dih.k &&
                 other_ang.k == dih.l)) {
              if (candidate_angles.find(a) == candidate_angles.end()) {
                all_angles_candidates = false;
                break;
              }
            }
          }

          if (!all_angles_candidates) {
            should_swap = false;
            break;
          }
        }
      }
    }

    if (should_swap) {
      // Swap angle: q[i] = 2π - q[i]
      double old_val = q(coord_idx);
      q(coord_idx) = 2.0 * M_PI - q(coord_idx);
      occ::log::trace("  Angle {} swapped: {:.8f} -> {:.8f}", angle_idx,
                      old_val, q(coord_idx));
    }
  }

  occ::log::trace("After discontinuity handling - q norm: {:.8f}, RMS: {:.8f}",
                  q.norm(), std::sqrt(q.squaredNorm() / q.size()));

  return q;
}

// Class methods for Wilson B-matrix
Mat InternalCoordinates::wilson_b_matrix(const Mat3N &positions) const {
  // B-matrix implementation
  int len_geom = positions.cols();
  int len_coords = size();

  // Line 372: B = np.zeros((len(self), len(geom), 3))
  Mat B = Mat::Zero(len_coords, 3 * len_geom);

  int row = 0;

  // Process bonds - use stored coordinate classes directly
  for (const auto &bond : m_bonds) {
    auto grad = bond.gradient(positions);

    // Line 377: B[i, j] += grad
    B.block(row, 3 * bond.i, 1, 3) = grad.col(0).transpose();
    B.block(row, 3 * bond.j, 1, 3) = grad.col(1).transpose();
    row++;
  }

  // Process angles - use stored coordinate classes directly
  for (const auto &angle : m_angles) {
    auto grad = angle.gradient(positions);

    B.block(row, 3 * angle.i, 1, 3) = grad.col(0).transpose();
    B.block(row, 3 * angle.j, 1, 3) = grad.col(1).transpose();
    B.block(row, 3 * angle.k, 1, 3) = grad.col(2).transpose();
    row++;
  }

  // Process dihedrals - use stored coordinate classes directly
  for (const auto &dihedral : m_dihedrals) {
    auto grad = dihedral.gradient(positions);

    B.block(row, 3 * dihedral.i, 1, 3) = grad.col(0).transpose();
    B.block(row, 3 * dihedral.j, 1, 3) = grad.col(1).transpose();
    B.block(row, 3 * dihedral.k, 1, 3) = grad.col(2).transpose();
    B.block(row, 3 * dihedral.l, 1, 3) = grad.col(3).transpose();
    row++;
  }

  return B;
}

Mat3N transform_step_to_cartesian(const Vec &internal_step,
                                  const InternalCoordinates &coords,
                                  const Mat3N &positions, const Mat &B_inv) {
  // Coordinate transformation algorithm
  // Use FULL coordinate system
  Vec current_q = coords.to_vector(positions);
  Vec dq = internal_step; // Start with target step
  Mat3N new_positions = positions;

  const int max_iter = 20; // Use 20 iterations
  const double tol = 1e-6; // Use 1e-6 threshold

  Mat3N first_positions = positions;
  Vec first_q = current_q;
  double first_dcart_rms = 0.0;
  double first_dq_rms = 0.0;

  for (int iter = 0; iter < max_iter; iter++) {
    // Update coordinates: coords_new = geom.coords + B_inv.dot(dq).reshape(-1,
    // 3) / angstrom
    Vec cart_step_flat = B_inv * dq;

    // Apply step with units conversion - divide by angstrom constant
    Mat3N coord_step(3, new_positions.cols());
    for (int i = 0; i < coord_step.cols(); i++) {
      // Using angstrom = 1 / 0.52917721092, division converts bohr to angstrom
      coord_step.col(i) =
          cart_step_flat.segment(3 * i, 3) / occ::units::ANGSTROM_TO_BOHR;
    }
    Mat3N coords_new = new_positions + coord_step;

    // Compute Cartesian RMS: dcart_rms = rms(coords_new - geom.coords)
    Mat3N coord_diff = coords_new - new_positions;
    double dcart_rms = std::sqrt(coord_diff.squaredNorm() / coord_diff.size());

    // Update geometry progressively: geom.coords = coords_new
    new_positions = coords_new;

    // Evaluate new coordinates: q_new = eval_geom(geom, template=q)
    Vec q_new = coords.to_vector_with_template(new_positions, current_q);

    // Compute internal coordinate RMS: dq_rms = rms(q_new - q)
    Vec q_diff = q_new - current_q;
    double dq_rms = std::sqrt(q_diff.squaredNorm() / q_diff.size());

    // Update coordinates and step: q, dq = q_new, dq - (q_new - q)
    Vec q_old = current_q;
    current_q = q_new;
    dq = dq - (q_new - q_old);

    // Convergence check: if dcart_rms < threshold
    if (dcart_rms < tol) {
      occ::log::debug("Perfect transformation to cartesians in {} iterations",
                      iter + 1);
      occ::log::debug("* RMS(dcart): {:.2e}, RMS(dq): {:.2e}", dcart_rms,
                      dq_rms);
      break;
    }

    // Keep first iteration as fallback
    if (iter == 0) {
      first_positions = new_positions;
      first_q = current_q;
      first_dcart_rms = dcart_rms;
      first_dq_rms = dq_rms;
    }

    // If we reach max iterations, use first iteration
    if (iter == max_iter - 1) {
      new_positions = first_positions;
      current_q = first_q;
      occ::log::debug("Transformation did not converge in {} iterations",
                      max_iter);
      occ::log::debug("RMS(dcart): {:.3e}, RMS(dq): {:.3e}", first_dcart_rms,
                      first_dq_rms);
    }
  }

  return new_positions;
}

std::pair<Vec, Mat3N> update_geometry(const Vec &current_q,
                                      const Vec &internal_step,
                                      const InternalCoordinates &coords,
                                      const Mat3N &positions,
                                      const Mat &B_inv) {
  // Use the fixed transform_step_to_cartesian function to avoid duplicate logic
  // Log inputs for debugging
  occ::log::trace("=== update_geometry INPUTS ===");
  occ::log::trace(
      "current_q size: {}, first 5: [{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}]",
      current_q.size(), current_q(0), current_q(1), current_q(2), current_q(3),
      current_q(4));
  occ::log::trace("internal_step size: {}, first 5: [{:.8f}, {:.8f}, {:.8f}, "
                  "{:.8f}, {:.8f}]",
                  internal_step.size(), internal_step(0), internal_step(1),
                  internal_step(2), internal_step(3), internal_step(4));
  occ::log::trace("dq norm: {:.8f}", std::sqrt(internal_step.squaredNorm() /
                                               internal_step.size()));
  occ::log::trace("positions shape: ({}, {}), atom 0: [{:.8f}, {:.8f}, {:.8f}]",
                  positions.rows(), positions.cols(), positions(0, 0),
                  positions(1, 0), positions(2, 0));
  occ::log::trace("B_inv shape: ({}, {}), norm: {:.6e}, max: {:.6e}",
                  B_inv.rows(), B_inv.cols(), B_inv.norm(),
                  B_inv.cwiseAbs().maxCoeff());

  // Call the fixed coordinate transformation function
  Mat3N new_positions =
      transform_step_to_cartesian(internal_step, coords, positions, B_inv);
  Vec new_q = coords.to_vector_with_template(new_positions, current_q);

  occ::log::trace("=== update_geometry OUTPUTS ===");
  occ::log::trace(
      "final q size: {}, first 5: [{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}]",
      new_q.size(), new_q(0), new_q(1), new_q(2), new_q(3), new_q(4));
  occ::log::trace("final positions atom 0: [{:.8f}, {:.8f}, {:.8f}]",
                  new_positions(0, 0), new_positions(1, 0),
                  new_positions(2, 0));

  return {new_q, new_positions};
}

Mat InternalCoordinates::hessian_guess() {
  // Hessian guess implementation
  // Use full coordinate space
  const int N = size();
  Mat H = Mat::Zero(N, N);

  int n_atoms = m_atomic_numbers.rows();

  // Build rho matrix
  Mat rho(n_atoms, n_atoms);
  for (int i = 0; i < n_atoms; i++) {
    for (int j = 0; j < n_atoms; j++) {
      if (i == j) {
        rho(i, j) = 1.0;
      } else {
        double dist = (m_positions.col(i) - m_positions.col(j)).norm();
        double sum_radii = get_covalent_radius(m_atomic_numbers(i)) +
                           get_covalent_radius(m_atomic_numbers(j));
        rho(i, j) = std::exp(-dist / sum_radii + 1.0); // Note the +1
      }
    }
  }

  // Build diagonal hessian using full coordinate space
  int idx = 0;

  // Bonds: Use stored coordinate classes directly
  for (const auto &bond : m_bonds) {
    H(idx, idx) = bond.hessian(rho);
    idx++;
  }

  // Angles: Use stored coordinate classes directly
  for (const auto &angle : m_angles) {
    H(idx, idx) = angle.hessian(rho);
    idx++;
  }

  // Dihedrals: Use stored coordinate classes directly
  for (const auto &dihedral : m_dihedrals) {
    H(idx, idx) = dihedral.hessian(rho);
    idx++;
  }

  return H;
}

} // namespace occ::opt
