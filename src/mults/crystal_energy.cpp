#include <occ/mults/crystal_energy.h>
#include <occ/mults/cartesian_hessian.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace occ::mults {

// ============================================================================
// MoleculeState
// ============================================================================

Mat3 MoleculeState::rotation_matrix() const {
    double angle = angle_axis.norm();
    if (angle < 1e-12) {
        return Mat3::Identity();
    }
    Vec3 axis = angle_axis / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

MoleculeState MoleculeState::from_rotation(const Vec3& pos, const Mat3& R) {
    MoleculeState state;
    state.position = pos;

    Eigen::AngleAxisd aa(R);
    state.angle_axis = aa.angle() * aa.axis();
    return state;
}

// ============================================================================
// CrystalEnergyResult
// ============================================================================

Vec CrystalEnergyResult::pack_gradient() const {
    const int n = static_cast<int>(forces.size());
    Vec grad(6 * n);
    for (int i = 0; i < n; ++i) {
        grad.segment<3>(6 * i) = forces[i];
        grad.segment<3>(6 * i + 3) = torques[i];
    }
    return grad;
}

// ============================================================================
// Williams DE Buckingham Parameters (kJ/mol, Angstrom)
// ============================================================================

std::map<std::pair<int,int>, BuckinghamParams> CrystalEnergy::williams_de_params() {
    // Reference: Williams & Cox, Acta Cryst. B40 (1984)
    // Parameters in kJ/mol and Angstrom
    std::map<std::pair<int,int>, BuckinghamParams> params;

    // H-H
    params[{1, 1}] = {2650.8, 3.74, 27.3};
    // C-C
    params[{6, 6}] = {369742.2, 3.60, 2439.8};
    // N-N
    params[{7, 7}] = {254501.2, 3.78, 1378.4};
    // O-O
    params[{8, 8}] = {230064.3, 3.96, 1123.6};

    // Cross-terms (geometric mean mixing for A and C, arithmetic for B)
    // H-C
    params[{1, 6}] = {31368.8, 3.67, 258.0};
    params[{6, 1}] = params[{1, 6}];
    // H-N
    params[{1, 7}] = {25988.3, 3.76, 194.0};
    params[{7, 1}] = params[{1, 7}];
    // H-O
    params[{1, 8}] = {24716.7, 3.85, 175.2};
    params[{8, 1}] = params[{1, 8}];
    // C-N
    params[{6, 7}] = {306739.8, 3.69, 1834.1};
    params[{7, 6}] = params[{6, 7}];
    // C-O
    params[{6, 8}] = {291770.4, 3.78, 1655.4};
    params[{8, 6}] = params[{6, 8}];
    // N-O
    params[{7, 8}] = {242022.9, 3.87, 1244.5};
    params[{8, 7}] = params[{7, 8}];

    return params;
}

// ============================================================================
// CrystalEnergy Constructor
// ============================================================================

CrystalEnergy::CrystalEnergy(const crystal::Crystal& crystal,
                             std::vector<MultipoleSource> multipoles,
                             double cutoff_radius,
                             ForceFieldType ff,
                             bool use_cartesian,
                             bool use_ewald,
                             double ewald_accuracy,
                             double ewald_eta,
                             int ewald_kmax)
    : m_crystal(crystal)
    , m_multipoles(std::move(multipoles))
    , m_cutoff_radius(cutoff_radius)
    , m_force_field(ff)
    , m_use_cartesian(use_cartesian)
    , m_use_ewald(use_ewald)
    , m_ewald_accuracy(ewald_accuracy)
    , m_ewald_eta(ewald_eta)
    , m_ewald_kmax(ewald_kmax) {

    if (m_multipoles.empty()) {
        throw std::invalid_argument("CrystalEnergy: no multipoles provided");
    }

    build_neighbor_list();
    build_molecule_geometry();
    initialize_force_field();
}

// ============================================================================
// Neighbor List Construction
// ============================================================================

void CrystalEnergy::build_neighbor_list() {
    m_neighbors.clear();

    auto dimers = m_crystal.symmetry_unique_dimers(m_cutoff_radius);

    const int num_unique = static_cast<int>(m_multipoles.size());

    for (size_t i = 0; i < dimers.molecule_neighbors.size(); ++i) {
        for (const auto& neighbor : dimers.molecule_neighbors[i]) {
            const auto& dimer = neighbor.dimer;

            // Get molecule indices
            int mol_i = static_cast<int>(i);
            // Get the asymmetric molecule index of the neighbor (molecule B in dimer)
            int mol_j = dimer.b().asymmetric_molecule_idx();

            // Validate indices
            if (mol_i < 0 || mol_i >= num_unique || mol_j < 0 || mol_j >= num_unique) {
                occ::log::debug("Skipping invalid neighbor pair: mol_i={}, mol_j={}, num_unique={}",
                               mol_i, mol_j, num_unique);
                continue;
            }

            // Compute cell shift from molecule B's position
            const auto& mol_a = dimer.a();
            const auto& mol_b = dimer.b();

            Vec3 center_a = mol_a.center_of_mass();
            Vec3 center_b = mol_b.center_of_mass();

            // Convert to fractional and find integer shift
            Vec3 frac_a = m_crystal.to_fractional(center_a);
            Vec3 frac_b = m_crystal.to_fractional(center_b);
            Vec3 diff = frac_b - frac_a;

            IVec3 cell_shift;
            cell_shift << static_cast<int>(std::round(diff[0])),
                         static_cast<int>(std::round(diff[1])),
                         static_cast<int>(std::round(diff[2]));

            // Weight: 0.5 for self-interaction, 1.0 otherwise
            double weight = (mol_i == mol_j && cell_shift == IVec3::Zero()) ? 0.5 : 1.0;

            // Skip true self (same molecule, no translation)
            if (mol_i == mol_j && cell_shift == IVec3::Zero()) {
                continue;
            }

            m_neighbors.push_back({mol_i, mol_j, cell_shift, weight});
        }
    }

    occ::log::debug("Built neighbor list with {} pairs", m_neighbors.size());
}

void CrystalEnergy::update_neighbors() {
    build_neighbor_list();
}

void CrystalEnergy::build_neighbor_list_from_positions(
        const std::vector<Vec3>& mol_coms, bool force_com_cutoff) {
    m_neighbors.clear();
    int n_mol = static_cast<int>(mol_coms.size());

    const auto& uc = m_crystal.unit_cell();
    double min_len = std::min({uc.a(), uc.b(), uc.c()});

    // Use atom-based cutoff: include pair if any atom-atom distance < cutoff.
    // When force_com_cutoff=true, use COM distance (matches DMACRYS TBLCNT).
    // Need extra margin for COM-to-atom extent when determining cell search range.
    bool use_atom_cutoff = !m_geometry.empty() && !force_com_cutoff;
    double max_atom_extent = 0.0;
    // Pre-compute crystal-frame atom positions for each molecule
    std::vector<std::vector<Vec3>> crystal_atoms(n_mol);
    if (use_atom_cutoff) {
        for (int m = 0; m < n_mol; ++m) {
            const auto& geom = m_geometry[m];
            Mat3 R = m_initial_states[m].rotation_matrix();
            crystal_atoms[m].reserve(geom.atom_positions.size());
            for (const auto& bp : geom.atom_positions) {
                Vec3 cart = mol_coms[m] + R * bp;
                crystal_atoms[m].push_back(cart);
                double ext = (R * bp).norm();
                if (ext > max_atom_extent)
                    max_atom_extent = ext;
            }
        }
    }

    double search_radius = m_cutoff_radius + 2.0 * max_atom_extent;
    int nmax = static_cast<int>(std::ceil(search_radius / min_len)) + 1;

    for (int i = 0; i < n_mol; ++i) {
        for (int j = 0; j < n_mol; ++j) {
            for (int nx = -nmax; nx <= nmax; ++nx) {
                for (int ny = -nmax; ny <= nmax; ++ny) {
                    for (int nz = -nmax; nz <= nmax; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;

                        IVec3 shift(nx, ny, nz);
                        Vec3 trans = uc.to_cartesian(shift.cast<double>());

                        bool include = false;
                        if (use_atom_cutoff) {
                            // Check minimum atom-atom distance
                            for (const auto& ai : crystal_atoms[i]) {
                                for (const auto& aj : crystal_atoms[j]) {
                                    double d = (aj + trans - ai).norm();
                                    if (d < m_cutoff_radius) {
                                        include = true;
                                        break;
                                    }
                                }
                                if (include) break;
                            }
                        } else {
                            double dist = (mol_coms[j] + trans - mol_coms[i]).norm();
                            include = (dist < m_cutoff_radius);
                        }

                        if (include) {
                            // Weight 0.5: lattice sum E = (1/2) sum_i sum_{j≠i}
                            // Our list includes both (i,j,shift) and (j,i,-shift)
                            double com_dist = (mol_coms[j] + trans - mol_coms[i]).norm();
                            m_neighbors.push_back({i, j, shift, 0.5, com_dist});
                        }
                    }
                }
            }
        }
    }

    occ::log::info("Built neighbor list from positions: {} pairs for {} molecules (atom-based cutoff: {})",
                   m_neighbors.size(), n_mol, use_atom_cutoff);
}

void CrystalEnergy::set_neighbor_list(const std::vector<NeighborPair>& neighbors) {
    m_neighbors = neighbors;
}

void CrystalEnergy::set_molecule_geometry(std::vector<MoleculeGeometry> geometry) {
    m_geometry = std::move(geometry);
}

void CrystalEnergy::set_initial_states(std::vector<MoleculeState> states) {
    m_initial_states = std::move(states);
}

// ============================================================================
// Molecule Geometry
// ============================================================================

void CrystalEnergy::build_molecule_geometry() {
    m_geometry.clear();
    m_geometry.reserve(m_multipoles.size());

    const auto& unique_mols = m_crystal.symmetry_unique_molecules();

    for (size_t i = 0; i < m_multipoles.size() && i < unique_mols.size(); ++i) {
        const auto& mol = unique_mols[i];
        MoleculeGeometry geom;

        geom.center_of_mass = mol.center_of_mass();

        for (int j = 0; j < mol.size(); ++j) {
            geom.atomic_numbers.push_back(mol.atomic_numbers()(j));
            // Store in body frame (relative to COM)
            geom.atom_positions.push_back(mol.positions().col(j) - geom.center_of_mass);
        }

        m_geometry.push_back(std::move(geom));
    }
}

// ============================================================================
// Force Field Initialization
// ============================================================================

void CrystalEnergy::initialize_force_field() {
    if (m_force_field == ForceFieldType::BuckinghamDE) {
        m_buckingham_params = williams_de_params();
    }
}

void CrystalEnergy::set_buckingham_params(int Z1, int Z2, const BuckinghamParams& params) {
    m_buckingham_params[{Z1, Z2}] = params;
    m_buckingham_params[{Z2, Z1}] = params;
}

BuckinghamParams CrystalEnergy::get_buckingham_params(int Z1, int Z2) const {
    auto it = m_buckingham_params.find({Z1, Z2});
    if (it != m_buckingham_params.end()) {
        return it->second;
    }
    // Default: small repulsion to avoid overlaps
    return {1000.0, 3.5, 10.0};
}

// ============================================================================
// Initial States
// ============================================================================

std::vector<MoleculeState> CrystalEnergy::initial_states() const {
    if (!m_initial_states.empty()) {
        return m_initial_states;
    }

    std::vector<MoleculeState> states;
    states.reserve(m_multipoles.size());

    const auto& unique_mols = m_crystal.symmetry_unique_molecules();

    for (size_t i = 0; i < m_multipoles.size() && i < unique_mols.size(); ++i) {
        MoleculeState state;
        state.position = unique_mols[i].center_of_mass();
        state.angle_axis = Vec3::Zero();  // Identity rotation
        states.push_back(state);
    }

    return states;
}

// ============================================================================
// Buckingham Site-Pair Masks
// ============================================================================

std::vector<std::vector<bool>> CrystalEnergy::compute_buckingham_site_masks(
    const std::vector<MoleculeState>& states) const {

    const double buck_cutoff = (m_buck_site_cutoff > 0) ? m_buck_site_cutoff : m_cutoff_radius;

    std::vector<std::vector<bool>> masks(m_neighbors.size());

    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        int mi = pair.mol_i;
        int mj = pair.mol_j;
        const auto& geom_i = m_geometry[mi];
        const auto& geom_j = m_geometry[mj];
        const size_t nA = geom_i.atom_positions.size();
        const size_t nB = geom_j.atom_positions.size();

        masks[pair_idx].resize(nA * nB, false);

        Mat3 R_i = states[mi].rotation_matrix();
        Mat3 R_j = states[mj].rotation_matrix();
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        for (size_t a = 0; a < nA; ++a) {
            Vec3 pos_a = states[mi].position + R_i * geom_i.atom_positions[a];
            for (size_t b = 0; b < nB; ++b) {
                Vec3 pos_b = states[mj].position + cell_translation +
                             R_j * geom_j.atom_positions[b];
                double r = (pos_b - pos_a).norm();
                if (r <= buck_cutoff && r >= 0.1) {
                    masks[pair_idx][a * nB + b] = true;
                }
            }
        }
    }

    return masks;
}

// ============================================================================
// Short-Range Pair Computation
// ============================================================================

void CrystalEnergy::compute_short_range_pair(
    int mol_i, int mol_j,
    const MoleculeState& state_i,
    const MoleculeState& state_j,
    const Vec3& translation,
    double weight,
    double& energy,
    Vec3& force_i, Vec3& force_j,
    Vec3& torque_i, Vec3& torque_j,
    int neighbor_idx) const {

    if (m_force_field == ForceFieldType::None) {
        return;
    }

    const auto& geom_i = m_geometry[mol_i];
    const auto& geom_j = m_geometry[mol_j];
    const size_t nB = geom_j.atom_positions.size();

    // Check if we have frozen site masks for this neighbor pair
    const bool use_frozen = (neighbor_idx >= 0 &&
                             static_cast<size_t>(neighbor_idx) < m_fixed_site_masks.size() &&
                             !m_fixed_site_masks[neighbor_idx].empty());

    Mat3 R_i = state_i.rotation_matrix();
    Mat3 R_j = state_j.rotation_matrix();

    // Loop over atom pairs
    for (size_t a = 0; a < geom_i.atom_positions.size(); ++a) {
        int Z_a = geom_i.atomic_numbers[a];
        Vec3 pos_a = state_i.position + R_i * geom_i.atom_positions[a];

        for (size_t b = 0; b < nB; ++b) {
            // Apply frozen mask or distance cutoff
            if (use_frozen) {
                if (!m_fixed_site_masks[neighbor_idx][a * nB + b]) {
                    continue;
                }
            }

            int Z_b = geom_j.atomic_numbers[b];
            Vec3 pos_b = state_j.position + translation + R_j * geom_j.atom_positions[b];

            Vec3 r_ab = pos_b - pos_a;
            double r = r_ab.norm();

            if (!use_frozen) {
                double buck_cutoff = (m_buck_site_cutoff > 0) ? m_buck_site_cutoff : m_cutoff_radius;
                if (r > buck_cutoff || r < 0.1) {
                    continue;
                }
            }

            ShortRangeInteraction::EnergyAndDerivatives sr;

            if (m_force_field == ForceFieldType::BuckinghamDE ||
                m_force_field == ForceFieldType::Custom) {
                auto params = get_buckingham_params(Z_a, Z_b);
                sr = ShortRangeInteraction::buckingham_all(r, params);
            } else if (m_force_field == ForceFieldType::LennardJones) {
                auto it = m_lj_params.find({Z_a, Z_b});
                if (it == m_lj_params.end()) {
                    continue;
                }
                sr = ShortRangeInteraction::lennard_jones_all(r, it->second);
            }

            energy += weight * sr.energy;

            // Force from derivative
            Vec3 force_on_a = ShortRangeInteraction::derivative_to_force(sr.first_derivative, r_ab);

            // Weighted forces
            Vec3 wf = weight * force_on_a;
            force_i += wf;
            force_j -= wf;

            // Torque from lever arm (body frame)
            Vec3 lever_a = R_i * geom_i.atom_positions[a];
            Vec3 lever_b = R_j * geom_j.atom_positions[b];

            // Torque in lab frame, then convert to body frame for angle-axis grad
            Vec3 torque_lab_a = lever_a.cross(wf);
            Vec3 torque_lab_b = lever_b.cross(-wf);

            // For small rotations, angle-axis gradient ≈ torque in body frame
            torque_i += R_i.transpose() * torque_lab_a;
            torque_j += R_j.transpose() * torque_lab_b;
        }
    }
}

// ============================================================================
// Main Energy Computation
// ============================================================================

CrystalEnergyResult CrystalEnergy::compute(const std::vector<MoleculeState>& molecules) {
    const int N = static_cast<int>(molecules.size());
    CrystalEnergyResult result;
    result.forces.resize(N, Vec3::Zero());
    result.torques.resize(N, Vec3::Zero());
    result.molecule_energies.resize(N, 0.0);

    // Update multipole orientations (modifies m_multipoles in place)
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            molecules[i].rotation_matrix(),
            molecules[i].position);
    }

    // Pre-build all CartesianMolecules ONCE (expensive conversion happens here)
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }

    // Loop over neighbor pairs
    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        int i = pair.mol_i;
        int j = pair.mol_j;

        // Translation for molecule j
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        // Create translated copy - just shift positions, don't rebuild multipoles
        CartesianMolecule mol_j_translated = cart_mols[j];
        for (auto& site : mol_j_translated.sites) {
            site.position += cell_translation;
        }

        // Electrostatic interaction.
        // COM gate: skip electrostatics for pairs with COM distance > cutoff.
        // This matches DMACRYS TBLCNT which selects molecule pairs by COM distance.
        // Ewald correction compensates for qq+qμ+μμ regardless of pair selection.
        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate &&
            pair.com_distance > m_cutoff_radius) {
            include_elec = false;
        }
        if (include_elec) {
            auto elec_result = compute_molecule_forces_torques(
                cart_mols[i],
                mol_j_translated,
                m_elec_site_cutoff,
                m_max_interaction_order);

            double e_elec = pair.weight * elec_result.energy;
            result.electrostatic_energy += e_elec;
            result.molecule_energies[i] += e_elec * 0.5;
            result.molecule_energies[j] += e_elec * 0.5;

            result.forces[i] += pair.weight * elec_result.force_A;
            result.forces[j] += pair.weight * elec_result.force_B;
            result.torques[i] += pair.weight * elec_result.grad_angle_axis_A;
            result.torques[j] += pair.weight * elec_result.grad_angle_axis_B;
        }

        // Short-range interaction
        double sr_energy = 0.0;
        Vec3 sr_force_i = Vec3::Zero();
        Vec3 sr_force_j = Vec3::Zero();
        Vec3 sr_torque_i = Vec3::Zero();
        Vec3 sr_torque_j = Vec3::Zero();

        compute_short_range_pair(
            i, j,
            molecules[i], molecules[j],
            cell_translation,
            pair.weight,
            sr_energy,
            sr_force_i, sr_force_j,
            sr_torque_i, sr_torque_j,
            static_cast<int>(pair_idx));

        result.repulsion_dispersion += sr_energy;
        result.molecule_energies[i] += sr_energy * 0.5;
        result.molecule_energies[j] += sr_energy * 0.5;

        result.forces[i] += sr_force_i;
        result.forces[j] += sr_force_j;
        result.torques[i] += sr_torque_i;
        result.torques[j] += sr_torque_j;
    }

    // Ewald correction for charge-charge electrostatics
    if (m_use_ewald) {
        auto ewald = compute_charge_ewald_correction(molecules, cart_mols);
        result.electrostatic_energy += ewald.energy;
        for (int i = 0; i < N; ++i) {
            result.forces[i] += ewald.forces[i];
            result.torques[i] += ewald.torques[i];
        }
    }

    result.total_energy = result.electrostatic_energy + result.repulsion_dispersion;
    return result;
}

double CrystalEnergy::compute_energy(const std::vector<MoleculeState>& molecules) {
    // Simplified version without gradient computation
    // For now, just call compute() and return energy
    // Could be optimized later to skip gradient calculation
    return compute(molecules).total_energy;
}

// ============================================================================
// Hessian Computation - Analytical for electrostatics, numerical for short-range
// ============================================================================

CrystalEnergyResultWithHessian CrystalEnergy::compute_with_hessian(
    const std::vector<MoleculeState>& molecules) {

    const int N = static_cast<int>(molecules.size());
    const int ndof = 6 * N;  // 3 translation + 3 rotation per molecule

    CrystalEnergyResultWithHessian result;

    // First compute energy and gradient at current point
    auto base_result = compute(molecules);
    result.total_energy = base_result.total_energy;
    result.electrostatic_energy = base_result.electrostatic_energy;
    result.repulsion_dispersion = base_result.repulsion_dispersion;
    result.forces = base_result.forces;
    result.torques = base_result.torques;
    result.molecule_energies = base_result.molecule_energies;

    // Initialize Hessian
    result.hessian = Mat::Zero(ndof, ndof);

    // Update multipole orientations
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            molecules[i].rotation_matrix(),
            molecules[i].position);
    }

    // Pre-build CartesianMolecules
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }

    // ========================================================================
    // Analytical Hessian for electrostatics (charge-charge and charge-dipole)
    // ========================================================================
    for (const auto& pair : m_neighbors) {
        int i = pair.mol_i;
        int j = pair.mol_j;

        // Translation for molecule j
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        // Create translated copy
        CartesianMolecule mol_j_translated = cart_mols[j];
        for (auto& site : mol_j_translated.sites) {
            site.position += cell_translation;
        }

        // Compute analytical Hessian for this pair
        auto pair_hess = compute_molecule_hessian_truncated(cart_mols[i], mol_j_translated);

        // Accumulate into full Hessian
        // Layout: mol i uses DOF [6*i, 6*i+6), mol j uses DOF [6*j, 6*j+6)
        // Position: [6*k, 6*k+3), Rotation: [6*k+3, 6*k+6)

        double w = pair.weight;

        // Position-Position blocks
        result.hessian.block<3, 3>(6 * i, 6 * i) += w * pair_hess.H_posA_posA;
        result.hessian.block<3, 3>(6 * i, 6 * j) += w * pair_hess.H_posA_posB;
        result.hessian.block<3, 3>(6 * j, 6 * i) += w * pair_hess.H_posA_posB.transpose();
        result.hessian.block<3, 3>(6 * j, 6 * j) += w * pair_hess.H_posB_posB;

        // Position-Rotation cross blocks
        result.hessian.block<3, 3>(6 * i, 6 * i + 3) += w * pair_hess.H_posA_rotA;
        result.hessian.block<3, 3>(6 * i + 3, 6 * i) += w * pair_hess.H_posA_rotA.transpose();
        result.hessian.block<3, 3>(6 * i, 6 * j + 3) += w * pair_hess.H_posA_rotB;
        result.hessian.block<3, 3>(6 * j + 3, 6 * i) += w * pair_hess.H_posA_rotB.transpose();
        result.hessian.block<3, 3>(6 * j, 6 * i + 3) += w * pair_hess.H_posB_rotA;
        result.hessian.block<3, 3>(6 * i + 3, 6 * j) += w * pair_hess.H_posB_rotA.transpose();
        result.hessian.block<3, 3>(6 * j, 6 * j + 3) += w * pair_hess.H_posB_rotB;
        result.hessian.block<3, 3>(6 * j + 3, 6 * j) += w * pair_hess.H_posB_rotB.transpose();

        // Rotation-Rotation blocks
        result.hessian.block<3, 3>(6 * i + 3, 6 * i + 3) += w * pair_hess.H_rotA_rotA;
        result.hessian.block<3, 3>(6 * i + 3, 6 * j + 3) += w * pair_hess.H_rotA_rotB;
        result.hessian.block<3, 3>(6 * j + 3, 6 * i + 3) += w * pair_hess.H_rotA_rotB.transpose();
        result.hessian.block<3, 3>(6 * j + 3, 6 * j + 3) += w * pair_hess.H_rotB_rotB;
    }

    // ========================================================================
    // Numerical Hessian for short-range (Buckingham/LJ) contributions
    // ========================================================================
    if (m_force_field != ForceFieldType::None) {
        const double h = 1e-5;

        // Helper to compute short-range gradient only
        auto compute_sr_gradient = [&](const std::vector<MoleculeState>& states) -> Vec {
            Vec grad = Vec::Zero(ndof);
            for (size_t pidx = 0; pidx < m_neighbors.size(); ++pidx) {
                const auto& pair = m_neighbors[pidx];
                int pi = pair.mol_i;
                int pj = pair.mol_j;
                Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
                    pair.cell_shift.cast<double>());

                double sr_energy = 0.0;
                Vec3 sr_force_i = Vec3::Zero(), sr_force_j = Vec3::Zero();
                Vec3 sr_torque_i = Vec3::Zero(), sr_torque_j = Vec3::Zero();

                compute_short_range_pair(pi, pj, states[pi], states[pj],
                                        cell_translation, pair.weight,
                                        sr_energy, sr_force_i, sr_force_j,
                                        sr_torque_i, sr_torque_j,
                                        static_cast<int>(pidx));

                grad.segment<3>(6 * pi) -= sr_force_i;
                grad.segment<3>(6 * pj) -= sr_force_j;
                grad.segment<3>(6 * pi + 3) += sr_torque_i;
                grad.segment<3>(6 * pj + 3) += sr_torque_j;
            }
            return grad;
        };

        // Numerical differentiation of short-range gradient
        for (int dof = 0; dof < ndof; ++dof) {
            int mol_idx = dof / 6;
            int comp = dof % 6;
            bool is_trans = (comp < 3);

            std::vector<MoleculeState> states_plus = molecules;
            std::vector<MoleculeState> states_minus = molecules;

            if (is_trans) {
                states_plus[mol_idx].position[comp] += h;
                states_minus[mol_idx].position[comp] -= h;
            } else {
                int rot_comp = comp - 3;
                Vec3 axis = Vec3::Zero();
                axis[rot_comp] = 1.0;
                double c = std::cos(h), s = std::sin(h);
                Mat3 dR_plus, dR_minus;
                if (rot_comp == 0) {
                    dR_plus << 1, 0, 0, 0, c, -s, 0, s, c;
                    dR_minus << 1, 0, 0, 0, c, s, 0, -s, c;
                } else if (rot_comp == 1) {
                    dR_plus << c, 0, s, 0, 1, 0, -s, 0, c;
                    dR_minus << c, 0, -s, 0, 1, 0, s, 0, c;
                } else {
                    dR_plus << c, -s, 0, s, c, 0, 0, 0, 1;
                    dR_minus << c, s, 0, -s, c, 0, 0, 0, 1;
                }
                Mat3 R_base = molecules[mol_idx].rotation_matrix();
                Eigen::AngleAxisd aa_plus(dR_plus * R_base);
                Eigen::AngleAxisd aa_minus(dR_minus * R_base);
                states_plus[mol_idx].angle_axis = aa_plus.angle() * aa_plus.axis();
                states_minus[mol_idx].angle_axis = aa_minus.angle() * aa_minus.axis();
            }

            Vec g_plus = compute_sr_gradient(states_plus);
            Vec g_minus = compute_sr_gradient(states_minus);
            result.hessian.col(dof) += (g_plus - g_minus) / (2.0 * h);
        }
    }

    // Symmetrize Hessian (handles any numerical noise)
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());

    return result;
}

// ============================================================================
// Pack Hessian for Reduced DOF
// ============================================================================

Mat CrystalEnergyResultWithHessian::pack_hessian(
    bool fix_first_translation, bool fix_first_rotation) const {

    int N = static_cast<int>(forces.size());
    int ndof_full = 6 * N;

    // Count reduced DOF
    int mol0_dof = 0;
    if (!fix_first_translation) mol0_dof += 3;
    if (!fix_first_rotation) mol0_dof += 3;
    int ndof_reduced = mol0_dof + 6 * (N - 1);

    if (ndof_reduced == ndof_full) {
        return hessian;
    }

    Mat H_reduced = Mat::Zero(ndof_reduced, ndof_reduced);

    // Build index mapping from reduced to full DOF
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(ndof_reduced);

    // Molecule 0
    if (!fix_first_translation) {
        reduced_to_full.push_back(0);
        reduced_to_full.push_back(1);
        reduced_to_full.push_back(2);
    }
    if (!fix_first_rotation) {
        reduced_to_full.push_back(3);
        reduced_to_full.push_back(4);
        reduced_to_full.push_back(5);
    }

    // Molecules 1..N-1
    for (int i = 1; i < N; ++i) {
        for (int c = 0; c < 6; ++c) {
            reduced_to_full.push_back(6 * i + c);
        }
    }

    // Extract submatrix
    for (int i = 0; i < ndof_reduced; ++i) {
        for (int j = 0; j < ndof_reduced; ++j) {
            H_reduced(i, j) = hessian(reduced_to_full[i], reduced_to_full[j]);
        }
    }

    return H_reduced;
}

int CrystalEnergy::max_multipole_rank() const {
    int max_rank = 0;
    for (const auto& mp : m_multipoles) {
        const auto& cart = mp.cartesian();
        for (const auto& site : cart.sites) {
            max_rank = std::max(max_rank, site.rank);
        }
    }
    return max_rank;
}

size_t CrystalEnergy::num_sites() const {
    size_t total = 0;
    for (const auto& mp : m_multipoles) {
        total += mp.cartesian().sites.size();
    }
    return total;
}

std::vector<PairEnergyDebug> CrystalEnergy::debug_pair_energies(const std::vector<MoleculeState>& molecules) const {
    const int N = static_cast<int>(molecules.size());
    std::vector<PairEnergyDebug> out;
    out.reserve(m_neighbors.size());

    // Update multipole orientations
    std::vector<MultipoleSource> multipoles = m_multipoles;  // copy to keep const correctness
    for (int i = 0; i < N; ++i) {
        multipoles[i].set_orientation(
            molecules[i].rotation_matrix(),
            molecules[i].position);
    }

    // Pre-build Cartesian molecules
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(multipoles[i].cartesian());
    }

    for (const auto& pair : m_neighbors) {
        int i = pair.mol_i;
        int j = pair.mol_j;

        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        CartesianMolecule mol_j_translated = cart_mols[j];
        for (auto& site : mol_j_translated.sites) {
            site.position += cell_translation;
        }

        PairEnergyDebug dbg;
        dbg.mol_i = i;
        dbg.mol_j = j;
        dbg.cell_shift = pair.cell_shift;
        dbg.weight = pair.weight;

        Vec3 com_i = molecules[i].position;
        Vec3 com_j = molecules[j].position + cell_translation;
        dbg.com_distance = (com_j - com_i).norm();

        if (m_use_cartesian) {
            auto elec = compute_molecule_forces_torques(cart_mols[i], mol_j_translated, 0.0);
            dbg.electrostatic = pair.weight * elec.energy;
            dbg.total += dbg.electrostatic;
        }

        if (m_force_field != ForceFieldType::None) {
            double sr_energy = 0.0;
            Vec3 dummyF, dummyF2, dummyT, dummyT2;
            compute_short_range_pair(i, j, molecules[i], molecules[j], cell_translation,
                                     pair.weight, sr_energy, dummyF, dummyF2, dummyT, dummyT2);
            dbg.short_range = sr_energy;
            dbg.total += sr_energy;
        }

        out.push_back(dbg);
    }

    return out;
}

std::vector<int> CrystalEnergy::neighbor_shell_histogram() const {
    std::vector<int> bins(5, 0);
    for (const auto& pair : m_neighbors) {
        Vec3 shift = m_crystal.unit_cell().to_cartesian(pair.cell_shift.cast<double>());
        Vec3 com_i = m_crystal.symmetry_unique_molecules()[pair.mol_i].center_of_mass();
        Vec3 com_j = m_crystal.symmetry_unique_molecules()[pair.mol_j].center_of_mass() + shift;
        double d = (com_j - com_i).norm();
        if (d < 3.0) bins[0]++; else if (d < 6.0) bins[1]++; else if (d < 10.0) bins[2]++; else if (d < 15.0) bins[3]++; else bins[4]++;
    }
    return bins;
}

// ============================================================================
// Charge-Charge Ewald Correction
// ============================================================================

CrystalEnergy::EwaldCorrectionResult CrystalEnergy::compute_charge_ewald_correction(
    const std::vector<MoleculeState>& molecules,
    const std::vector<CartesianMolecule>& cart_mols) const {

    const int N = static_cast<int>(molecules.size());
    EwaldCorrectionResult ewald_result;
    ewald_result.forces.resize(N, Vec3::Zero());
    ewald_result.torques.resize(N, Vec3::Zero());

    // 1. Gather charge+dipole sites (position in Bohr/Ang, charge/dipole in a.u.)
    struct ChargeSite {
        Vec3 pos_bohr;
        Vec3 pos_ang;
        double charge;     // Q00 (a.u.)
        Vec3 dipole;       // (μx, μy, μz) in a.u. (e·Bohr)
        int mol_index;
    };

    std::vector<ChargeSite> all_sites;
    std::vector<std::vector<size_t>> mol_site_indices(N);

    for (int m = 0; m < N; ++m) {
        const auto& cart = cart_mols[m];
        for (size_t s = 0; s < cart.sites.size(); ++s) {
            const auto& site = cart.sites[s];
            if (site.rank < 0) continue;

            ChargeSite cs;
            cs.pos_ang = site.position;
            cs.pos_bohr = site.position * occ::units::ANGSTROM_TO_BOHR;
            cs.charge = site.cart.data[0];  // Q00
            // Dipole: data[1]=μx, data[2]=μy, data[3]=μz (a.u.)
            if (m_ewald_dipole && site.rank >= 1) {
                cs.dipole = Vec3(site.cart.data[1], site.cart.data[2], site.cart.data[3]);
            } else {
                cs.dipole = Vec3::Zero();
            }
            cs.mol_index = m;
            size_t idx = all_sites.size();
            all_sites.push_back(cs);
            mol_site_indices[m].push_back(idx);
        }
    }

    if (all_sites.empty()) return ewald_result;

    // 2. Select Ewald parameters
    double alpha;  // in 1/Angstrom
    int kmax;

    if (m_ewald_eta > 0) {
        alpha = m_ewald_eta;
    } else {
        // erfc(alpha * r_cut) ≈ accuracy
        double x = std::sqrt(-std::log(m_ewald_accuracy));
        alpha = x / m_cutoff_radius;
    }

    const auto& uc = m_crystal.unit_cell();
    if (m_ewald_kmax > 0) {
        kmax = m_ewald_kmax;
    } else {
        double min_len = std::min({uc.a(), uc.b(), uc.c()});
        double G_max = 2.0 * alpha * std::sqrt(-std::log(m_ewald_accuracy));
        kmax = std::max(1, static_cast<int>(std::ceil(
            G_max * min_len / (2.0 * M_PI))));
    }

    double alpha_bohr = alpha * occ::units::BOHR_TO_ANGSTROM;
    double two_alpha_over_sqrt_pi = 2.0 * alpha_bohr / std::sqrt(M_PI);

    occ::log::debug("Ewald correction: alpha = {:.4f} /Ang ({:.6f} /Bohr), kmax = {}",
                    alpha, alpha_bohr, kmax);

    // Per-site force accumulator (in Hartree/Bohr)
    std::vector<Vec3> site_forces(all_sites.size(), Vec3::Zero());
    double energy_ha = 0.0;  // in Hartree

    // Diagnostic accumulators (in Hartree)
    double diag_erf_inter_qq = 0.0, diag_erf_inter_qmu = 0.0, diag_erf_inter_mumu = 0.0;
    double diag_erf_intra_qq = 0.0, diag_erf_intra_qmu = 0.0, diag_erf_intra_mumu = 0.0;
    double diag_recip_qq = 0.0, diag_recip_qmu = 0.0, diag_recip_mumu = 0.0;
    double diag_self_qq = 0.0, diag_self_mumu = 0.0;

    // Helper: compute erf correction for a site pair (qq + qμ + μμ).
    // R_vec = r_b - r_a (Bohr). Returns energy correction in Hartree.
    // Also accumulates into diag_qq, diag_qmu, diag_mumu for diagnostics.
    auto erf_pair_correction = [&](const ChargeSite& sa, const ChargeSite& sb,
                                   const Vec3& R_vec, double w,
                                   Vec3& f_a, Vec3& f_b,
                                   double& diag_qq, double& diag_qmu, double& diag_mumu) {
        double r = R_vec.norm();
        if (r < 1e-10) return 0.0;

        double ar = alpha_bohr * r;
        double r2 = r * r;
        double r3 = r2 * r;
        double erf_val = std::erf(ar);
        double exp_val = std::exp(-ar * ar);
        double qa = sa.charge, qb = sb.charge;
        const Vec3& da = sa.dipole;
        const Vec3& db = sb.dipole;

        // Charge-charge erf correction: -q_a * q_b * erf(αr) / r
        double e_qq = -w * qa * qb * erf_val / r;

        // Charge-charge force correction
        double f_over_r = qa * qb *
            (-erf_val / r2 + two_alpha_over_sqrt_pi * exp_val / r);
        Vec3 f_corr = w * f_over_r * (R_vec / r);
        f_a -= f_corr;
        f_b += f_corr;

        // Charge-dipole erf correction:
        // The bare qμ energy from T-tensor contraction is:
        //   E_qμ = A * [q_A(R·μ_B) - q_B(R·μ_A)]
        // where A = ψ'(r)/r, ψ = erf(αr)/r
        // (The sign comes from sign_inv_fact having (-1)^1 for the A-side dipole)
        // Correction = -w * E_erf_qμ
        double f1 = two_alpha_over_sqrt_pi * exp_val / r2 - erf_val / r3;
        double muA_dot_R = da.dot(R_vec);
        double muB_dot_R = db.dot(R_vec);
        double e_qmu = -w * (qa * muB_dot_R - qb * muA_dot_R) * f1;

        // Dipole-dipole erf correction:
        // T_αβ^erf = δ_αβ A(r) + R_α R_β B(r)
        // where A = ψ'(r)/r, B = ψ''(r)/r² - ψ'(r)/r³
        // The bare μμ energy from T-tensor contraction is:
        //   E_μμ = -[(μ_a·μ_b)A + (μ_a·R)(μ_b·R)B]
        // (The minus comes from sign_inv_fact having (-1)^1 for the A-side dipole)
        // Correction = -w * E_erf_μμ = w * [(μ_a·μ_b)A + (μ_a·R)(μ_b·R)B]
        double r4 = r2 * r2;
        double r5 = r4 * r;
        double A = f1;
        double Ap_over_r = -two_alpha_over_sqrt_pi * exp_val *
            (2.0 * alpha_bohr * alpha_bohr / r2 + 3.0 / r4) +
            3.0 * erf_val / r5;
        double e_mumu = w * (da.dot(db) * A + muA_dot_R * muB_dot_R * Ap_over_r);

        diag_qq += e_qq;
        diag_qmu += e_qmu;
        diag_mumu += e_mumu;

        return e_qq + e_qmu + e_mumu;
    };

    // 3. Real-space erf correction over neighbor pairs (inter-molecular)
    // Must match the main electrostatic loop's COM gate and per-site cutoff.
    const bool use_erf_site_cutoff = (m_elec_site_cutoff > 0.0);
    const double erf_site_cutoff_bohr = m_elec_site_cutoff * occ::units::ANGSTROM_TO_BOHR;
    for (const auto& pair : m_neighbors) {
        // Apply same COM gate as main electrostatic loop
        if (m_use_com_elec_gate && pair.com_distance > m_cutoff_radius)
            continue;

        int mi = pair.mol_i;
        int mj = pair.mol_j;

        Vec3 cell_trans_bohr = uc.to_cartesian(
            pair.cell_shift.cast<double>()) * occ::units::ANGSTROM_TO_BOHR;

        for (size_t ai : mol_site_indices[mi]) {
            for (size_t bj : mol_site_indices[mj]) {
                Vec3 R_vec = all_sites[bj].pos_bohr + cell_trans_bohr
                           - all_sites[ai].pos_bohr;
                if (use_erf_site_cutoff && R_vec.norm() > erf_site_cutoff_bohr)
                    continue;
                energy_ha += erf_pair_correction(
                    all_sites[ai], all_sites[bj], R_vec, pair.weight,
                    site_forces[ai], site_forces[bj],
                    diag_erf_inter_qq, diag_erf_inter_qmu, diag_erf_inter_mumu);
            }
        }
    }

    // 3b. Intra-molecular erf correction
    // The reciprocal space includes all pairs (inter + intra).
    // The erf correction must also cover intra-molecular pairs so they
    // cancel, giving a net correction only for inter-molecular interactions.
    for (int m = 0; m < N; ++m) {
        const auto& indices = mol_site_indices[m];
        for (size_t ii = 0; ii < indices.size(); ++ii) {
            size_t ai = indices[ii];
            for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
                size_t bi = indices[jj];
                Vec3 R_vec = all_sites[bi].pos_bohr - all_sites[ai].pos_bohr;
                // Unique pair (i<j), weight = 1.0
                energy_ha += erf_pair_correction(
                    all_sites[ai], all_sites[bi], R_vec, 1.0,
                    site_forces[ai], site_forces[bi],
                    diag_erf_intra_qq, diag_erf_intra_qmu, diag_erf_intra_mumu);
            }
        }
    }

    // 4. Reciprocal-space sum
    Mat3 A_bohr = uc.direct() * occ::units::ANGSTROM_TO_BOHR;
    double volume_bohr = uc.volume() *
        std::pow(occ::units::ANGSTROM_TO_BOHR, 3);
    Mat3 B_bohr = 2.0 * M_PI * A_bohr.inverse().transpose();

    double four_pi_over_vol = 4.0 * M_PI / volume_bohr;
    double inv_4alpha2 = 1.0 / (4.0 * alpha_bohr * alpha_bohr);

    for (int hx = -kmax; hx <= kmax; ++hx) {
        for (int hy = -kmax; hy <= kmax; ++hy) {
            for (int hz = -kmax; hz <= kmax; ++hz) {
                if (hx == 0 && hy == 0 && hz == 0) continue;

                Vec3 G = B_bohr * Vec3(hx, hy, hz);
                double G2 = G.squaredNorm();
                double coeff = std::exp(-G2 * inv_4alpha2) / G2;

                // Charge structure factor: S_q(G) = Σ q_j exp(iG·r_j)
                // Dipole structure factor: S_μ(G) = Σ (μ_j·G) exp(iG·r_j)
                double Sq_re = 0.0, Sq_im = 0.0;
                double Smu_re = 0.0, Smu_im = 0.0;
                for (const auto& s : all_sites) {
                    double phase = G.dot(s.pos_bohr);
                    double cos_p = std::cos(phase);
                    double sin_p = std::sin(phase);
                    Sq_re += s.charge * cos_p;
                    Sq_im += s.charge * sin_p;
                    double mu_dot_G = s.dipole.dot(G);
                    Smu_re += mu_dot_G * cos_p;
                    Smu_im += mu_dot_G * sin_p;
                }

                // Combined reciprocal energy: (2π/V) * coeff * |S_q + i·S_μ|²
                // = (2π/V) * coeff * [|S_q|² - 2·Im(S_q*·S_μ) + |S_μ|²]
                // where Im(S_q*·S_μ) = Sq_re·Smu_im - Sq_im·Smu_re
                double qq_recip = Sq_re * Sq_re + Sq_im * Sq_im;
                double qmu_cross = -2.0 * (Sq_re * Smu_im - Sq_im * Smu_re);
                double mumu_recip = Smu_re * Smu_re + Smu_im * Smu_im;
                double prefactor = 0.5 * four_pi_over_vol * coeff;
                energy_ha += prefactor * (qq_recip + qmu_cross + mumu_recip);
                diag_recip_qq += prefactor * qq_recip;
                diag_recip_qmu += prefactor * qmu_cross;
                diag_recip_mumu += prefactor * mumu_recip;

                // Force on site k from combined structure factor:
                // F_k = (4π/V) f(G) G × [q_k(P sin_k - Q cos_k) + mk(P cos_k + Q sin_k)]
                // where P = Sq_re - Smu_im, Q = Sq_im + Smu_re, mk = μ_k·G
                double P = Sq_re - Smu_im;
                double Q = Sq_im + Smu_re;
                for (size_t k = 0; k < all_sites.size(); ++k) {
                    double q_k = all_sites[k].charge;
                    double mk = all_sites[k].dipole.dot(G);

                    double phase_k = G.dot(all_sites[k].pos_bohr);
                    double sin_k = std::sin(phase_k);
                    double cos_k = std::cos(phase_k);

                    double force_factor = q_k * (P * sin_k - Q * cos_k)
                                        + mk * (P * cos_k + Q * sin_k);

                    site_forces[k] += four_pi_over_vol * coeff * force_factor * G;
                }
            }
        }
    }

    // 4b. Verify Ewald identity: reciprocal self-contribution should equal
    //     (α/√π) Σ q² for qq and (2α³/(3√π)) Σ |μ|² for μμ
    {
        double recip_self_qq = 0.0, recip_self_mumu = 0.0;
        for (int hx = -kmax; hx <= kmax; ++hx) {
            for (int hy = -kmax; hy <= kmax; ++hy) {
                for (int hz = -kmax; hz <= kmax; ++hz) {
                    if (hx == 0 && hy == 0 && hz == 0) continue;
                    Vec3 G = B_bohr * Vec3(hx, hy, hz);
                    double G2 = G.squaredNorm();
                    double coeff = std::exp(-G2 * inv_4alpha2) / G2;
                    double prefactor = 0.5 * four_pi_over_vol * coeff;
                    for (const auto& s : all_sites) {
                        recip_self_qq += prefactor * s.charge * s.charge;
                        double muG = s.dipole.dot(G);
                        recip_self_mumu += prefactor * muG * muG;
                    }
                }
            }
        }
        double sum_q2 = 0.0, sum_mu2 = 0.0;
        for (const auto& s : all_sites) {
            sum_q2 += s.charge * s.charge;
            sum_mu2 += s.dipole.squaredNorm();
        }
        double alpha3_loc = alpha_bohr * alpha_bohr * alpha_bohr;
        double expected_self_qq = (alpha_bohr / std::sqrt(M_PI)) * sum_q2;
        double expected_self_mumu = (2.0 * alpha3_loc / (3.0 * std::sqrt(M_PI))) * sum_mu2;
        occ::log::info("  Ewald identity check (Hartree):");
        occ::log::info("    Recip self qq:  {:.6f}, Expected (α/√π)Σq²: {:.6f}, ratio: {:.6f}",
                       recip_self_qq, expected_self_qq, recip_self_qq / expected_self_qq);
        occ::log::info("    Recip self mumu: {:.6f}, Expected (2α³/(3√π))Σ|μ|²: {:.6f}, ratio: {:.6f}",
                       recip_self_mumu, expected_self_mumu,
                       expected_self_mumu > 1e-15 ? recip_self_mumu / expected_self_mumu : 0.0);
        occ::log::info("    Σq²={:.6f}, Σ|μ|²={:.6f}", sum_q2, sum_mu2);
        // Print first few site dipoles
        for (size_t i = 0; i < std::min(all_sites.size(), size_t(5)); ++i) {
            occ::log::info("    Site {} (mol {}): q={:.6f}, μ=({:.6f},{:.6f},{:.6f}), |μ|²={:.6f}",
                          i, all_sites[i].mol_index, all_sites[i].charge,
                          all_sites[i].dipole[0], all_sites[i].dipole[1], all_sites[i].dipole[2],
                          all_sites[i].dipole.squaredNorm());
        }

        // Cross-check: compute erf-screened + bare intra mumu for comparison
        double test_erf_intra_mumu = 0.0, test_bare_intra_mumu = 0.0;
        for (int m = 0; m < N; ++m) {
            const auto& indices = mol_site_indices[m];
            for (size_t ii = 0; ii < indices.size(); ++ii) {
                size_t ai = indices[ii];
                for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
                    size_t bi = indices[jj];
                    Vec3 R_vec = all_sites[bi].pos_bohr - all_sites[ai].pos_bohr;
                    double r = R_vec.norm();
                    if (r < 1e-10) continue;
                    double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r;
                    double ar = alpha_bohr * r;
                    double erf_v = std::erf(ar), exp_v = std::exp(-ar*ar);
                    double f1_v = two_alpha_over_sqrt_pi * exp_v / r2 - erf_v / r3;
                    double ap_v = -two_alpha_over_sqrt_pi * exp_v * (2.0*alpha_bohr*alpha_bohr/r2 + 3.0/r4) + 3.0*erf_v/r5;
                    const Vec3& da = all_sites[ai].dipole;
                    const Vec3& db = all_sites[bi].dipole;
                    double muA_R = da.dot(R_vec), muB_R = db.dot(R_vec);
                    test_erf_intra_mumu += da.dot(db) * f1_v + muA_R * muB_R * ap_v;
                    // Bare T-tensor: T_ab = -delta/r³ + 3 RaRb/r⁵
                    test_bare_intra_mumu += -da.dot(db)/r3 + 3.0*muA_R*muB_R/r5;
                }
            }
        }
        occ::log::info("    Intra mumu erf-screened: {:.6f} Ha = {:.4f} kJ/mol",
                       test_erf_intra_mumu, test_erf_intra_mumu * occ::units::AU_TO_KJ_PER_MOL);
        occ::log::info("    Intra mumu bare:         {:.6f} Ha = {:.4f} kJ/mol",
                       test_bare_intra_mumu, test_bare_intra_mumu * occ::units::AU_TO_KJ_PER_MOL);
    }

    // 5. Self correction
    double alpha3 = alpha_bohr * alpha_bohr * alpha_bohr;
    for (const auto& s : all_sites) {
        // Charge-charge: -(α/√π) * q²
        double self_qq = (alpha_bohr / std::sqrt(M_PI)) * s.charge * s.charge;
        energy_ha -= self_qq;
        diag_self_qq -= self_qq;
        // Dipole-dipole: -(2α³/(3√π)) * |μ|²
        double self_mumu = (2.0 * alpha3 / (3.0 * std::sqrt(M_PI))) *
            s.dipole.squaredNorm();
        energy_ha -= self_mumu;
        diag_self_mumu -= self_mumu;
    }

    // 6. Convert to kJ/mol and map to rigid-body forces/torques
    ewald_result.energy = energy_ha * occ::units::AU_TO_KJ_PER_MOL;

    double force_conv = occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;

    for (size_t k = 0; k < all_sites.size(); ++k) {
        int m = all_sites[k].mol_index;
        Vec3 f_kJ = site_forces[k] * force_conv;

        ewald_result.forces[m] += f_kJ;

        // Torque from lever arm (charge-charge has no multipole rotation torque)
        Vec3 lever = all_sites[k].pos_ang - molecules[m].position;
        Vec3 torque_lab = lever.cross(f_kJ);
        ewald_result.torques[m] -= torque_lab;
    }

    double au2kj = occ::units::AU_TO_KJ_PER_MOL;
    occ::log::info("Ewald correction (qq+qmu+mumu): {:.4f} kJ/mol", ewald_result.energy);
    occ::log::info("  Recip:  qq={:.4f}  qmu={:.4f}  mumu={:.4f}  total={:.4f} kJ/mol",
                   diag_recip_qq * au2kj, diag_recip_qmu * au2kj,
                   diag_recip_mumu * au2kj,
                   (diag_recip_qq + diag_recip_qmu + diag_recip_mumu) * au2kj);
    occ::log::info("  Self:   qq={:.4f}  mumu={:.4f}  total={:.4f} kJ/mol",
                   diag_self_qq * au2kj, diag_self_mumu * au2kj,
                   (diag_self_qq + diag_self_mumu) * au2kj);
    occ::log::info("  Erf-inter: qq={:.4f}  qmu={:.4f}  mumu={:.4f}  total={:.4f} kJ/mol",
                   diag_erf_inter_qq * au2kj, diag_erf_inter_qmu * au2kj,
                   diag_erf_inter_mumu * au2kj,
                   (diag_erf_inter_qq + diag_erf_inter_qmu + diag_erf_inter_mumu) * au2kj);
    occ::log::info("  Erf-intra: qq={:.4f}  qmu={:.4f}  mumu={:.4f}  total={:.4f} kJ/mol",
                   diag_erf_intra_qq * au2kj, diag_erf_intra_qmu * au2kj,
                   diag_erf_intra_mumu * au2kj,
                   (diag_erf_intra_qq + diag_erf_intra_qmu + diag_erf_intra_mumu) * au2kj);
    occ::log::info("  Net qq:   {:.4f} kJ/mol",
                   (diag_recip_qq + diag_self_qq + diag_erf_inter_qq + diag_erf_intra_qq) * au2kj);
    occ::log::info("  Net qmu:  {:.4f} kJ/mol",
                   (diag_recip_qmu + diag_erf_inter_qmu + diag_erf_intra_qmu) * au2kj);
    occ::log::info("  Net mumu: {:.4f} kJ/mol",
                   (diag_recip_mumu + diag_self_mumu + diag_erf_inter_mumu + diag_erf_intra_mumu) * au2kj);
    occ::log::info("  {} sites, {} neighbor pairs", all_sites.size(), m_neighbors.size());

    return ewald_result;
}

CrystalEnergy::EwaldBreakdown CrystalEnergy::charge_ewald_breakdown(
    const std::vector<MoleculeState>& molecules,
    double alpha, int kmax) const {

    // Gather all charge sites (rank 0 terms) for each molecule
    struct Site { Vec3 pos_bohr; double q; };
    std::vector<Site> sites;
    sites.reserve(num_sites());

    std::vector<MultipoleSource> multipoles = m_multipoles; // copy for const correctness
    for (size_t i = 0; i < molecules.size(); ++i) {
        multipoles[i].set_orientation(molecules[i].rotation_matrix(), molecules[i].position);
        const auto cart = multipoles[i].cartesian();
        for (const auto& s : cart.sites) {
            double q = (s.rank >= 0) ? s.cart.data[0] : 0.0;  // first coefficient is charge (a.u.)
            if (std::abs(q) < 1e-12) continue;
            sites.push_back({s.position * occ::units::ANGSTROM_TO_BOHR, q});
        }
    }

    EwaldBreakdown out;
    if (sites.empty()) return out;

    const auto& uc = m_crystal.unit_cell();
    Mat3 A_ang = uc.direct();
    Mat3 A = A_ang * occ::units::ANGSTROM_TO_BOHR;  // bohr
    double volume = uc.volume() * std::pow(occ::units::ANGSTROM_TO_BOHR, 3);  // bohr^3
    Mat3 B = 2.0 * M_PI * A.inverse().transpose();  // reciprocal lattice vectors (columns, 1/bohr)

    double alpha_bohr = alpha * occ::units::BOHR_TO_ANGSTROM;

    // Real-space sum over lattice vectors within cutoff (use existing neighbor cutoff)
    double rcut = m_cutoff_radius * occ::units::ANGSTROM_TO_BOHR;
    // naive double loop over translations near origin
    double min_len = std::min({uc.a(), uc.b(), uc.c()}) * occ::units::ANGSTROM_TO_BOHR;
    int tmax = static_cast<int>(std::ceil(rcut / min_len)) + 1;
    for (int nx = -tmax; nx <= tmax; ++nx) {
        for (int ny = -tmax; ny <= tmax; ++ny) {
            for (int nz = -tmax; nz <= tmax; ++nz) {
                Vec3 nvec = A * Vec3(nx, ny, nz);  // bohr
                bool self_image = (nx == 0 && ny == 0 && nz == 0);
                for (size_t i = 0; i < sites.size(); ++i) {
                    for (size_t j = self_image ? i + 1 : 0; j < sites.size(); ++j) {
                        Vec3 rvec = sites[j].pos_bohr + nvec - sites[i].pos_bohr;
                        double r = rvec.norm();
                        if (r < 1e-10 || r > rcut) continue;
                        double erfc_term = std::erfc(alpha_bohr * r) / r;
                        out.real_space += sites[i].q * sites[j].q * erfc_term;
                    }
                }
            }
        }
    }
    out.real_space *= 0.5;  // double-count correction

    // Reciprocal-space sum
    for (int hx = -kmax; hx <= kmax; ++hx) {
        for (int hy = -kmax; hy <= kmax; ++hy) {
            for (int hz = -kmax; hz <= kmax; ++hz) {
                if (hx == 0 && hy == 0 && hz == 0) continue;
                Vec3 h = B * Vec3(hx, hy, hz);  // 1/bohr
                double h2 = h.squaredNorm();
                double coeff = std::exp(-h2 / (4.0 * alpha_bohr * alpha_bohr)) / h2;

                // Structure factor
                std::complex<double> S(0.0, 0.0);
                for (const auto& s : sites) {
                    double phase = h.dot(s.pos_bohr);
                    S += std::complex<double>(std::cos(phase), std::sin(phase)) * s.q;
                }
                double S2 = std::norm(S);
                out.reciprocal += coeff * S2;
            }
        }
    }
    out.reciprocal *= (2.0 * M_PI) / volume;

    // Self term
    double qsum = 0.0;
    for (const auto& s : sites) {
        out.self -= (alpha_bohr / std::sqrt(M_PI)) * s.q * s.q;
        qsum += s.q;
    }

    // Net charge correction (background)
    if (std::abs(qsum) > 1e-10) {
        out.self += -M_PI * qsum * qsum / (alpha_bohr * alpha_bohr * volume);
    }

    // Convert to kJ/mol
    out.real_space *= occ::units::AU_TO_KJ_PER_MOL;
    out.reciprocal *= occ::units::AU_TO_KJ_PER_MOL;
    out.self *= occ::units::AU_TO_KJ_PER_MOL;

    return out;
}

} // namespace occ::mults
