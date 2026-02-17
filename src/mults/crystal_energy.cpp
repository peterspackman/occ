#include <occ/mults/crystal_energy.h>
#include <occ/mults/ewald_sum.h>
#include <occ/mults/cartesian_hessian.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <Eigen/Geometry>
#include <cmath>

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

    // ========================================================================
    // Full numerical Hessian via central differences of the total gradient.
    // This captures all contributions: pairwise electrostatics (all multipole
    // ranks), Ewald correction (qq+qmu+mumu), and short-range (Buckingham/LJ).
    //
    // TODO: Replace with analytical Hessian as compute_molecule_hessian_truncated
    // (in cartesian_hessian.h) is extended to cover all multipole ranks and
    // rotation terms. The analytical code exists but currently only handles
    // charge-charge and partial charge-dipole position terms.
    // ========================================================================
    const double h = 1e-5;

    // Helper to compute total gradient for a given set of states.
    // compute() returns forces (= -dE/dr) and torques (= dE/dtheta).
    // The gradient vector is [-force, +torque] for each molecule.
    auto compute_total_gradient = [&](const std::vector<MoleculeState>& states) -> Vec {
        auto r = compute(states);
        Vec grad(ndof);
        for (int i = 0; i < N; ++i) {
            grad.segment<3>(6 * i) = -r.forces[i];
            grad.segment<3>(6 * i + 3) = r.torques[i];
        }
        return grad;
    };

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

        Vec g_plus = compute_total_gradient(states_plus);
        Vec g_minus = compute_total_gradient(states_minus);
        result.hessian.col(dof) = (g_plus - g_minus) / (2.0 * h);
    }

    // Symmetrize Hessian (handles numerical noise)
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

std::vector<PairEnergyDebug> CrystalEnergy::debug_pair_energies(const std::vector<MoleculeState>& molecules) {
    const int N = static_cast<int>(molecules.size());
    std::vector<PairEnergyDebug> out;
    out.reserve(m_neighbors.size());

    // Update multipole orientations
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            molecules[i].rotation_matrix(),
            molecules[i].position);
    }

    // Pre-build Cartesian molecules
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
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
// Charge-Charge Ewald Correction (thin wrapper around standalone engine)
// ============================================================================

CrystalEnergy::EwaldCorrectionResult CrystalEnergy::compute_charge_ewald_correction(
    const std::vector<MoleculeState>& molecules,
    const std::vector<CartesianMolecule>& cart_mols) const {

    const int N = static_cast<int>(molecules.size());
    EwaldCorrectionResult ewald_result;
    ewald_result.forces.resize(N, Vec3::Zero());
    ewald_result.torques.resize(N, Vec3::Zero());

    // Gather sites and indices from CartesianMolecules
    auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
    auto mol_site_indices = build_mol_site_indices(cart_mols);

    if (ewald_sites.empty()) return ewald_result;

    // Select Ewald parameters
    EwaldParams params;
    if (m_ewald_eta > 0) {
        params.alpha = m_ewald_eta;
    } else {
        double x = std::sqrt(-std::log(m_ewald_accuracy));
        params.alpha = x / m_cutoff_radius;
    }

    if (m_ewald_kmax > 0) {
        params.kmax = m_ewald_kmax;
    } else {
        const auto& uc = m_crystal.unit_cell();
        double min_len = std::min({uc.a(), uc.b(), uc.c()});
        double G_max = 2.0 * params.alpha * std::sqrt(-std::log(m_ewald_accuracy));
        params.kmax = std::max(1, static_cast<int>(std::ceil(
            G_max * min_len / (2.0 * M_PI))));
    }
    params.include_dipole = m_ewald_dipole;

    // Call standalone Ewald engine
    auto raw = compute_ewald_correction(
        ewald_sites, m_crystal.unit_cell(), m_neighbors,
        mol_site_indices, m_cutoff_radius,
        m_use_com_elec_gate, m_elec_site_cutoff, params);

    ewald_result.energy = raw.energy;

    // Map per-site forces to rigid-body forces and torques
    for (size_t k = 0; k < ewald_sites.size(); ++k) {
        int m = ewald_sites[k].mol_index;
        const Vec3& f_kJ = raw.site_forces[k];

        ewald_result.forces[m] += f_kJ;

        // Torque from lever arm
        Vec3 lever = ewald_sites[k].position - molecules[m].position;
        Vec3 torque_lab = lever.cross(f_kJ);
        ewald_result.torques[m] -= torque_lab;
    }

    return ewald_result;
}

} // namespace occ::mults
