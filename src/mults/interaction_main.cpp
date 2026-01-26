/**
 * @file interaction_main.cpp
 * @brief Command-line program for computing multipole interactions, forces, and geometry optimizations
 *
 * This program provides a comprehensive interface for:
 * - Computing multipole-multipole interaction energies
 * - Evaluating short-range potentials (Lennard-Jones, Buckingham)
 * - Computing forces and gradients on all sites
 * - Performing geometry optimizations
 * - Systematic comparison with Orient reference calculations
 *
 * Input: JSON file with molecular system, multipoles, and calculation parameters
 * Output: Energies, forces, optimized geometries in human-readable or Orient-compatible format
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/esp.h>
#include <occ/mults/sfunctions.h>
#include <occ/mults/derivative_transform.h>
#include <occ/mults/short_range.h>
#include <occ/mults/rotation.h>
#include <occ/mults/rigid_body.h>
#include <occ/mults/rigid_body_dynamics.h>
#include <occ/mults/optimization_projection.h>

#include <LBFGSB.h>

using json = nlohmann::json;
using namespace occ;
using namespace occ::mults;
using namespace occ::dma;

// ==================== Data Structures ====================

/**
 * @brief Represents a molecule with position, orientation, and multipole moments
 *
 * This struct wraps RigidBodyState for use in static calculations.
 * For optimization, we use the RigidBodyState directly.
 */
struct Molecule {
    RigidBodyState state;  // Complete rigid body state

    Molecule(const std::string& n, const Vec3& pos, int max_rank)
        : state(n, -1, pos, 1.0, max_rank) {}

    // Convenient accessors for compatibility
    std::string name() const { return state.name; }
    Vec3& position() { return state.position; }
    const Vec3& position() const { return state.position; }

    Mult& multipole_body() { return state.multipole_body; }
    const Mult& multipole_body() const { return state.multipole_body; }

    // Apply rotation to transform body frame multipoles to lab frame
    void update_lab_frame_multipoles() {
        state.update_lab_multipoles();
    }

    // Get the multipoles to use for calculations (always lab frame)
    const Mult& get_multipoles() const {
        const_cast<RigidBodyState&>(state).update_lab_multipoles();
        return state.multipole_lab;
    }

    // Euler angle setters
    void set_euler_angles(double alpha, double beta, double gamma) {
        state.set_euler_angles(alpha, beta, gamma);
    }

    Vec3 get_euler_angles() {
        return state.get_euler_angles();
    }

    // Angle-axis interface
    Vec3 get_angle_axis() const {
        return state.get_angle_axis();
    }

    void set_angle_axis(const Vec3& p) {
        state.set_angle_axis(p);
    }

    Mat3 angle_axis_jacobian() const {
        return state.angle_axis_jacobian();
    }
};

/**
 * @brief Short-range pair interaction specification
 */
struct ShortRangePair {
    enum class Type { LennardJones, Buckingham };

    Type type;
    int site_a;
    int site_b;

    // LJ parameters
    LennardJonesParams lj_params;

    // Buckingham parameters
    BuckinghamParams buck_params;
};

/**
 * @brief Complete molecular system with calculation settings
 */
struct System {
    std::vector<Molecule> molecules;
    std::vector<ShortRangePair> short_range_pairs;
    int max_rank = 4;
    bool compute_forces = false;
    bool optimize = false;
    bool verbose = false;
    bool orient_style = false;

    // Optimization parameters
    int max_opt_iterations = 100;
    double convergence_threshold = 1e-5;  // Force convergence in au
    double step_size = 0.01;  // Initial step size in bohr

    // Finite difference gradient options
    bool use_finite_diff_gradients = false;  // Use finite diff instead of analytical
    bool compare_gradients = false;           // Compare analytical vs finite diff
    double finite_diff_delta = 1e-5;          // Step size for finite differences
};

/**
 * @brief Energy components
 */
struct EnergyComponents {
    double multipole = 0.0;
    double lennard_jones = 0.0;
    double buckingham = 0.0;

    double total() const {
        return multipole + lennard_jones + buckingham;
    }
};

/**
 * @brief Forces and torques on all molecules
 */
struct Forces {
    std::vector<Vec3> forces;             // Forces on each molecule (Cartesian)
    std::vector<Vec3> torques_euler;      // Torques w.r.t. Euler angles
    std::vector<Vec3> torques_space;      // Torques in space-fixed (lab) frame
    std::vector<Vec3> grad_angle_axis;    // Gradients w.r.t. angle-axis parameters

    explicit Forces(size_t n)
        : forces(n, Vec3::Zero()), torques_euler(n, Vec3::Zero()),
          torques_space(n, Vec3::Zero()), grad_angle_axis(n, Vec3::Zero()) {}

    double max_magnitude() const {
        double max_mag = 0.0;
        for (const auto& f : forces) {
            max_mag = std::max(max_mag, f.norm());
        }
        for (const auto& t : torques_euler) {
            max_mag = std::max(max_mag, t.norm());
        }
        for (const auto& g : grad_angle_axis) {
            max_mag = std::max(max_mag, g.norm());
        }
        return max_mag;
    }
};

// ==================== JSON Input Parsing ====================

/**
 * @brief Parse multipole from JSON
 */
Mult parse_multipole(const json& mult_json, int max_rank) {
    Mult mult(max_rank);
    mult.q.setZero();

    // Charge (Q00)
    if (mult_json.contains("charge")) {
        mult.Q00() = mult_json["charge"].get<double>();
    }

    // Dipole (Q10, Q11c, Q11s)
    // JSON format: [Q10, Q11c, Q11s] to match Orient convention
    if (mult_json.contains("dipole")) {
        auto dip = mult_json["dipole"];
        if (dip.is_array() && dip.size() == 3) {
            mult.Q10() = dip[0].get<double>();    // Q10 (z component)
            mult.Q11c() = dip[1].get<double>();   // Q11c (x component)
            mult.Q11s() = dip[2].get<double>();   // Q11s (y component)
        }
    }

    // Quadrupole (Q20, Q21c, Q21s, Q22c, Q22s)
    if (mult_json.contains("quadrupole") && max_rank >= 2) {
        auto quad = mult_json["quadrupole"];
        if (quad.is_array() && quad.size() == 5) {
            // Array format: [Q20, Q21c, Q21s, Q22c, Q22s]
            mult.Q20() = quad[0].get<double>();
            mult.Q21c() = quad[1].get<double>();
            mult.Q21s() = quad[2].get<double>();
            mult.Q22c() = quad[3].get<double>();
            mult.Q22s() = quad[4].get<double>();
        } else if (quad.is_object()) {
            // Object format: {"Q20": ..., "Q21c": ..., ...}
            if (quad.contains("Q20")) mult.Q20() = quad["Q20"].get<double>();
            if (quad.contains("Q21c")) mult.Q21c() = quad["Q21c"].get<double>();
            if (quad.contains("Q21s")) mult.Q21s() = quad["Q21s"].get<double>();
            if (quad.contains("Q22c")) mult.Q22c() = quad["Q22c"].get<double>();
            if (quad.contains("Q22s")) mult.Q22s() = quad["Q22s"].get<double>();
        }
    }

    // Octapole (Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s)
    if (mult_json.contains("octapole") && max_rank >= 3) {
        auto oct = mult_json["octapole"];
        if (oct.is_array() && oct.size() == 7) {
            // Array format: [Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s] (Orient convention)
            mult.Q30() = oct[0].get<double>();
            mult.Q31c() = oct[1].get<double>();
            mult.Q31s() = oct[2].get<double>();
            mult.Q32c() = oct[3].get<double>();
            mult.Q32s() = oct[4].get<double>();
            mult.Q33c() = oct[5].get<double>();
            mult.Q33s() = oct[6].get<double>();
        } else if (oct.is_object()) {
            // Object format: {"Q30": ..., "Q31c": ..., ...}
            if (oct.contains("Q30")) mult.Q30() = oct["Q30"].get<double>();
            if (oct.contains("Q31c")) mult.Q31c() = oct["Q31c"].get<double>();
            if (oct.contains("Q31s")) mult.Q31s() = oct["Q31s"].get<double>();
            if (oct.contains("Q32c")) mult.Q32c() = oct["Q32c"].get<double>();
            if (oct.contains("Q32s")) mult.Q32s() = oct["Q32s"].get<double>();
            if (oct.contains("Q33c")) mult.Q33c() = oct["Q33c"].get<double>();
            if (oct.contains("Q33s")) mult.Q33s() = oct["Q33s"].get<double>();
        }
    }

    // Hexadecapole (Q40, Q41c, Q41s, Q42c, Q42s, Q43c, Q43s, Q44c, Q44s)
    if (mult_json.contains("hexadecapole") && max_rank >= 4) {
        auto hex = mult_json["hexadecapole"];
        if (hex.is_array() && hex.size() == 9) {
            // Array format: [Q40, Q41c, Q41s, Q42c, Q42s, Q43c, Q43s, Q44c, Q44s] (Orient convention)
            mult.Q40() = hex[0].get<double>();
            mult.Q41c() = hex[1].get<double>();
            mult.Q41s() = hex[2].get<double>();
            mult.Q42c() = hex[3].get<double>();
            mult.Q42s() = hex[4].get<double>();
            mult.Q43c() = hex[5].get<double>();
            mult.Q43s() = hex[6].get<double>();
            mult.Q44c() = hex[7].get<double>();
            mult.Q44s() = hex[8].get<double>();
        } else if (hex.is_object()) {
            // Object format: {"Q40": ..., "Q41c": ..., ...}
            if (hex.contains("Q40")) mult.Q40() = hex["Q40"].get<double>();
            if (hex.contains("Q41c")) mult.Q41c() = hex["Q41c"].get<double>();
            if (hex.contains("Q41s")) mult.Q41s() = hex["Q41s"].get<double>();
            if (hex.contains("Q42c")) mult.Q42c() = hex["Q42c"].get<double>();
            if (hex.contains("Q42s")) mult.Q42s() = hex["Q42s"].get<double>();
            if (hex.contains("Q43c")) mult.Q43c() = hex["Q43c"].get<double>();
            if (hex.contains("Q43s")) mult.Q43s() = hex["Q43s"].get<double>();
            if (hex.contains("Q44c")) mult.Q44c() = hex["Q44c"].get<double>();
            if (hex.contains("Q44s")) mult.Q44s() = hex["Q44s"].get<double>();
        }
    }

    return mult;
}

/**
 * @brief Read complete system from JSON file
 */
System read_input(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filename);
    }

    json j;
    file >> j;

    System sys;

    // Get calculation settings
    if (j.contains("calculation")) {
        auto calc = j["calculation"];
        if (calc.contains("max_rank")) {
            sys.max_rank = calc["max_rank"].get<int>();
        }
        if (calc.contains("optimize")) {
            sys.optimize = calc["optimize"].get<bool>();
        }
        if (calc.contains("compute_forces")) {
            sys.compute_forces = calc["compute_forces"].get<bool>();
        }
        if (calc.contains("max_opt_iterations")) {
            sys.max_opt_iterations = calc["max_opt_iterations"].get<int>();
        }
        if (calc.contains("convergence_threshold")) {
            sys.convergence_threshold = calc["convergence_threshold"].get<double>();
        }
        if (calc.contains("step_size")) {
            sys.step_size = calc["step_size"].get<double>();
        }
        if (calc.contains("use_finite_diff_gradients")) {
            sys.use_finite_diff_gradients = calc["use_finite_diff_gradients"].get<bool>();
        }
        if (calc.contains("compare_gradients")) {
            sys.compare_gradients = calc["compare_gradients"].get<bool>();
        }
        if (calc.contains("finite_diff_delta")) {
            sys.finite_diff_delta = calc["finite_diff_delta"].get<double>();
        }
    }

    // Read molecules
    if (!j.contains("molecules")) {
        throw std::runtime_error("Input file must contain 'molecules' array");
    }

    for (const auto& mol_json : j["molecules"]) {
        std::string name = mol_json["name"].get<std::string>();

        Vec3 position;
        auto pos = mol_json["position"];
        position << pos[0].get<double>(),
                    pos[1].get<double>(),
                    pos[2].get<double>();

        Molecule mol(name, position, sys.max_rank);

        if (mol_json.contains("multipoles")) {
            mol.multipole_body() = parse_multipole(mol_json["multipoles"], sys.max_rank);
        }

        // Parse orientation if provided
        if (mol_json.contains("orientation")) {
            auto orient = mol_json["orientation"];
            double alpha = 0.0, beta = 0.0, gamma = 0.0;

            // Support Euler angles (ZYZ convention, in degrees by default)
            if (orient.contains("euler_angles")) {
                auto euler = orient["euler_angles"];
                bool use_radians = orient.value("radians", false);
                double deg_to_rad = use_radians ? 1.0 : M_PI / 180.0;

                alpha = euler[0].get<double>() * deg_to_rad;
                beta = euler[1].get<double>() * deg_to_rad;
                gamma = euler[2].get<double>() * deg_to_rad;
            }
            // Support individual angle specification
            else if (orient.contains("alpha") || orient.contains("beta") || orient.contains("gamma")) {
                bool use_radians = orient.value("radians", false);
                double deg_to_rad = use_radians ? 1.0 : M_PI / 180.0;

                alpha = orient.value("alpha", 0.0) * deg_to_rad;
                beta = orient.value("beta", 0.0) * deg_to_rad;
                gamma = orient.value("gamma", 0.0) * deg_to_rad;
            }

            mol.set_euler_angles(alpha, beta, gamma);
        }

        // Transform body frame to lab frame
        mol.update_lab_frame_multipoles();

        sys.molecules.push_back(mol);
    }

    // Read short-range pairs
    if (j.contains("short_range")) {
        auto sr = j["short_range"];
        std::string type_str = sr.value("type", "lennard_jones");

        if (sr.contains("pairs")) {
            for (const auto& pair_json : sr["pairs"]) {
                ShortRangePair pair;
                pair.site_a = pair_json["site_a"].get<int>();
                pair.site_b = pair_json["site_b"].get<int>();

                if (type_str == "lennard_jones") {
                    pair.type = ShortRangePair::Type::LennardJones;
                    pair.lj_params.epsilon = pair_json["epsilon"].get<double>();
                    pair.lj_params.sigma = pair_json["sigma"].get<double>();
                } else if (type_str == "buckingham") {
                    pair.type = ShortRangePair::Type::Buckingham;
                    pair.buck_params.A = pair_json["A"].get<double>();
                    pair.buck_params.B = pair_json["B"].get<double>();
                    pair.buck_params.C = pair_json["C"].get<double>();
                }

                sys.short_range_pairs.push_back(pair);
            }
        }
    }

    return sys;
}

// ==================== Energy Calculations ====================

/**
 * @brief Compute multipole-multipole interaction energy for all pairs
 */
double compute_multipole_energy(const System& sys) {
    MultipoleESP esp(sys.max_rank);
    double total_energy = 0.0;

    for (size_t i = 0; i < sys.molecules.size(); ++i) {
        for (size_t j = i + 1; j < sys.molecules.size(); ++j) {
            double energy = esp.compute_interaction_energy(
                sys.molecules[i].get_multipoles(), sys.molecules[i].position(),
                sys.molecules[j].get_multipoles(), sys.molecules[j].position()
            );
            total_energy += energy;

            if (sys.verbose) {
                std::cout << "  Multipole interaction " << sys.molecules[i].name()
                         << " - " << sys.molecules[j].name()
                         << ": " << std::setprecision(10) << energy << " au\n";
            }
        }
    }

    return total_energy;
}

/**
 * @brief Compute short-range interaction energies for all pairs
 */
EnergyComponents compute_short_range_energies(const System& sys) {
    EnergyComponents energies;

    for (const auto& pair : sys.short_range_pairs) {
        if (pair.site_a >= sys.molecules.size() || pair.site_b >= sys.molecules.size()) {
            std::cerr << "Warning: Invalid site indices in short-range pair\n";
            continue;
        }

        Vec3 r_vec = sys.molecules[pair.site_b].position() - sys.molecules[pair.site_a].position();
        double r = r_vec.norm();

        if (pair.type == ShortRangePair::Type::LennardJones) {
            double energy = ShortRangeInteraction::lennard_jones_energy(r, pair.lj_params);
            energies.lennard_jones += energy;

            if (sys.verbose) {
                std::cout << "  LJ interaction " << sys.molecules[pair.site_a].name()
                         << " - " << sys.molecules[pair.site_b].name()
                         << ": " << std::setprecision(10) << energy << " au\n";
            }
        } else if (pair.type == ShortRangePair::Type::Buckingham) {
            double energy = ShortRangeInteraction::buckingham_energy(r, pair.buck_params);
            energies.buckingham += energy;

            if (sys.verbose) {
                std::cout << "  Buckingham interaction " << sys.molecules[pair.site_a].name()
                         << " - " << sys.molecules[pair.site_b].name()
                         << ": " << std::setprecision(10) << energy << " au\n";
            }
        }
    }

    return energies;
}

/**
 * @brief Compute total energy of the system
 */
EnergyComponents compute_energy(const System& sys) {
    if (sys.verbose) {
        std::cout << "\nComputing energy components...\n";
    }

    EnergyComponents energies;
    energies.multipole = compute_multipole_energy(sys);

    auto sr_energies = compute_short_range_energies(sys);
    energies.lennard_jones = sr_energies.lennard_jones;
    energies.buckingham = sr_energies.buckingham;

    return energies;
}

// ==================== Force Calculations ====================

/**
 * @brief Compute forces and torques using analytical derivatives
 *
 * Uses TorqueCalculation to compute exact forces and torques from
 * multipole-multipole interactions.
 */
Forces compute_forces_analytical(const System& sys) {
    Forces result(sys.molecules.size());
    MultipoleESP esp(sys.max_rank);

    // Compute pairwise forces and torques
    for (size_t i = 0; i < sys.molecules.size(); ++i) {
        for (size_t j = i + 1; j < sys.molecules.size(); ++j) {
            // Get Euler angles for torque calculation
            Vec3 euler_i = const_cast<Molecule&>(sys.molecules[i]).get_euler_angles();
            Vec3 euler_j = const_cast<Molecule&>(sys.molecules[j]).get_euler_angles();

            // Compute torque on molecule i
            TorqueResult torque_i = TorqueCalculation::compute_torque_analytical(
                sys.molecules[i].multipole_body(), sys.molecules[i].position(), euler_i,
                sys.molecules[j].multipole_body(), sys.molecules[j].position(), euler_j,
                1  // molecule 1
            );

            // Compute torque on molecule j
            TorqueResult torque_j = TorqueCalculation::compute_torque_analytical(
                sys.molecules[i].multipole_body(), sys.molecules[i].position(), euler_i,
                sys.molecules[j].multipole_body(), sys.molecules[j].position(), euler_j,
                2  // molecule 2
            );

            // Accumulate forces (Newton's 3rd law)
            result.forces[i] += torque_i.force;
            result.forces[j] += torque_j.force;

            // Accumulate torques (Euler angles and space frame)
            result.torques_euler[i] += torque_i.torque_euler;
            result.torques_euler[j] += torque_j.torque_euler;
            result.torques_space[i] += torque_i.torque_space;
            result.torques_space[j] += torque_j.torque_space;

            // Accumulate angle-axis gradients
            result.grad_angle_axis[i] += torque_i.grad_angle_axis;
            result.grad_angle_axis[j] += torque_j.grad_angle_axis;
        }
    }

    // Add short-range contributions
    for (const auto& pair : sys.short_range_pairs) {
        if (pair.site_a >= sys.molecules.size() || pair.site_b >= sys.molecules.size()) {
            continue;
        }

        Vec3 r_vec = sys.molecules[pair.site_b].position() - sys.molecules[pair.site_a].position();
        double r = r_vec.norm();

        double dE_dr = 0.0;

        if (pair.type == ShortRangePair::Type::LennardJones) {
            dE_dr = ShortRangeInteraction::lennard_jones_derivative(r, pair.lj_params);
        } else if (pair.type == ShortRangePair::Type::Buckingham) {
            dE_dr = ShortRangeInteraction::buckingham_derivative(r, pair.buck_params);
        }

        // Force = -dE/dr * r_hat
        Vec3 force = ShortRangeInteraction::derivative_to_force(dE_dr, r_vec);

        result.forces[pair.site_a] += force;
        result.forces[pair.site_b] -= force;
    }

    return result;
}

/**
 * @brief Compute forces using finite differences (for validation)
 */
Forces compute_forces_finite_diff(const System& sys, double delta = 1e-5) {
    Forces forces(sys.molecules.size());

    System sys_plus = sys;
    System sys_minus = sys;

    // Position derivatives (forces)
    for (size_t i = 0; i < sys.molecules.size(); ++i) {
        for (int coord = 0; coord < 3; ++coord) {
            // Forward step
            sys_plus.molecules[i].position()[coord] = sys.molecules[i].position()[coord] + delta;
            sys_plus.molecules[i].update_lab_frame_multipoles();
            double E_plus = compute_energy(sys_plus).total();

            // Backward step
            sys_minus.molecules[i].position()[coord] = sys.molecules[i].position()[coord] - delta;
            sys_minus.molecules[i].update_lab_frame_multipoles();
            double E_minus = compute_energy(sys_minus).total();

            // Central difference: F = -dE/dx
            forces.forces[i][coord] = -(E_plus - E_minus) / (2.0 * delta);

            // Reset
            sys_plus.molecules[i].position()[coord] = sys.molecules[i].position()[coord];
            sys_minus.molecules[i].position()[coord] = sys.molecules[i].position()[coord];
            sys_plus.molecules[i].update_lab_frame_multipoles();
            sys_minus.molecules[i].update_lab_frame_multipoles();
        }
    }

    // Euler angle derivatives (torques)
    for (size_t i = 0; i < sys.molecules.size(); ++i) {
        Vec3 euler = const_cast<Molecule&>(sys.molecules[i]).get_euler_angles();

        for (int coord = 0; coord < 3; ++coord) {
            // Forward step
            Vec3 euler_plus = euler;
            euler_plus[coord] += delta;
            sys_plus.molecules[i].set_euler_angles(euler_plus[0], euler_plus[1], euler_plus[2]);
            sys_plus.molecules[i].update_lab_frame_multipoles();
            double E_plus = compute_energy(sys_plus).total();

            // Backward step
            Vec3 euler_minus = euler;
            euler_minus[coord] -= delta;
            sys_minus.molecules[i].set_euler_angles(euler_minus[0], euler_minus[1], euler_minus[2]);
            sys_minus.molecules[i].update_lab_frame_multipoles();
            double E_minus = compute_energy(sys_minus).total();

            // Central difference: tau_euler = -dE/d(euler)
            forces.torques_euler[i][coord] = -(E_plus - E_minus) / (2.0 * delta);

            // Reset
            sys_plus.molecules[i].set_euler_angles(euler[0], euler[1], euler[2]);
            sys_minus.molecules[i].set_euler_angles(euler[0], euler[1], euler[2]);
            sys_plus.molecules[i].update_lab_frame_multipoles();
            sys_minus.molecules[i].update_lab_frame_multipoles();
        }
    }

    return forces;
}

/**
 * @brief Compute forces and torques on all molecules
 */
Forces compute_forces(const System& sys) {
    if (sys.verbose) {
        std::cout << "\nComputing forces and torques...\n";
    }

    // Use analytical derivatives (much faster than finite differences)
    return compute_forces_analytical(sys);
}

// ==================== Geometry Optimization ====================

/**
 * @brief Pack molecule positions into a flat vector for LBFGS (position-only)
 */
Vec pack_positions(const System& sys) {
    int ndof = 3 * sys.molecules.size();
    Vec x(ndof);
    for (size_t i = 0; i < sys.molecules.size(); i++) {
        x[3*i + 0] = sys.molecules[i].position()[0];
        x[3*i + 1] = sys.molecules[i].position()[1];
        x[3*i + 2] = sys.molecules[i].position()[2];
    }
    return x;
}

/**
 * @brief Unpack flat vector back into molecule positions (position-only)
 */
void unpack_positions(const Vec& x, System& sys) {
    for (size_t i = 0; i < sys.molecules.size(); i++) {
        sys.molecules[i].position()[0] = x[3*i + 0];
        sys.molecules[i].position()[1] = x[3*i + 1];
        sys.molecules[i].position()[2] = x[3*i + 2];
    }
    // Update lab-frame multipoles after moving
    for (auto& mol : sys.molecules) {
        mol.update_lab_frame_multipoles();
    }
}

/**
 * @brief Pack all DOF (positions + orientations) into flat vector for LBFGS
 *
 * Packing format:
 *   [x1, y1, z1, px1, py1, pz1,  // molecule 1: position + angle-axis
 *    x2, y2, z2, px2, py2, pz2,  // molecule 2: position + angle-axis
 *    ...]
 */
Vec pack_all_dof(const System& sys) {
    size_t n_molecules = sys.molecules.size();
    Vec x(6 * n_molecules);

    for (size_t i = 0; i < n_molecules; i++) {
        // Position (3 DOF)
        x.segment<3>(6*i) = sys.molecules[i].position();

        // Orientation as angle-axis (3 DOF)
        Vec3 p = sys.molecules[i].get_angle_axis();
        x.segment<3>(6*i + 3) = p;
    }

    return x;
}

/**
 * @brief Unpack flat vector back into system (full 6N DOF)
 */
void unpack_all_dof(const Vec& x, System& sys) {
    size_t n_molecules = sys.molecules.size();

    for (size_t i = 0; i < n_molecules; i++) {
        // Update position
        sys.molecules[i].position() = x.segment<3>(6*i);

        // Update orientation from angle-axis
        Vec3 p = x.segment<3>(6*i + 3);
        sys.molecules[i].set_angle_axis(p);

        // Update lab-frame multipoles
        sys.molecules[i].update_lab_frame_multipoles();
    }
}

/**
 * @brief Convert forces to gradient (gradient = -force for minimization, position-only)
 */
Vec gradient_from_forces(const Forces& forces) {
    Vec grad(forces.forces.size() * 3);
    for (size_t i = 0; i < forces.forces.size(); i++) {
        grad[3*i + 0] = -forces.forces[i][0];
        grad[3*i + 1] = -forces.forces[i][1];
        grad[3*i + 2] = -forces.forces[i][2];
    }
    return grad;
}

/**
 * @brief Compute full gradient for 6N-DOF optimization (analytical method)
 *
 * This computes gradients with respect to both positions and angle-axis parameters
 * using analytical derivatives throughout.
 *
 * For each molecule:
 *   - Position gradient: grad_pos = -force (from analytical force calculation)
 *   - Orientation gradient: grad_aa = analytical gradient w.r.t. angle-axis
 */
Vec compute_full_gradient(const System& sys) {
    size_t n_molecules = sys.molecules.size();
    Vec grad(6 * n_molecules);

    // Compute analytical forces and angle-axis gradients
    Forces forces = compute_forces(sys);

    for (size_t i = 0; i < n_molecules; i++) {
        // Position gradient: -force (minimization convention)
        grad.segment<3>(6*i) = -forces.forces[i];

        // Orientation gradient: analytical angle-axis gradient
        grad.segment<3>(6*i + 3) = forces.grad_angle_axis[i];
    }

    return grad;
}

/**
 * @brief Compute full gradient for 6N-DOF optimization using finite differences
 *
 * This computes gradients by numerically differentiating the energy with respect
 * to all degrees of freedom (positions and angle-axis parameters).
 * Uses central differences for accuracy.
 *
 * @param sys System configuration
 * @param delta Step size for finite differences (default: 1e-5)
 * @return Gradient vector with respect to [x, y, z, px, py, pz] for each molecule
 */
Vec compute_full_gradient_finite_diff(const System& sys, double delta = 1e-5) {
    size_t n_molecules = sys.molecules.size();
    Vec grad(6 * n_molecules);

    // Pack current configuration
    Vec x = pack_all_dof(sys);

    // Compute gradient for each DOF using central differences
    for (size_t i = 0; i < n_molecules; i++) {
        System sys_plus = sys;
        System sys_minus = sys;

        // Position derivatives (3 DOF per molecule)
        for (int coord = 0; coord < 3; coord++) {
            // Forward step
            sys_plus.molecules[i].position()[coord] = sys.molecules[i].position()[coord] + delta;
            sys_plus.molecules[i].update_lab_frame_multipoles();
            double E_plus = compute_energy(sys_plus).total();

            // Backward step
            sys_minus.molecules[i].position()[coord] = sys.molecules[i].position()[coord] - delta;
            sys_minus.molecules[i].update_lab_frame_multipoles();
            double E_minus = compute_energy(sys_minus).total();

            // Central difference: dE/dx
            grad[6*i + coord] = (E_plus - E_minus) / (2.0 * delta);

            // Reset
            sys_plus.molecules[i].position()[coord] = sys.molecules[i].position()[coord];
            sys_minus.molecules[i].position()[coord] = sys.molecules[i].position()[coord];
            sys_plus.molecules[i].update_lab_frame_multipoles();
            sys_minus.molecules[i].update_lab_frame_multipoles();
        }

        // Angle-axis derivatives (3 DOF per molecule)
        Vec3 p_current = sys.molecules[i].get_angle_axis();

        for (int coord = 0; coord < 3; coord++) {
            // Forward step
            Vec3 p_plus = p_current;
            p_plus[coord] += delta;
            sys_plus.molecules[i].set_angle_axis(p_plus);
            sys_plus.molecules[i].update_lab_frame_multipoles();
            double E_plus = compute_energy(sys_plus).total();

            // Backward step
            Vec3 p_minus = p_current;
            p_minus[coord] -= delta;
            sys_minus.molecules[i].set_angle_axis(p_minus);
            sys_minus.molecules[i].update_lab_frame_multipoles();
            double E_minus = compute_energy(sys_minus).total();

            // Central difference: dE/dp
            grad[6*i + 3 + coord] = (E_plus - E_minus) / (2.0 * delta);

            // Reset
            sys_plus.molecules[i].set_angle_axis(p_current);
            sys_minus.molecules[i].set_angle_axis(p_current);
            sys_plus.molecules[i].update_lab_frame_multipoles();
            sys_minus.molecules[i].update_lab_frame_multipoles();
        }
    }

    return grad;
}

/**
 * @brief Compare analytical and finite difference gradients
 *
 * Computes both analytical and finite difference gradients and prints
 * a detailed comparison showing component-wise differences.
 *
 * For constrained optimization (molecule 0 fixed), we compute finite differences
 * only for the free DOFs (molecule 1).
 */
void compare_gradients(const System& sys) {
    std::cout << "\n======================================================\n";
    std::cout << "Gradient Comparison: Analytical vs Finite Difference\n";
    std::cout << "======================================================\n";
    std::cout << "Finite difference step size: " << sys.finite_diff_delta << "\n";
    std::cout << "Note: Computing FD gradients for molecule 1 only (molecule 0 fixed)\n\n";

    // Compute analytical gradient (full 12-DOF space)
    Vec grad_analytical = compute_full_gradient(sys);

    // For constrained optimization, compute finite differences only for molecule 1's DOFs
    // This matches what the optimizer actually sees
    size_t n_molecules = sys.molecules.size();
    Vec grad_finite_diff = Vec::Zero(6 * n_molecules);

    if (n_molecules == 2) {
        double delta = sys.finite_diff_delta;
        System sys_plus = sys;
        System sys_minus = sys;

        // Pack current configuration
        Vec x_base = pack_all_dof(sys);

        // Compute finite differences for molecule 1's position (DOFs 6,7,8)
        for (int coord = 0; coord < 3; coord++) {
            // Forward step
            sys_plus.molecules[1].position()[coord] = sys.molecules[1].position()[coord] + delta;
            sys_plus.molecules[1].update_lab_frame_multipoles();
            double E_plus = compute_energy(sys_plus).total();

            // Backward step
            sys_minus.molecules[1].position()[coord] = sys.molecules[1].position()[coord] - delta;
            sys_minus.molecules[1].update_lab_frame_multipoles();
            double E_minus = compute_energy(sys_minus).total();

            // Central difference: dE/dx
            grad_finite_diff[6 + coord] = (E_plus - E_minus) / (2.0 * delta);

            // Reset
            sys_plus.molecules[1].position()[coord] = sys.molecules[1].position()[coord];
            sys_minus.molecules[1].position()[coord] = sys.molecules[1].position()[coord];
            sys_plus.molecules[1].update_lab_frame_multipoles();
            sys_minus.molecules[1].update_lab_frame_multipoles();
        }

        // Compute finite differences for molecule 1's orientation (DOFs 9,10,11)
        Vec3 p_current = sys.molecules[1].get_angle_axis();

        for (int coord = 0; coord < 3; coord++) {
            // Forward step
            Vec3 p_plus = p_current;
            p_plus[coord] += delta;
            sys_plus.molecules[1].set_angle_axis(p_plus);
            sys_plus.molecules[1].update_lab_frame_multipoles();
            double E_plus = compute_energy(sys_plus).total();

            // Backward step
            Vec3 p_minus = p_current;
            p_minus[coord] -= delta;
            sys_minus.molecules[1].set_angle_axis(p_minus);
            sys_minus.molecules[1].update_lab_frame_multipoles();
            double E_minus = compute_energy(sys_minus).total();

            // Central difference: dE/dp
            grad_finite_diff[9 + coord] = (E_plus - E_minus) / (2.0 * delta);

            // Reset
            sys_plus.molecules[1].set_angle_axis(p_current);
            sys_minus.molecules[1].set_angle_axis(p_current);
            sys_plus.molecules[1].update_lab_frame_multipoles();
            sys_minus.molecules[1].update_lab_frame_multipoles();
        }
    } else {
        // For N>2, fall back to full finite difference computation
        grad_finite_diff = compute_full_gradient_finite_diff(sys, sys.finite_diff_delta);
    }

    // Compute differences in full space
    Vec diff = grad_analytical - grad_finite_diff;
    double max_diff = diff.cwiseAbs().maxCoeff();
    double grad_norm = grad_analytical.norm();
    double rel_error = (grad_norm > 1e-12) ? (max_diff / grad_norm) : max_diff;

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Summary (Full 12-DOF space):\n";
    std::cout << "  Analytical gradient norm:     " << grad_analytical.norm() << "\n";
    std::cout << "  Finite diff gradient norm:    " << grad_finite_diff.norm() << "\n";
    std::cout << "  Max absolute difference:      " << max_diff << "\n";
    std::cout << "  Relative error (max/norm):    " << rel_error << "\n\n";

    // ALSO compare in filtered space (what optimizer actually uses)
    if (n_molecules == 2) {
        // Simple DOF filtering: Extract molecule 1's gradient (DOFs 6-11)
        Vec grad_anal_filtered = grad_analytical.segment<6>(6);
        Vec grad_fd_filtered = grad_finite_diff.segment<6>(6);
        Vec diff_filtered = grad_anal_filtered - grad_fd_filtered;

        double max_diff_filtered = diff_filtered.cwiseAbs().maxCoeff();
        double grad_norm_filtered = grad_anal_filtered.norm();
        double rel_error_filtered = (grad_norm_filtered > 1e-12) ? (max_diff_filtered / grad_norm_filtered) : max_diff_filtered;

        std::cout << "Summary (Filtered 6-DOF space - what optimizer uses):\n";
        std::cout << "  Analytical gradient norm:     " << grad_anal_filtered.norm() << "\n";
        std::cout << "  Finite diff gradient norm:    " << grad_fd_filtered.norm() << "\n";
        std::cout << "  Max absolute difference:      " << max_diff_filtered << "\n";
        std::cout << "  Relative error (max/norm):    " << rel_error_filtered << "\n\n";

        std::cout << "Filtered gradient components (molecule 1 only):\n";
        std::cout << std::setw(16) << "DOF" << " "
                  << std::setw(16) << "Analytical" << " "
                  << std::setw(16) << "Finite Diff" << " "
                  << std::setw(16) << "Difference\n";
        std::cout << std::string(68, '-') << "\n";
        const char* dof_names[] = {"x", "y", "z", "px", "py", "pz"};
        for (int i = 0; i < 6; i++) {
            std::cout << std::setw(16) << dof_names[i] << " "
                      << std::setw(16) << grad_anal_filtered[i] << " "
                      << std::setw(16) << grad_fd_filtered[i] << " "
                      << std::setw(16) << diff_filtered[i] << "\n";
        }
        std::cout << std::string(68, '-') << "\n\n";
    }

    // Print component-wise comparison
    const char* coord_names[] = {"x", "y", "z", "px", "py", "pz"};

    std::cout << "Component-wise comparison:\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(8) << "Mol" << " "
              << std::setw(4) << "DOF" << " "
              << std::setw(16) << "Analytical" << " "
              << std::setw(16) << "Finite Diff" << " "
              << std::setw(16) << "Difference" << " "
              << std::setw(16) << "Rel Error\n";
    std::cout << std::string(100, '-') << "\n";

    for (size_t i = 0; i < n_molecules; i++) {
        for (int j = 0; j < 6; j++) {
            int idx = 6*i + j;
            double anal = grad_analytical[idx];
            double fd = grad_finite_diff[idx];
            double d = diff[idx];
            double rel = (std::abs(anal) > 1e-12) ? (std::abs(d) / std::abs(anal)) : std::abs(d);

            std::cout << std::setw(8) << i << " "
                      << std::setw(4) << coord_names[j] << " "
                      << std::setw(16) << anal << " "
                      << std::setw(16) << fd << " "
                      << std::setw(16) << d << " "
                      << std::setw(16) << rel << "\n";
        }
    }

    std::cout << std::string(100, '-') << "\n";

    // Warn if discrepancy is large
    if (rel_error > 1e-4) {
        std::cout << "\n*** WARNING: Large gradient discrepancy detected! ***\n";
        std::cout << "    This may indicate an error in analytical gradients.\n";
        std::cout << "    Relative error: " << rel_error << "\n\n";
    } else if (rel_error > 1e-6) {
        std::cout << "\n*** NOTICE: Moderate gradient discrepancy ***\n";
        std::cout << "    Analytical and finite diff gradients differ slightly.\n";
        std::cout << "    This may be acceptable depending on finite diff step size.\n\n";
    } else {
        std::cout << "\n*** GOOD: Gradients agree well ***\n";
        std::cout << "    Analytical gradients appear correct.\n\n";
    }

    std::cout << "======================================================\n\n";
}

/**
 * @brief Perform LBFGS-B optimization (position-only, for compatibility)
 */
System optimize_geometry_positions_only(System sys) {
    using LBFGSpp::LBFGSBParam;
    using LBFGSpp::LBFGSBSolver;

    std::cout << "\n========================================\n";
    std::cout << "Starting LBFGS-B Position Optimization\n";
    std::cout << "========================================\n";
    std::cout << "Degrees of freedom: " << 3 * sys.molecules.size() << " (position only)\n";
    std::cout << "Max iterations: " << sys.max_opt_iterations << "\n";
    std::cout << "Convergence threshold: " << sys.convergence_threshold << " au\n\n";

    // LBFGS parameters
    LBFGSBParam<double> param;
    param.epsilon = sys.convergence_threshold;
    param.max_iterations = sys.max_opt_iterations;
    param.max_linesearch = 40;

    LBFGSBSolver<double> solver(param);

    // Count degrees of freedom (3 per molecule)
    int ndof = 3 * sys.molecules.size();

    // Set up bounds (optional - use generous bounds for now)
    Vec lb = Vec::Constant(ndof, -1000.0);  // Lower bound (bohr)
    Vec ub = Vec::Constant(ndof, 1000.0);   // Upper bound (bohr)

    // Pack initial positions
    Vec x = pack_positions(sys);

    // Iteration counter for progress reporting
    int iter_count = 0;

    // Define objective function: returns energy and computes gradient
    auto objective = [&](const Vec& x_vec, Vec& grad) -> double {
        iter_count++;

        // Update system with new positions
        System sys_trial = sys;
        unpack_positions(x_vec, sys_trial);

        // Compute energy
        EnergyComponents energies = compute_energy(sys_trial);
        double energy = energies.total();

        // Compute forces and convert to gradient
        Forces forces = compute_forces(sys_trial);
        grad = gradient_from_forces(forces);

        double max_grad = grad.array().abs().maxCoeff();

        // Print progress every iteration
        std::cout << "Iteration " << std::setw(4) << iter_count
                  << "  Energy: " << std::setw(16) << std::setprecision(10) << energy
                  << " au  Max Gradient: " << std::setw(12) << std::setprecision(6)
                  << max_grad << " au\n";

        return energy;
    };

    // Run optimization
    double final_energy;
    int status = solver.minimize(objective, x, final_energy, lb, ub);

    // Unpack optimized positions
    unpack_positions(x, sys);

    std::cout << "\n";
    if (status < 0) {
        std::cout << "Optimization stopped with status: " << status << "\n";
    } else {
        std::cout << "Optimization converged!\n";
    }
    std::cout << "Final energy: " << std::setprecision(12) << final_energy << " au\n";
    std::cout << "Total iterations: " << iter_count << "\n";
    std::cout << "========================================\n\n";

    return sys;
}

/**
 * @brief Perform LBFGS-B optimization with full 6N-DOF (position + orientation)
 *
 * ORIENT-STYLE DOF FILTERING APPROACH:
 * ===================================
 * This implementation uses a simple and robust DOF filtering strategy inspired by Orient:
 *
 * 1. Fix molecule 0 completely (6 DOF frozen at initial values)
 * 2. Optimize only molecule 1's 6 DOF (position + orientation)
 * 3. Extract/insert the 6 free DOFs directly - no complex projection matrices
 *
 * Why this approach?
 * - Simpler than RigidBodyProjection (which had subtle bugs)
 * - More efficient (no matrix inversions or projections per iteration)
 * - Matches Orient's proven methodology for dimer optimization
 * - Direct angle-axis gradients are more accurate than Euler-based approaches
 *
 * The optimizer sees only 6 DOF, but we compute gradients in the full 12-DOF space
 * and simply extract the molecule 1 components.
 */
System optimize_geometry(System sys) {
    using LBFGSpp::LBFGSBParam;
    using LBFGSpp::LBFGSBSolver;

    size_t n_molecules = sys.molecules.size();
    size_t full_dof = 6 * n_molecules;

    std::cout << "\n========================================\n";
    std::cout << "Starting L-BFGS Orient-Style Optimization\n";
    std::cout << "========================================\n";
    std::cout << "Number of molecules: " << n_molecules << "\n";
    std::cout << "Full degrees of freedom: " << full_dof
              << " (3 position + 3 orientation per molecule)\n";

    // For now, only support 2 molecules (Orient's typical case)
    if (n_molecules != 2) {
        std::cerr << "Error: Orient-style optimization currently only supports 2 molecules\n";
        return sys;
    }

    std::cout << "Using Orient-style DOF filtering: 12 DOF → 6 DOF\n";
    std::cout << "  (Molecule 0 fixed, molecule 1 free)\n";
    std::cout << "Max iterations: " << sys.max_opt_iterations << "\n";
    std::cout << "Convergence threshold: " << sys.convergence_threshold << " au\n\n";

    // Save the base configuration (molecule 0 will stay fixed at this)
    Vec x_base = pack_all_dof(sys);

    // Setup LBFGS parameters
    LBFGSBParam<double> param;
    param.epsilon = sys.convergence_threshold;
    param.max_iterations = sys.max_opt_iterations;
    param.max_linesearch = 40;

    LBFGSBSolver<double> solver(param);

    // Only 6 DOF for molecule 1 (position + orientation)
    size_t free_dof = 6;
    Vec lb(free_dof), ub(free_dof);

    // Position bounds (generous)
    lb.segment<3>(0).setConstant(-1000.0);
    ub.segment<3>(0).setConstant(1000.0);

    // Angle-axis bounds: |p| ≤ π
    lb.segment<3>(3).setConstant(-M_PI);
    ub.segment<3>(3).setConstant(M_PI);

    // Perform gradient comparison on initial configuration if requested
    if (sys.compare_gradients) {
        std::cout << "\nPerforming initial gradient comparison...\n";
        compare_gradients(sys);
    }

    // Define objective function - operates directly on molecule 1's 6 DOFs
    int iter_count = 0;
    auto objective = [&](const Vec& x_free, Vec& grad_free) -> double {
        // Reconstruct full coordinates:
        // Molecule 0: Keep at base (DOFs 0-5)
        // Molecule 1: Use optimizer's values (DOFs 6-11)
        Vec x_full(12);
        x_full.segment<6>(0) = x_base.segment<6>(0);  // Molecule 0 fixed
        x_full.segment<6>(6) = x_base.segment<6>(6) + x_free;  // Molecule 1 relative to base

        // Unpack into trial system
        System sys_trial = sys;
        unpack_all_dof(x_full, sys_trial);

        // Compute energy
        EnergyComponents energies = compute_energy(sys_trial);
        double E = energies.total();

        // Compute full gradient (analytical or finite difference)
        Vec grad_full;
        if (sys.use_finite_diff_gradients) {
            grad_full = compute_full_gradient_finite_diff(sys_trial, sys.finite_diff_delta);

            // Optionally compare with analytical on first iteration
            if (iter_count == 0 && sys.verbose) {
                std::cout << "\n=== Using Finite Difference Gradients ===\n";
                std::cout << "Step size: " << sys.finite_diff_delta << "\n";
                std::cout << "Gradient norm: " << grad_full.norm() << "\n";
                std::cout << "=========================================\n\n";
            }
        } else {
            grad_full = compute_full_gradient(sys_trial);
        }

        // Extract gradient for molecule 1 only (simple filtering)
        // Molecule 0 is fixed, so we ignore its gradient components
        grad_free = grad_full.segment<6>(6);

        // Print progress
        iter_count++;
        double max_grad = grad_free.array().abs().maxCoeff();

        std::cout << "Iteration " << std::setw(4) << iter_count
                  << "  Energy: " << std::setw(16) << std::setprecision(10) << E
                  << " au  Max Gradient: " << std::setw(12) << std::setprecision(6)
                  << max_grad << " au";

        if (sys.use_finite_diff_gradients) {
            std::cout << " (FD)";
        }
        std::cout << "\n";

        return E;
    };

    // Initial guess: Start at zero displacement from base
    Vec x0_free = Vec::Zero(free_dof);

    // Optimize
    double fval;
    int status = solver.minimize(objective, x0_free, fval, lb, ub);

    // Reconstruct final geometry
    Vec x_final(12);
    x_final.segment<6>(0) = x_base.segment<6>(0);  // Molecule 0 unchanged
    x_final.segment<6>(6) = x_base.segment<6>(6) + x0_free;  // Molecule 1 updated
    unpack_all_dof(x_final, sys);

    std::cout << "\n========================================\n";
    if (status < 0) {
        std::cout << "Optimization failed (status " << status << ")\n";
    } else {
        std::cout << "Optimization converged!\n";
        std::cout << "Final energy: " << std::setprecision(12) << fval << " au\n";
        std::cout << "Status code: " << status << "\n";
    }
    std::cout << "========================================\n\n";

    return sys;
}

/**
 * @brief Perform simple steepest descent optimization (fallback/comparison)
 */
System optimize_geometry_steepest_descent(System sys) {
    std::cout << "\n========================================\n";
    std::cout << "Starting geometry optimization\n";
    std::cout << "========================================\n";
    std::cout << "Max iterations: " << sys.max_opt_iterations << "\n";
    std::cout << "Convergence threshold: " << sys.convergence_threshold << " au\n";
    std::cout << "Initial step size: " << sys.step_size << " bohr\n\n";

    for (int iter = 0; iter < sys.max_opt_iterations; ++iter) {
        // Compute energy and forces
        EnergyComponents energies = compute_energy(sys);
        Forces forces = compute_forces(sys);

        double max_force = forces.max_magnitude();

        std::cout << "Iteration " << std::setw(4) << iter + 1
                  << "  Energy: " << std::setw(16) << std::setprecision(10) << energies.total()
                  << " au  Max Force: " << std::setw(12) << std::setprecision(6) << max_force
                  << " au\n";

        // Check convergence
        if (max_force < sys.convergence_threshold) {
            std::cout << "\nOptimization converged!\n";
            std::cout << "Final energy: " << std::setprecision(12) << energies.total() << " au\n";
            break;
        }

        // Update positions using steepest descent
        double step = sys.step_size;

        // Simple backtracking line search
        System sys_trial = sys;
        double E_current = energies.total();
        bool step_accepted = false;

        for (int ls_iter = 0; ls_iter < 10; ++ls_iter) {
            for (size_t i = 0; i < sys.molecules.size(); ++i) {
                sys_trial.molecules[i].position() = sys.molecules[i].position() + step * forces.forces[i];
            }

            double E_trial = compute_energy(sys_trial).total();

            if (E_trial < E_current) {
                sys = sys_trial;
                step_accepted = true;
                break;
            }

            step *= 0.5;
        }

        if (!step_accepted && sys.verbose) {
            std::cout << "  Warning: Line search failed, reducing step size\n";
            sys.step_size *= 0.5;
        }
    }

    std::cout << "========================================\n\n";
    return sys;
}

// ==================== Output Functions ====================

/**
 * @brief Print energy components
 */
void print_energy(const EnergyComponents& energies, bool orient_style = false) {
    if (orient_style) {
        // Format similar to Orient output
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Energy breakdown:\n";
        std::cout << "  Multipole:      " << std::setw(16) << energies.multipole << " au\n";
        if (energies.lennard_jones != 0.0) {
            std::cout << "  Lennard-Jones:  " << std::setw(16) << energies.lennard_jones << " au\n";
        }
        if (energies.buckingham != 0.0) {
            std::cout << "  Buckingham:     " << std::setw(16) << energies.buckingham << " au\n";
        }
        std::cout << "  Total:          " << std::setw(16) << energies.total() << " au\n";
    } else {
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "\n========================================\n";
        std::cout << "Energy Components\n";
        std::cout << "========================================\n";
        std::cout << "Multipole interaction:  " << energies.multipole << " au\n";
        if (energies.lennard_jones != 0.0) {
            std::cout << "Lennard-Jones:          " << energies.lennard_jones << " au\n";
        }
        if (energies.buckingham != 0.0) {
            std::cout << "Buckingham:             " << energies.buckingham << " au\n";
        }
        std::cout << "----------------------------------------\n";
        std::cout << "Total energy:           " << energies.total() << " au\n";
        std::cout << "========================================\n";
    }
}

/**
 * @brief Print forces and torques on all molecules
 */
void print_forces(const System& sys, const Forces& forces, bool orient_style = false) {
    if (orient_style) {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "\nForces (au):\n";
        for (size_t i = 0; i < forces.forces.size(); ++i) {
            std::cout << "  " << std::setw(10) << sys.molecules[i].name() << ": "
                     << std::setw(16) << forces.forces[i][0] << " "
                     << std::setw(16) << forces.forces[i][1] << " "
                     << std::setw(16) << forces.forces[i][2] << "\n";
        }
        std::cout << "\nTorques (Euler, au):\n";
        for (size_t i = 0; i < forces.torques_euler.size(); ++i) {
            std::cout << "  " << std::setw(10) << sys.molecules[i].name() << ": "
                     << std::setw(16) << forces.torques_euler[i][0] << " "
                     << std::setw(16) << forces.torques_euler[i][1] << " "
                     << std::setw(16) << forces.torques_euler[i][2] << "\n";
        }
    } else {
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "\n========================================\n";
        std::cout << "Forces and Torques on Molecules\n";
        std::cout << "========================================\n";
        for (size_t i = 0; i < forces.forces.size(); ++i) {
            std::cout << sys.molecules[i].name() << ":\n";
            std::cout << "  Force:\n";
            std::cout << "    Fx: " << forces.forces[i][0] << " au\n";
            std::cout << "    Fy: " << forces.forces[i][1] << " au\n";
            std::cout << "    Fz: " << forces.forces[i][2] << " au\n";
            std::cout << "    |F|: " << forces.forces[i].norm() << " au\n";
            std::cout << "  Torque (w.r.t. Euler angles):\n";
            std::cout << "    dE/dα: " << forces.torques_euler[i][0] << " au\n";
            std::cout << "    dE/dβ: " << forces.torques_euler[i][1] << " au\n";
            std::cout << "    dE/dγ: " << forces.torques_euler[i][2] << " au\n";
            std::cout << "    |τ|: " << forces.torques_euler[i].norm() << " au\n";
        }

        // Verify Newton's 3rd law
        Vec3 total_force = Vec3::Zero();
        Vec3 total_torque = Vec3::Zero();
        for (const auto& f : forces.forces) {
            total_force += f;
        }
        for (const auto& t : forces.torques_euler) {
            total_torque += t;
        }
        std::cout << "----------------------------------------\n";
        std::cout << "Total force (should be ~0):\n";
        std::cout << "  [" << total_force[0] << ", " << total_force[1] << ", " << total_force[2] << "]\n";
        std::cout << "  Magnitude: " << total_force.norm() << "\n";
        std::cout << "Total torque (should be ~0):\n";
        std::cout << "  [" << total_torque[0] << ", " << total_torque[1] << ", " << total_torque[2] << "]\n";
        std::cout << "  Magnitude: " << total_torque.norm() << "\n";
        std::cout << "========================================\n";
    }
}

/**
 * @brief Print optimized geometry
 */
void print_geometry(const System& sys, bool orient_style = false) {
    if (orient_style) {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "\nOptimized geometry (bohr, radians):\n";
        for (const auto& mol : sys.molecules) {
            Vec3 euler = const_cast<Molecule&>(mol).get_euler_angles();
            std::cout << "  " << std::setw(10) << mol.name() << ": "
                     << "pos=(" << std::setw(12) << mol.position()[0] << ","
                     << std::setw(12) << mol.position()[1] << ","
                     << std::setw(12) << mol.position()[2] << ")  "
                     << "euler=(" << std::setw(10) << euler[0] << ","
                     << std::setw(10) << euler[1] << ","
                     << std::setw(10) << euler[2] << ")\n";
        }
    } else {
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "\n========================================\n";
        std::cout << "Molecular Positions and Orientations\n";
        std::cout << "========================================\n";
        for (const auto& mol : sys.molecules) {
            Vec3 euler = const_cast<Molecule&>(mol).get_euler_angles();
            std::cout << mol.name() << ":\n";
            std::cout << "  Position: [" << mol.position()[0] << ", "
                     << mol.position()[1] << ", " << mol.position()[2] << "] bohr\n";
            std::cout << "  Euler angles (ZYZ): [" << euler[0] << ", "
                     << euler[1] << ", " << euler[2] << "] radians\n";
            std::cout << "  Euler angles (deg): [" << euler[0]*180.0/M_PI << ", "
                     << euler[1]*180.0/M_PI << ", " << euler[2]*180.0/M_PI << "] degrees\n";
        }
        std::cout << "========================================\n";
    }
}

/**
 * @brief Print complete results
 */
void print_results(const System& sys, const EnergyComponents& energies,
                  const Forces* forces = nullptr) {
    if (!sys.orient_style) {
        std::cout << "\n========================================\n";
        std::cout << "MULTIPOLE INTERACTION CALCULATION\n";
        std::cout << "========================================\n";
        std::cout << "System: " << sys.molecules.size() << " molecules\n";
        std::cout << "Max multipole rank: " << sys.max_rank << "\n";
        std::cout << "Short-range pairs: " << sys.short_range_pairs.size() << "\n";
    }

    print_energy(energies, sys.orient_style);

    if (forces) {
        print_forces(sys, *forces, sys.orient_style);
    }

    if (!sys.orient_style) {
        print_geometry(sys, false);
    }
}

// ==================== Main Program ====================

// ==================== Molecular Dynamics ====================

/**
 * @brief Convert Molecule to RigidBodyState for MD
 */
RigidBodyState molecule_to_rigid_body(const Molecule& mol, int id) {
    RigidBodyState rb(mol.name(), id, mol.position(), 1.0);  // Default mass = 1 amu
    rb.multipole_body = mol.multipole_body();

    // Copy orientation from the molecule's state
    rb.quaternion = mol.state.quaternion;
    rb.euler_valid = mol.state.euler_valid;
    rb.euler_angles = mol.state.euler_angles;

    // Make sure lab multipoles are computed
    rb.update_lab_multipoles();

    return rb;
}

/**
 * @brief Parse MD configuration from JSON
 */
RigidBodyDynamics::Config parse_md_config(const json& md_json) {
    RigidBodyDynamics::Config config;

    if (md_json.contains("timestep")) {
        config.timestep = md_json["timestep"].get<double>();
    }
    if (md_json.contains("num_steps")) {
        config.num_steps = md_json["num_steps"].get<int>();
    }
    if (md_json.contains("output_frequency")) {
        config.output_frequency = md_json["output_frequency"].get<int>();
    }

    return config;
}

/**
 * @brief Parse inertia tensor from JSON
 */
void parse_inertia(RigidBodyState& rb, const json& inertia_json) {
    std::string type = inertia_json.value("type", "spherical");

    if (type == "spherical") {
        double I = inertia_json["moment"].get<double>();
        rb.set_spherical_inertia(I);
    } else if (type == "diagonal") {
        double Ixx = inertia_json["Ixx"].get<double>();
        double Iyy = inertia_json["Iyy"].get<double>();
        double Izz = inertia_json["Izz"].get<double>();
        rb.set_diagonal_inertia(Ixx, Iyy, Izz);
    } else if (type == "general") {
        auto tensor = inertia_json["tensor"];
        Mat3 I;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                I(i, j) = tensor[i][j].get<double>();
            }
        }
        rb.set_inertia_tensor(I);
    }
}

/**
 * @brief Write trajectory frame
 */
void write_trajectory_frame(std::ofstream& traj,
                            const std::vector<RigidBodyState>& molecules,
                            int step, double time, double energy) {
    traj << molecules.size() << "\n";
    traj << "Step: " << step << " Time: " << std::fixed << std::setprecision(4)
         << time << " fs  Energy: " << std::setprecision(8) << energy << " Eh\n";

    for (const auto& mol : molecules) {
        Vec3 euler = const_cast<RigidBodyState&>(mol).get_euler_angles();
        traj << mol.name
             << " " << std::setw(12) << mol.position(0)
             << " " << std::setw(12) << mol.position(1)
             << " " << std::setw(12) << mol.position(2)
             << "  v=(" << mol.velocity(0) << "," << mol.velocity(1) << "," << mol.velocity(2) << ")"
             << "  euler=(" << euler(0) << "," << euler(1) << "," << euler(2) << ")"
             << "\n";
    }
}

/**
 * @brief Run molecular dynamics simulation
 */
void run_molecular_dynamics(const System& sys, const json& input_json) {
    std::cout << "\n========== Starting Molecular Dynamics Simulation ==========\n\n";

    // Parse MD configuration
    RigidBodyDynamics::Config md_config = parse_md_config(input_json["dynamics"]);

    std::cout << "MD Parameters:\n";
    std::cout << "  Timestep: " << md_config.timestep << " fs\n";
    std::cout << "  Number of steps: " << md_config.num_steps << "\n";
    std::cout << "  Output frequency: " << md_config.output_frequency << "\n\n";

    // Convert molecules to rigid bodies
    std::vector<RigidBodyState> molecules;
    int id = 0;

    for (const auto& mol : sys.molecules) {
        RigidBodyState rb = molecule_to_rigid_body(mol, id++);

        // Parse additional MD parameters from JSON
        const auto& mol_json = input_json["molecules"][rb.id];

        if (mol_json.contains("mass")) {
            rb.mass = mol_json["mass"].get<double>();
        }

        if (mol_json.contains("velocity")) {
            auto vel = mol_json["velocity"];
            rb.velocity << vel[0].get<double>(), vel[1].get<double>(), vel[2].get<double>();
        }

        if (mol_json.contains("angular_velocity")) {
            auto omega = mol_json["angular_velocity"];
            rb.angular_velocity_body << omega[0].get<double>(), omega[1].get<double>(), omega[2].get<double>();
        }

        if (mol_json.contains("inertia_tensor")) {
            parse_inertia(rb, mol_json["inertia_tensor"]);
        }

        molecules.push_back(rb);
    }

    // Open trajectory file
    std::string traj_file = input_json["dynamics"].value("trajectory_file", "trajectory.xyz");
    std::ofstream traj(traj_file);

    std::cout << "Initial configuration:\n";
    for (const auto& mol : molecules) {
        std::cout << "  " << mol.name << ": pos=" << mol.position.transpose()
                  << " vel=" << mol.velocity.transpose() << "\n";
    }

    // Compute initial forces
    RigidBodyDynamics::compute_forces_torques(molecules);

    // Initial energies
    double KE0 = RigidBodyDynamics::compute_kinetic_energy(molecules);
    double PE0 = RigidBodyDynamics::compute_potential_energy(molecules);
    double E0 = KE0 + PE0;

    std::cout << "\nInitial energies:\n";
    std::cout << "  Kinetic:   " << std::setprecision(10) << KE0 << " Eh\n";
    std::cout << "  Potential: " << PE0 << " Eh\n";
    std::cout << "  Total:     " << E0 << " Eh\n\n";

    std::cout << "Running MD simulation...\n\n";
    std::cout << std::setw(8) << "Step"
              << std::setw(12) << "Time (fs)"
              << std::setw(16) << "KE (Eh)"
              << std::setw(16) << "PE (Eh)"
              << std::setw(16) << "Total (Eh)"
              << std::setw(16) << "dE/E0"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    // Write initial frame
    write_trajectory_frame(traj, molecules, 0, 0.0, E0);

    // MD loop
    for (int step = 1; step <= md_config.num_steps; step++) {
        // Integrate one step
        RigidBodyDynamics::velocity_verlet_step(molecules, md_config.timestep);

        // Output
        if (step % md_config.output_frequency == 0) {
            double time = step * md_config.timestep;
            double KE = RigidBodyDynamics::compute_kinetic_energy(molecules);
            double PE = RigidBodyDynamics::compute_potential_energy(molecules);
            double E = KE + PE;
            double dE = (E - E0) / std::abs(E0);

            std::cout << std::setw(8) << step
                      << std::setw(12) << std::fixed << std::setprecision(2) << time
                      << std::setw(16) << std::scientific << std::setprecision(6) << KE
                      << std::setw(16) << PE
                      << std::setw(16) << E
                      << std::setw(16) << std::setprecision(3) << dE
                      << "\n";

            write_trajectory_frame(traj, molecules, step, time, E);
        }
    }

    traj.close();

    std::cout << "\n========== MD Simulation Complete ==========\n";
    std::cout << "Trajectory written to: " << traj_file << "\n\n";

    // Final statistics
    double KE_final = RigidBodyDynamics::compute_kinetic_energy(molecules);
    double PE_final = RigidBodyDynamics::compute_potential_energy(molecules);
    double E_final = KE_final + PE_final;
    double dE_final = E_final - E0;

    std::cout << "Energy conservation:\n";
    std::cout << "  Initial total energy: " << std::setprecision(10) << E0 << " Eh\n";
    std::cout << "  Final total energy:   " << E_final << " Eh\n";
    std::cout << "  Drift:                " << dE_final << " Eh ("
              << std::setprecision(4) << (dE_final/std::abs(E0))*100 << "%)\n";
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <input.json> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --forces            Compute forces on all molecules\n";
    std::cout << "  --optimize          Perform geometry optimization (LBFGS-B)\n";
    std::cout << "  --optimize-sd       Perform geometry optimization (steepest descent)\n";
    std::cout << "  --md                Run molecular dynamics simulation\n";
    std::cout << "  --verbose           Enable verbose output\n";
    std::cout << "  --orient-style      Output in Orient-compatible format\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog_name << " water_dimer.json --forces --verbose\n";
    std::cout << "  " << prog_name << " argon_dimer.json --optimize\n";
    std::cout << "  " << prog_name << " argon_dimer.json --optimize-sd\n";
    std::cout << "  " << prog_name << " water_dimer.json --md\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_file;
    bool force_flag = false;
    bool optimize_flag = false;
    bool optimize_sd_flag = false;
    bool verbose_flag = false;
    bool orient_flag = false;
    bool md_flag = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--forces") {
            force_flag = true;
        } else if (arg == "--optimize") {
            optimize_flag = true;
        } else if (arg == "--optimize-sd") {
            optimize_sd_flag = true;
        } else if (arg == "--md") {
            md_flag = true;
        } else if (arg == "--verbose") {
            verbose_flag = true;
        } else if (arg == "--orient-style") {
            orient_flag = true;
        } else if (input_file.empty()) {
            input_file = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input file specified\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        // Read input file (need to keep JSON for MD)
        std::ifstream file(input_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open input file: " + input_file);
        }
        json input_json;
        file >> input_json;
        file.close();

        // Parse system
        System sys = read_input(input_file);

        // Override flags from command line
        if (force_flag) sys.compute_forces = true;
        if (optimize_flag) sys.optimize = true;
        if (optimize_sd_flag) sys.optimize = true;
        if (verbose_flag) sys.verbose = true;
        if (orient_flag) sys.orient_style = true;

        if (!sys.orient_style) {
            std::cout << "Reading input from: " << input_file << "\n";
            std::cout << "Loaded " << sys.molecules.size() << " molecules\n";
        }

        // Run molecular dynamics if requested
        if (md_flag || input_json.contains("dynamics")) {
            run_molecular_dynamics(sys, input_json);
            return 0;
        }

        // Perform optimization if requested
        if (sys.optimize) {
            if (optimize_sd_flag) {
                sys = optimize_geometry_steepest_descent(sys);
            } else {
                sys = optimize_geometry(sys);
            }
        }

        // Compute energy
        EnergyComponents energies = compute_energy(sys);

        // Compute forces if requested
        Forces* forces_ptr = nullptr;
        Forces forces(0);
        if (sys.compute_forces) {
            forces = compute_forces(sys);
            forces_ptr = &forces;
        }

        // Perform gradient comparison if requested (standalone mode)
        if (sys.compare_gradients && !sys.optimize) {
            compare_gradients(sys);
        }

        // Print results
        print_results(sys, energies, forces_ptr);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
