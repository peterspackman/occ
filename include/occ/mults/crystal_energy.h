#pragma once
#include <occ/mults/multipole_source.h>
#include <occ/mults/short_range.h>
#include <occ/mults/cartesian_force.h>
#include <occ/crystal/crystal.h>
#include <occ/core/linear_algebra.h>
#include <map>
#include <vector>

namespace occ::mults {

/**
 * @brief Force field type for repulsion-dispersion interactions.
 */
enum class ForceFieldType {
    None,           ///< No short-range interactions
    LennardJones,   ///< LJ 12-6 potential
    BuckinghamDE,   ///< Williams DE Buckingham parameters
    Custom          ///< User-provided parameters
};

/**
 * @brief State of a rigid molecule (COM position + orientation).
 */
struct MoleculeState {
    Vec3 position = Vec3::Zero();    ///< COM position (Angstrom)
    Vec3 angle_axis = Vec3::Zero();  ///< Rotation as angle-axis (radians)

    /// Compute rotation matrix from angle-axis representation.
    Mat3 rotation_matrix() const;

    /// Create from rotation matrix (extracts angle-axis).
    static MoleculeState from_rotation(const Vec3& pos, const Mat3& R);
};

/**
 * @brief Result from crystal energy evaluation.
 */
struct CrystalEnergyResult {
    double total_energy = 0.0;           ///< Total energy (kJ/mol)
    double electrostatic_energy = 0.0;   ///< Electrostatic component
    double repulsion_dispersion = 0.0;   ///< Short-range component

    std::vector<Vec3> forces;            ///< Force on each molecule (kJ/mol/Ang)
    std::vector<Vec3> torques;           ///< Angle-axis gradient on each molecule

    /// Pack into single gradient vector (6 components per molecule).
    Vec pack_gradient() const;

    /// Per-molecule energies for analysis.
    std::vector<double> molecule_energies;
};

/**
 * @brief Result from crystal energy evaluation with Hessian.
 */
struct CrystalEnergyResultWithHessian : CrystalEnergyResult {
    /// Full Hessian matrix (6N x 6N for N molecules)
    /// Layout: [x1, y1, z1, θx1, θy1, θz1, x2, ...]
    Mat hessian;

    /// Pack Hessian with only the optimizable DOF
    /// (excluding fixed molecule translations/rotations)
    Mat pack_hessian(bool fix_first_translation, bool fix_first_rotation) const;
};

struct PairEnergyDebug {
    int mol_i;
    int mol_j;
    IVec3 cell_shift;
    double com_distance = 0.0;
    double electrostatic = 0.0;
    double short_range = 0.0;
    double total = 0.0;
    double weight = 1.0;
};

/**
 * @brief Neighbor pair information for crystal energy computation.
 */
struct NeighborPair {
    int mol_i;          ///< Index of first molecule
    int mol_j;          ///< Index of second molecule
    IVec3 cell_shift;   ///< Lattice translation of mol_j
    double weight;      ///< Weight (0.5 for self-images, 1.0 otherwise)
};

/**
 * @brief Crystal energy evaluator for rigid molecule assemblies.
 *
 * Computes electrostatic energy using multipole interactions and
 * short-range repulsion-dispersion using Buckingham or LJ potentials.
 *
 * Usage:
 * @code
 * Crystal crystal = ...;  // Load from CIF
 * std::vector<MultipoleSource> multipoles = ...;  // From DMA
 *
 * CrystalEnergy energy_calc(crystal, multipoles, 20.0, ForceFieldType::BuckinghamDE);
 *
 * std::vector<MoleculeState> states = energy_calc.initial_states();
 * auto result = energy_calc.compute(states);
 * @endcode
 */
class CrystalEnergy {
public:
    /// Cached molecule geometry (atom positions in body frame).
    struct MoleculeGeometry {
        std::vector<int> atomic_numbers;
        std::vector<Vec3> atom_positions;  // Body-frame
        Vec3 center_of_mass;
    };

    /**
     * @brief Construct crystal energy evaluator.
     *
     * @param crystal Crystal structure (provides geometry and symmetry)
     * @param multipoles Body-frame multipoles for each symmetry-unique molecule
     * @param cutoff_radius Neighbor cutoff in Angstrom (default 20.0)
     * @param ff Force field for short-range (default BuckinghamDE)
     * @param use_cartesian Use Cartesian T-tensor engine (default true)
     */
    CrystalEnergy(const crystal::Crystal& crystal,
                  std::vector<MultipoleSource> multipoles,
                  double cutoff_radius = 20.0,
                  ForceFieldType ff = ForceFieldType::BuckinghamDE,
                  bool use_cartesian = true,
                  bool use_ewald = true,
                  double ewald_accuracy = 1e-6,
                  double ewald_eta = 0.0,
                  int ewald_kmax = 0);

    /// Compute energy and gradient for given molecular states.
    CrystalEnergyResult compute(const std::vector<MoleculeState>& molecules);

    /// Compute energy, gradient, and Hessian for given molecular states.
    CrystalEnergyResultWithHessian compute_with_hessian(
        const std::vector<MoleculeState>& molecules);

    /// Compute energy only (faster, for line search).
    double compute_energy(const std::vector<MoleculeState>& molecules);

    /// Get initial states from crystal geometry.
    std::vector<MoleculeState> initial_states() const;

    /// Number of molecules in asymmetric unit.
    int num_molecules() const { return static_cast<int>(m_multipoles.size()); }

    /// Get neighbor list.
    const std::vector<NeighborPair>& neighbor_pairs() const { return m_neighbors; }

    /// Get/set Buckingham parameters for atom pair.
    void set_buckingham_params(int Z1, int Z2, const BuckinghamParams& params);
    BuckinghamParams get_buckingham_params(int Z1, int Z2) const;

    /// Williams DE Buckingham parameters (built-in).
    static std::map<std::pair<int,int>, BuckinghamParams> williams_de_params();

    /// Get the underlying crystal.
    const crystal::Crystal& crystal() const { return m_crystal; }

    /// Update neighbor list (called if molecules move significantly).
    void update_neighbors();

    /// Build neighbor list from explicit molecule COM positions.
    /// Bypasses Crystal's molecule detection (useful when loading external data).
    void build_neighbor_list_from_positions(const std::vector<Vec3>& mol_coms);

    /// Set molecule geometry directly (bypasses Crystal's molecule detection).
    void set_molecule_geometry(std::vector<MoleculeGeometry> geometry);

    /// Set initial molecule states directly.
    void set_initial_states(std::vector<MoleculeState> states);

    /// Enable/disable dipole Ewald (charge-dipole + dipole-dipole).
    /// Default is true (DMACRYS computes Ewald for qq, qμ, and μμ).
    void set_ewald_dipole(bool enable) { m_ewald_dipole = enable; }

    /// Get maximum multipole rank across all sites.
    int max_multipole_rank() const;

    /// Get number of neighbor pairs.
    size_t num_neighbor_pairs() const { return m_neighbors.size(); }

    /// Compute per-pair debug breakdown at current states.
    std::vector<PairEnergyDebug> debug_pair_energies(const std::vector<MoleculeState>& molecules) const;

    /// Neighbor shell histogram for quick sanity checks.
    std::vector<int> neighbor_shell_histogram() const;  // bins: [0-3), [3-6), [6-10), [10-15), [15+]

    struct EwaldBreakdown {
        double real_space = 0.0;
        double reciprocal = 0.0;
        double self = 0.0;
        double total() const { return real_space + reciprocal + self; }
    };

    /// Charge-only Ewald energy breakdown (energy only, no forces; for debugging).
    EwaldBreakdown charge_ewald_breakdown(const std::vector<MoleculeState>& molecules,
                                          double alpha = 0.35,
                                          int kmax = 6) const;

    /// Get total number of multipole sites.
    size_t num_sites() const;

private:
    // Ewald settings
    bool m_use_ewald = true;
    bool m_ewald_dipole = true; // Include qμ/μμ in Ewald (DMACRYS does all three)
    double m_ewald_accuracy = 1e-6;
    double m_ewald_eta = 0.0;
    int m_ewald_kmax = 0;

    crystal::Crystal m_crystal;
    std::vector<MultipoleSource> m_multipoles;
    double m_cutoff_radius;
    ForceFieldType m_force_field;
    bool m_use_cartesian;

    std::vector<NeighborPair> m_neighbors;
    std::map<std::pair<int,int>, BuckinghamParams> m_buckingham_params;
    std::map<std::pair<int,int>, LennardJonesParams> m_lj_params;

    std::vector<MoleculeGeometry> m_geometry;

    void build_neighbor_list();
    void build_molecule_geometry();
    void initialize_force_field();

    std::vector<MoleculeState> m_initial_states;  // optional override

    /// Result from charge-charge Ewald correction computation.
    struct EwaldCorrectionResult {
        double energy = 0.0;           ///< Energy correction (kJ/mol)
        std::vector<Vec3> forces;      ///< Force correction per molecule (kJ/mol/Å)
        std::vector<Vec3> torques;     ///< Torque correction per molecule
    };

    /// Compute charge-charge Ewald correction (energy + forces + torques).
    /// Returns the difference between Ewald charge-charge and truncated charge-charge.
    EwaldCorrectionResult compute_charge_ewald_correction(
        const std::vector<MoleculeState>& molecules,
        const std::vector<CartesianMolecule>& cart_mols) const;

    /// Compute short-range energy and forces for a molecule pair.
    void compute_short_range_pair(
        int mol_i, int mol_j,
        const MoleculeState& state_i,
        const MoleculeState& state_j,
        const Vec3& translation,
        double weight,
        double& energy,
        Vec3& force_i, Vec3& force_j,
        Vec3& torque_i, Vec3& torque_j) const;
};

} // namespace occ::mults
