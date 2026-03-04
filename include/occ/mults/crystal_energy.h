#pragma once
#include <occ/mults/multipole_source.h>
#include <occ/mults/short_range.h>
#include <occ/mults/force_field_params.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/cutoff_spline.h>
#include <occ/crystal/crystal.h>
#include <occ/core/linear_algebra.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace occ::mults {

struct EwaldLatticeCache;  // Forward declaration (defined in ewald_sum.h)

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
    Vec3 angle_axis = Vec3::Zero();  ///< Proper rotation part as angle-axis (radians)
    int parity = 1;                  ///< Orientation parity (+1 proper, -1 improper)

    /// Compute full orientation matrix in O(3) (det = parity).
    Mat3 rotation_matrix() const;

    /// Compute proper SO(3) component (det = +1) from angle-axis.
    Mat3 proper_rotation_matrix() const;

    /// Create from rotation matrix (extracts parity + proper angle-axis part).
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
    std::vector<Vec3> torques;           ///< Lab-frame rotational gradient dE/dψ on each molecule

    /// Affine cell-strain gradient dE/dE_i (Voigt E1..E6, kJ/mol per unit cell).
    /// Computed from pair virial terms; Ewald contribution is force-mapped only.
    Vec6 strain_gradient = Vec6::Zero();

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

    /// True only when the Hessian matches the full current energy model.
    bool exact_for_model = false;

    /// True when Ewald second-derivative terms are included.
    bool includes_ewald_terms = false;

    /// Clamped affine cell-strain Hessian d^2E/dE_i dE_j (Voigt, kJ/mol per unit cell).
    Mat6 strain_hessian = Mat6::Zero();

    /// Strain-state coupling d^2E/dE_i dq_j (6 x 6N, kJ/mol).
    /// q layout matches `hessian`: [x1,y1,z1,θx1,θy1,θz1,...].
    Mat strain_state_hessian;

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
    int short_range_site_pairs = 0;
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
    double weight;      ///< Pair weight (typically 1.0 for canonical unique pairs)
    double com_distance = 0.0; ///< COM-COM distance (Angstrom), for electrostatic COM gate
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
        /// Optional short-range atom types (e.g. DMACRYS/NEIGHCRYS codes).
        /// 0 means untyped/unknown and falls back to element-based parameters.
        std::vector<int> short_range_type_codes;
        std::vector<Vec3> atom_positions;  // Body-frame
        std::vector<Vec3> aniso_body_axes; // Body-frame aniso z-axis per atom (zero if not aniso)
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

    /// Destructor (defined in .cpp for unique_ptr with forward-declared type).
    ~CrystalEnergy();

    /// Move operations (defined in .cpp where EwaldLatticeCache is complete).
    CrystalEnergy(CrystalEnergy&&) noexcept;
    CrystalEnergy& operator=(CrystalEnergy&&) noexcept;

    /// Compute energy and gradient for given molecular states.
    CrystalEnergyResult compute(const std::vector<MoleculeState>& molecules);

    /// Compute energy, gradient, and Hessian for given molecular states.
    /// Hessian is assembled analytically from pair terms (short-range full rigid-body
    /// and full Cartesian multipole electrostatics). Ewald site-position Hessian
    /// terms are included when enabled; exactness for the full model is reported by
    /// `exact_for_model`.
    CrystalEnergyResultWithHessian compute_with_hessian(
        const std::vector<MoleculeState>& molecules);

    /// Returns true when compute_with_hessian provides an exact Hessian for
    /// the current configured energy model.
    bool can_compute_exact_hessian() const;

    /// Compute energy only (faster, for line search).
    double compute_energy(const std::vector<MoleculeState>& molecules);

    /// Get initial states from crystal geometry.
    std::vector<MoleculeState> initial_states() const;

    /// Update the crystal lattice for strained evaluation.
    /// Keeps: neighbors, force field, geometry (body-frame), multipoles.
    /// Updates: crystal (for cell translations + Ewald), initial states.
    void update_lattice(const crystal::Crystal& strained_crystal,
                        std::vector<MoleculeState> new_states);

    /// Number of molecules in asymmetric unit.
    int num_molecules() const { return static_cast<int>(m_multipoles.size()); }

    /// Access molecule geometry (body-frame atom positions).
    const std::vector<MoleculeGeometry>& molecule_geometry() const { return m_geometry; }

    /// Get neighbor list.
    const std::vector<NeighborPair>& neighbor_pairs() const { return m_neighbors; }

    /// Access force field parameters.
    ForceFieldParams& force_field() { return m_ff; }
    const ForceFieldParams& force_field() const { return m_ff; }

    /// Set/query anisotropic repulsion parameters (typed).
    void set_typed_aniso_params(
        const std::map<std::pair<int,int>, AnisotropicRepulsionParams>& params) {
        m_ff.set_typed_aniso(params);
    }
    bool has_aniso_params(int type1, int type2) const { return m_ff.has_aniso(type1, type2); }
    AnisotropicRepulsionParams get_aniso_params(int type1, int type2) const { return m_ff.get_aniso(type1, type2); }
    bool has_any_aniso_params() const { return m_ff.has_any_aniso(); }

    /// Get/set Buckingham parameters for atom pair.
    void set_buckingham_params(int Z1, int Z2, const BuckinghamParams& params) { m_ff.set_buckingham(Z1, Z2, params); }
    void set_typed_buckingham_params(int type1, int type2,
                                     const BuckinghamParams& params) { m_ff.set_typed_buckingham(type1, type2, params); }
    void set_typed_buckingham_params(
        const std::map<std::pair<int,int>, BuckinghamParams>& params) { m_ff.set_typed_buckingham(params); }
    void clear_typed_buckingham_params() { m_ff.clear_typed_buckingham(); }
    void set_short_range_type_labels(
        const std::map<int, std::string>& labels) { m_ff.set_type_labels(labels); }
    BuckinghamParams get_buckingham_params(int Z1, int Z2) const { return m_ff.get_buckingham(Z1, Z2); }
    bool has_buckingham_params(int Z1, int Z2) const { return m_ff.has_buckingham(Z1, Z2); }
    bool uses_williams_atom_typing() const { return m_ff.use_williams_atom_typing(); }
    bool uses_short_range_typing() const { return m_ff.use_short_range_typing(); }
    bool has_typed_buckingham_params(int type1, int type2) const { return m_ff.has_typed_buckingham(type1, type2); }
    BuckinghamParams get_buckingham_params_for_types(int type1, int type2) const { return m_ff.get_buckingham_for_types(type1, type2); }
    std::string short_range_type_name(int type_code) const { return m_ff.type_name(type_code); }

    /// Convert a Williams/NEIGHCRYS type code to a short label (e.g. 512 -> C_W3).
    static const char* short_range_type_label(int type_code) { return ForceFieldParams::short_range_type_label(type_code); }

    /// Map a Williams/NEIGHCRYS type code to element Z, or 0 if unknown.
    static int short_range_type_atomic_number(int type_code) { return ForceFieldParams::short_range_type_atomic_number(type_code); }

    /// Williams DE Buckingham parameters (built-in).
    static std::map<std::pair<int,int>, BuckinghamParams> williams_de_params() { return ForceFieldParams::williams_de_params(); }

    /// Get the underlying crystal.
    const crystal::Crystal& crystal() const { return m_crystal; }
    double cutoff_radius() const { return m_cutoff_radius; }
    void set_cutoff_radius(double cutoff);

    /// Update neighbor list (called if molecules move significantly).
    void update_neighbors();

    /// Update neighbor list using current molecule states.
    /// For explicit neighbor mode (all UC molecules), this rebuilds the list
    /// from current COM positions and orientations.
    void update_neighbors(const std::vector<MoleculeState>& states);

    /// Build neighbor list from explicit molecule COM positions.
    /// Bypasses Crystal's molecule detection (useful when loading external data).
    /// When force_com_cutoff=true, uses COM distance for the molecule pair gate
    /// even when atom geometry is available (matches DMACRYS TBLCNT behavior).
    void build_neighbor_list_from_positions(const std::vector<Vec3>& mol_coms,
                                            bool force_com_cutoff = false,
                                            const std::vector<MoleculeState>* orientation_states = nullptr);

    /// Set neighbor list directly (e.g. to reuse a list from another CrystalEnergy).
    void set_neighbor_list(const std::vector<NeighborPair>& neighbors);

    /// Get the neighbor list.
    const std::vector<NeighborPair>& neighbor_list() const { return m_neighbors; }

    /// Set molecule geometry directly (bypasses Crystal's molecule detection).
    void set_molecule_geometry(std::vector<MoleculeGeometry> geometry);

    /// Set initial molecule states directly.
    void set_initial_states(std::vector<MoleculeState> states);

    /// Enable/disable dipole Ewald (charge-dipole + dipole-dipole).
    /// Default is true (DMACRYS computes Ewald for qq, qμ, and μμ).
    void set_ewald_dipole(bool enable) { m_ewald_dipole = enable; }
    bool use_ewald() const { return m_use_ewald; }

    /// Set maximum interaction order for electrostatics (rankA + rankB <= max).
    /// Default -1 means no truncation (compute all orders up to 2*max_rank).
    /// Set to 4 to match DMACRYS truncation.
    void set_max_interaction_order(int max_order) { m_max_interaction_order = max_order; }
    int max_interaction_order() const { return m_max_interaction_order; }

    /// Set per-site cutoff for electrostatic interactions (Angstrom).
    /// Default 0.0 means no per-site cutoff (all site pairs within included
    /// molecule pairs are computed).  DMACRYS applies RANG2 per-site cutoff
    /// in its PAIR module for higher multipole terms.
    void set_elec_site_cutoff(double cutoff) {
        m_elec_site_cutoff = cutoff;
        invalidate_ewald_params();
    }
    double elec_site_cutoff() const { return m_elec_site_cutoff; }

    /// Enable COM-based gate for electrostatic interactions.
    /// When enabled, only molecule pairs with COM distance <= cutoff_radius
    /// contribute to electrostatics (matching DMACRYS TBLCNT behavior).
    /// Buckingham is unaffected — uses full atom-based neighbor list.
    /// Default is true for DMACRYS compatibility.
    void set_use_com_elec_gate(bool enable) {
        m_use_com_elec_gate = enable;
        invalidate_ewald_params();
    }
    bool use_com_elec_gate() const { return m_use_com_elec_gate; }

    /// Set a separate per-site cutoff for Buckingham (default: same as neighbor cutoff).
    /// Use a larger value than the neighbor cutoff to reduce derivative
    /// discontinuities near the neighbor boundary.
    void set_buckingham_site_cutoff(double cutoff) {
        m_buck_site_cutoff = cutoff;
        invalidate_ewald_params();
    }
    double buckingham_site_cutoff() const { return m_buck_site_cutoff; }

    /// Set DMACRYS-style electrostatic spline taper.
    /// f=1 for r<=r_on, spline to 0 over (r_on,r_off], zero beyond r_off.
    /// Order must be 3 (cubic) or 5 (quintic).
    void set_electrostatic_taper(double r_on, double r_off, int order = 3);
    void clear_electrostatic_taper() {
        m_electrostatic_taper = {};
        invalidate_ewald_params();
    }
    const CutoffSpline& electrostatic_taper() const { return m_electrostatic_taper; }

    /// When false, the electrostatic taper is applied to energy and gradient
    /// but NOT to the pair Hessian.  This matches DMACRYS behaviour where
    /// the multipole SEC array does not include spline chain-rule terms.
    void set_electrostatic_taper_in_hessian(bool enable) { m_elec_taper_hessian = enable; }
    bool electrostatic_taper_in_hessian() const { return m_elec_taper_hessian; }

    /// Set DMACRYS-style short-range spline taper.
    void set_short_range_taper(double r_on, double r_off, int order = 3);
    void clear_short_range_taper() {
        m_short_range_taper = {};
        invalidate_ewald_params();
    }
    const CutoffSpline& short_range_taper() const { return m_short_range_taper; }

    /// Compute which atom-atom pairs are within the Buckingham cutoff for each
    /// neighbor pair at the given molecular states.  Returns a per-neighbor-pair
    /// boolean mask (row-major, nA*nB) indicating inclusion.
    std::vector<std::vector<bool>> compute_buckingham_site_masks(
        const std::vector<MoleculeState>& states) const;

    /// Set frozen site-pair masks.  When non-empty, compute_short_range_pair
    /// uses only the atom pairs marked true, bypassing the distance cutoff.
    /// This ensures a smoother energy surface under cell/molecular perturbations.
    void set_fixed_site_masks(std::vector<std::vector<bool>> masks) {
        m_fixed_site_masks = std::move(masks);
    }

    /// Clear frozen site-pair masks (revert to distance-based cutoff).
    void clear_fixed_site_masks() { m_fixed_site_masks.clear(); }

    /// Get maximum multipole rank across all sites.
    int max_multipole_rank() const;

    /// Get number of neighbor pairs.
    size_t num_neighbor_pairs() const { return m_neighbors.size(); }

    /// Compute per-pair debug breakdown at current states.
    std::vector<PairEnergyDebug> debug_pair_energies(const std::vector<MoleculeState>& molecules);

    /// Neighbor shell histogram for quick sanity checks.
    std::vector<int> neighbor_shell_histogram() const;  // bins: [0-3), [3-6), [6-10), [10-15), [15+]

    /// Get total number of multipole sites.
    size_t num_sites() const;

    /// Cached per-molecule data to avoid redundant recomputation.
    struct MoleculeCache {
        Mat3 rotation;
        std::vector<Vec3> lab_atom_positions; // state.position + R * body_pos
    };

private:
    // Ewald settings
    bool m_use_ewald = true;
    bool m_ewald_dipole = true; // Include qμ/μμ in Ewald (DMACRYS does all three)
    double m_ewald_accuracy = 1e-6;
    double m_ewald_eta = 0.0;
    int m_ewald_kmax = 0;
    mutable bool m_ewald_params_initialized = false;
    mutable double m_ewald_alpha_fixed = 0.0;
    mutable int m_ewald_kmax_fixed = 0;

    crystal::Crystal m_crystal;
    std::vector<MultipoleSource> m_multipoles;
    double m_cutoff_radius;
    ForceFieldType m_force_field;
    bool m_use_cartesian;
    int m_max_interaction_order = -1; // -1 = no truncation
    double m_elec_site_cutoff = 0.0;  // 0 = no per-site cutoff for electrostatics
    bool m_use_com_elec_gate = true;  // Skip electrostatics for COM > cutoff
    double m_buck_site_cutoff = -1.0; // -1 = use m_cutoff_radius
    CutoffSpline m_electrostatic_taper; // optional radial taper for electrostatics
    CutoffSpline m_short_range_taper;   // optional radial taper for short-range
    bool m_elec_taper_hessian = false;  // false = match DMACRYS (no taper in elec Hessian)
    bool m_explicit_neighbors = false; // true when neighbor list was set externally

    /// Per-neighbor-pair atom inclusion masks for frozen Buckingham cutoff.
    /// m_fixed_site_masks[pair_idx] has size nA*nB (row-major), true = include.
    std::vector<std::vector<bool>> m_fixed_site_masks;

    std::vector<NeighborPair> m_neighbors;
    ForceFieldParams m_ff;

    std::vector<MoleculeGeometry> m_geometry;

    void build_neighbor_list();
    void invalidate_ewald_params();
    void ensure_ewald_params_initialized() const;
    double effective_neighbor_pair_cutoff() const;
    double effective_electrostatic_com_cutoff() const;
    double effective_electrostatic_site_cutoff() const;
    double effective_buckingham_site_cutoff() const;
    void build_molecule_geometry();
    void assign_williams_atom_types();
    void initialize_force_field();

    std::vector<MoleculeState> m_initial_states;  // optional override

    /// Cached Ewald lattice (G-vectors + coefficients). Lazy-built on first
    /// Ewald call and reused while the unit cell and Ewald params are unchanged.
    mutable std::unique_ptr<EwaldLatticeCache> m_ewald_lattice_cache;

    /// Result from Ewald correction mapped to rigid-body DOF.
    struct EwaldCorrectionResult {
        double energy = 0.0;           ///< Energy correction (kJ/mol)
        std::vector<Vec3> forces;      ///< Force correction per molecule (kJ/mol/Ang)
        std::vector<Vec3> torques;     ///< Torque correction per molecule
    };

    /// Compute Ewald correction and map site forces to rigid-body forces/torques.
    EwaldCorrectionResult compute_charge_ewald_correction(
        const std::vector<MoleculeState>& molecules,
        const std::vector<CartesianMolecule>& cart_mols) const;

    /// Compute short-range energy and forces for a molecule pair.
    /// @param neighbor_idx Index into m_neighbors (used for fixed site masks).
    ///        Pass -1 if no fixed masks should be applied.
    /// @param cache_i Precomputed rotation and lab-frame atom positions for mol_i (or nullptr).
    /// @param cache_j Precomputed rotation and lab-frame atom positions for mol_j (or nullptr).
    void compute_short_range_pair(
        int mol_i, int mol_j,
        const MoleculeState& state_i,
        const MoleculeState& state_j,
        const Vec3& translation,
        double weight,
        double& energy,
        Vec3& force_i, Vec3& force_j,
        Vec3& torque_i, Vec3& torque_j,
        int neighbor_idx = -1,
        const MoleculeCache* cache_i = nullptr,
        const MoleculeCache* cache_j = nullptr,
        int* short_range_site_pairs = nullptr) const;
};

const std::array<Mat3, 6>& voigt_basis_matrices();

} // namespace occ::mults
