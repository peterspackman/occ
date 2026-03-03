#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/multipole_source.h>
#include <occ/mults/short_range.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace occ::mults {

/// Data loaded from a DMACRYS benchmark JSON file.
struct DmacrysInput {
    std::string title;
    std::string source;

    struct CrystalData {
        double a, b, c, alpha, beta, gamma;
        std::string space_group;
        int Z = 1;
        struct Atom {
            std::string label;
            std::string element;
            Vec3 frac_xyz;
        };
        std::vector<Atom> atoms;
        std::vector<std::string> symops;
    } crystal;

    struct MoleculeSites {
        struct Site {
            std::string label;
            std::string element;
            std::string atom_type;
            int atomic_number;
            Vec3 position_bohr;
            int rank;
            std::vector<double> components; // Stone convention
            Vec3 aniso_axis_body = Vec3::Zero(); ///< Body-frame aniso z-axis (zero = not aniso)
        };
        std::vector<Site> sites;
    } molecule;

    struct BuckPair {
        std::string type1, type2;
        std::string el1, el2;
        std::string kind;
        double A_eV, rho_ang, C6_eV_ang6;
    };
    std::vector<BuckPair> potentials;

    struct AnisoPair {
        std::string type1, type2;
        double alpha;   ///< Å⁻¹ (1/(1/alpha))
        double rho_00;  ///< Å
        double rho_20;  ///< Å (site 1 P₂)
        double rho_02;  ///< Å (site 2 P₂)
    };
    std::vector<AnisoPair> aniso_potentials;

    double cutoff_radius = 15.0;
    bool has_spline = false;
    double spline_min = 0.0;  // SPLI arg 1 (Angstrom)
    double spline_max = 0.0;  // SPLI arg 2 (Angstrom)
    bool has_ewald_accuracy = false;
    double ewald_accuracy = 1e-6; // target accuracy for auto eta/kmax
    bool has_ewald_eta = false;
    double ewald_eta = 0.0; // Angstrom^-1
    bool has_ewald_kmax = false;
    int ewald_kmax = 0; // reciprocal cutoff integer extent
    bool has_pressure = false;
    double pressure_pa = 0.0; // PRES converted to Pa

    struct Reference {
        double total_kJ_per_mol = 0.0;
        double repulsion_dispersion_eV = 0.0;
        double repulsion_dispersion_kJ = 0.0;
        double total_eV_per_cell = 0.0;
        double charge_charge_inter_eV = 0.0;
        double charge_charge_ewald_summed_eV = 0.0;
        double charge_charge_intra_eV = 0.0;
        double charge_dipole_eV = 0.0;
        double dipole_dipole_eV = 0.0;
        double higher_multipole_eV = 0.0;
        std::vector<double> strain_derivatives_eV;  // 6 Voigt components
        Mat6 elastic_constants_GPa = Mat6::Zero();   // 6x6 Voigt stiffness
        bool has_strain_derivatives = false;
        bool has_elastic_constants = false;
    };
    Reference initial_ref, optimized_ref;

    // Optional fixed-point crystal geometry (typically from DMACRYS optimized
    // structure) for like-for-like benchmark comparisons.
    std::optional<CrystalData> optimized_crystal;
};

/// Read a DMACRYS benchmark JSON file.
DmacrysInput read_dmacrys_json(const std::string &json_path);

/// Build an OCC Crystal from JSON crystal data.
crystal::Crystal build_crystal(const DmacrysInput::CrystalData &data);

/// Build MultipoleSource objects from body-frame data + crystal geometry.
/// One MultipoleSource per symmetry-unique molecule.
std::vector<MultipoleSource> build_multipole_sources(
    const DmacrysInput &input,
    const crystal::Crystal &crystal);

/// Convert Buckingham params: DMACRYS (A_eV, rho) -> OCC (A_kJ, B=1/rho, C_kJ).
/// Key: pair of atomic numbers (Z1, Z2) with Z1 <= Z2.
std::map<std::pair<int, int>, BuckinghamParams>
convert_buckingham_params(const std::vector<DmacrysInput::BuckPair> &pairs);

/// Convert typed Buckingham params using explicit type-code mapping.
/// Type code keys are canonicalized with code1 <= code2.
std::map<std::pair<int, int>, BuckinghamParams>
convert_typed_buckingham_params(
    const std::vector<DmacrysInput::BuckPair> &pairs,
    const std::map<std::string, int> &type_codes);

/// Convert anisotropic repulsion params using explicit type-code mapping.
/// Keys are canonicalized with code1 <= code2 (rho_20/rho_02 swapped as needed).
std::map<std::pair<int, int>, AnisotropicRepulsionParams>
convert_typed_aniso_params(
    const std::vector<DmacrysInput::AnisoPair> &pairs,
    const std::map<std::string, int> &type_codes);

/// Prepare CrystalEnergy from DMACRYS input, bypassing Crystal's molecule
/// detection.  Sets neighbor list, geometry and initial states directly.
/// When build_neighbors=false, skips the neighbor list build (caller will
/// set it via set_neighbor_list).
void setup_crystal_energy_from_dmacrys(
    CrystalEnergy &calc,
    const DmacrysInput &input,
    const crystal::Crystal &crystal,
    const std::vector<MultipoleSource> &multipoles,
    bool build_neighbors = true);

/// Compute molecule states (COM positions + rotations) for a crystal.
/// Useful for strained evaluations where geometry/neighbors don't change.
std::vector<MoleculeState> compute_molecule_states(
    const DmacrysInput &input,
    const crystal::Crystal &crystal,
    const std::vector<MultipoleSource> &multipoles);

} // namespace occ::mults
