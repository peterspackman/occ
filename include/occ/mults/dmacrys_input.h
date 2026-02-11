#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/multipole_source.h>
#include <occ/mults/short_range.h>
#include <map>
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
            int atomic_number;
            Vec3 position_bohr;
            int rank;
            std::vector<double> components; // Stone convention
        };
        std::vector<Site> sites;
    } molecule;

    struct BuckPair {
        std::string el1, el2;
        double A_eV, rho_ang, C6_eV_ang6;
    };
    std::vector<BuckPair> potentials;

    double cutoff_radius = 15.0;

    struct Reference {
        double total_kJ_per_mol = 0.0;
        double repulsion_dispersion_eV = 0.0;
        double repulsion_dispersion_kJ = 0.0;
        double total_eV_per_cell = 0.0;
        double charge_charge_inter_eV = 0.0;
        double charge_dipole_eV = 0.0;
        double dipole_dipole_eV = 0.0;
        double higher_multipole_eV = 0.0;
    };
    Reference initial_ref, optimized_ref;
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

/// Prepare CrystalEnergy from DMACRYS input, bypassing Crystal's molecule
/// detection.  Sets neighbor list, geometry and initial states directly.
void setup_crystal_energy_from_dmacrys(
    CrystalEnergy &calc,
    const DmacrysInput &input,
    const crystal::Crystal &crystal,
    const std::vector<MultipoleSource> &multipoles);

} // namespace occ::mults
