#pragma once
#include <array>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace occ::io {

struct SiteMultipoles {
    double charge = 0.0;
    std::vector<double> dipole;       // [3] or empty
    std::vector<double> quadrupole;   // [5] or empty
    std::vector<double> octupole;     // [7] or empty
    std::vector<double> hexadecapole; // [9] or empty

    /// Highest non-zero rank present (0=charge only, 4=hexadecapole).
    int max_rank() const;

    /// Flatten to Stone convention array: [Q00, Q10, Q11c, Q11s, ...].
    /// Length = (max_rank+1)^2.
    std::vector<double> to_flat() const;

    /// Construct from flat Stone array.
    static SiteMultipoles from_flat(const std::vector<double> &flat);
};

struct MoleculeSite {
    std::string label;
    std::string element;
    std::string type;                   // defaults to element if empty
    std::array<double, 3> position{};   // Angstrom, body frame
    SiteMultipoles multipoles;
};

struct MoleculeType {
    std::string name;
    std::vector<MoleculeSite> sites;
};

struct SymmetryEntry {
    std::string op;    // CIF string, e.g. "-x+1/2, y, -z+1/2"
    int molecule = 0;  // index into molecules array
};

struct IndependentMolecule {
    std::string type;                        // references MoleculeType::name
    std::array<double, 3> translation{};     // fractional COM
    std::array<double, 3> orientation{};     // angle-axis, radians (proper rotation part)
    int parity = 1;                          // +1 proper, -1 improper (inversion/mirror)
};

struct BuckinghamPair {
    std::array<std::string, 2> types;
    std::array<std::string, 2> elements;  // element symbols (e.g. "C", "Cl")
    double A = 0.0;    // eV
    double rho = 0.0;  // Angstrom
    double C6 = 0.0;   // eV·Å⁶
};

struct Potentials {
    double cutoff = 15.0;
    std::vector<BuckinghamPair> buckingham;
};

struct Settings {
    double ewald_accuracy = 1e-8;
    int max_interaction_order = 4;
    int max_iterations = 200;
    double gradient_tolerance = 1e-5;
    double neighbor_radius = 20.0;      // Angstrom
    double spline_min = 0.0;            // SPLI taper width (0 = no taper)
    double spline_max = 0.0;            // SPLI neighbor shell extension
    int spline_order = 3;               // taper polynomial order
    double pressure_gpa = 0.0;          // external pressure (GPa), 0 = none
    bool use_ewald = true;              // Ewald electrostatics
};

struct ReferenceEnergies {
    double total = 0.0;
    std::map<std::string, double> components;
};

struct StructureInput {
    std::string title;

    // Cell parameters (Angstrom, degrees)
    double a = 0, b = 0, c = 0;
    double alpha = 90, beta = 90, gamma = 90;

    // Molecule templates
    std::vector<MoleculeType> molecule_types;

    // Symmetry (optional — empty + empty space_group = P1)
    std::vector<SymmetryEntry> symmetry;
    std::string space_group; // alternative to symmetry array

    // Independent molecules (Z')
    std::vector<IndependentMolecule> molecules;

    // Optional sections
    Potentials potentials;
    Settings settings;
    ReferenceEnergies reference;
};

// --- JSON serialization (ADL) ---

void to_json(nlohmann::json &j, const SiteMultipoles &m);
void from_json(const nlohmann::json &j, SiteMultipoles &m);

void to_json(nlohmann::json &j, const MoleculeSite &s);
void from_json(const nlohmann::json &j, MoleculeSite &s);

void to_json(nlohmann::json &j, const MoleculeType &mt);
void from_json(const nlohmann::json &j, MoleculeType &mt);

void to_json(nlohmann::json &j, const SymmetryEntry &se);
void from_json(const nlohmann::json &j, SymmetryEntry &se);

void to_json(nlohmann::json &j, const IndependentMolecule &im);
void from_json(const nlohmann::json &j, IndependentMolecule &im);

void to_json(nlohmann::json &j, const BuckinghamPair &bp);
void from_json(const nlohmann::json &j, BuckinghamPair &bp);

void to_json(nlohmann::json &j, const Potentials &p);
void from_json(const nlohmann::json &j, Potentials &p);

void to_json(nlohmann::json &j, const Settings &s);
void from_json(const nlohmann::json &j, Settings &s);

void to_json(nlohmann::json &j, const ReferenceEnergies &r);
void from_json(const nlohmann::json &j, ReferenceEnergies &r);

void to_json(nlohmann::json &j, const StructureInput &si);
void from_json(const nlohmann::json &j, StructureInput &si);

// --- File I/O ---

/// Read a StructureInput from a JSON file.
StructureInput read_structure_json(const std::string &path);

/// Write a StructureInput to a JSON file (pretty-printed).
void write_structure_json(const std::string &path, const StructureInput &input);

/// Detect whether a JSON file uses the structure format (has "molecule_types" key).
bool is_structure_format(const std::string &json_path);

} // namespace occ::io
