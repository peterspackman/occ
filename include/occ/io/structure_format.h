#pragma once
#include <array>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace occ::io {

// ============================================================================
// Shared types (used by both Basis and Crystal)
// ============================================================================

struct SiteMultipoles {
    double charge = 0.0;
    std::vector<double> dipole;       // [3] or empty
    std::vector<double> quadrupole;   // [5] or empty
    std::vector<double> octupole;     // [7] or empty
    std::vector<double> hexadecapole; // [9] or empty

    int max_rank() const;
    std::vector<double> to_flat() const;
    static SiteMultipoles from_flat(const std::vector<double> &flat);
};

struct MoleculeSite {
    std::string label;
    std::string element;
    std::string type;                   // force-field type (e.g. "C_W3")
    std::array<double, 3> position{};   // Angstrom, body frame
    SiteMultipoles multipoles;
};

struct MoleculeType {
    std::string name;
    std::vector<MoleculeSite> sites;
};

struct BuckinghamPair {
    std::array<std::string, 2> types;
    std::array<std::string, 2> elements;
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
    double neighbor_radius = 20.0;
    double spline_min = 0.0;
    double spline_max = 0.0;
    int spline_order = 3;
    double pressure_gpa = 0.0;
    bool use_ewald = true;
};

// ============================================================================
// Basis — molecule definitions + potentials (crystal-independent)
// ============================================================================

/// Everything needed to define the energy model, independent of crystal packing.
/// Can be written by `occ dma` and consumed by CSP programs.
struct Basis {
    std::vector<MoleculeType> molecule_types;
    Potentials potentials;
    Settings settings;
};

// ============================================================================
// Crystal — cell + molecule placements (requires a Basis to interpret)
// ============================================================================

struct IndependentMolecule {
    std::string type;                        // references MoleculeType::name
    std::array<double, 3> translation{};     // fractional COM
    std::array<double, 3> orientation{};     // angle-axis, radians
    int parity = 1;                          // +1 proper, -1 improper
};

struct CrystalData {
    double a = 0, b = 0, c = 0;
    double alpha = 90, beta = 90, gamma = 90;
    std::string space_group;
    std::vector<IndependentMolecule> molecules;
};

struct ReferenceEnergies {
    double total = 0.0;
    std::map<std::string, double> components;
};

// ============================================================================
// StructureInput — the full file (basis + crystal)
// ============================================================================

/// Complete input for crystal energy calculations.
/// The `basis` section is always present. The `crystal` section is optional —
/// a file without it is a "basis file" (molecule definitions + potentials only).
struct StructureInput {
    std::string title;
    Basis basis;
    CrystalData crystal;
    ReferenceEnergies reference;

    /// True if crystal data is populated (cell params > 0).
    bool has_crystal() const { return crystal.a > 0; }

    // --- Convenience accessors (forward to basis) ---
    const std::vector<MoleculeType> &molecule_types() const {
        return basis.molecule_types;
    }
    const Potentials &potentials() const { return basis.potentials; }
    const Settings &settings() const { return basis.settings; }
};

// --- JSON serialization (ADL) ---

void to_json(nlohmann::json &j, const SiteMultipoles &m);
void from_json(const nlohmann::json &j, SiteMultipoles &m);

void to_json(nlohmann::json &j, const MoleculeSite &s);
void from_json(const nlohmann::json &j, MoleculeSite &s);

void to_json(nlohmann::json &j, const MoleculeType &mt);
void from_json(const nlohmann::json &j, MoleculeType &mt);

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

void to_json(nlohmann::json &j, const Basis &b);
void from_json(const nlohmann::json &j, Basis &b);

void to_json(nlohmann::json &j, const CrystalData &c);
void from_json(const nlohmann::json &j, CrystalData &c);

void to_json(nlohmann::json &j, const StructureInput &si);
void from_json(const nlohmann::json &j, StructureInput &si);

// --- File I/O ---

StructureInput read_structure_json(const std::string &path);
void write_structure_json(const std::string &path, const StructureInput &input);

/// Write only the basis section to a JSON file.
void write_basis_json(const std::string &path, const Basis &basis,
                      const std::string &title = "");

/// Detect whether a JSON file uses the structure format.
bool is_structure_format(const std::string &json_path);

} // namespace occ::io
