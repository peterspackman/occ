#include <occ/io/structure_format.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <fstream>
#include <stdexcept>

namespace occ::io {

using nlohmann::json;

// --- SiteMultipoles ---

int SiteMultipoles::max_rank() const {
    if (!hexadecapole.empty()) return 4;
    if (!octupole.empty()) return 3;
    if (!quadrupole.empty()) return 2;
    if (!dipole.empty()) return 1;
    return 0;
}

std::vector<double> SiteMultipoles::to_flat() const {
    int rank = max_rank();
    int n = (rank + 1) * (rank + 1);
    std::vector<double> flat(n, 0.0);
    flat[0] = charge;
    if (rank >= 1 && !dipole.empty()) {
        for (int i = 0; i < 3; ++i) flat[1 + i] = dipole[i];
    }
    if (rank >= 2 && !quadrupole.empty()) {
        for (int i = 0; i < 5; ++i) flat[4 + i] = quadrupole[i];
    }
    if (rank >= 3 && !octupole.empty()) {
        for (int i = 0; i < 7; ++i) flat[9 + i] = octupole[i];
    }
    if (rank >= 4 && !hexadecapole.empty()) {
        for (int i = 0; i < 9; ++i) flat[16 + i] = hexadecapole[i];
    }
    return flat;
}

SiteMultipoles SiteMultipoles::from_flat(const std::vector<double> &flat) {
    SiteMultipoles m;
    if (flat.empty()) return m;
    m.charge = flat[0];
    int n = static_cast<int>(flat.size());
    // Determine rank from flat size: n = (rank+1)^2
    int rank = 0;
    if (n >= 25) rank = 4;
    else if (n >= 16) rank = 3;
    else if (n >= 9) rank = 2;
    else if (n >= 4) rank = 1;

    if (rank >= 1) {
        m.dipole.assign(flat.begin() + 1, flat.begin() + 4);
    }
    if (rank >= 2) {
        m.quadrupole.assign(flat.begin() + 4, flat.begin() + 9);
    }
    if (rank >= 3) {
        m.octupole.assign(flat.begin() + 9, flat.begin() + 16);
    }
    if (rank >= 4) {
        m.hexadecapole.assign(flat.begin() + 16, flat.begin() + 25);
    }
    return m;
}

// --- JSON serialization ---

void to_json(json &j, const SiteMultipoles &m) {
    j["charge"] = m.charge;
    if (!m.dipole.empty()) j["dipole"] = m.dipole;
    if (!m.quadrupole.empty()) j["quadrupole"] = m.quadrupole;
    if (!m.octupole.empty()) j["octupole"] = m.octupole;
    if (!m.hexadecapole.empty()) j["hexadecapole"] = m.hexadecapole;
}

void from_json(const json &j, SiteMultipoles &m) {
    m.charge = j.value("charge", 0.0);
    if (j.contains("dipole"))
        m.dipole = j.at("dipole").get<std::vector<double>>();
    if (j.contains("quadrupole"))
        m.quadrupole = j.at("quadrupole").get<std::vector<double>>();
    if (j.contains("octupole"))
        m.octupole = j.at("octupole").get<std::vector<double>>();
    if (j.contains("hexadecapole"))
        m.hexadecapole = j.at("hexadecapole").get<std::vector<double>>();
}

void to_json(json &j, const MoleculeSite &s) {
    if (!s.label.empty()) j["label"] = s.label;
    j["element"] = s.element;
    if (!s.type.empty() && s.type != s.element) j["type"] = s.type;
    j["position"] = s.position;
    if (s.multipoles.max_rank() > 0 ||
        std::abs(s.multipoles.charge) > 0.0) {
        j["multipoles"] = s.multipoles;
    }
}

void from_json(const json &j, MoleculeSite &s) {
    s.label = j.value("label", "");
    s.element = j.at("element").get<std::string>();
    s.type = j.value("type", s.element);
    auto pos = j.at("position");
    for (int i = 0; i < 3; ++i) s.position[i] = pos[i].get<double>();
    if (j.contains("multipoles"))
        s.multipoles = j.at("multipoles").get<SiteMultipoles>();
}

void to_json(json &j, const MoleculeType &mt) {
    j["name"] = mt.name;
    j["sites"] = mt.sites;
}

void from_json(const json &j, MoleculeType &mt) {
    mt.name = j.at("name").get<std::string>();
    mt.sites = j.at("sites").get<std::vector<MoleculeSite>>();
}

void to_json(json &j, const IndependentMolecule &im) {
    j["type"] = im.type;
    j["translation"] = im.translation;
    bool has_orientation = false;
    for (double v : im.orientation) {
        if (std::abs(v) > 0.0) { has_orientation = true; break; }
    }
    if (has_orientation) j["orientation"] = im.orientation;
    if (im.parity != 1) j["parity"] = im.parity;
}

void from_json(const json &j, IndependentMolecule &im) {
    im.type = j.at("type").get<std::string>();
    auto t = j.at("translation");
    for (int i = 0; i < 3; ++i) im.translation[i] = t[i].get<double>();
    if (j.contains("orientation")) {
        auto o = j.at("orientation");
        for (int i = 0; i < 3; ++i) im.orientation[i] = o[i].get<double>();
    } else {
        im.orientation = {0.0, 0.0, 0.0};
    }
    im.parity = j.value("parity", 1);
}

void to_json(json &j, const BuckinghamPair &bp) {
    j["types"] = bp.types;
    if (!bp.elements[0].empty()) j["elements"] = bp.elements;
    j["A"] = bp.A;
    j["rho"] = bp.rho;
    j["C6"] = bp.C6;
}

void from_json(const json &j, BuckinghamPair &bp) {
    bp.types = j.at("types").get<std::array<std::string, 2>>();
    if (j.contains("elements"))
        bp.elements = j.at("elements").get<std::array<std::string, 2>>();
    bp.A = j.at("A").get<double>();
    bp.rho = j.at("rho").get<double>();
    bp.C6 = j.at("C6").get<double>();
}

void to_json(json &j, const Potentials &p) {
    j["cutoff"] = p.cutoff;
    if (!p.buckingham.empty()) j["buckingham"] = p.buckingham;
}

void from_json(const json &j, Potentials &p) {
    p.cutoff = j.value("cutoff", 15.0);
    if (j.contains("buckingham"))
        p.buckingham = j.at("buckingham").get<std::vector<BuckinghamPair>>();
}

void to_json(json &j, const Settings &s) {
    j["ewald_accuracy"] = s.ewald_accuracy;
    j["use_ewald"] = s.use_ewald;
    j["max_interaction_order"] = s.max_interaction_order;
    j["max_iterations"] = s.max_iterations;
    j["gradient_tolerance"] = s.gradient_tolerance;
    j["neighbor_radius"] = s.neighbor_radius;
    if (s.spline_min > 0.0) {
        j["spline_min"] = s.spline_min;
        j["spline_max"] = s.spline_max;
        j["spline_order"] = s.spline_order;
    }
    if (std::abs(s.pressure_gpa) > 1e-16) {
        j["pressure_gpa"] = s.pressure_gpa;
    }
}

void from_json(const json &j, Settings &s) {
    s.ewald_accuracy = j.value("ewald_accuracy", 1e-8);
    s.use_ewald = j.value("use_ewald", true);
    s.max_interaction_order = j.value("max_interaction_order", 4);
    s.max_iterations = j.value("max_iterations", 200);
    s.gradient_tolerance = j.value("gradient_tolerance", 1e-5);
    s.neighbor_radius = j.value("neighbor_radius", 20.0);
    s.spline_min = j.value("spline_min", 0.0);
    s.spline_max = j.value("spline_max", 0.0);
    s.spline_order = j.value("spline_order", 3);
    s.pressure_gpa = j.value("pressure_gpa", 0.0);
}

void to_json(json &j, const ReferenceEnergies &r) {
    j["total"] = r.total;
    if (!r.components.empty()) j["components"] = r.components;
}

void from_json(const json &j, ReferenceEnergies &r) {
    r.total = j.value("total", 0.0);
    if (j.contains("components"))
        r.components = j.at("components").get<std::map<std::string, double>>();
}

// --- Basis ---

void to_json(json &j, const Basis &b) {
    j["multipole_convention"] = "spherical_gdma";
    j["molecule_types"] = b.molecule_types;
    if (!b.potentials.buckingham.empty()) {
        j["potentials"] = b.potentials;
    }
    j["settings"] = b.settings;
}

void from_json(const json &j, Basis &b) {
    // Read and validate multipole convention if present
    if (j.contains("multipole_convention")) {
        std::string convention = j.at("multipole_convention").get<std::string>();
        if (convention != "spherical_gdma") {
            throw std::runtime_error(
                "Unsupported multipole convention: '" + convention +
                "' (expected 'spherical_gdma')");
        }
    }
    if (j.contains("molecule_types"))
        b.molecule_types =
            j.at("molecule_types").get<std::vector<MoleculeType>>();
    if (j.contains("potentials"))
        b.potentials = j.at("potentials").get<Potentials>();
    if (j.contains("settings"))
        b.settings = j.at("settings").get<Settings>();
}

// --- CrystalData ---

void to_json(json &j, const CrystalData &c) {
    j["cell"] = {{"a", c.a},     {"b", c.b},    {"c", c.c},
                 {"alpha", c.alpha}, {"beta", c.beta}, {"gamma", c.gamma}};
    if (!c.space_group.empty()) {
        j["space_group"] = c.space_group;
    }
    j["molecules"] = c.molecules;
}

void from_json(const json &j, CrystalData &c) {
    if (j.contains("cell")) {
        const auto &cell = j.at("cell");
        c.a = cell.at("a").get<double>();
        c.b = cell.at("b").get<double>();
        c.c = cell.at("c").get<double>();
        c.alpha = cell.value("alpha", 90.0);
        c.beta = cell.value("beta", 90.0);
        c.gamma = cell.value("gamma", 90.0);
    }
    c.space_group = j.value("space_group", "");
    if (j.contains("molecules"))
        c.molecules =
            j.at("molecules").get<std::vector<IndependentMolecule>>();
}

// --- StructureInput ---

void to_json(json &j, const StructureInput &si) {
    j["title"] = si.title;
    j["basis"] = si.basis;
    if (si.has_crystal()) {
        j["crystal"] = si.crystal;
    }
    if (std::abs(si.reference.total) > 0.0 ||
        !si.reference.components.empty()) {
        j["reference"] = si.reference;
    }
}

void from_json(const json &j, StructureInput &si) {
    si.title = j.value("title", "");

    if (j.contains("basis")) {
        // New nested format
        si.basis = j.at("basis").get<Basis>();
    } else if (j.contains("molecule_types")) {
        // Legacy flat format — migrate
        si.basis = j.get<Basis>();
    }

    if (j.contains("crystal")) {
        si.crystal = j.at("crystal").get<CrystalData>();
    } else {
        // Legacy flat format — migrate cell + molecules
        if (j.contains("cell")) {
            const auto &cell = j.at("cell");
            si.crystal.a = cell.at("a").get<double>();
            si.crystal.b = cell.at("b").get<double>();
            si.crystal.c = cell.at("c").get<double>();
            si.crystal.alpha = cell.value("alpha", 90.0);
            si.crystal.beta = cell.value("beta", 90.0);
            si.crystal.gamma = cell.value("gamma", 90.0);
        }
        si.crystal.space_group = j.value("space_group", "");
        if (j.contains("molecules"))
            si.crystal.molecules =
                j.at("molecules").get<std::vector<IndependentMolecule>>();
    }

    if (j.contains("reference"))
        si.reference = j.at("reference").get<ReferenceEnergies>();
}

// --- File I/O ---

StructureInput read_structure_json(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open structure JSON file: " + path);
    }
    json j;
    file >> j;
    return j.get<StructureInput>();
}

void write_structure_json(const std::string &path, const StructureInput &input) {
    json j = input;
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    file << j.dump(2) << "\n";
}

void write_basis_json(const std::string &path, const Basis &basis,
                      const std::string &title) {
    json j;
    if (!title.empty()) j["title"] = title;
    j["basis"] = basis;
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    file << j.dump(2) << "\n";
}

// --- Format detection ---

bool is_structure_format(const std::string &json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) return false;
    json j;
    file >> j;
    // New nested format or legacy flat format
    return j.contains("basis") || j.contains("molecule_types");
}

} // namespace occ::io
