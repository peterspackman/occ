#include <occ/mults/dmacrys_input.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/asymmetric_unit.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/crystal/unitcell.h>
#include <occ/dma/mult.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <Eigen/SVD>

namespace occ::mults {

using nlohmann::json;

/// Compute mass-weighted center of mass for a set of positions.
/// \param positions  3 x N matrix of Cartesian positions
/// \param mol_data   molecule data (for atomic numbers → masses)
static Vec3 mass_weighted_com(const Mat3N &positions,
                              const DmacrysInput::MoleculeSites &mol_data) {
    int n = static_cast<int>(positions.cols());
    Vec3 com = Vec3::Zero();
    double total_mass = 0.0;
    for (int i = 0; i < n; ++i) {
        double m = occ::core::Element(mol_data.sites[i].atomic_number).mass();
        com += m * positions.col(i);
        total_mass += m;
    }
    return com / total_mass;
}

// --- JSON reader ---

static DmacrysInput::Reference parse_reference(const json &j) {
    DmacrysInput::Reference ref;
    if (j.contains("total_kJ_per_mol"))
        ref.total_kJ_per_mol = j.at("total_kJ_per_mol").get<double>();
    if (j.contains("total_eV_per_cell"))
        ref.total_eV_per_cell = j.at("total_eV_per_cell").get<double>();
    if (j.contains("repulsion_dispersion_eV"))
        ref.repulsion_dispersion_eV = j.at("repulsion_dispersion_eV").get<double>();
    if (j.contains("repulsion_dispersion_kJ"))
        ref.repulsion_dispersion_kJ = j.at("repulsion_dispersion_kJ").get<double>();
    if (j.contains("charge_charge_inter_eV"))
        ref.charge_charge_inter_eV = j.at("charge_charge_inter_eV").get<double>();
    if (j.contains("charge_dipole_eV"))
        ref.charge_dipole_eV = j.at("charge_dipole_eV").get<double>();
    if (j.contains("dipole_dipole_eV"))
        ref.dipole_dipole_eV = j.at("dipole_dipole_eV").get<double>();
    if (j.contains("higher_multipole_eV"))
        ref.higher_multipole_eV = j.at("higher_multipole_eV").get<double>();
    if (j.contains("strain_derivatives_eV")) {
        ref.strain_derivatives_eV =
            j.at("strain_derivatives_eV").get<std::vector<double>>();
        ref.has_strain_derivatives = (ref.strain_derivatives_eV.size() == 6);
    }
    if (j.contains("elastic_constants_GPa")) {
        const auto &ec = j.at("elastic_constants_GPa");
        ref.elastic_constants_GPa = Mat6::Zero();
        // Read upper triangle in Voigt notation: C11..C66
        const char* keys[] = {
            "C11","C12","C13","C14","C15","C16",
                  "C22","C23","C24","C25","C26",
                        "C33","C34","C35","C36",
                              "C44","C45","C46",
                                    "C55","C56",
                                          "C66"
        };
        int k = 0;
        for (int i = 0; i < 6; ++i) {
            for (int jj = i; jj < 6; ++jj) {
                if (ec.contains(keys[k])) {
                    double val = ec.at(keys[k]).get<double>();
                    ref.elastic_constants_GPa(i, jj) = val;
                    ref.elastic_constants_GPa(jj, i) = val;
                }
                ++k;
            }
        }
        ref.has_elastic_constants = true;
    }
    return ref;
}

DmacrysInput read_dmacrys_json(const std::string &json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open DMACRYS JSON file: " + json_path);
    }

    json j;
    file >> j;

    DmacrysInput input;
    input.title = j.at("title").get<std::string>();
    input.source = j.at("source").get<std::string>();

    // Crystal data
    const auto &jc = j.at("crystal");
    const auto &cell = jc.at("cell");
    input.crystal.a = cell.at("a").get<double>();
    input.crystal.b = cell.at("b").get<double>();
    input.crystal.c = cell.at("c").get<double>();
    input.crystal.alpha = cell.at("alpha").get<double>();
    input.crystal.beta = cell.at("beta").get<double>();
    input.crystal.gamma = cell.at("gamma").get<double>();
    input.crystal.space_group = jc.at("space_group").get<std::string>();
    input.crystal.Z = jc.at("Z").get<int>();

    for (const auto &ja : jc.at("atoms")) {
        DmacrysInput::CrystalData::Atom atom;
        atom.label = ja.at("label").get<std::string>();
        atom.element = ja.at("element").get<std::string>();
        auto xyz = ja.at("frac_xyz");
        atom.frac_xyz = Vec3(xyz[0].get<double>(), xyz[1].get<double>(),
                             xyz[2].get<double>());
        input.crystal.atoms.push_back(std::move(atom));
    }

    for (const auto &js : jc.at("symmetry_operations")) {
        input.crystal.symops.push_back(js.get<std::string>());
    }

    // Molecule sites
    const auto &jm = j.at("molecule").at("sites");
    for (const auto &js : jm) {
        DmacrysInput::MoleculeSites::Site site;
        site.label = js.at("label").get<std::string>();
        site.element = js.at("element").get<std::string>();
        site.atomic_number = js.at("atomic_number").get<int>();
        auto pos = js.at("position_bohr");
        site.position_bohr =
            Vec3(pos[0].get<double>(), pos[1].get<double>(),
                 pos[2].get<double>());
        const auto &mp = js.at("multipoles");
        site.rank = mp.at("rank").get<int>();
        site.components = mp.at("components").get<std::vector<double>>();
        input.molecule.sites.push_back(std::move(site));
    }

    // Potentials
    const auto &jp = j.at("potentials").at("pairs");
    for (const auto &jpair : jp) {
        DmacrysInput::BuckPair bp;
        bp.el1 = jpair.at("el1").get<std::string>();
        bp.el2 = jpair.at("el2").get<std::string>();
        bp.A_eV = jpair.at("A_eV").get<double>();
        bp.rho_ang = jpair.at("rho_ang").get<double>();
        bp.C6_eV_ang6 = jpair.at("C6_eV_ang6").get<double>();
        input.potentials.push_back(std::move(bp));
    }

    // Settings
    if (j.contains("settings")) {
        const auto &jset = j.at("settings");
        if (jset.contains("repulsion_dispersion_cutoff_angstrom"))
            input.cutoff_radius =
                jset.at("repulsion_dispersion_cutoff_angstrom").get<double>();
    }

    // Reference energies
    if (j.contains("reference")) {
        const auto &jref = j.at("reference");
        if (jref.contains("initial"))
            input.initial_ref = parse_reference(jref.at("initial"));
        if (jref.contains("optimized"))
            input.optimized_ref = parse_reference(jref.at("optimized"));
    }

    return input;
}

// --- Crystal builder ---

/// Normalize a SHELX-style symop string to standard triplet format.
/// e.g., "+X +1/2, -Y +1/2, -Z" -> "x+1/2,-y+1/2,-z"
static std::string normalize_symop(const std::string &raw) {
    std::string result;
    result.reserve(raw.size());
    for (char c : raw) {
        if (c == ' ')
            continue; // strip all spaces
        result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

crystal::Crystal build_crystal(const DmacrysInput::CrystalData &data) {
    // Build unit cell (angles in JSON are degrees, UnitCell expects radians)
    crystal::UnitCell unit_cell(data.a, data.b, data.c,
                                units::radians(data.alpha),
                                units::radians(data.beta),
                                units::radians(data.gamma));

    // Build space group - try by name first, then from normalized symops.
    crystal::SpaceGroup space_group;
    bool found = false;
    if (!data.space_group.empty()) {
        space_group = crystal::SpaceGroup(data.space_group);
        if (!space_group.symbol().empty() && space_group.symbol() != "XX") {
            found = true;
            occ::log::info("Space group from name: {} -> {}",
                           data.space_group, space_group.symbol());
        }
    }
    if (!found && !data.symops.empty()) {
        // Build SymmetryOperation objects; this constructor uses
        // gemmi::split_centering_vectors for reliable space group lookup.
        std::vector<crystal::SymmetryOperation> symop_objs;
        symop_objs.emplace_back("x,y,z");
        for (const auto &op : data.symops) {
            std::string norm = normalize_symop(op);
            occ::log::debug("  symop: {}", norm);
            symop_objs.emplace_back(norm);
        }
        occ::log::info("Building space group from {} symops",
                       symop_objs.size());
        space_group = crystal::SpaceGroup(symop_objs);
    }
    occ::log::info("Space group: {}", space_group.symbol());

    // Build asymmetric unit
    int n = static_cast<int>(data.atoms.size());
    Mat3N positions(3, n);
    IVec atomic_numbers(n);
    std::vector<std::string> labels;
    labels.reserve(n);

    for (int i = 0; i < n; ++i) {
        const auto &atom = data.atoms[i];
        positions.col(i) = atom.frac_xyz;
        occ::core::Element el(atom.element);
        atomic_numbers(i) = el.atomic_number();
        labels.push_back(atom.label);
    }

    crystal::AsymmetricUnit asym(positions, atomic_numbers, labels);

    return crystal::Crystal(asym, space_group, unit_cell);
}

// --- Kabsch alignment ---

/// Compute optimal rotation R mapping body-frame positions P to crystal-frame Q.
/// Both P and Q must be centered (zero mean).
static Mat3 kabsch_rotation(const Mat3N &P, const Mat3N &Q) {
    // H = P * Q^T
    Mat3 H = P * Q.transpose();

    Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 V = svd.matrixV();
    Mat3 U = svd.matrixU();

    // Ensure proper rotation (det = +1, not reflection)
    double d = (V * U.transpose()).determinant();
    Mat3 D = Mat3::Identity();
    D(2, 2) = (d > 0) ? 1.0 : -1.0;

    return V * D * U.transpose();
}

// --- MultipoleSource builder ---

/// Build body-frame BodySite objects from DMA data.
static std::pair<std::vector<MultipoleSource::BodySite>, Mat3N>
build_body_sites(const DmacrysInput::MoleculeSites &mol_data) {
    int n_sites = static_cast<int>(mol_data.sites.size());
    std::vector<MultipoleSource::BodySite> body_sites;
    body_sites.reserve(n_sites);
    Mat3N body_pos_ang(3, n_sites);

    for (int i = 0; i < n_sites; ++i) {
        const auto &site = mol_data.sites[i];

        Vec3 pos_ang = site.position_bohr * units::BOHR_TO_ANGSTROM;
        body_pos_ang.col(i) = pos_ang;

        occ::dma::Mult mult(site.rank);
        int expected = (site.rank + 1) * (site.rank + 1);
        int actual = static_cast<int>(site.components.size());
        if (actual != expected) {
            throw std::runtime_error(
                fmt::format("Site {} has {} multipole components, expected {}",
                            site.label, actual, expected));
        }
        for (int j = 0; j < expected; ++j) {
            mult.q(j) = site.components[j];
        }

        MultipoleSource::BodySite bs;
        bs.multipole = std::move(mult);
        body_sites.push_back(std::move(bs));
    }

    Vec3 body_com = mass_weighted_com(body_pos_ang, mol_data);
    for (int i = 0; i < n_sites; ++i) {
        body_sites[i].offset = body_pos_ang.col(i) - body_com;
    }

    return {std::move(body_sites), std::move(body_pos_ang)};
}

std::vector<MultipoleSource> build_multipole_sources(
    const DmacrysInput &input, const crystal::Crystal &crystal) {

    int n_sites = static_cast<int>(input.molecule.sites.size());
    auto [body_sites, body_pos_ang] = build_body_sites(input.molecule);

    Vec3 body_com = mass_weighted_com(body_pos_ang, input.molecule);

    // Center body-frame positions at mass-weighted COM
    Mat3N body_centered(3, n_sites);
    for (int i = 0; i < n_sites; ++i) {
        body_centered.col(i) = body_pos_ang.col(i) - body_com;
    }

    // Build fractional coordinates for the asymmetric unit atoms
    Mat3N frac_asym(3, n_sites);
    for (int i = 0; i < n_sites; ++i) {
        frac_asym.col(i) = input.crystal.atoms[i].frac_xyz;
    }

    // Generate one MultipoleSource per symmetry image in the cell.
    // Each symmetry operation produces a distinct molecular copy.
    const auto &symops = crystal.space_group().symmetry_operations();
    int n_mol = static_cast<int>(symops.size());

    occ::log::info("Building multipole sources: {} molecules in cell "
                   "({} symops x {} sites)",
                   n_mol, n_mol, n_sites);

    std::vector<MultipoleSource> sources;
    sources.reserve(n_mol);

    for (int m = 0; m < n_mol; ++m) {
        // Apply symmetry operation to asymmetric unit fractional coords
        Mat3N frac_image = symops[m].apply(frac_asym);

        // Convert to Cartesian
        Mat3N cart_image = crystal.unit_cell().to_cartesian(frac_image);
        Vec3 crystal_com = mass_weighted_com(cart_image, input.molecule);

        // Center crystal-frame positions at mass-weighted COM
        Mat3N crystal_centered(3, n_sites);
        for (int i = 0; i < n_sites; ++i) {
            crystal_centered.col(i) = cart_image.col(i) - crystal_com;
        }

        // Kabsch alignment: find R such that R * body_centered ≈ crystal_centered
        Mat3 R = kabsch_rotation(body_centered, crystal_centered);

        // Verify alignment quality
        double rmsd = 0.0;
        for (int i = 0; i < n_sites; ++i) {
            Vec3 diff = R * body_centered.col(i) - crystal_centered.col(i);
            rmsd += diff.squaredNorm();
        }
        rmsd = std::sqrt(rmsd / n_sites);
        occ::log::debug("Molecule {} (symop {}): Kabsch RMSD = {:.6f} Angstrom",
                        m, symops[m].to_string(), rmsd);
        if (rmsd > 0.01) {
            occ::log::warn("Large alignment RMSD for molecule {} ({:.6f} A) - "
                           "check atom ordering",
                           m, rmsd);
        }

        sources.emplace_back(body_sites);
        sources.back().set_orientation(R, crystal_com);
    }

    return sources;
}

// --- CrystalEnergy setup ---

void setup_crystal_energy_from_dmacrys(
    CrystalEnergy &calc,
    const DmacrysInput &input,
    const crystal::Crystal &crystal,
    const std::vector<MultipoleSource> &multipoles,
    bool build_neighbors) {

    int n_mol = static_cast<int>(multipoles.size());
    int n_sites = static_cast<int>(input.molecule.sites.size());

    // Build molecule geometry (atomic numbers + body-frame positions)
    std::vector<CrystalEnergy::MoleculeGeometry> geom_vec;
    geom_vec.reserve(n_mol);

    // All molecules share the same body-frame geometry (same molecule type)
    CrystalEnergy::MoleculeGeometry base_geom;
    for (int i = 0; i < n_sites; ++i) {
        const auto &site = input.molecule.sites[i];
        base_geom.atomic_numbers.push_back(site.atomic_number);

        // Body-frame atom position in Angstrom, relative to COM
        Vec3 pos_ang = site.position_bohr * units::BOHR_TO_ANGSTROM;
        base_geom.atom_positions.push_back(pos_ang);
    }
    // Center relative to mass-weighted COM (consistent with build_multipole_sources
    // and DMACRYS COFMAS convention)
    {
        Vec3 mw_com = Vec3::Zero();
        double total_mass = 0.0;
        for (int i = 0; i < n_sites; ++i) {
            double m = occ::core::Element(
                input.molecule.sites[i].atomic_number).mass();
            mw_com += m * base_geom.atom_positions[i];
            total_mass += m;
        }
        mw_com /= total_mass;
        for (auto &p : base_geom.atom_positions)
            p -= mw_com;
    }

    // Build initial states and geometry for each molecule
    std::vector<MoleculeState> states;
    std::vector<Vec3> mol_coms;
    states.reserve(n_mol);
    mol_coms.reserve(n_mol);

    const auto &symops = crystal.space_group().symmetry_operations();

    // Build fractional coordinates for asymmetric unit
    Mat3N frac_asym(3, n_sites);
    for (int i = 0; i < n_sites; ++i) {
        frac_asym.col(i) = input.crystal.atoms[i].frac_xyz;
    }

    for (int m = 0; m < n_mol; ++m) {
        // Get crystal-frame COM for this molecule
        Mat3N frac_image = symops[m].apply(frac_asym);
        Mat3N cart_image = crystal.unit_cell().to_cartesian(frac_image);
        Vec3 com = mass_weighted_com(cart_image, input.molecule);

        CrystalEnergy::MoleculeGeometry mol_geom = base_geom;
        mol_geom.center_of_mass = com;
        geom_vec.push_back(std::move(mol_geom));

        // Initial state: position = COM, rotation from MultipoleSource
        const auto &src = multipoles[m];
        states.push_back(MoleculeState::from_rotation(com, src.rotation()));
        mol_coms.push_back(com);
    }

    calc.set_molecule_geometry(std::move(geom_vec));
    calc.set_initial_states(std::move(states));
    // Build atom-based neighbor list (needed for Buckingham short-range).
    // COM distances are stored per pair and used as a COM gate for
    // electrostatics, matching DMACRYS TBLCNT behavior.
    // Skip when caller will override via set_neighbor_list().
    if (build_neighbors) {
        calc.build_neighbor_list_from_positions(mol_coms);
    }
}

// --- Buckingham converter ---

static constexpr double EV_TO_KJ_PER_MOL = 96.4853329;

std::map<std::pair<int, int>, BuckinghamParams>
convert_buckingham_params(const std::vector<DmacrysInput::BuckPair> &pairs) {
    std::map<std::pair<int, int>, BuckinghamParams> result;

    for (const auto &p : pairs) {
        occ::core::Element el1(p.el1);
        occ::core::Element el2(p.el2);
        int z1 = el1.atomic_number();
        int z2 = el2.atomic_number();

        // Canonical ordering: smaller Z first
        if (z1 > z2)
            std::swap(z1, z2);

        BuckinghamParams bp;
        bp.A = p.A_eV * EV_TO_KJ_PER_MOL;    // eV -> kJ/mol
        bp.B = 1.0 / p.rho_ang;                // rho (Angstrom) -> B (Angstrom^-1)
        bp.C = p.C6_eV_ang6 * EV_TO_KJ_PER_MOL; // eV*Ang^6 -> kJ/mol*Ang^6

        auto key = std::make_pair(z1, z2);
        if (result.count(key)) {
            occ::log::warn("Duplicate Buckingham params for Z={}-Z={}, "
                           "overwriting",
                           z1, z2);
        }
        result[key] = bp;

        occ::log::info("Buckingham {}-{} (Z={}-{}): A={:.2f} kJ/mol, "
                       "B={:.4f} Ang^-1, C={:.2f} kJ/mol*Ang^6",
                       p.el1, p.el2, z1, z2, bp.A, bp.B, bp.C);
    }

    return result;
}

} // namespace occ::mults
