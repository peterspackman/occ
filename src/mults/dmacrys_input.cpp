#include <occ/mults/dmacrys_input.h>
#include <occ/mults/dmacrys_type_codes.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/molecular_symmetry.h>
#include <occ/core/units.h>
#include <occ/crystal/asymmetric_unit.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/crystal/unitcell.h>
#include <occ/dma/mult.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <Eigen/SVD>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

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

// Type code management functions (normalize_type_key, infer_site_atom_type_from_label,
// infer_atomic_number_from_type, DmacrysTypeCodeTables, build_dmacrys_type_code_tables)
// moved to dmacrys_type_codes.h/cpp.

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
    if (j.contains("charge_charge_ewald_summed_eV"))
        ref.charge_charge_ewald_summed_eV =
            j.at("charge_charge_ewald_summed_eV").get<double>();
    if (j.contains("charge_charge_intra_eV"))
        ref.charge_charge_intra_eV =
            j.at("charge_charge_intra_eV").get<double>();
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

static DmacrysInput::CrystalData parse_crystal_data(const json &jc) {
    DmacrysInput::CrystalData crystal_data;
    const auto &cell = jc.at("cell");
    crystal_data.a = cell.at("a").get<double>();
    crystal_data.b = cell.at("b").get<double>();
    crystal_data.c = cell.at("c").get<double>();
    crystal_data.alpha = cell.at("alpha").get<double>();
    crystal_data.beta = cell.at("beta").get<double>();
    crystal_data.gamma = cell.at("gamma").get<double>();
    crystal_data.space_group = jc.value("space_group", "");
    crystal_data.Z = jc.value("Z", 1);

    for (const auto &ja : jc.at("atoms")) {
        DmacrysInput::CrystalData::Atom atom;
        atom.label = ja.at("label").get<std::string>();
        atom.element = ja.at("element").get<std::string>();
        auto xyz = ja.at("frac_xyz");
        atom.frac_xyz = Vec3(xyz[0].get<double>(), xyz[1].get<double>(),
                             xyz[2].get<double>());
        crystal_data.atoms.push_back(std::move(atom));
    }

    if (jc.contains("symmetry_operations")) {
        for (const auto &js : jc.at("symmetry_operations")) {
            crystal_data.symops.push_back(js.get<std::string>());
        }
    }
    return crystal_data;
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
    input.crystal = parse_crystal_data(jc);

    // Molecule sites
    const auto &jm = j.at("molecule").at("sites");
    for (const auto &js : jm) {
        DmacrysInput::MoleculeSites::Site site;
        site.label = js.at("label").get<std::string>();
        site.element = js.at("element").get<std::string>();
        site.atom_type = js.value("atom_type",
                                  infer_site_atom_type_from_label(site.label));
        site.atomic_number = js.at("atomic_number").get<int>();
        auto pos = js.at("position_bohr");
        site.position_bohr =
            Vec3(pos[0].get<double>(), pos[1].get<double>(),
                 pos[2].get<double>());
        const auto &mp = js.at("multipoles");
        site.rank = mp.at("rank").get<int>();
        site.components = mp.at("components").get<std::vector<double>>();
        if (js.contains("aniso_axis_body")) {
            auto ax = js.at("aniso_axis_body");
            site.aniso_axis_body = Vec3(ax[0].get<double>(),
                                        ax[1].get<double>(),
                                        ax[2].get<double>());
        }
        input.molecule.sites.push_back(std::move(site));
    }

    // Potentials
    const auto &jp = j.at("potentials").at("pairs");
    for (const auto &jpair : jp) {
        DmacrysInput::BuckPair bp;
        bp.type1 = jpair.value("type1", "");
        bp.type2 = jpair.value("type2", "");
        bp.el1 = jpair.at("el1").get<std::string>();
        bp.el2 = jpair.at("el2").get<std::string>();
        bp.kind = jpair.value("kind", "BUCK");
        bp.A_eV = jpair.at("A_eV").get<double>();
        bp.rho_ang = jpair.at("rho_ang").get<double>();
        bp.C6_eV_ang6 = jpair.at("C6_eV_ang6").get<double>();
        input.potentials.push_back(std::move(bp));
    }

    // Anisotropic potentials
    if (j.at("potentials").contains("aniso_pairs")) {
        for (const auto &ja : j.at("potentials").at("aniso_pairs")) {
            DmacrysInput::AnisoPair ap;
            ap.type1 = ja.at("type1").get<std::string>();
            ap.type2 = ja.at("type2").get<std::string>();
            ap.alpha = 1.0 / ja.at("alpha_inv_ang").get<double>();  // 1/α_inv → α
            ap.rho_00 = ja.at("rho_00").get<double>();
            ap.rho_20 = ja.at("rho_20").get<double>();
            ap.rho_02 = ja.at("rho_02").get<double>();
            input.aniso_potentials.push_back(std::move(ap));
        }
    }

    // Settings
    if (j.contains("settings")) {
        const auto &jset = j.at("settings");
        if (jset.contains("cutoff")) {
            const auto& jcut = jset.at("cutoff");
            if (jcut.contains("potential")) {
                input.cutoff_radius = jcut.at("potential").get<double>();
            }
        }
        if (jset.contains("repulsion_dispersion_cutoff_angstrom"))
            input.cutoff_radius =
                jset.at("repulsion_dispersion_cutoff_angstrom").get<double>();
        if (jset.contains("spline")) {
            const auto& jspl = jset.at("spline");
            if (jspl.contains("min") && jspl.contains("max")) {
                input.spline_min = jspl.at("min").get<double>();
                input.spline_max = jspl.at("max").get<double>();
                input.has_spline = true;
            }
        }
        if (jset.contains("ewald")) {
            const auto& jew = jset.at("ewald");
            if (jew.contains("accuracy")) {
                input.ewald_accuracy = jew.at("accuracy").get<double>();
                input.has_ewald_accuracy = true;
            }
            if (jew.contains("eta")) {
                input.ewald_eta = jew.at("eta").get<double>();
                input.has_ewald_eta = std::abs(input.ewald_eta) > 0.0;
            }
            if (jew.contains("kmax")) {
                input.ewald_kmax = jew.at("kmax").get<int>();
                input.has_ewald_kmax = input.ewald_kmax > 0;
            }
        }
        if (jset.contains("ewald_accuracy")) {
            input.ewald_accuracy = jset.at("ewald_accuracy").get<double>();
            input.has_ewald_accuracy = true;
        }
        if (jset.contains("ewald_eta")) {
            input.ewald_eta = jset.at("ewald_eta").get<double>();
            input.has_ewald_eta = std::abs(input.ewald_eta) > 0.0;
        }
        if (jset.contains("ewald_kmax")) {
            input.ewald_kmax = jset.at("ewald_kmax").get<int>();
            input.has_ewald_kmax = input.ewald_kmax > 0;
        }
        if (jset.contains("pressure")) {
            const auto& jp = jset.at("pressure");
            double value = 0.0;
            std::string units = "Pa";
            if (jp.is_object()) {
                value = jp.value("value", 0.0);
                units = jp.value("units", "Pa");
            } else if (jp.is_number()) {
                value = jp.get<double>();
            }
            const auto u = normalize_type_key(units);
            if (u == "GPA") {
                value *= 1.0e9;
            } else if (u == "MPA") {
                value *= 1.0e6;
            } else if (u == "KPA") {
                value *= 1.0e3;
            }
            if (std::abs(value) > 0.0) {
                input.has_pressure = true;
                input.pressure_pa = value;
            }
        }
        if (jset.contains("pressure_pa")) {
            input.pressure_pa = jset.at("pressure_pa").get<double>();
            input.has_pressure = (std::abs(input.pressure_pa) > 0.0);
        }
    }

    // Reference energies
    if (j.contains("reference")) {
        const auto &jref = j.at("reference");
        if (jref.contains("initial"))
            input.initial_ref = parse_reference(jref.at("initial"));
        if (jref.contains("optimized")) {
            input.optimized_ref = parse_reference(jref.at("optimized"));
            if (jref.at("optimized").contains("crystal")) {
                input.optimized_crystal =
                    parse_crystal_data(jref.at("optimized").at("crystal"));
            }
        }
    }

    // Optional top-level alias for optimized fixed-point crystal geometry.
    if (j.contains("optimized_crystal")) {
        input.optimized_crystal = parse_crystal_data(j.at("optimized_crystal"));
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

/// Compute optimal orthogonal transform R mapping body-frame positions P to
/// crystal-frame Q. Both P and Q must be centered (zero mean).
/// When force_proper=true, restrict to SO(3); otherwise allow O(3) (improper).
static Mat3 kabsch_rotation(const Mat3N &P, const Mat3N &Q, bool force_proper = true) {
    // H = P * Q^T
    Mat3 H = P * Q.transpose();

    Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 V = svd.matrixV();
    Mat3 U = svd.matrixU();

    Mat3 R = V * U.transpose();
    if (!force_proper) {
        return R;
    }

    // Ensure proper rotation (det = +1, not reflection)
    double d = R.determinant();
    Mat3 D = Mat3::Identity();
    D(2, 2) = (d > 0) ? 1.0 : -1.0;
    return V * D * U.transpose();
}

// --- MultipoleSource builder ---

/// Build body-frame BodySite objects from DMA data.
static std::pair<std::vector<MultipoleSource::BodySite>, Mat3N>
build_body_sites(const DmacrysInput::MoleculeSites &mol_data,
                 const std::map<std::string, int>& type_codes) {
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
        bs.atomic_number = site.atomic_number;
        bs.aniso_axis = site.aniso_axis_body;
        const std::string atom_type = !site.atom_type.empty()
                                          ? site.atom_type
                                          : infer_site_atom_type_from_label(site.label);
        const auto key = normalize_type_key(atom_type);
        auto it_type = type_codes.find(key);
        if (it_type != type_codes.end()) {
            bs.short_range_type_code = it_type->second;
        }
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

    const int total_sites = static_cast<int>(input.molecule.sites.size());

    // Use Crystal's correctly assembled unit cell molecules.
    // This handles Z'<1 (molecules on symmetry centres) and Z'>1
    // co-crystal/multi-independent-molecule cases.
    const auto &uc_mols = crystal.unit_cell_molecules();
    const auto &asym_mols = crystal.symmetry_unique_molecules();
    const int n_mol = static_cast<int>(uc_mols.size());
    const int zprime = static_cast<int>(asym_mols.size());

    if (asym_mols.empty() || uc_mols.empty()) {
        throw std::runtime_error(
            "Crystal has no assembled molecules — cannot build multipole sources");
    }

    occ::log::info("Building multipole sources: {} UC molecules, {} total DMA sites, Z'={}",
                   n_mol, total_sites, zprime);

    const auto type_tables = build_dmacrys_type_code_tables(input);

    int sum_asym_atoms = 0;
    std::vector<int> asym_sizes;
    asym_sizes.reserve(zprime);
    for (const auto &am : asym_mols) {
        asym_sizes.push_back(am.size());
        sum_asym_atoms += am.size();
    }
    if (sum_asym_atoms != total_sites) {
        throw std::runtime_error(fmt::format(
            "DMACRYS site count ({}) does not match total asymmetric-molecule atom count ({})",
            total_sites, sum_asym_atoms));
    }

    struct BodyTemplate {
        std::vector<MultipoleSource::BodySite> ordered_body_sites;
        Mat3 R_base = Mat3::Identity();
        int n_sites = 0;
    };
    std::vector<BodyTemplate> templates;
    templates.reserve(zprime);

    int offset = 0;
    for (int ai = 0; ai < zprime; ++ai) {
        const auto &asym_mol = asym_mols[ai];
        const int n_sites = asym_mol.size();

        DmacrysInput::MoleculeSites subset;
        subset.sites.reserve(n_sites);
        for (int i = 0; i < n_sites; ++i) {
            subset.sites.push_back(input.molecule.sites[offset + i]);
        }
        offset += n_sites;

        auto [body_sites, body_pos_ang] =
            build_body_sites(subset, type_tables.type_to_code);
        Vec3 body_com = mass_weighted_com(body_pos_ang, subset);

        Vec3 asym_com = asym_mol.center_of_mass();
        Mat3N asym_centered(3, n_sites);
        for (int i = 0; i < n_sites; ++i) {
            asym_centered.col(i) = asym_mol.positions().col(i) - asym_com;
        }

        Mat3N body_centered(3, n_sites);
        for (int i = 0; i < n_sites; ++i) {
            body_centered.col(i) = body_pos_ang.col(i) - body_com;
        }

        // Allow O(3) fit: crystallographic mappings can be improper.
        Mat3 R_base = kabsch_rotation(body_centered, asym_centered, false);

        auto compute_rmsd = [&](const std::vector<MultipoleSource::BodySite>& sites,
                                const Mat3& R) {
            double rmsd = 0.0;
            for (int i = 0; i < n_sites; ++i) {
                const Vec3 d = R * sites[i].offset - asym_centered.col(i);
                rmsd += d.squaredNorm();
            }
            return std::sqrt(rmsd / std::max(1, n_sites));
        };

        IVec body_labels(n_sites);
        for (int i = 0; i < n_sites; ++i) {
            body_labels(i) = subset.sites[i].atomic_number;
        }

        auto match = occ::core::try_transformation_with_grouped_permutations(
            body_labels, body_pos_ang,
            asym_mol.atomic_numbers(), asym_mol.positions(),
            R_base.transpose(),
            0.25);

        std::vector<MultipoleSource::BodySite> ordered_body_sites = body_sites;
        double rmsd_best = compute_rmsd(ordered_body_sites, R_base);

        if (match.success) {
            std::vector<MultipoleSource::BodySite> candidate(n_sites);
            for (int i = 0; i < n_sites; ++i) {
                candidate[i] = body_sites[match.permutation[i]];
            }
            Mat3N body_reordered(3, n_sites);
            for (int i = 0; i < n_sites; ++i) {
                body_reordered.col(i) = candidate[i].offset;
            }
            const Mat3 R_candidate =
                kabsch_rotation(body_reordered, asym_centered, false);
            const double rmsd_candidate = compute_rmsd(candidate, R_candidate);

            // Prefer the input ordering unless permutation gives a clear
            // geometric improvement. This avoids unstable swaps among
            // chemically equivalent atoms (common for Z'>1 systems).
            if (rmsd_candidate + 1e-4 < rmsd_best) {
                ordered_body_sites = std::move(candidate);
                R_base = R_candidate;
                rmsd_best = rmsd_candidate;
                occ::log::debug("Asym molecule {}: accepted permuted mapping RMSD = {:.6e}",
                                ai, rmsd_best);
            } else {
                occ::log::debug("Asym molecule {}: kept input ordering RMSD = {:.6e} (perm {:.6e})",
                                ai, rmsd_best, rmsd_candidate);
            }
        } else {
            occ::log::warn("Asym molecule {}: could not match body-frame atoms to crystal — "
                           "assuming same ordering",
                           ai);
        }

        templates.push_back({std::move(ordered_body_sites), R_base, n_sites});
    }

    // Per UC molecule: compose crystallographic R_sym with asym-template Kabsch R_base
    std::vector<MultipoleSource> sources;
    sources.reserve(n_mol);

    for (int m = 0; m < n_mol; ++m) {
        const auto &uc_mol = uc_mols[m];
        const int asym_idx = uc_mol.asymmetric_molecule_idx();
        if (asym_idx < 0 || asym_idx >= static_cast<int>(templates.size())) {
            throw std::runtime_error(
                fmt::format("UC molecule {} has invalid asymmetric molecule index {}",
                            m, asym_idx));
        }
        const auto &templ = templates[asym_idx];
        if (uc_mol.size() != templ.n_sites) {
            throw std::runtime_error(
                fmt::format("UC molecule {} has {} atoms, template {} has {}",
                            m, uc_mol.size(), asym_idx, templ.n_sites));
        }

        auto [R_sym, t_sym] = uc_mol.asymmetric_unit_transformation();
        Mat3 R_total = R_sym * templ.R_base;
        Vec3 com = uc_mol.center_of_mass();

        // Verify alignment quality
        double rmsd = 0.0;
        for (int i = 0; i < templ.n_sites; ++i) {
            Vec3 rotated = R_total * templ.ordered_body_sites[i].offset;
            Vec3 crystal_offset = uc_mol.positions().col(i) - com;
            rmsd += (rotated - crystal_offset).squaredNorm();
        }
        rmsd = std::sqrt(rmsd / templ.n_sites);
        occ::log::debug("Molecule {}: alignment RMSD = {:.6f} Angstrom", m, rmsd);
        if (rmsd > 0.01) {
            occ::log::warn("Large alignment RMSD for molecule {} ({:.6f} A) — "
                           "check crystal structure",
                           m, rmsd);
        }

        sources.emplace_back(templ.ordered_body_sites);
        sources.back().set_orientation(R_total, com);
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
    const auto type_tables = build_dmacrys_type_code_tables(input);
    const auto typed_buck =
        convert_typed_buckingham_params(input.potentials, type_tables.type_to_code);
    const auto &uc_mols = crystal.unit_cell_molecules();
    if (static_cast<int>(uc_mols.size()) != n_mol) {
        throw std::runtime_error(fmt::format(
            "UC molecule count ({}) does not match multipole source count ({})",
            uc_mols.size(), n_mol));
    }

    // Build molecule geometry (atomic numbers + body-frame positions) per molecule.
    std::vector<CrystalEnergy::MoleculeGeometry> geom_vec;
    geom_vec.reserve(n_mol);

    // Build initial states and geometry for each molecule using Crystal's
    // correctly assembled UC molecules (handles Z'<1).
    //
    // Use an O(3) fit so improper crystallographic images are preserved.
    std::vector<MoleculeState> states;
    std::vector<Vec3> mol_coms;
    states.reserve(n_mol);
    mol_coms.reserve(n_mol);

    for (int m = 0; m < n_mol; ++m) {
        const auto &src = multipoles[m];
        const auto &body_sites = src.body_sites();
        const int n_sites = static_cast<int>(body_sites.size());
        if (uc_mols[m].size() != n_sites) {
            throw std::runtime_error(fmt::format(
                "Molecule {} size mismatch: UC has {} atoms, multipole source has {} sites",
                m, uc_mols[m].size(), n_sites));
        }

        const Vec3 com = uc_mols[m].center_of_mass();

        CrystalEnergy::MoleculeGeometry mol_geom;
        mol_geom.center_of_mass = com;
        mol_geom.atomic_numbers.reserve(n_sites);
        mol_geom.atom_positions.reserve(n_sites);
        mol_geom.short_range_type_codes.reserve(n_sites);

        Mat3N uc_centered(3, n_sites);
        IVec uc_labels(n_sites);
        for (int i = 0; i < n_sites; ++i) {
            uc_centered.col(i) = uc_mols[m].positions().col(i) - com;
            uc_labels(i) = uc_mols[m].atomic_numbers()(i);
        }

        Mat3N body_centered(3, n_sites);
        IVec body_labels(n_sites);
        for (int i = 0; i < n_sites; ++i) {
            body_centered.col(i) = body_sites[i].offset;
            body_labels(i) = (body_sites[i].atomic_number > 0)
                                 ? body_sites[i].atomic_number
                                 : uc_labels(i);
        }

        Mat3 R_fit = kabsch_rotation(body_centered, uc_centered, false);

        auto compute_rmsd = [&](const std::vector<int>& perm, const Mat3& R) {
            double rmsd = 0.0;
            for (int i = 0; i < n_sites; ++i) {
                const auto& site = body_sites[perm[i]];
                const Vec3 d = R * site.offset - uc_centered.col(i);
                rmsd += d.squaredNorm();
            }
            return std::sqrt(rmsd / std::max(1, n_sites));
        };

        // We reorder body_sites into UC atom order:
        // permutation[uc_index] = body_index.
        std::vector<int> permutation(n_sites);
        for (int i = 0; i < n_sites; ++i) {
            permutation[i] = i;
        }
        double rmsd_best = compute_rmsd(permutation, R_fit);
        auto match = occ::core::try_transformation_with_grouped_permutations(
            uc_labels, uc_centered,
            body_labels, body_centered,
            R_fit,
            0.25);
        if (match.success &&
            static_cast<int>(match.permutation.size()) == n_sites) {
            // try_transformation_with_grouped_permutations returns
            // transformed(B) -> reference(A), i.e. body_index -> uc_index here.
            // Invert to uc_index -> body_index for local reordering.
            std::vector<int> inv_perm(n_sites, -1);
            bool valid_perm = true;
            for (int body_idx = 0; body_idx < n_sites; ++body_idx) {
                const int uc_idx = match.permutation[body_idx];
                if (uc_idx < 0 || uc_idx >= n_sites || inv_perm[uc_idx] != -1) {
                    valid_perm = false;
                    break;
                }
                inv_perm[uc_idx] = body_idx;
            }
            if (valid_perm) {
                Mat3N body_reordered(3, n_sites);
                for (int i = 0; i < n_sites; ++i) {
                    body_reordered.col(i) = body_sites[inv_perm[i]].offset;
                }
                const Mat3 R_candidate =
                    kabsch_rotation(body_reordered, uc_centered, true);
                const double rmsd_candidate = compute_rmsd(inv_perm, R_candidate);

                // Preserve input ordering unless permutation significantly
                // improves fit quality (avoids equivalent-atom swapping noise).
                if (rmsd_candidate + 1e-4 < rmsd_best) {
                    permutation = std::move(inv_perm);
                    R_fit = R_candidate;
                    rmsd_best = rmsd_candidate;
                }
            } else {
                occ::log::warn("DMACRYS setup: invalid atom permutation for molecule {} "
                               "(falling back to identity ordering)",
                               m);
            }
        }

        const double rmsd = compute_rmsd(permutation, R_fit);
        if (rmsd > 0.05) {
            occ::log::warn("DMACRYS setup: molecule {} fit RMSD {:.6f} A", m, rmsd);
        }

        // Use crystal geometry expressed in fitted body frame for short-range sites.
        for (int i = 0; i < n_sites; ++i) {
            const auto &site = body_sites[permutation[i]];
            mol_geom.atomic_numbers.push_back(uc_labels(i));
            mol_geom.short_range_type_codes.push_back(site.short_range_type_code);
            mol_geom.atom_positions.push_back(R_fit.transpose() * uc_centered.col(i));
            mol_geom.aniso_body_axes.push_back(site.aniso_axis);
        }

        geom_vec.push_back(std::move(mol_geom));

        // Initial state: position = COM, rotation from orthogonal Procrustes fit.
        states.push_back(MoleculeState::from_rotation(com, R_fit));
        mol_coms.push_back(com);
    }

    calc.set_molecule_geometry(std::move(geom_vec));
    calc.set_initial_states(std::move(states));
    calc.set_short_range_type_labels(type_tables.code_to_type);
    calc.set_typed_buckingham_params(typed_buck);
    if (!input.aniso_potentials.empty()) {
        const auto typed_aniso =
            convert_typed_aniso_params(input.aniso_potentials,
                                       type_tables.type_to_code);
        calc.set_typed_aniso_params(typed_aniso);
        occ::log::info("Set {} anisotropic repulsion pair types", typed_aniso.size());
    }
    if (input.has_spline && input.spline_min > 0.0) {
        const double taper_on = input.cutoff_radius;
        const double taper_off = input.cutoff_radius + input.spline_min;
        calc.set_electrostatic_taper(taper_on, taper_off, 3);
        calc.set_short_range_taper(taper_on, taper_off, 3);

        // DMACRYS PAIRTAB filters site-site electrostatics by RANG2.
        // With SPLI: RANG2 = RANG + SPLI_min.
        calc.set_elec_site_cutoff(taper_off);
        calc.set_buckingham_site_cutoff(taper_off);

        // DMACRYS expands the COM table range by SPLI_max, while site terms
        // are still filtered by RANG2 and spline value. Keep that shell in OCC.
        const double ext = (input.spline_max > 0.0) ? input.spline_max
                                                    : input.spline_min;
        calc.set_cutoff_radius(input.cutoff_radius + ext);

        // DMACRYS uses TBLCNT+PAIRTAB (COM table + per-site RANG2 filter).
        // In OCC explicit-neighbor mode, rely on per-site cutoff for parity.
        calc.set_use_com_elec_gate(false);

        occ::log::info("Applied DMACRYS SPLI: on={:.3f} off={:.3f}, "
                       "table_cutoff={:.3f}, elec_site_cutoff={:.3f}",
                       taper_on, taper_off, calc.cutoff_radius(), taper_off);
        occ::log::warn("DMACRYS SPLI uses cubic taper (order=3): second-derivative "
                       "quantities can be non-robust for pairs near taper boundaries");
    }

    // Build atom-based neighbor list (needed for Buckingham short-range).
    // COM distances are stored per pair and used as a COM gate for
    // electrostatics, matching DMACRYS TBLCNT behavior.
    // Skip when caller will override via set_neighbor_list().
    if (build_neighbors) {
        calc.build_neighbor_list_from_positions(mol_coms);
    }
}

std::vector<MoleculeState> compute_molecule_states(
    const DmacrysInput &input,
    const crystal::Crystal &crystal,
    const std::vector<MultipoleSource> &multipoles) {

    int n_mol = static_cast<int>(multipoles.size());

    // Use Crystal's correctly assembled UC molecules for COM positions.
    const auto &uc_mols = crystal.unit_cell_molecules();
    if (static_cast<int>(uc_mols.size()) != n_mol) {
        throw std::runtime_error(fmt::format(
            "UC molecule count ({}) does not match multipole source count ({})",
            uc_mols.size(), n_mol));
    }

    std::vector<MoleculeState> states;
    states.reserve(n_mol);

    for (int m = 0; m < n_mol; ++m) {
        Vec3 com = uc_mols[m].center_of_mass();
        const auto &body_sites = multipoles[m].body_sites();
        (void)body_sites;
        states.push_back(MoleculeState::from_rotation(com, multipoles[m].rotation()));
    }

    return states;
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
                           "keeping first and ignoring later duplicate",
                           z1, z2);
            continue;
        }
        result[key] = bp;

        occ::log::info("Buckingham {}-{} (Z={}-{}): A={:.2f} kJ/mol, "
                       "B={:.4f} Ang^-1, C={:.2f} kJ/mol*Ang^6",
                       p.el1, p.el2, z1, z2, bp.A, bp.B, bp.C);
    }

    return result;
}

std::map<std::pair<int, int>, BuckinghamParams>
convert_typed_buckingham_params(
    const std::vector<DmacrysInput::BuckPair> &pairs,
    const std::map<std::string, int> &type_codes) {

    std::map<std::pair<int, int>, BuckinghamParams> result;
    for (const auto &p : pairs) {
        if (p.type1.empty() || p.type2.empty()) {
            continue;
        }

        const auto it1 = type_codes.find(normalize_type_key(p.type1));
        const auto it2 = type_codes.find(normalize_type_key(p.type2));
        if (it1 == type_codes.end() || it2 == type_codes.end()) {
            occ::log::warn("Skipping typed Buckingham pair {}-{}: missing type code mapping",
                           p.type1, p.type2);
            continue;
        }

        int t1 = it1->second;
        int t2 = it2->second;
        if (t1 > t2) {
            std::swap(t1, t2);
        }

        BuckinghamParams bp;
        bp.A = p.A_eV * EV_TO_KJ_PER_MOL;
        bp.B = 1.0 / p.rho_ang;
        bp.C = p.C6_eV_ang6 * EV_TO_KJ_PER_MOL;

        const auto key = std::make_pair(t1, t2);
        auto it = result.find(key);
        if (it != result.end()) {
            const double dA = std::abs(it->second.A - bp.A);
            const double dB = std::abs(it->second.B - bp.B);
            const double dC = std::abs(it->second.C - bp.C);
            if (dA > 1e-8 || dB > 1e-12 || dC > 1e-8) {
                occ::log::warn("Conflicting typed Buckingham params for {}-{}; "
                               "keeping first",
                               p.type1, p.type2);
            }
            continue;
        }

        result[key] = bp;
        result[{t2, t1}] = bp;
    }
    return result;
}

std::map<std::pair<int, int>, AnisotropicRepulsionParams>
convert_typed_aniso_params(
    const std::vector<DmacrysInput::AnisoPair> &pairs,
    const std::map<std::string, int> &type_codes) {

    std::map<std::pair<int, int>, AnisotropicRepulsionParams> result;
    for (const auto &p : pairs) {
        if (p.type1.empty() || p.type2.empty()) {
            continue;
        }

        const auto it1 = type_codes.find(normalize_type_key(p.type1));
        const auto it2 = type_codes.find(normalize_type_key(p.type2));
        if (it1 == type_codes.end() || it2 == type_codes.end()) {
            occ::log::warn("Skipping typed aniso pair {}-{}: missing type code mapping",
                           p.type1, p.type2);
            continue;
        }

        int t1 = it1->second;
        int t2 = it2->second;
        AnisotropicRepulsionParams ap;
        ap.alpha = p.alpha;
        ap.rho_00 = p.rho_00;
        if (t1 <= t2) {
            ap.rho_20 = p.rho_20;
            ap.rho_02 = p.rho_02;
        } else {
            // Swap: type ordering flips, so rho_20/rho_02 swap too
            std::swap(t1, t2);
            ap.rho_20 = p.rho_02;
            ap.rho_02 = p.rho_20;
        }

        const auto key = std::make_pair(t1, t2);
        if (result.count(key)) {
            continue;
        }
        result[key] = ap;
        // Also store reverse for quick lookup
        if (t1 != t2) {
            AnisotropicRepulsionParams ap_rev = ap;
            ap_rev.rho_20 = ap.rho_02;
            ap_rev.rho_02 = ap.rho_20;
            result[{t2, t1}] = ap_rev;
        }

        occ::log::info("Aniso repulsion {}-{}: alpha={:.4f} Ang^-1, "
                       "rho00={:.6f}, rho20={:.6f}, rho02={:.6f}",
                       p.type1, p.type2, ap.alpha, ap.rho_00,
                       p.rho_20, p.rho_02);
    }
    return result;
}

} // namespace occ::mults
