#include <occ/mults/crystal_energy_setup.h>
#include <occ/mults/multipole_source.h>
#include <occ/io/structure_format.h>
#include <occ/crystal/crystal.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <Eigen/Geometry>
#include <set>
#include <stdexcept>

namespace occ::mults {

namespace {

/// Angle-axis to rotation matrix (Rodrigues formula).
Mat3 angle_axis_to_rotation(const std::array<double, 3> &aa) {
    Vec3 v(aa[0], aa[1], aa[2]);
    double theta = v.norm();
    if (theta < 1e-15) {
        return Mat3::Identity();
    }
    return Eigen::AngleAxisd(theta, v / theta).toRotationMatrix();
}

} // anonymous namespace

// ============================================================================
// From StructureInput (explicit rigid bodies — no Crystal needed)
// ============================================================================

CrystalEnergySetup
from_structure_input(const occ::io::StructureInput &si) {
    using namespace occ::units;

    CrystalEnergySetup setup;
    setup.force_field = ForceFieldType::Custom;
    setup.unit_cell = crystal::UnitCell(si.a, si.b, si.c, radians(si.alpha),
                                        radians(si.beta), radians(si.gamma));

    // Build type name -> index map
    std::map<std::string, int> type_map;
    for (int i = 0; i < static_cast<int>(si.molecule_types.size()); ++i) {
        type_map[si.molecule_types[i].name] = i;
    }

    Mat3 cell_mat = setup.unit_cell.direct();

    // Build molecules
    for (const auto &mol : si.molecules) {
        auto it = type_map.find(mol.type);
        if (it == type_map.end()) {
            throw std::runtime_error("Unknown molecule type: " + mol.type);
        }
        const auto &mt = si.molecule_types[it->second];

        RigidMolecule rm;

        // Placement
        Vec3 frac_com(mol.translation[0], mol.translation[1],
                      mol.translation[2]);
        rm.com = cell_mat * frac_com;
        rm.angle_axis =
            Vec3(mol.orientation[0], mol.orientation[1], mol.orientation[2]);
        rm.parity = mol.parity;

        // Body-frame COM (mass-weighted)
        Vec3 body_com = Vec3::Zero();
        double total_mass = 0.0;
        for (const auto &site : mt.sites) {
            double mass = occ::core::Element(site.element).mass();
            body_com += mass * Vec3(site.position[0], site.position[1],
                                    site.position[2]);
            total_mass += mass;
        }
        if (total_mass > 0.0)
            body_com /= total_mass;

        // Build sites and atoms
        for (int si_idx = 0; si_idx < static_cast<int>(mt.sites.size());
             ++si_idx) {
            const auto &s = mt.sites[si_idx];
            int z = occ::core::Element(s.element).atomic_number();

            // Atom
            int atom_idx = static_cast<int>(rm.atoms.size());
            RigidMolecule::Atom atom;
            atom.atomic_number = z;
            atom.position =
                Vec3(s.position[0], s.position[1], s.position[2]) - body_com;
            rm.atoms.push_back(atom);

            // Site (multipole on the atom)
            RigidMolecule::Site site;
            site.position = atom.position;
            site.atom_index = atom_idx;

            // Multipoles
            auto flat = s.multipoles.to_flat();
            int rank = s.multipoles.max_rank();
            int n_comp = (rank + 1) * (rank + 1);
            site.multipole = occ::dma::Mult(rank);
            for (int j = 0; j < n_comp && j < static_cast<int>(flat.size());
                 ++j) {
                site.multipole.q(j) = flat[j];
            }

            rm.sites.push_back(std::move(site));
        }

        setup.molecules.push_back(std::move(rm));
    }

    // Resolve type codes from type strings and build Buckingham params.
    // Use the Williams code table for known type labels.
    std::map<std::string, int> type_name_to_code;
    {
        // Build reverse lookup from ForceFieldParams type labels
        // Williams codes: H_W1=501..H_Wa=505, C_W4=511..C_W2=513, etc.
        for (int code = 501; code <= 545; ++code) {
            const char *label = ForceFieldParams::short_range_type_label(code);
            if (label && label[0] != '\0') {
                type_name_to_code[label] = code;
            }
        }
    }

    // Assign type codes to sites
    for (auto &rm : setup.molecules) {
        for (auto &site : rm.sites) {
            if (site.atom_index < 0) continue;
            // Find the type string from the StructureInput
            // (sites were built in the same order as mt.sites)
        }
    }

    // Re-iterate molecules to assign type codes from the MoleculeType sites
    {
        std::map<std::string, int> type_map;
        for (int i = 0; i < static_cast<int>(si.molecule_types.size()); ++i) {
            type_map[si.molecule_types[i].name] = i;
        }
        for (int mi = 0; mi < static_cast<int>(si.molecules.size()); ++mi) {
            auto it = type_map.find(si.molecules[mi].type);
            if (it == type_map.end()) continue;
            const auto &mt = si.molecule_types[it->second];
            auto &rm = setup.molecules[mi];
            for (int si_idx = 0;
                 si_idx < static_cast<int>(mt.sites.size()) &&
                 si_idx < static_cast<int>(rm.sites.size());
                 ++si_idx) {
                const auto &s = mt.sites[si_idx];
                if (!s.type.empty()) {
                    auto code_it = type_name_to_code.find(s.type);
                    if (code_it != type_name_to_code.end()) {
                        rm.sites[si_idx].short_range_type = code_it->second;
                    }
                }
            }
        }
    }

    // Buckingham parameters (eV -> kJ/mol)
    // Check if pairs have type names — if so, store as typed Buckingham.
    bool has_typed_pairs = false;
    for (const auto &bp : si.potentials.buckingham) {
        if (!bp.types[0].empty() && !bp.types[1].empty() &&
            type_name_to_code.count(bp.types[0]) &&
            type_name_to_code.count(bp.types[1])) {
            has_typed_pairs = true;
            break;
        }
    }

    for (const auto &bp : si.potentials.buckingham) {
        BuckinghamParams params;
        params.A = bp.A * EV_TO_KJ_PER_MOL;
        params.B = 1.0 / bp.rho;
        params.C = bp.C6 * EV_TO_KJ_PER_MOL;

        if (has_typed_pairs) {
            auto it1 = type_name_to_code.find(bp.types[0]);
            auto it2 = type_name_to_code.find(bp.types[1]);
            if (it1 != type_name_to_code.end() &&
                it2 != type_name_to_code.end()) {
                int t1 = it1->second, t2 = it2->second;
                setup.typed_buckingham[{t1, t2}] = params;
                setup.typed_buckingham[{t2, t1}] = params;
                setup.type_labels[t1] = bp.types[0];
                setup.type_labels[t2] = bp.types[1];
            }
        }

        // Also store element-based as fallback
        std::string el1 =
            !bp.elements[0].empty() ? bp.elements[0] : bp.types[0];
        std::string el2 =
            !bp.elements[1].empty() ? bp.elements[1] : bp.types[1];
        int z1 = occ::core::Element(el1).atomic_number();
        int z2 = occ::core::Element(el2).atomic_number();
        if (z1 == 0 || z2 == 0) continue;
        if (z1 > z2) std::swap(z1, z2);
        if (!setup.buckingham_params.count({z1, z2})) {
            setup.buckingham_params[{z1, z2}] = params;
        }
    }

    // Settings
    setup.cutoff_radius = si.potentials.cutoff;
    setup.use_ewald = si.settings.use_ewald;
    setup.ewald_accuracy = si.settings.ewald_accuracy;
    setup.max_interaction_order = si.settings.max_interaction_order;

    // Tapering
    if (si.settings.spline_min > 0.0) {
        setup.taper_on = si.potentials.cutoff;
        setup.taper_off = si.potentials.cutoff + si.settings.spline_min;
        setup.taper_order = si.settings.spline_order;
        if (si.settings.spline_max > 0.0) {
            setup.cutoff_radius =
                si.potentials.cutoff + si.settings.spline_max;
        } else {
            setup.cutoff_radius =
                si.potentials.cutoff + si.settings.spline_min;
        }
    }

    occ::log::info("Built CrystalEnergySetup from StructureInput: "
                   "{} molecules, {} Buckingham pairs",
                   setup.molecules.size(), setup.buckingham_params.size());

    return setup;
}

// ============================================================================
// From Crystal + MultipleSources (CIF path)
// ============================================================================

CrystalEnergySetup from_crystal_and_multipoles(
    const occ::crystal::Crystal &crystal,
    const std::vector<MultipoleSource> &multipoles) {

    CrystalEnergySetup setup;
    setup.unit_cell = crystal.unit_cell();

    const auto &uc_mols = crystal.unit_cell_molecules();
    int n_mol = static_cast<int>(uc_mols.size());

    if (n_mol != static_cast<int>(multipoles.size())) {
        throw std::runtime_error(
            "UC molecule count does not match multipole source count");
    }

    for (int m = 0; m < n_mol; ++m) {
        const auto &uc_mol = uc_mols[m];
        const auto &src = multipoles[m];
        const auto &body_sites = src.body_sites();
        int n_sites = static_cast<int>(body_sites.size());

        RigidMolecule rm;
        rm.com = uc_mol.center_of_mass();

        // Extract orientation from MultipoleSource rotation
        RigidMolecule::set_from_rotation(rm, rm.com, src.rotation());

        // Build atoms from UC molecule geometry
        for (int i = 0; i < uc_mol.size(); ++i) {
            RigidMolecule::Atom atom;
            atom.atomic_number = uc_mol.atomic_numbers()(i);
            // Body-frame position: rotate crystal position back
            Vec3 crystal_offset = uc_mol.positions().col(i) - rm.com;
            atom.position = rm.rotation_matrix().transpose() * crystal_offset;
            rm.atoms.push_back(atom);
        }

        // Build sites from MultipoleSource body sites
        for (int i = 0; i < n_sites; ++i) {
            const auto &bs = body_sites[i];
            RigidMolecule::Site site;
            site.position = bs.offset;
            site.multipole = bs.multipole;
            site.short_range_type = bs.short_range_type_code;
            site.aniso_axis = bs.aniso_axis;

            // Match site to atom by proximity
            if (i < static_cast<int>(rm.atoms.size())) {
                double d = (site.position - rm.atoms[i].position).norm();
                if (d < 0.1) {
                    site.atom_index = i;
                }
            }
            if (site.atom_index < 0) {
                // Search all atoms
                for (int ai = 0; ai < static_cast<int>(rm.atoms.size());
                     ++ai) {
                    double d = (site.position - rm.atoms[ai].position).norm();
                    if (d < 0.1) {
                        site.atom_index = ai;
                        break;
                    }
                }
            }

            rm.sites.push_back(std::move(site));
        }

        setup.molecules.push_back(std::move(rm));
    }

    // Run Williams atom typing on each molecule and assign type codes.
    for (auto &rm : setup.molecules) {
        std::vector<int> atomic_numbers;
        std::vector<Vec3> positions;
        for (const auto &atom : rm.atoms) {
            atomic_numbers.push_back(atom.atomic_number);
            positions.push_back(atom.position);
        }
        auto neighbors =
            ForceFieldParams::bonded_neighbors(atomic_numbers, positions);
        for (int i = 0; i < static_cast<int>(atomic_numbers.size()); ++i) {
            int type_code = ForceFieldParams::classify_williams_type(
                i, neighbors, atomic_numbers);
            // Propagate to sites linked to this atom
            for (auto &site : rm.sites) {
                if (site.atom_index == i) {
                    site.short_range_type = type_code;
                }
            }
        }
    }

    // Populate Buckingham params: element-based (fallback) + typed (Williams)
    for (const auto &[key, params] : ForceFieldParams::williams_de_params()) {
        setup.buckingham_params[key] = params;
    }
    setup.typed_buckingham = ForceFieldParams::williams_typed_params();

    // Build type labels
    for (const auto &[code, label] : setup.typed_buckingham) {
        if (!setup.type_labels.count(code.first)) {
            setup.type_labels[code.first] =
                ForceFieldParams::short_range_type_label(code.first);
        }
        if (!setup.type_labels.count(code.second)) {
            setup.type_labels[code.second] =
                ForceFieldParams::short_range_type_label(code.second);
        }
    }

    occ::log::info("Built CrystalEnergySetup from Crystal: "
                   "{} molecules, {} typed Buckingham pairs",
                   setup.molecules.size(), setup.typed_buckingham.size());

    return setup;
}

// ============================================================================
// Convert CrystalEnergySetup back to StructureInput for serialization
// ============================================================================

occ::io::StructureInput
to_structure_input(const CrystalEnergySetup &setup,
                   const std::string &title) {
    using namespace occ::io;
    using namespace occ::units;

    StructureInput si;
    si.title = title;

    // Cell
    const auto &uc = setup.unit_cell;
    si.a = uc.a();
    si.b = uc.b();
    si.c = uc.c();
    si.alpha = degrees(uc.alpha());
    si.beta = degrees(uc.beta());
    si.gamma = degrees(uc.gamma());

    Mat3 inv_cell = uc.direct().inverse();

    // Identify unique molecule types by comparing body-frame site positions.
    // Two molecules are the same type if they have the same number of sites
    // and matching atomic numbers in the same order.
    struct TypeKey {
        int n_sites;
        std::vector<int> atomic_numbers;
        bool operator==(const TypeKey &o) const {
            return n_sites == o.n_sites && atomic_numbers == o.atomic_numbers;
        }
    };

    std::vector<TypeKey> type_keys;
    std::vector<int> mol_to_type; // molecule index -> type index

    for (const auto &mol : setup.molecules) {
        TypeKey key;
        key.n_sites = static_cast<int>(mol.atoms.size());
        for (const auto &atom : mol.atoms) {
            key.atomic_numbers.push_back(atom.atomic_number);
        }

        int type_idx = -1;
        for (int i = 0; i < static_cast<int>(type_keys.size()); ++i) {
            if (type_keys[i] == key) {
                type_idx = i;
                break;
            }
        }
        if (type_idx < 0) {
            type_idx = static_cast<int>(type_keys.size());
            type_keys.push_back(key);

            // Build molecule type from this molecule's sites
            MoleculeType mt;
            mt.name = title.empty()
                          ? "mol_" + std::to_string(type_idx)
                          : title + "_" + std::to_string(type_idx);

            for (const auto &site : mol.sites) {
                MoleculeSite ms;
                int z = (site.atom_index >= 0 &&
                         site.atom_index < static_cast<int>(mol.atoms.size()))
                            ? mol.atoms[site.atom_index].atomic_number
                            : 0;
                ms.element = occ::core::Element(z).symbol();
                ms.label = ms.element;
                ms.position = {site.position(0), site.position(1),
                               site.position(2)};

                // Type label from type code
                if (site.short_range_type > 0) {
                    auto it = setup.type_labels.find(site.short_range_type);
                    if (it != setup.type_labels.end()) {
                        ms.type = it->second;
                    }
                }
                if (ms.type.empty()) {
                    ms.type = ms.element;
                }

                // Multipoles
                int n_comp = site.multipole.num_components();
                std::vector<double> flat(n_comp);
                for (int j = 0; j < n_comp; ++j) {
                    flat[j] = site.multipole.q(j);
                }
                ms.multipoles = SiteMultipoles::from_flat(flat);
                mt.sites.push_back(std::move(ms));
            }
            si.molecule_types.push_back(std::move(mt));
        }
        mol_to_type.push_back(type_idx);
    }

    // Independent molecules
    for (int m = 0; m < static_cast<int>(setup.molecules.size()); ++m) {
        const auto &mol = setup.molecules[m];
        IndependentMolecule im;
        im.type = si.molecule_types[mol_to_type[m]].name;
        Vec3 frac = inv_cell * mol.com;
        im.translation = {frac(0), frac(1), frac(2)};
        im.orientation = {mol.angle_axis(0), mol.angle_axis(1),
                          mol.angle_axis(2)};
        im.parity = mol.parity;
        si.molecules.push_back(std::move(im));
    }

    // Buckingham params — prefer typed if available, else element-based.
    // Only write unique pairs (type1 <= type2).
    if (!setup.typed_buckingham.empty()) {
        std::set<std::pair<int, int>> written;
        for (const auto &[key, params] : setup.typed_buckingham) {
            auto canonical = (key.first <= key.second) ? key : std::make_pair(key.second, key.first);
            if (written.count(canonical)) continue;
            written.insert(canonical);

            BuckinghamPair bp;
            auto label1 = setup.type_labels.find(canonical.first);
            auto label2 = setup.type_labels.find(canonical.second);
            bp.types = {
                label1 != setup.type_labels.end() ? label1->second : std::to_string(canonical.first),
                label2 != setup.type_labels.end() ? label2->second : std::to_string(canonical.second)};
            // Infer elements from type labels
            bp.elements = {
                occ::core::Element(ForceFieldParams::short_range_type_atomic_number(canonical.first)).symbol(),
                occ::core::Element(ForceFieldParams::short_range_type_atomic_number(canonical.second)).symbol()};
            bp.A = params.A * KJ_PER_MOL_TO_EV;
            bp.rho = 1.0 / params.B;
            bp.C6 = params.C * KJ_PER_MOL_TO_EV;
            si.potentials.buckingham.push_back(std::move(bp));
        }
    } else {
        for (const auto &[key, params] : setup.buckingham_params) {
            BuckinghamPair bp;
            bp.elements = {occ::core::Element(key.first).symbol(),
                           occ::core::Element(key.second).symbol()};
            bp.types = bp.elements;
            bp.A = params.A * KJ_PER_MOL_TO_EV;
            bp.rho = 1.0 / params.B;
            bp.C6 = params.C * KJ_PER_MOL_TO_EV;
            si.potentials.buckingham.push_back(std::move(bp));
        }
    }

    si.potentials.cutoff = setup.cutoff_radius;
    si.settings.use_ewald = setup.use_ewald;
    si.settings.ewald_accuracy = setup.ewald_accuracy;
    si.settings.max_interaction_order = setup.max_interaction_order;

    if (setup.taper_on > 0.0 && setup.taper_off > setup.taper_on) {
        si.settings.spline_min = setup.taper_off - setup.taper_on;
        si.settings.spline_max =
            setup.cutoff_radius - setup.taper_on;
        si.settings.spline_order = setup.taper_order;
    }

    return si;
}

} // namespace occ::mults
