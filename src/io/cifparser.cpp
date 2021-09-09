#include <occ/core/element.h>
#include <gemmi/numb.hpp>
#include <iostream>
#include <occ/core/logger.h>
#include <occ/core/units.h>
#include <occ/io/cifparser.h>

namespace occ::io {

CifParser::CifParser() {}

void CifParser::extract_atom_sites(const gemmi::cif::Loop &loop) {
    int label_idx = loop.find_tag("_atom_site_label");
    int symbol_idx = loop.find_tag("_atom_site_type_symbol");
    int x_idx = loop.find_tag("_atom_site_fract_x");
    int y_idx = loop.find_tag("_atom_site_fract_y");
    int z_idx = loop.find_tag("_atom_site_fract_z");
    for (size_t i = 0; i < loop.length(); i++) {
        AtomData atom;
        bool info_found = false;
        if (label_idx >= 0) {
            atom.site_label = loop.val(i, label_idx);
            info_found = true;
        }
        if (symbol_idx >= 0) {
            atom.element = loop.val(i, symbol_idx);
            info_found = true;
        }
        if (x_idx >= 0) {
            atom.position[0] = gemmi::cif::as_number(loop.val(i, x_idx));
            info_found = true;
        }
        if (y_idx >= 0) {
            atom.position[1] = gemmi::cif::as_number(loop.val(i, y_idx));
            info_found = true;
        }
        if (z_idx >= 0) {
            atom.position[2] = gemmi::cif::as_number(loop.val(i, z_idx));
            info_found = true;
        }
        if (info_found)
            m_atoms.push_back(atom);
    }
}

void CifParser::extract_cell_parameter(const gemmi::cif::Pair &pair) {
    const auto &tag = pair.front();
    if (tag == "_cell_length_a")
        m_cell.a = gemmi::cif::as_number(pair.back());
    else if (tag == "_cell_length_b")
        m_cell.b = gemmi::cif::as_number(pair.back());
    else if (tag == "_cell_length_c")
        m_cell.c = gemmi::cif::as_number(pair.back());
    else if (tag == "_cell_angle_alpha")
        m_cell.alpha = occ::units::radians(gemmi::cif::as_number(pair.back()));
    else if (tag == "_cell_angle_beta")
        m_cell.beta = occ::units::radians(gemmi::cif::as_number(pair.back()));
    else if (tag == "_cell_angle_gamma")
        m_cell.gamma = occ::units::radians(gemmi::cif::as_number(pair.back()));
}

void remove_quotes(std::string &s) {
    const auto &f = s.front();
    if (f == '"' || f == '\'' || f == '`') {
        s.erase(0, 1); // erase the first character
    }
    const auto &b = s.back();
    if (b == '"' || b == '\'' || b == '`') {
        s.erase(s.size() - 1); // erase the last character
    }
}

void CifParser::extract_symmetry_operations(const gemmi::cif::Loop &loop) {
    int idx = loop.find_tag("_symmetry_equiv_pos_as_xyz");
    if (idx < 0)
        return;

    for (size_t i = 0; i < loop.length(); i++) {
        std::string symop = gemmi::cif::as_string(loop.val(i, idx));
        m_sym.symops.push_back(symop);
    }
}

void CifParser::extract_symmetry_data(const gemmi::cif::Pair &pair) {
    const auto &tag = pair.front();
    if (tag == "_symmetry_space_group_name_Hall")
        m_sym.nameHall = gemmi::cif::as_string(pair.back());
    else if (tag == "_symmetry_space_group_name_H-M")
        m_sym.nameHM = gemmi::cif::as_string(pair.back());
    else if (tag == "_space_group_IT_number" ||
             tag == "_symmetry_Int_Tables_number")
        m_sym.number = gemmi::cif::as_number(pair.back());
}

std::optional<occ::crystal::Crystal>
CifParser::parse_crystal(const std::string &filename) {
    try {
        auto doc = gemmi::cif::read_file(filename);
        auto block = doc.blocks.front();
        for (const auto &item : block.items) {
            if (item.type == gemmi::cif::ItemType::Pair) {
                if (item.has_prefix("_cell"))
                    extract_cell_parameter(item.pair);
                if (item.has_prefix("_symmetry") ||
                    item.has_prefix("_space_group")) {
                    extract_symmetry_data(item.pair);
                }
            }
            if (item.type == gemmi::cif::ItemType::Loop) {
                if (item.has_prefix("_atom_site"))
                    extract_atom_sites(item.loop);
                else if (item.has_prefix("_symmetry_equiv_pos"))
                    extract_symmetry_operations(item.loop);
            }
        }
        if (!cell_valid()) {
            m_failure_desc = "Missing unit cell data";
            return std::nullopt;
        }
        if (!symmetry_valid()) {
            m_failure_desc = "Missing symmetry data";
            return std::nullopt;
        }
        occ::crystal::AsymmetricUnit asym;
        if (num_atoms() > 0) {
            asym.atomic_numbers.conservativeResize(num_atoms());
            asym.positions.conservativeResize(3, num_atoms());
            int i = 0;

            for (const auto &atom : m_atoms) {
                asym.positions(0, i) = atom.position[0];
                asym.positions(1, i) = atom.position[1];
                asym.positions(2, i) = atom.position[2];
                asym.atomic_numbers(i) = occ::chem::Element(atom.element).n();
                asym.labels.push_back(atom.site_label);
                i++;
            }
        }
        occ::crystal::UnitCell uc(m_cell.a, m_cell.b, m_cell.c, m_cell.alpha,
                                  m_cell.beta, m_cell.gamma);

        if (m_sym.valid()) {
            if (m_sym.nameHall != "Not set") {
                auto sg = occ::crystal::SpaceGroup(m_sym.nameHall);
                return occ::crystal::Crystal(asym, sg, uc);
            }
            if (m_sym.nameHM != "Not set") {
                auto sg = occ::crystal::SpaceGroup(m_sym.nameHM);
                return occ::crystal::Crystal(asym, sg, uc);
            } else if (m_sym.number > 0) {
                auto sg = occ::crystal::SpaceGroup(m_sym.number);
                return occ::crystal::Crystal(asym, sg, uc);
            } else if (m_sym.symops.size() > 0) {
                auto sg = occ::crystal::SpaceGroup(m_sym.symops);
                return occ::crystal::Crystal(asym, sg, uc);
            }
        }
        occ::crystal::SpaceGroup sg("P 1");
        return occ::crystal::Crystal(asym, sg, uc);
    } catch (const std::exception &e) {
        m_failure_desc = e.what();
        occ::log::error("Exception encountered when parsing CIF: {}",
                        m_failure_desc);
        return std::nullopt;
    }
}

} // namespace occ::io
