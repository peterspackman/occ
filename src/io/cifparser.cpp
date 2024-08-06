#include <filesystem>
#include <gemmi/numb.hpp>
#include <iostream>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/io/cifparser.h>

namespace fs = std::filesystem;
using gemmi::cif::Loop;
using gemmi::cif::Pair;

namespace occ::io {

const ankerl::unordered_dense::map<std::string, CifParser::AtomField>
    CifParser::m_known_atom_fields{
        {"_atom_site_label", AtomField::Label},
        {"_atom_site_type_symbol", AtomField::Element},
        {"_atom_site_fract_x", AtomField::FracX},
        {"_atom_site_fract_y", AtomField::FracY},
        {"_atom_site_fract_z", AtomField::FracZ},
    };

const ankerl::unordered_dense::map<std::string, CifParser::CellField>
    CifParser::m_known_cell_fields{
        {"_cell_length_a", CellField::LengthA},
        {"_cell_length_b", CellField::LengthB},
        {"_cell_length_c", CellField::LengthC},
        {"_cell_angle_alpha", CellField::AngleAlpha},
        {"_cell_angle_beta", CellField::AngleBeta},
        {"_cell_angle_gamma", CellField::AngleGamma},
    };

const ankerl::unordered_dense::map<std::string, CifParser::SymmetryField>
    CifParser::m_known_symmetry_fields{
        {"_symmetry_space_group_name_hall", SymmetryField::HallSymbol},
        {"_symmetry_space_group_name_h-m", SymmetryField::HMSymbol},
        {"_space_group_it_number", SymmetryField::Number},
        {"_symmetry_int_tables_number", SymmetryField::Number},
    };

CifParser::CifParser() {}

void CifParser::set_atom_data(int index, const std::vector<AtomField> &fields,
                              const Loop &loop, AtomData &atom) {
  using enum CifParser::AtomField;
  using gemmi::cif::as_number;

  for (int field_index = 0; field_index < fields.size(); field_index++) {
    const auto &field = fields[field_index];
    const auto &value = loop.val(index, field_index);
    switch (field) {
    case Label:
      atom.site_label = value;
      break;
    case Element:
      atom.element = value;
      break;
    case FracX:
      atom.position[0] = as_number(value);
      break;
    case FracY:
      atom.position[1] = as_number(value);
      break;
    case FracZ:
      atom.position[2] = as_number(value);
      break;
    default:
      break;
    }
  }
}

void CifParser::extract_atom_sites(const Loop &loop) {
  std::vector<AtomField> fields(loop.tags.size(), AtomField::Ignore);

  bool found_info = false;
  // Map tags to fields
  for (size_t i = 0; i < loop.tags.size(); ++i) {
    const auto &tag = loop.tags[i];
    const auto kv = m_known_atom_fields.find(tag);
    if (kv != m_known_atom_fields.end()) {
      fields[i] = kv->second;
      occ::log::debug("{} at loop field {}", kv->first, i);
      found_info = true;
    }
  }

  if (!found_info)
    return;
  for (size_t i = 0; i < loop.length(); i++) {
    AtomData atom;
    set_atom_data(i, fields, loop, atom);
    // use the site label if we have no element type
    if (atom.element.empty())
      atom.element = atom.site_label;
    m_atoms.push_back(atom);
  }
  occ::log::debug("Found {} atom sites", m_atoms.size());
}

void CifParser::extract_cell_parameter(const gemmi::cif::Pair &pair) {
  using enum CifParser::CellField;
  using gemmi::cif::as_number;
  using occ::units::radians;

  const auto &tag = pair.front();
  const auto kv = m_known_cell_fields.find(tag);
  if (kv == m_known_cell_fields.end()) {
    occ::log::debug("Skipping unused cell parameter field: {}", tag);
    return;
  }
  const auto &value = pair.back();
  switch (kv->second) {
  case LengthA:
    m_cell.a = as_number(value);
    break;
  case LengthB:
    m_cell.b = as_number(value);
    break;
  case LengthC:
    m_cell.c = as_number(value);
    break;
  case AngleAlpha:
    m_cell.alpha = radians(as_number(value));
    break;
  case AngleBeta:
    m_cell.beta = radians(as_number(value));
    break;
  case AngleGamma:
    m_cell.gamma = radians(as_number(value));
    break;
  default:
    break;
  }
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
  const std::array<const char *, 2> keys{"_symmetry_equiv_pos_as_xyz",
                                         "_space_group_symop_operation_xyz"};
  int idx = -1;
  for (const auto &tag : keys) {
    idx = loop.find_tag(tag);
    if (idx > -1) {
      occ::log::debug("Found symmetry operations tag: {}", tag);
      break;
    }
  }
  if (idx < 0)
    return;

  for (size_t i = 0; i < loop.length(); i++) {
    std::string symop = gemmi::cif::as_string(loop.val(i, idx));
    m_sym.symops.push_back(symop);
  }
}

void CifParser::extract_symmetry_data(const gemmi::cif::Pair &pair) {
  using enum CifParser::SymmetryField;
  using gemmi::cif::as_number;

  const auto &tag = occ::util::to_lower_copy(pair.front());
  const auto kv = m_known_symmetry_fields.find(tag);
  if (kv == m_known_symmetry_fields.end()) {
    occ::log::debug("Skipping unused symmetry field: {}", tag);
    return;
  }
  const auto &value = pair.back();
  switch (kv->second) {
  case HallSymbol:
    m_sym.nameHall = value;
    break;
  case HMSymbol:
    m_sym.nameHM = value;
    break;
  case Number:
    m_sym.number = as_number(value);
    break;
  default:
    break;
  }

  // Tidy up symbols
  if (m_sym.nameHall.find('_') != std::string::npos) {
    occ::log::debug("Removing '_' characters from name Hall");
    occ::util::remove_character_occurences(m_sym.nameHall, '_');
  }

  if (m_sym.nameHM.find('_') != std::string::npos) {
    occ::log::debug("Removing '_' characters from name HM");
    occ::util::remove_character_occurences(m_sym.nameHM, '_');
  }
}

std::optional<occ::crystal::Crystal>
CifParser::parse_crystal(const std::string &filename) {
  using gemmi::cif::ItemType;
  try {
    auto doc = gemmi::cif::read_file(filename);
    occ::log::debug("Gemmi read cif: {}", filename);
    auto block = doc.blocks.front();
    for (const auto &item : block.items) {
      if (item.type == ItemType::Pair) {
        if (item.has_prefix("_cell")) {
          occ::log::debug("Extracting {}", item.pair[0]);
          extract_cell_parameter(item.pair);
        } else if (item.has_prefix("_sym") || item.has_prefix("_space_group")) {
          occ::log::debug("Extracting {}", item.pair[0]);
          extract_symmetry_data(item.pair);
        } else {
          occ::log::debug("Ignoring item: {}", item.pair[0]);
        }
      }
      if (item.type == ItemType::Loop) {
        if (item.has_prefix("_atom_site")) {
          occ::log::debug("Extracting _atom_site loop with {} items",
                          item.loop.length());
          extract_atom_sites(item.loop);
        } else if (item.has_prefix("_sym") | item.has_prefix("_space_group")) {
          occ::log::debug("Extracting _symmetry or _space_group loop with {} items", item.loop.length());
          extract_symmetry_operations(item.loop);
        }
      }
    }
    if (!cell_valid()) {
      m_failure_desc = "Missing unit cell data";
      occ::log::debug("Failed reading crystal: {}", m_failure_desc);
      return std::nullopt;
    }
    if (!symmetry_valid()) {
      m_failure_desc = "Missing symmetry data";
      occ::log::debug("Failed reading crystal: {}", m_failure_desc);
      return std::nullopt;
    }
    occ::crystal::AsymmetricUnit asym;
    if (num_atoms() > 0) {
      occ::log::debug("Found {} atoms _atom_site data block", num_atoms());
      asym.atomic_numbers.conservativeResize(num_atoms());
      asym.positions.conservativeResize(3, num_atoms());
      int i = 0;

      for (const auto &atom : m_atoms) {
        occ::log::debug("Atom element = {}, label = {} position = {} {} {}",
                        atom.element, atom.site_label, atom.position[0],
                        atom.position[1], atom.position[2]);
        asym.positions(0, i) = atom.position[0];
        asym.positions(1, i) = atom.position[1];
        asym.positions(2, i) = atom.position[2];
        asym.atomic_numbers(i) =
            occ::core::Element(atom.element).atomic_number();
        asym.labels.push_back(atom.site_label);
        i++;
      }
    }
    occ::crystal::UnitCell uc(m_cell.a, m_cell.b, m_cell.c, m_cell.alpha,
                              m_cell.beta, m_cell.gamma);

    occ::crystal::SpaceGroup sg(1);
    bool found = false;
    if (m_sym.valid()) {
      if (!found && m_sym.nameHM != "Not set") {
        occ::log::debug("Try using space group HM name: {}", m_sym.nameHM);
        const auto *sgdata = gemmi::find_spacegroup_by_name(m_sym.nameHM);
        if (sgdata) {
          occ::log::debug("Found space group: {}", sgdata->number);
          sg = occ::crystal::SpaceGroup(m_sym.nameHM);
          found = true;
        }
      }
      if (!found && m_sym.nameHall != "Not set") {
        const auto *sgdata = gemmi::find_spacegroup_by_name(m_sym.nameHall);
        occ::log::debug("Try using space group Hall name: {}", m_sym.nameHall);
        if (sgdata) {
          occ::log::debug("Found space group: {}", sgdata->number);
          sg = occ::crystal::SpaceGroup(m_sym.nameHall);
          found = true;
        }
      }
      if (!found && m_sym.symops.size() > 0) {
        occ::log::debug("Try using space group symops (len = {})",
                        m_sym.symops.size());
        std::vector<gemmi::Op> operations;
        for (const auto &symop : m_sym.symops) {
          operations.push_back(gemmi::parse_triplet(symop));
          occ::log::debug("Symop: {}", operations.back().triplet());
        }
        gemmi::GroupOps ops = gemmi::split_centering_vectors(operations);
        const auto *sgdata = gemmi::find_spacegroup_by_ops(ops);
        if (sgdata) {
          occ::log::debug("Found space group: {}", sgdata->hm);
          sg = occ::crystal::SpaceGroup(sgdata->hm);
          found = true;
        }
      }
      if (!found && m_sym.number > 0) {
        occ::log::debug("Try using space group number: {}", m_sym.number);
        const auto *sgdata = gemmi::find_spacegroup_by_number(m_sym.number);
        if (sgdata) {
          sg = occ::crystal::SpaceGroup(m_sym.number);
          found = true;
        }
      }
    }
    if (!found)
      occ::log::warn("Could not determine space group from CIF, using P "
                     "1 symmetry");
    return occ::crystal::Crystal(asym, sg, uc);
  } catch (const std::exception &e) {
    m_failure_desc = e.what();
    occ::log::error("Exception encountered when parsing CIF: {}",
                    m_failure_desc);
    return std::nullopt;
  }
}

bool CifParser::is_likely_cif_filename(const std::string &filename) {
  fs::path path(filename);
  std::string ext = path.extension().string();
  if (ext == ".cif")
    return true;
  return false;
}

} // namespace occ::io
