#include <cmath>
#include <filesystem>
#include <gemmi/numb.hpp>
#include <occ/core/element.h>
#include <occ/core/format_matrix.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/crystal/symmetryoperation.h>
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
        {"_atom_site_adp_type", AtomField::AdpType},
        {"_atom_site_u_iso_or_equiv", AtomField::Uiso},
        {"_atom_site_b_iso_or_equiv", AtomField::Biso},
        {"_atom_site_occupancy", AtomField::Occupancy},
        {"_atom_site_aniso_label", AtomField::AdpLabel},
        {"_atom_site_aniso_u_11", AtomField::AdpU11},
        {"_atom_site_aniso_u_22", AtomField::AdpU22},
        {"_atom_site_aniso_u_33", AtomField::AdpU33},
        {"_atom_site_aniso_u_12", AtomField::AdpU12},
        {"_atom_site_aniso_u_13", AtomField::AdpU13},
        {"_atom_site_aniso_u_23", AtomField::AdpU23},
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
        // The current spellings; the four above are the deprecated ones, still
        // by far the most common in the wild.
        {"_space_group_name_hall", SymmetryField::HallSymbol},
        {"_space_group_name_h-m_alt", SymmetryField::HMSymbol},
    };

namespace {

/// Assign a CIF numeric field, leaving the destination alone for the '?' and
/// '.' placeholders a file is entitled to use for a value it does not report.
/// Letting those through as NaN poisons every structure factor computed from
/// the result.
bool set_if_number(double &destination, const std::string &value) {
  const double number = gemmi::cif::as_number(value);
  if (std::isnan(number))
    return false;
  destination = number;
  return true;
}

/// The atomic number of a CIF scattering type. That is an element symbol,
/// perhaps with a charge, and files do not agree on its case, so it is
/// capitalized before matching -- which a site label must not be. Deuterium and
/// tritium are written as their own symbols but are hydrogen as far as anything
/// occ stores is concerned: only the atomic number survives into the asymmetric
/// unit.
int atomic_number_of(const std::string &symbol) {
  const std::string trimmed = occ::util::trim_copy(symbol);
  const std::string upper = occ::util::to_upper_copy(trimmed);
  if (upper == "D" || upper == "T")
    return 1;
  return occ::core::Element(occ::util::capitalize_copy(trimmed))
      .atomic_number();
}

/// Whether a list of operations is a whole group: closed under composition, up
/// to lattice translations. A list that is not -- an operation missing or
/// mistyped -- cannot be used as given.
bool is_closed_group(
    const std::vector<occ::crystal::SymmetryOperation> &operations) {
  const auto canonical = [](const occ::crystal::SymmetryOperation &op) {
    return op.translated(occ::Vec3::Zero(), true).to_string();
  };
  ankerl::unordered_dense::set<std::string> present;
  for (const auto &op : operations)
    present.insert(canonical(op));
  for (const auto &a : operations)
    for (const auto &b : operations)
      if (!present.contains(canonical(a * b)))
        return false;
  return true;
}

/// Translate the setting suffixes ICSD-derived files put on a Hermann-Mauguin
/// name into the ':' form gemmi understands: 'S' and 'Z' are the first and
/// second origin choices, 'H' and 'R' the hexagonal and rhombohedral axes.
/// Without this, 'F d -3 m Z' is read at the wrong origin.
std::string normalized_hm_symbol(const std::string &symbol) {
  const std::string trimmed = occ::util::trim_copy(symbol);
  if (trimmed.size() < 3 || trimmed[trimmed.size() - 2] != ' ')
    return trimmed;
  const std::string stem = trimmed.substr(0, trimmed.size() - 2);
  switch (std::toupper(static_cast<unsigned char>(trimmed.back()))) {
  case 'S':
    return stem + ":1";
  case 'Z':
    return stem + ":2";
  case 'H':
    return stem + ":H";
  case 'R':
    return stem + ":R";
  default:
    return trimmed;
  }
}

} // namespace

CifParser::CifParser() {}

void CifParser::set_atom_data(int index, const std::vector<AtomField> &fields,
                              const Loop &loop, AtomData &atom, AdpData &adp) {
  using Field = CifParser::AtomField;
  using gemmi::cif::as_number;

  for (int field_index = 0; field_index < fields.size(); field_index++) {
    const auto &field = fields[field_index];
    const auto &value = loop.val(index, field_index);
    switch (field) {
    case Field::Label:
      atom.site_label = value;
      break;
    case Field::Element:
      atom.element = value;
      break;
    case Field::FracX:
      atom.x = as_number(value);
      break;
    case Field::FracY:
      atom.y = as_number(value);
      break;
    case Field::FracZ:
      atom.z = as_number(value);
      break;
    case Field::AdpType:
      atom.adp_type = value;
      break;
    case Field::Uiso:
      set_if_number(atom.uiso, value);
      break;
    case Field::Biso:
      // B = 8 pi^2 U. Powder refinements in particular tend to report B.
      if (double b = 0.0; set_if_number(b, value))
        atom.uiso = b / (8.0 * occ::units::PI * occ::units::PI);
      break;
    case Field::Occupancy:
      set_if_number(atom.occupancy, value);
      break;
    case Field::AdpLabel:
      adp.aniso_label = value;
      break;
    case Field::AdpU11:
      set_if_number(adp.u11, value);
      break;
    case Field::AdpU22:
      set_if_number(adp.u22, value);
      break;
    case Field::AdpU33:
      set_if_number(adp.u33, value);
      break;
    case Field::AdpU12:
      set_if_number(adp.u12, value);
      break;
    case Field::AdpU13:
      set_if_number(adp.u13, value);
      break;
    case Field::AdpU23:
      set_if_number(adp.u23, value);
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
    const auto tag = occ::util::to_lower_copy(loop.tags[i]);
    const auto kv = m_known_atom_fields.find(tag);
    if (kv != m_known_atom_fields.end()) {
      fields[i] = kv->second;
      occ::log::debug("{} at loop field {}", kv->first, i);
      found_info = true;
    }
  }

  if (!found_info)
    return;

  m_atoms.resize(std::max(m_atoms.size(), loop.length()));
  for (size_t i = 0; i < loop.length(); i++) {
    AdpData adp;
    auto &atom = m_atoms[i];
    set_atom_data(i, fields, loop, atom, adp);

    if (!adp.aniso_label.empty()) {
      m_adps.insert({adp.aniso_label, adp});
    }
  }
  occ::log::debug("Found {} atom sites", m_atoms.size());
}

void CifParser::extract_cell_parameter(const gemmi::cif::Pair &pair) {
  using Field = CifParser::CellField;
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
  case Field::LengthA:
    m_cell.a = as_number(value);
    break;
  case Field::LengthB:
    m_cell.b = as_number(value);
    break;
  case Field::LengthC:
    m_cell.c = as_number(value);
    break;
  case Field::AngleAlpha:
    m_cell.alpha = radians(as_number(value));
    break;
  case Field::AngleBeta:
    m_cell.beta = radians(as_number(value));
    break;
  case Field::AngleGamma:
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
  using Field = CifParser::SymmetryField;
  using gemmi::cif::as_number;

  const auto &tag = occ::util::to_lower_copy(pair.front());
  const auto kv = m_known_symmetry_fields.find(tag);
  if (kv == m_known_symmetry_fields.end()) {
    occ::log::debug("Skipping unused symmetry field: {}", tag);
    return;
  }
  const auto &value = pair.back();
  switch (kv->second) {
  case Field::HallSymbol:
    m_sym.nameHall = value;
    break;
  case Field::HMSymbol:
    m_sym.nameHM = value;
    break;
  case Field::Number:
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

  // Remove quotes from symbols
  remove_quotes(m_sym.nameHall);
  remove_quotes(m_sym.nameHM);
  m_sym.nameHM = normalized_hm_symbol(m_sym.nameHM);
}

std::optional<occ::crystal::Crystal>
CifParser::parse_crystal_from_file(const std::string &filename) {
  auto doc = gemmi::cif::read_file(filename);
  occ::log::debug("Gemmi read cif: {}", filename);
  return parse_crystal_from_document(doc);
}

std::optional<occ::crystal::Crystal>
CifParser::parse_crystal_from_string(const std::string &contents) {
  auto doc = gemmi::cif::read_string(contents);
  occ::log::debug("Gemmi read cif from string (size = {})", contents.size());
  return parse_crystal_from_document(doc);
}

std::optional<occ::crystal::Crystal>
CifParser::parse_crystal_from_document(const gemmi::cif::Document &doc) {
  using gemmi::cif::ItemType;
  try {
    if (doc.blocks.empty()) {
      m_failure_desc = "No data block found";
      occ::log::debug("Failed reading crystal: {}", m_failure_desc);
      return std::nullopt;
    }
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
          occ::log::debug(
              "Extracting _symmetry or _space_group loop with {} items",
              item.loop.length());
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

    occ::crystal::UnitCell uc(m_cell.a, m_cell.b, m_cell.c, m_cell.alpha,
                              m_cell.beta, m_cell.gamma);

    occ::crystal::AsymmetricUnit asym;
    if (num_atoms() > 0) {
      occ::log::debug("Found {} atoms _atom_site data block", num_atoms());
      // resize() sizes every member, with neutral defaults for the ones we
      // don't set below (charge 0, occupancy 1, zero ADPs)
      asym.resize(num_atoms());

      int i = 0;

      for (const auto &atom : m_atoms) {
        occ::log::debug("Atom element = {}, label = {} position = {} {} {}",
                        atom.element, atom.site_label, atom.x, atom.y, atom.z);
        asym.positions(0, i) = atom.x;
        asym.positions(1, i) = atom.y;
        asym.positions(2, i) = atom.z;
        // Without a type symbol the label is all there is, and a label is
        // matched as written: HO1 is a hydrogen, not holmium.
        asym.atomic_numbers(i) =
            atom.element.empty()
                ? occ::core::Element(atom.site_label).atomic_number()
                : atomic_number_of(atom.element);
        asym.labels.push_back(atom.site_label);
        asym.occupations(i) = atom.occupancy;

        // Overwritten below if the atom has a full anisotropic tensor.
        asym.adps.col(i) = uc.isotropic_adp(atom.uiso);

        const auto kv = m_adps.find(atom.site_label);
        if (kv != m_adps.end()) {
          const auto &adp = kv->second;
          asym.adps(0, i) = adp.u11;
          asym.adps(1, i) = adp.u22;
          asym.adps(2, i) = adp.u33;
          asym.adps(3, i) = adp.u12;
          asym.adps(4, i) = adp.u13;
          asym.adps(5, i) = adp.u23;
          const auto &c = asym.adps.col(i);
          occ::log::debug("Have ADP for atom {}: [{:.3f}, {:.3f}, {:.3f}, "
                          "{:.3f}, {:.3f}, {:.3f}]",
                          i, c(0), c(1), c(2), c(3), c(4), c(5));
        }
        i++;
      }
    }
    occ::log::debug("Cartesian ADPs\n{}",
                    format_matrix(uc.to_cartesian_adp(asym.adps).transpose()));

    occ::crystal::SpaceGroup sg(1);
    bool found = false;
    // In order of how much they pin down. A Hall symbol and a list of
    // operations each name one setting of one group; a Hermann-Mauguin name or
    // an International Tables number does not -- around twenty groups have two
    // origin choices, and taking the name at face value silently puts the
    // structure at the wrong origin.
    if (m_sym.valid()) {
      if (!found && m_sym.nameHall != "Not set") {
        occ::log::debug("Try using space group Hall name: {}", m_sym.nameHall);
        // find_spacegroup_by_name only knows Hermann-Mauguin names, so the
        // Hall symbol has to go via the operations it generates.
        try {
          const gemmi::GroupOps ops =
              gemmi::symops_from_hall(m_sym.nameHall.c_str());
          const auto *sgdata = gemmi::find_spacegroup_by_ops(ops);
          if (sgdata) {
            occ::log::debug("Found space group: {}", sgdata->xhm());
            sg = occ::crystal::SpaceGroup(sgdata->xhm());
            found = true;
          }
        } catch (const std::exception &e) {
          occ::log::debug("Could not interpret Hall symbol '{}': {}",
                          m_sym.nameHall, e.what());
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
          occ::log::debug("Found space group: {}", sgdata->xhm());
          // The extended name, which carries the setting: going back through
          // the plain name would throw away what the operations established.
          sg = occ::crystal::SpaceGroup(sgdata->xhm());
          found = true;
        } else {
          // A setting gemmi has no table entry for, such as a centred cell
          // chosen to line up with a related structure. The name describes a
          // different setting, so a whole group of operations is the better
          // description.
          std::vector<occ::crystal::SymmetryOperation> listed;
          for (const auto &op : operations)
            listed.emplace_back(op.triplet());
          if (is_closed_group(listed)) {
            occ::log::warn("CIF symmetry operations match no tabulated "
                           "setting; using the {} operations as given",
                           listed.size());
            sg = occ::crystal::SpaceGroup(listed);
            found = true;
          } else {
            occ::log::warn("CIF symmetry operations do not form a group; "
                           "ignoring them");
          }
        }
      }
      if (!found && m_sym.nameHM != "Not set") {
        occ::log::debug("Try using space group HM name: {}", m_sym.nameHM);
        const auto *sgdata = gemmi::find_spacegroup_by_name(m_sym.nameHM);
        if (sgdata) {
          occ::log::debug("Found space group: {}", sgdata->number);
          sg = occ::crystal::SpaceGroup(m_sym.nameHM);
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
