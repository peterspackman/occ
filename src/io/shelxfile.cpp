#include <algorithm>
#include <filesystem>
#include <gemmi/symmetry.hpp>
#include <iomanip>
#include <iostream>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/crystal/asymmetric_unit.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/crystal/unitcell.h>
#include <occ/io/shelxfile.h>
#include <sstream>

namespace fs = std::filesystem;

namespace occ::io {

const ankerl::unordered_dense::set<std::string> ShelxFile::m_ignored_keywords{
    "FVAR", "UNIT", "REM",  "MORE", "TIME", "OMIT", "ESEL",
    "EGEN", "LIST", "FMAP", "PLAN", "MOLE", "HKLF", "ZERR"};

std::optional<occ::crystal::Crystal>
ShelxFile::read_crystal_from_file(const std::string &filename) {
  try {
    std::ifstream file(filename);
    if (!file.is_open()) {
      m_error_message = "Could not open file: " + filename;
      return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return read_crystal_from_string(buffer.str());
  } catch (const std::exception &e) {
    m_error_message = e.what();
    occ::log::error("Exception when reading SHELX file {}: {}", filename,
                    m_error_message);
    return std::nullopt;
  }
}

std::optional<occ::crystal::Crystal>
ShelxFile::read_crystal_from_string(const std::string &contents) {
  clear_data();

  try {
    std::istringstream stream(contents);
    std::string line;

    // Initialize with identity operation
    m_sym.symops.push_back("x,y,z");

    while (std::getline(stream, line)) {
      occ::util::trim(line);
      if (line.empty())
        continue;

      LineType type = classify_line(line);

      switch (type) {
      case LineType::Title:
        parse_title_line(line);
        break;
      case LineType::Cell:
        parse_cell_line(line);
        break;
      case LineType::Latt:
        parse_latt_line(line);
        break;
      case LineType::Sfac:
        parse_sfac_line(line);
        break;
      case LineType::Symm:
        parse_symm_line(line);
        break;
      case LineType::Atom:
        parse_atom_line(line);
        break;
      case LineType::End:
        occ::log::debug("Reached END statement");
        goto end_parsing;
      case LineType::Zerr:
      case LineType::Ignored:
        occ::log::debug("Ignoring line: {}", line);
        break;
      }
    }

  end_parsing:

    if (!cell_valid()) {
      m_error_message = "Missing or invalid unit cell data";
      occ::log::debug("Failed reading crystal: {}", m_error_message);
      return std::nullopt;
    }

    occ::crystal::UnitCell uc(m_cell.a, m_cell.b, m_cell.c, m_cell.alpha,
                              m_cell.beta, m_cell.gamma);

    occ::crystal::AsymmetricUnit asym;
    if (num_atoms() > 0) {
      occ::log::debug("Found {} atoms in SHELX file", num_atoms());
      asym.atomic_numbers.resize(num_atoms());
      asym.positions.resize(3, num_atoms());
      asym.adps = Mat6N::Zero(6, num_atoms());

      for (size_t i = 0; i < m_atoms.size(); ++i) {
        const auto &atom = m_atoms[i];
        asym.positions(0, i) = atom.x;
        asym.positions(1, i) = atom.y;
        asym.positions(2, i) = atom.z;
        asym.atomic_numbers(i) =
            occ::core::Element(atom.element).atomic_number();
        asym.labels.push_back(atom.label);

        // Set default isotropic ADP
        asym.adps(0, i) = 0.05;
        asym.adps(1, i) = 0.05;
        asym.adps(2, i) = 0.05;
      }
    }

    // Generate symmetry operations and determine space group
    occ::crystal::SpaceGroup sg(1); // Default to P1

    try {
      // Generate base operations
      std::vector<gemmi::Op> base_operations;
      base_operations.push_back(gemmi::parse_triplet("x,y,z"));

      for (const auto &symop : m_sym.symops) {
        if (symop != "x,y,z") {
          base_operations.push_back(gemmi::parse_triplet(symop));
          occ::log::debug("Added SYMM operation: {}", symop);
        }
      }

      // Generate centering translations based on |LATT| value
      int latt_abs = std::abs(m_sym.latt);
      std::vector<std::array<double, 3>> centering_translations;
      centering_translations.push_back(
          {0.0, 0.0, 0.0}); // Always include origin

      switch (latt_abs) {
      case 1: // P - Primitive, no additional centering
        break;
      case 2: // I - Body centered
        centering_translations.push_back({0.5, 0.5, 0.5});
        break;
      case 3: // R - Rhombohedral
        centering_translations.push_back({2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
        centering_translations.push_back({1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0});
        break;
      case 4: // F - Face centered
        centering_translations.push_back({0.0, 0.5, 0.5});
        centering_translations.push_back({0.5, 0.0, 0.5});
        centering_translations.push_back({0.5, 0.5, 0.0});
        break;
      case 5: // A - A face centered
        centering_translations.push_back({0.0, 0.5, 0.5});
        break;
      case 6: // B - B face centered
        centering_translations.push_back({0.5, 0.0, 0.5});
        break;
      case 7: // C - C face centered
        centering_translations.push_back({0.5, 0.5, 0.0});
        break;
      }

      // Generate all operations using SymmetryOperation for centering
      std::vector<occ::crystal::SymmetryOperation> base_symops;
      base_symops.push_back(occ::crystal::SymmetryOperation("x,y,z"));

      for (const auto &symop : m_sym.symops) {
        if (symop != "x,y,z") {
          base_symops.push_back(occ::crystal::SymmetryOperation(symop));
        }
      }

      // Generate all centered operations as SymmetryOperation objects
      std::vector<occ::crystal::SymmetryOperation> all_operations;
      for (size_t cen_idx = 0; cen_idx < centering_translations.size();
           ++cen_idx) {
        for (size_t base_idx = 0; base_idx < base_symops.size(); ++base_idx) {
          Vec3 centering_vec(centering_translations[cen_idx][0],
                             centering_translations[cen_idx][1],
                             centering_translations[cen_idx][2]);
          auto centered_symop =
              base_symops[base_idx].translated(centering_vec, true);
          all_operations.push_back(centered_symop);
        }
      }

      // If LATT is positive, add inversion operations (centrosymmetric
      // structure)
      if (m_sym.latt > 0) {
        std::vector<occ::crystal::SymmetryOperation> inverted_ops =
            all_operations;
        for (const auto &op : all_operations) {
          auto inverted = op.inverted();
          inverted_ops.push_back(inverted);
        }
        all_operations = inverted_ops;
      }

      occ::log::debug("Generated {} total symmetry operations",
                      all_operations.size());

      // Convert SymmetryOperation objects to gemmi::Op using to_string()
      std::vector<gemmi::Op> gemmi_operations;
      for (const auto &symop : all_operations) {
        gemmi_operations.push_back(gemmi::parse_triplet(symop.to_string()));
      }

      // Use gemmi split_centering_vectors to find space group
      try {
        gemmi::GroupOps ops = gemmi::split_centering_vectors(gemmi_operations);
        occ::log::debug(
            "GroupOps created: {} unique operations, {} centering vectors",
            ops.sym_ops.size(), ops.cen_ops.size());

        const auto *sgdata = gemmi::find_spacegroup_by_ops(ops);
        if (sgdata) {
          occ::log::debug("Found space group: {} (#{}) from operations",
                          sgdata->hm, sgdata->number);
          sg = occ::crystal::SpaceGroup(sgdata->hm);
        } else {
          occ::log::warn(
              "Could not determine space group from {} operations, using P1",
              all_operations.size());
        }
      } catch (const std::exception &e) {
        occ::log::warn("Error creating GroupOps: {}, using P1", e.what());
      }
    } catch (const std::exception &e) {
      occ::log::warn("Error parsing symmetry operations: {}, using P1",
                     e.what());
    }

    return occ::crystal::Crystal(asym, sg, uc);

  } catch (const std::exception &e) {
    m_error_message = e.what();
    occ::log::error("Exception encountered when parsing SHELX: {}",
                    m_error_message);
    return std::nullopt;
  }
}

bool ShelxFile::write_crystal_to_file(const occ::crystal::Crystal &crystal,
                                      const std::string &filename) {
  try {
    std::ofstream file(filename);
    if (!file.is_open()) {
      m_error_message = "Could not open file for writing: " + filename;
      return false;
    }

    write_crystal_to_stream(crystal, file);
    return true;
  } catch (const std::exception &e) {
    m_error_message = e.what();
    occ::log::error("Exception when writing SHELX file {}: {}", filename,
                    m_error_message);
    return false;
  }
}

std::string
ShelxFile::write_crystal_to_string(const occ::crystal::Crystal &crystal) {
  std::ostringstream ss;
  write_crystal_to_stream(crystal, ss);
  return ss.str();
}

void ShelxFile::write_crystal_to_stream(const occ::crystal::Crystal &crystal,
                                        std::ostream &stream) {
  write_title_line(stream);
  write_cell_line(crystal, stream);
  write_latt_line(crystal, stream);
  write_symm_lines(crystal, stream);
  write_sfac_line(crystal, stream);
  write_atom_lines(crystal, stream);
  write_end_line(stream);
}

ShelxFile::LineType ShelxFile::classify_line(const std::string &line) const {
  if (line.empty())
    return LineType::Ignored;

  std::string key = line.substr(0, 4);
  std::transform(key.begin(), key.end(), key.begin(), ::toupper);

  if (key == "TITL")
    return LineType::Title;
  if (key == "CELL")
    return LineType::Cell;
  if (key == "ZERR")
    return LineType::Zerr;
  if (key == "LATT")
    return LineType::Latt;
  if (key == "SFAC")
    return LineType::Sfac;
  if (key == "SYMM")
    return LineType::Symm;
  if (key == "END ")
    return LineType::End;

  if (m_ignored_keywords.contains(key)) {
    return LineType::Ignored;
  }

  // Check if this might be an atom line
  std::istringstream iss(line);
  std::string first_token;
  iss >> first_token;

  if (!first_token.empty() && std::isalpha(first_token[0])) {
    std::string second_token;
    iss >> second_token;
    try {
      int sfac_idx = std::stoi(second_token);
      if (sfac_idx > 0) {
        return LineType::Atom;
      }
    } catch (...) {
      // Not a valid sfac index, not an atom line
    }
  }

  return LineType::Ignored;
}

void ShelxFile::parse_title_line(const std::string &line) {
  std::istringstream iss(line);
  std::string titl;
  iss >> titl; // skip "TITL"
  std::getline(iss, m_title);
  occ::util::trim(m_title);
  occ::log::debug("Parsed title: {}", m_title);
}

void ShelxFile::parse_cell_line(const std::string &line) {
  std::istringstream iss(line);
  std::string cell;
  iss >> cell; // skip "CELL"

  iss >> m_cell.wavelength >> m_cell.a >> m_cell.b >> m_cell.c >>
      m_cell.alpha >> m_cell.beta >> m_cell.gamma;

  // Convert angles from degrees to radians
  m_cell.alpha = occ::units::radians(m_cell.alpha);
  m_cell.beta = occ::units::radians(m_cell.beta);
  m_cell.gamma = occ::units::radians(m_cell.gamma);

  occ::log::debug("Parsed cell: wavelength={}, a={}, b={}, c={}, alpha={}, "
                  "beta={}, gamma={}",
                  m_cell.wavelength, m_cell.a, m_cell.b, m_cell.c, m_cell.alpha,
                  m_cell.beta, m_cell.gamma);
}

void ShelxFile::parse_latt_line(const std::string &line) {
  std::istringstream iss(line);
  std::string latt;
  iss >> latt; // skip "LATT"
  iss >> m_sym.latt;
  occ::log::debug("Parsed LATT: {}", m_sym.latt);
}

void ShelxFile::parse_sfac_line(const std::string &line) {
  std::istringstream iss(line);
  std::string sfac;
  iss >> sfac; // skip "SFAC"

  std::string element;
  while (iss >> element) {
    m_sfac.push_back(element);
  }

  occ::log::debug("Parsed SFAC: {} elements", m_sfac.size());
  for (const auto &elem : m_sfac) {
    occ::log::debug("  Element: {}", elem);
  }
}

void ShelxFile::parse_symm_line(const std::string &line) {
  std::string symop = line.substr(4); // skip "SYMM"
  occ::util::trim(symop);
  m_sym.symops.push_back(symop);
  occ::log::debug("Parsed SYMM: {}", symop);
}

void ShelxFile::parse_atom_line(const std::string &line) {
  std::istringstream iss(line);
  AtomData atom;

  iss >> atom.label >> atom.sfac_index >> atom.x >> atom.y >> atom.z;

  // Optional occupation factor
  if (iss >> atom.occupation) {
    // Successfully read occupation
  } else {
    atom.occupation = 1.0;
  }

  // Get element from SFAC array
  if (atom.sfac_index > 0 && atom.sfac_index <= m_sfac.size()) {
    atom.element = m_sfac[atom.sfac_index - 1];
  } else {
    occ::log::warn("Invalid SFAC index {} for atom {}", atom.sfac_index,
                   atom.label);
    atom.element = "X"; // Unknown element
  }

  m_atoms.push_back(atom);
  occ::log::debug("Parsed atom: {} ({}) at ({}, {}, {}) occ={}", atom.label,
                  atom.element, atom.x, atom.y, atom.z, atom.occupation);
}

void ShelxFile::write_title_line(std::ostream &stream) {
  stream << "TITL " << m_title << "\n";
}

void ShelxFile::write_cell_line(const occ::crystal::Crystal &crystal,
                                std::ostream &stream) {
  const auto &uc = crystal.unit_cell();

  stream << std::fixed << std::setprecision(6);
  stream << "CELL " << m_wavelength << " " << uc.a() << " " << uc.b() << " "
         << uc.c() << " " << occ::units::degrees(uc.alpha()) << " "
         << occ::units::degrees(uc.beta()) << " "
         << occ::units::degrees(uc.gamma()) << "\n";
}

void ShelxFile::write_latt_line(const occ::crystal::Crystal &crystal,
                                std::ostream &stream) {
  int latt_type = determine_latt_type(crystal);
  stream << "LATT " << latt_type << "\n";
}

void ShelxFile::write_symm_lines(const occ::crystal::Crystal &crystal,
                                 std::ostream &stream) {
  const auto &sg = crystal.space_group();

  // Get all symmetry operations from the space group
  auto symops = sg.symmetry_operations();

  // Write SYMM lines for all operations except identity
  for (const auto &op : symops) {
    std::string op_str = op.to_string();
    if (op_str != "x,y,z") {
      stream << "SYMM " << op_str << "\n";
    }
  }
}

void ShelxFile::write_sfac_line(const occ::crystal::Crystal &crystal,
                                std::ostream &stream) {
  auto elements = get_unique_elements(crystal);

  stream << "SFAC";
  for (const auto &elem : elements) {
    stream << " " << elem;
  }
  stream << "\n";
}

void ShelxFile::write_atom_lines(const occ::crystal::Crystal &crystal,
                                 std::ostream &stream) {
  const auto &asym = crystal.asymmetric_unit();
  auto elements = get_unique_elements(crystal);

  // Create element to index mapping
  ankerl::unordered_dense::map<std::string, int> elem_to_index;
  for (size_t i = 0; i < elements.size(); ++i) {
    elem_to_index[elements[i]] = i + 1;
  }

  stream << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < asym.size(); ++i) {
    std::string element_symbol =
        occ::core::Element(asym.atomic_numbers(i)).symbol();
    int sfac_index = elem_to_index[element_symbol];

    std::string label = (i < asym.labels.size())
                            ? asym.labels[i]
                            : (element_symbol + std::to_string(i + 1));

    stream << label << " " << sfac_index << " " << asym.positions(0, i) << " "
           << asym.positions(1, i) << " " << asym.positions(2, i) << "\n";
  }
}

void ShelxFile::write_end_line(std::ostream &stream) { stream << "END\n"; }

int ShelxFile::determine_latt_type(const occ::crystal::Crystal &crystal) {
  const auto &sg = crystal.space_group();
  std::string hm_symbol = sg.symbol();

  // Determine lattice type from space group symbol
  int latt_type = 1; // Default to primitive

  if (hm_symbol[0] == 'P')
    latt_type = 1;
  else if (hm_symbol[0] == 'I')
    latt_type = 2;
  else if (hm_symbol[0] == 'R')
    latt_type = 3;
  else if (hm_symbol[0] == 'F')
    latt_type = 4;
  else if (hm_symbol[0] == 'A')
    latt_type = 5;
  else if (hm_symbol[0] == 'B')
    latt_type = 6;
  else if (hm_symbol[0] == 'C')
    latt_type = 7;

  // Check if space group is centrosymmetric by checking if it has inversion
  // Get the space group data from gemmi
  const auto *sgdata = gemmi::find_spacegroup_by_name(hm_symbol);
  bool is_centrosymmetric = false;
  if (sgdata) {
    is_centrosymmetric = sgdata->is_centrosymmetric();
  }

  // If centrosymmetric, LATT should be positive
  // If non-centrosymmetric, LATT should be negative
  return is_centrosymmetric ? latt_type : -latt_type;
}

std::vector<std::string>
ShelxFile::get_unique_elements(const occ::crystal::Crystal &crystal) {
  const auto &asym = crystal.asymmetric_unit();
  ankerl::unordered_dense::set<std::string> unique_elements;

  for (size_t i = 0; i < asym.size(); ++i) {
    std::string element = occ::core::Element(asym.atomic_numbers(i)).symbol();
    unique_elements.insert(element);
  }

  std::vector<std::string> result(unique_elements.begin(),
                                  unique_elements.end());
  std::sort(result.begin(), result.end());
  return result;
}

bool ShelxFile::cell_valid() const {
  return m_cell.a > 0 && m_cell.b > 0 && m_cell.c > 0 && m_cell.alpha > 0 &&
         m_cell.beta > 0 && m_cell.gamma > 0;
}

void ShelxFile::clear_data() {
  m_title = "Crystal structure";
  m_cell = CellData{};
  m_sym = SymmetryData{};
  m_sym.symops.clear();
  m_sfac.clear();
  m_atoms.clear();
  m_error_message.clear();
}

bool ShelxFile::is_likely_shelx_filename(const std::string &filename) {
  fs::path path(filename);
  std::string ext = path.extension().string();
  return ext == ".res" || ext == ".ins";
}

} // namespace occ::io
