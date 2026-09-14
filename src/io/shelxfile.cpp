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

// Every instruction ShelXL understands. A line is an atom only if its first
// word is not one of these -- guessing from the shape of the arguments instead
// turns `AFIX 137` and `RESI 1 GLY` into atoms.
const ankerl::unordered_dense::set<std::string> ShelxFile::m_instructions{
    "ABIN", "ACTA", "AFIX", "ANIS", "ANSC", "ANSR", "BASF", "BIND", "BLOC",
    "BOND", "BUMP", "CELL", "CGLS", "CHIV", "CONF", "CONN", "DAMP", "DANG",
    "DEFS", "DELU", "DFIX", "DISP", "EADP", "EGEN", "END",  "EQIV", "ESEL",
    "EXTI", "EXYZ", "FEND", "FLAT", "FMAP", "FRAG", "FREE", "FVAR", "GRID",
    "HFIX", "HKLF", "HOPE", "HTAB", "ISOR", "L.S.", "LATT", "LAUE", "LIST",
    "MERG", "MOLE", "MORE", "MOVE", "MPLA", "MUST", "NCSY", "NEUT", "OMIT",
    "PART", "PLAN", "PRIG", "REM",  "RESI", "RIGU", "RTAB", "SADI", "SAME",
    "SFAC", "SHEL", "SIMU", "SIZE", "SPEC", "STIR", "SUMP", "SWAT", "SYMM",
    "TEMP", "TIME", "TITL", "TWIN", "TWST", "UNIT", "WGHT", "WPDB", "XNPD",
    "ZERR"};

namespace {

/// Split a coded SHELX parameter into its free-variable index and multiplier.
/// The division truncates towards zero, which is what makes -21 mean
/// "1 - fv_2" rather than something involving fv_3.
std::pair<int, double> split_variable(double coded) {
  const double m = std::trunc(coded / 10.0);
  return {static_cast<int>(m), coded - 10.0 * m};
}

/// The lattice centring vectors implied by |LATT|, always including the origin.
std::vector<Vec3> centering_translations(int latt) {
  std::vector<Vec3> result{Vec3(0.0, 0.0, 0.0)};
  switch (std::abs(latt)) {
  case 1: // P - primitive
    break;
  case 2: // I - body centred
    result.emplace_back(0.5, 0.5, 0.5);
    break;
  case 3: // R - rhombohedral, on hexagonal axes
    result.emplace_back(2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    result.emplace_back(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
    break;
  case 4: // F - all faces centred
    result.emplace_back(0.0, 0.5, 0.5);
    result.emplace_back(0.5, 0.0, 0.5);
    result.emplace_back(0.5, 0.5, 0.0);
    break;
  case 5: // A
    result.emplace_back(0.0, 0.5, 0.5);
    break;
  case 6: // B
    result.emplace_back(0.5, 0.0, 0.5);
    break;
  case 7: // C
    result.emplace_back(0.5, 0.5, 0.0);
    break;
  default:
    occ::log::warn("Unknown LATT value {}, treating the lattice as primitive",
                   latt);
    break;
  }
  return result;
}

/// A symmetry operation with its translation reduced into [0, 1), so that
/// operations differing by a lattice vector compare equal.
std::string canonical_triplet(const occ::crystal::SymmetryOperation &op) {
  return op.translated(Vec3::Zero(), true).to_string();
}

/// U_eq, the isotropic equivalent of a displacement tensor: one third of the
/// trace of its Cartesian form.
double u_equivalent(const occ::crystal::UnitCell &cell, const Vec6 &u_cif) {
  Mat6N packed(6, 1);
  packed.col(0) = u_cif;
  const Vec6 cartesian = cell.to_cartesian_adp(packed).col(0);
  return (cartesian(0) + cartesian(1) + cartesian(2)) / 3.0;
}

} // namespace

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
    // Initialize with identity operation
    m_sym.symops.push_back("x,y,z");

    for (const std::string &line : logical_lines(contents)) {
      switch (classify_line(line)) {
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
      case LineType::Fvar:
        parse_fvar_line(line);
        break;
      case LineType::Part:
        parse_part_line(line);
        break;
      case LineType::Atom:
        parse_atom_line(line);
        break;
      case LineType::Peak:
        occ::log::debug("Skipping difference map peak: {}", line);
        break;
      case LineType::End:
        occ::log::debug("Reached END statement");
        goto end_parsing;
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
    resolve_displacement_parameters(uc);

    occ::crystal::AsymmetricUnit asym;
    if (num_atoms() > 0) {
      occ::log::debug("Found {} atoms in SHELX file", num_atoms());
      asym.resize(num_atoms());

      for (size_t i = 0; i < m_atoms.size(); ++i) {
        const auto &atom = m_atoms[i];
        asym.positions(0, i) = atom.x;
        asym.positions(1, i) = atom.y;
        asym.positions(2, i) = atom.z;
        // SHELX writes its scattering types in upper case, and NA matched
        // as written would be nitrogen.
        asym.atomic_numbers(i) =
            occ::core::Element(occ::util::capitalize_copy(atom.element))
                .atomic_number();
        asym.occupations(i) = atom.occupation;
        asym.labels.push_back(atom.label);
        asym.adps.col(i) = atom.adp;
      }
    }

    // Generate symmetry operations and determine space group
    occ::crystal::SpaceGroup sg(1); // Default to P1

    try {
      // SHELX gives only the operations that generate the group: the SYMM
      // lines, the centring implied by |LATT|, and -- when LATT is positive --
      // the inversion centre.
      std::vector<occ::crystal::SymmetryOperation> base_symops;
      base_symops.push_back(occ::crystal::SymmetryOperation("x,y,z"));
      for (const auto &symop : m_sym.symops) {
        if (symop != "x,y,z") {
          base_symops.push_back(occ::crystal::SymmetryOperation(symop));
          occ::log::debug("Added SYMM operation: {}", symop);
        }
      }

      const std::vector<Vec3> centering = centering_translations(m_sym.latt);

      std::vector<occ::crystal::SymmetryOperation> all_operations;
      for (const Vec3 &translation : centering)
        for (const auto &symop : base_symops)
          all_operations.push_back(symop.translated(translation, true));

      // A positive LATT means the structure is centrosymmetric, with the
      // inversion centre at the origin. That is the operation composed with
      // -x,-y,-z -- not its inverse, which for a two-fold screw axis is the
      // operation itself.
      if (m_sym.latt > 0) {
        const occ::crystal::SymmetryOperation inversion("-x,-y,-z");
        const size_t without_inversion = all_operations.size();
        for (size_t i = 0; i < without_inversion; i++)
          all_operations.push_back(
              (inversion * all_operations[i]).translated(Vec3::Zero(), true));
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

    // SHELX's site occupation factor already carries the site's own symmetry: a
    // fully occupied atom on a two-fold axis is written as 0.5, because the
    // structure factor sums over every general-position image whether or not
    // they coincide. occ keeps only the distinct positions, so that factor has
    // to be multiplied back out or the atom ends up at half weight.
    if (num_atoms() > 0) {
      const occ::crystal::Crystal provisional(asym, sg, uc);
      const auto &images = provisional.unit_cell_atoms();
      std::vector<int> multiplicity(asym.size(), 0);
      for (Eigen::Index i = 0; i < images.asym_idx.size(); i++)
        multiplicity[images.asym_idx(i)]++;

      const double order = static_cast<double>(sg.symmetry_operations().size());
      for (size_t i = 0; i < multiplicity.size(); i++) {
        if (multiplicity[i] > 0)
          asym.occupations(i) *= order / multiplicity[i];
      }
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

std::vector<std::string> ShelxFile::logical_lines(const std::string &contents) {
  std::vector<std::string> result;
  std::istringstream stream(contents);
  std::string line;
  bool continued = false;

  while (std::getline(stream, line)) {
    // '!' starts a comment, except in the free text of a title or remark.
    const std::string keyword = occ::util::to_upper_copy(line.substr(0, 4));
    if (keyword.rfind("TITL", 0) != 0 && keyword.rfind("REM", 0) != 0) {
      const auto comment = line.find('!');
      if (comment != std::string::npos)
        line.erase(comment);
    }
    occ::util::trim(line);

    // A trailing '=' means the instruction carries on over the next line, which
    // is how anisotropic atoms are written.
    const bool continues = !line.empty() && line.back() == '=';
    if (continues) {
      line.pop_back();
      occ::util::rtrim(line);
    }

    if (continued && !result.empty()) {
      result.back() += ' ';
      result.back() += line;
    } else if (!line.empty()) {
      result.push_back(line);
    }
    continued = continues;
  }
  return result;
}

ShelxFile::LineType ShelxFile::classify_line(const std::string &line) const {
  std::istringstream iss(line);
  std::string keyword;
  if (!(iss >> keyword))
    return LineType::Ignored;

  const std::string upper = occ::util::to_upper_copy(keyword);
  if (upper == "TITL")
    return LineType::Title;
  if (upper == "CELL")
    return LineType::Cell;
  if (upper == "LATT")
    return LineType::Latt;
  if (upper == "SFAC")
    return LineType::Sfac;
  if (upper == "SYMM")
    return LineType::Symm;
  if (upper == "FVAR")
    return LineType::Fvar;
  if (upper == "PART")
    return LineType::Part;
  if (upper == "END")
    return LineType::End;
  if (m_instructions.contains(upper))
    return LineType::Ignored;

  if (!std::isalpha(static_cast<unsigned char>(keyword[0])))
    return LineType::Ignored;

  // Whatever is left is an atom, or one of the Q1, Q2, ... peaks a refinement
  // leaves in the difference map. Those are written exactly like atoms, so the
  // label is the only thing telling them apart.
  if (upper.size() > 1 && upper[0] == 'Q' &&
      std::all_of(upper.begin() + 1, upper.end(),
                  [](unsigned char c) { return std::isdigit(c) != 0; }))
    return LineType::Peak;

  // An atom is `label sfac x y z ...` -- a scattering factor index and at least
  // a position.
  int sfac_index = 0;
  double coordinate = 0.0;
  if (!(iss >> sfac_index) || sfac_index < 1)
    return LineType::Ignored;
  for (int i = 0; i < 3; i++)
    if (!(iss >> coordinate))
      return LineType::Ignored;
  return LineType::Atom;
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
    // SHELX has a second form of SFAC that spells out the scattering factor
    // coefficients for one element: `SFAC label a1 b1 ... c f' f" mu r wt`.
    // Only the label is usable here.
    try {
      size_t consumed = 0;
      std::stod(element, &consumed);
      if (consumed == element.size()) {
        occ::log::warn("Ignoring the numeric part of an SFAC line: occ uses "
                       "its own scattering factors");
        break;
      }
    } catch (const std::exception &) {
      // not a number, so it is an element symbol
    }
    m_sfac.push_back(element);
  }

  occ::log::debug("Parsed SFAC: {} elements", m_sfac.size());
  for (const auto &elem : m_sfac) {
    occ::log::debug("  Element: {}", elem);
  }
}

void ShelxFile::parse_symm_line(const std::string &line) {
  std::string symop = line.substr(4); // skip "SYMM"
  // SHELX allows spaces inside an operation, `-x, 1/2+y, 1/2-z`; the triplet
  // parsers do not.
  symop.erase(std::remove_if(symop.begin(), symop.end(),
                             [](unsigned char c) { return std::isspace(c); }),
              symop.end());
  if (symop.empty()) {
    occ::log::warn("Ignoring a SYMM instruction with no operation");
    return;
  }
  m_sym.symops.push_back(symop);
  occ::log::debug("Parsed SYMM: {}", symop);
}

void ShelxFile::parse_fvar_line(const std::string &line) {
  std::istringstream iss(line);
  std::string keyword;
  iss >> keyword; // skip "FVAR"

  // ShelXL permits several FVAR instructions, which extend the same list. The
  // first value is the overall scale factor, free variable 1.
  double value;
  while (iss >> value)
    m_free_variables.push_back(value);
  occ::log::debug("Parsed FVAR: {} free variables in total",
                  m_free_variables.size());
}

void ShelxFile::parse_part_line(const std::string &line) {
  std::istringstream iss(line);
  std::string keyword;
  iss >> keyword; // skip "PART"

  m_part_number = 0;
  m_part_occupancy.reset();
  if (!(iss >> m_part_number))
    return; // a bare PART closes the current component

  double occupancy;
  if (iss >> occupancy)
    m_part_occupancy = decode_variable(occupancy);
  occ::log::debug("Parsed PART {}{}", m_part_number,
                  m_part_occupancy
                      ? fmt::format(" with occupancy {}", *m_part_occupancy)
                      : std::string{});
}

double ShelxFile::decode_variable(double coded) const {
  const auto [m, p] = split_variable(coded);
  // 0 is a parameter free to refine and 1 one that is fixed: either way the
  // value is what was written.
  if (m == 0 || m == 1)
    return p;
  // Undocumented, but this is what ShelXL itself does with m == -1.
  if (m == -1)
    return 0.0;

  const size_t index = static_cast<size_t>(std::abs(m));
  if (index > m_free_variables.size()) {
    occ::log::warn("SHELX parameter {} refers to free variable {}, but only {} "
                   "were declared; taking it literally",
                   coded, index, m_free_variables.size());
    return p;
  }
  const double free_variable = m_free_variables[index - 1];
  return m < 0 ? p * (free_variable - 1.0) : p * free_variable;
}

void ShelxFile::parse_atom_line(const std::string &line) {
  std::istringstream iss(line);
  AtomData atom;
  iss >> atom.label >> atom.sfac_index;

  std::vector<double> values;
  double value;
  while (iss >> value)
    values.push_back(value);

  if (values.size() < 3) {
    occ::log::warn("Skipping atom {}: it has no position", atom.label);
    return;
  }

  atom.x = decode_variable(values[0]);
  atom.y = decode_variable(values[1]);
  atom.z = decode_variable(values[2]);
  if (values.size() > 3)
    atom.occupation = decode_variable(values[3]);

  if (values.size() >= 10) {
    // `... sof U11 U22 U33 U23 U13 U12`. SHELX orders the off-diagonals the
    // other way round from the u_cif convention used everywhere else here.
    atom.anisotropic = true;
    atom.adp << decode_variable(values[4]), decode_variable(values[5]),
        decode_variable(values[6]), decode_variable(values[9]),
        decode_variable(values[8]), decode_variable(values[7]);
  } else if (values.size() >= 5) {
    // A negative U_iso is not a displacement parameter at all: it makes the
    // atom ride on the last one that had a U of its own, at that multiple of
    // its U_eq. This is how SHELX handles hydrogens.
    const auto [m, p] = split_variable(values[4]);
    if (m == 0 && p < -0.5) {
      atom.u_eq_multiple = -p;
      atom.u_eq_pivot = m_u_eq_pivot;
      if (atom.u_eq_pivot < 0)
        occ::log::warn("Atom {} rides on the previous atom's U_eq, but no atom "
                       "precedes it; using zero",
                       atom.label);
    } else {
      const double u_iso = decode_variable(values[4]);
      atom.adp(0) = u_iso;
      atom.adp(1) = u_iso;
      atom.adp(2) = u_iso;
    }
  }

  if (m_part_number != 0 && m_part_occupancy)
    atom.occupation = *m_part_occupancy;

  // Get element from SFAC array
  if (atom.sfac_index > 0 &&
      static_cast<size_t>(atom.sfac_index) <= m_sfac.size()) {
    atom.element = m_sfac[atom.sfac_index - 1];
  } else {
    occ::log::warn("Invalid SFAC index {} for atom {}", atom.sfac_index,
                   atom.label);
    atom.element = "X"; // Unknown element
  }

  if (atom.u_eq_multiple == 0.0)
    m_u_eq_pivot = static_cast<int>(m_atoms.size());
  m_atoms.push_back(atom);
  occ::log::debug("Parsed atom: {} ({}) at ({}, {}, {}) occ={}", atom.label,
                  atom.element, atom.x, atom.y, atom.z, atom.occupation);
}

void ShelxFile::resolve_displacement_parameters(
    const occ::crystal::UnitCell &cell) {
  // Riding atoms first: their pivots never ride themselves, and always come
  // earlier in the file, so a single pass in order is enough.
  for (auto &atom : m_atoms) {
    if (atom.u_eq_multiple == 0.0 || atom.u_eq_pivot < 0)
      continue;
    const AtomData &pivot = m_atoms[atom.u_eq_pivot];
    const double u_eq =
        pivot.anisotropic ? u_equivalent(cell, pivot.adp) : pivot.adp(0);
    atom.adp(0) = atom.u_eq_multiple * u_eq;
    atom.adp(1) = atom.adp(0);
    atom.adp(2) = atom.adp(0);
  }

  // Then give every isotropic atom the off-diagonals its U implies, which are
  // only zero for an orthogonal cell.
  for (auto &atom : m_atoms) {
    if (!atom.anisotropic)
      atom.adp = cell.isotropic_adp(atom.adp(0));
  }
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
  const int latt = determine_latt_type(crystal);
  const std::vector<Vec3> centering = centering_translations(latt);
  const occ::crystal::SymmetryOperation inversion("-x,-y,-z");

  // SHELX expects only the operations the LATT line does not already imply: the
  // centring translations and, when LATT is positive, the inversion centre.
  // Listing them again does not describe the same group any more.
  ankerl::unordered_dense::set<std::string> implied;
  const auto mark_implied = [&](const occ::crystal::SymmetryOperation &op) {
    for (const Vec3 &translation : centering) {
      const auto shifted = op.translated(translation, true);
      implied.insert(canonical_triplet(shifted));
      if (latt > 0)
        implied.insert(canonical_triplet(inversion * shifted));
    }
  };

  mark_implied(occ::crystal::SymmetryOperation("x,y,z"));
  for (const auto &op : crystal.space_group().symmetry_operations()) {
    const std::string triplet = canonical_triplet(op);
    if (implied.contains(triplet))
      continue;
    stream << "SYMM " << triplet << "\n";
    mark_implied(op);
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

  // The inverse of what the reader undoes: SHELX wants the occupancy scaled by
  // how much of the site's own symmetry it sits on.
  const auto &images = crystal.unit_cell_atoms();
  std::vector<int> multiplicity(asym.size(), 0);
  for (Eigen::Index i = 0; i < images.asym_idx.size(); i++)
    multiplicity[images.asym_idx(i)]++;
  const double order =
      static_cast<double>(crystal.space_group().symmetry_operations().size());

  const bool have_adps =
      asym.adps.cols() == asym.positions.cols() && !asym.adps.isZero(0.0);

  stream << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < asym.size(); ++i) {
    std::string element_symbol =
        occ::core::Element(asym.atomic_numbers(i)).symbol();
    int sfac_index = elem_to_index[element_symbol];

    std::string label = (i < asym.labels.size())
                            ? asym.labels[i]
                            : (element_symbol + std::to_string(i + 1));

    const double symmetry_factor =
        multiplicity[i] > 0 ? multiplicity[i] / order : 1.0;
    // 10 + x is SHELX for "fixed at x", as opposed to a value free to refine.
    const double sof = 10.0 + asym.occupations(i) * symmetry_factor;

    stream << label << " " << sfac_index << " " << asym.positions(0, i) << " "
           << asym.positions(1, i) << " " << asym.positions(2, i) << " " << sof;
    if (have_adps) {
      // SHELX orders the off-diagonals U23 U13 U12, the other way round from
      // the u_cif convention the ADPs are stored in.
      stream << " " << asym.adps(0, i) << " " << asym.adps(1, i) << " "
             << asym.adps(2, i) << " =\n    " << asym.adps(5, i) << " "
             << asym.adps(4, i) << " " << asym.adps(3, i);
    }
    stream << "\n";
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
  m_free_variables.clear();
  m_part_number = 0;
  m_part_occupancy.reset();
  m_u_eq_pivot = -1;
  m_error_message.clear();
}

bool ShelxFile::is_likely_shelx_filename(const std::string &filename) {
  fs::path path(filename);
  std::string ext = path.extension().string();
  return ext == ".res" || ext == ".ins";
}

} // namespace occ::io
