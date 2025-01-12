#include <ankerl/unordered_dense.h>
#include <filesystem>
#include <fstream>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/io/dftb_gen.h>
#include <scn/scan.h>

namespace fs = std::filesystem;

using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::Crystal;

namespace occ::io {

DftbGenFormat::DftbGenFormat() { m_origin = Vec3::Zero(); }

bool DftbGenFormat::is_likely_gen_filename(const std::string &filename) {
  fs::path path(filename);
  std::string ext = path.extension().string();
  if (ext == ".gen")
    return true;
  return false;
}

void DftbGenFormat::parse(const std::string &filename) {
  std::ifstream stream(filename);
  parse(stream);
}

auto parse_atoms_line(const std::string &line) {
  auto result = scn::scan<int, char>(line, "{} {}");
  if (!result) {
    throw std::runtime_error(
        "failure reading atom count line in DFTB gen format");
  }
  return result->values();
}

std::vector<Element> parse_elements_line(const std::string &line) {

  std::vector<Element> element_map;

  auto input = scn::ranges::subrange{line};

  while (auto result = scn::scan<std::string>(input, "{}")) {
    element_map.push_back(Element(result->value()));
    input = result->range();
  }

  occ::log::info("Found {} element symbols", element_map.size());
  return element_map;
}

auto parse_atom_line(const std::string &line) {
  auto result =
      scn::scan<int, int, double, double, double>(line, "{} {} {} {} {}");
  if (!result) {
    throw std::runtime_error("failure reading atom line in DFTB gen format");
  }
  return result->values();
}

auto parse_vector_line(const std::string &line) {
  auto result = scn::scan<double, double, double>(line, "{} {} {}");
  if (!result) {
    throw std::runtime_error("failure reading vector line in DFTB gen format");
  }
  return result->values();
}

void DftbGenFormat::parse(std::istream &stream) {

  std::string line;

  // read num atoms line
  std::getline(stream, line);
  auto [num_atoms, fmt_character] = parse_atoms_line(line);

  if (fmt_character == 'f' || fmt_character == 'F') {
    m_fractional = true;
    m_periodic = true;
  } else {
    if (fmt_character == 'c' || fmt_character == 'C') {
      m_fractional = false;
      m_periodic = false;
    } else {
      m_periodic = true;
      m_fractional = false;
    }
  }

  m_atomic_numbers.resize(num_atoms);
  m_positions.resize(3, num_atoms);
  m_symbol_index.resize(num_atoms);

  // read element symbols line
  std::getline(stream, line);
  auto element_map = parse_elements_line(line);
  std::vector<int> element_count(element_map.size());

  for (int i = 0; i < num_atoms; i++) {
    std::getline(stream, line);
    auto [idx, el, x, y, z] = parse_atom_line(line);
    m_symbol_index(i) = el - 1;
    // dftb gen indexes are start from 1
    m_atomic_numbers(i) = element_map[m_symbol_index(i)].atomic_number();
    m_positions(0, i) = x;
    m_positions(1, i) = y;
    m_positions(2, i) = z;
  }

  // skip origin line etc if we're not a periodic system
  if (!m_periodic)
    return;

  std::getline(stream, line);
  auto [x, y, z] = parse_vector_line(line);
  m_origin(0) = x;
  m_origin(1) = y;
  m_origin(2) = z;

  for (int i = 0; i < 3; i++) {
    std::getline(stream, line);
    auto [x, y, z] = parse_vector_line(line);
    m_lattice(0, i) = x;
    m_lattice(1, i) = y;
    m_lattice(2, i) = z;
  }
}

std::optional<occ::crystal::Crystal> DftbGenFormat::crystal() const {
  if (!m_periodic)
    return {};

  occ::crystal::UnitCell uc(m_lattice);
  occ::crystal::AsymmetricUnit asym(
      m_fractional ? m_positions : uc.to_fractional(m_positions),
      m_atomic_numbers);

  return occ::crystal::Crystal(asym, occ::crystal::SpaceGroup(1), uc);
}

std::optional<occ::core::Molecule> DftbGenFormat::molecule() const {
  if (m_periodic || m_fractional)
    return {};
  return Molecule(m_atomic_numbers, m_positions);
}

void DftbGenFormat::build_symbol_mapping() {
  m_symbols.clear();
  m_symbol_index.resize(num_atoms());

  ankerl::unordered_dense::map<std::string, size_t> index_map;

  for (int i = 0; i < num_atoms(); i++) {
    auto el = Element(m_atomic_numbers(i));
    const auto &symbol = el.symbol();
    const auto loc = index_map.find(symbol);
    if (loc != index_map.end()) {
      m_symbol_index(i) = loc->second;
    } else {
      m_symbol_index(i) = m_symbols.size();
      index_map.insert({symbol, m_symbols.size()});
      m_symbols.push_back(symbol);
    }
  }
}

void DftbGenFormat::set_molecule(const Molecule &mol) {
  m_positions = mol.positions();
  m_atomic_numbers = mol.atomic_numbers();
  m_periodic = false;
  m_fractional = false;
  build_symbol_mapping();
}

void DftbGenFormat::set_crystal(const Crystal &crystal) {
  auto c = Crystal::create_primitive_supercell(crystal, {1, 1, 1});
  const auto &asym = c.asymmetric_unit();
  m_positions = asym.positions;
  m_atomic_numbers = asym.atomic_numbers;
  m_lattice = c.unit_cell().direct();
  m_origin = Vec3::Zero();
  m_periodic = true;
  m_fractional = true;
  build_symbol_mapping();
}

char DftbGenFormat::format_character() const {
  if (m_periodic) {
    if (m_fractional)
      return 'F';
    else
      return 'S';
  }
  return 'C';
}

void DftbGenFormat::write(const std::string &filename) {
  std::ofstream stream(filename);
  write(stream);
}

void DftbGenFormat::write(std::ostream &stream) {
  fmt::print(stream, "{} {}\n", num_atoms(), format_character());
  for (size_t i = 0; i < m_symbols.size(); i++) {
    if (i != 0)
      fmt::print(stream, " ");
    fmt::print(stream, "{}", m_symbols[i]);
  }
  fmt::print(stream, "\n");

  for (size_t i = 0; i < num_atoms(); i++) {
    fmt::print(stream, "{:5d} {:3d} {:20.12e} {:20.12e} {:20.12e}\n", i + 1,
               m_symbol_index(i) + 1, m_positions(0, i), m_positions(1, i),
               m_positions(2, i));
  }

  if (m_periodic) {
    fmt::print(stream, "{:20.12e} {:20.12e} {:20.12e}\n", m_origin(0),
               m_origin(1), m_origin(2));
    for (size_t i = 0; i < 3; i++) {
      fmt::print(stream, "{:20.12e} {:20.12e} {:20.12e}\n", m_lattice(0, i),
                 m_lattice(1, i), m_lattice(2, i));
    }
  }
}

} // namespace occ::io
