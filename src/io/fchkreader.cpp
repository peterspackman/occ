#include <fmt/ostream.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/fchkreader.h>
#include <scn/scan.h>

namespace occ::io {

using occ::util::startswith;
using occ::util::trim_copy;

template <typename T>
void read_matrix_block(std::istream &stream, std::vector<T> &destination,
                       size_t count) {
  destination.reserve(count);
  std::string line;
  while (destination.size() < count) {
    std::getline(stream, line);
    auto input = scn::ranges::subrange{line};
    while (auto result = scn::scan<T>(input, "{}")) {
      destination.push_back(result->value());
      input = result->range();
    }
  }
}

FchkReader::FchkReader(const std::string &filename) {
  occ::timing::start(occ::timing::category::io);
  open(filename);
  parse(m_fchk_file);
  occ::timing::stop(occ::timing::category::io);
}

FchkReader::FchkReader(std::istream &filehandle) {
  occ::timing::start(occ::timing::category::io);
  parse(filehandle);
  occ::timing::stop(occ::timing::category::io);
}

void FchkReader::open(const std::string &filename) {
  m_fchk_file.open(filename);
  if (m_fchk_file.fail() || m_fchk_file.bad()) {
    throw std::runtime_error("Unable to open fchk file: " + filename);
  }
}

void FchkReader::close() { m_fchk_file.close(); }

void FchkReader::warn_about_ecp_reading() {
  if (!m_have_ecps) {
    occ::log::warn("Reading ECP basis is not supported - expect bad results.");
    m_have_ecps = true;
  }
}

FchkReader::LineLabel FchkReader::resolve_line(const std::string &line) const {
  std::string lt = trim_copy(line);
  if (startswith(lt, "Number of electrons", false))
    return LineLabel::NumElectrons;
  if (startswith(lt, "Atomic numbers", false))
    return LineLabel::AtomicNumbers;
  if (startswith(lt, "Nuclear charges", false))
    return LineLabel::NuclearCharges;
  if (startswith(lt, "Current cartesian coordinates", false))
    return LineLabel::AtomicPositions;
  if (startswith(lt, "Number of basis functions", false))
    return LineLabel::NumBasisFunctions;
  if (startswith(lt, "Number of electrons", false))
    return LineLabel::NumElectrons;
  if (startswith(lt, "Number of alpha electrons", false))
    return LineLabel::NumAlpha;
  if (startswith(lt, "Number of beta electrons", false))
    return LineLabel::NumBeta;
  if (startswith(lt, "SCF Energy", false))
    return LineLabel::SCFEnergy;
  if (startswith(lt, "Alpha MO coefficients", false))
    return LineLabel::AlphaMO;
  if (startswith(lt, "Beta MO coefficients", false))
    return LineLabel::BetaMO;
  if (startswith(lt, "Alpha Orbital Energies", false))
    return LineLabel::AlphaMOEnergies;
  if (startswith(lt, "Beta Orbital Energies", false))
    return LineLabel::BetaMOEnergies;
  if (startswith(lt, "Number of contracted shells", false))
    return LineLabel::NumShells;
  if (startswith(lt, "Number of primitive shells", false))
    return LineLabel::NumPrimitiveShells;
  if (startswith(lt, "Shell types", false))
    return LineLabel::ShellTypes;
  if (startswith(lt, "Number of primitives per shell", false))
    return LineLabel::PrimitivesPerShell;
  if (startswith(lt, "Shell to atom map", false))
    return LineLabel::ShellToAtomMap;
  if (startswith(lt, "Primitive exponents", false))
    return LineLabel::PrimitiveExponents;
  if (startswith(lt, "Contraction coefficients", false))
    return LineLabel::ContractionCoefficients;
  if (startswith(lt, "P(S=P) Contraction coefficients", false))
    return LineLabel::SPContractionCoefficients;
  if (startswith(lt, "Coordinates of each shell", false))
    return LineLabel::ShellCoordinates;
  if (startswith(lt, "Total SCF Density", false))
    return LineLabel::SCFDensity;
  if (startswith(lt, "Total MP2 Density", false))
    return LineLabel::MP2Density;
  if (startswith(lt, "Pure/Cartesian d shells", false))
    return LineLabel::PureCartesianD;
  if (startswith(lt, "Pure/Cartesian f shells", false))
    return LineLabel::PureCartesianF;
  if (startswith(lt, "ECP-RNFroz", false))
    return LineLabel::ECP_RNFroz;
  if (startswith(lt, "ECP-NLP", false))
    return LineLabel::ECP_NLP;
  if (startswith(lt, "ECP-CLP1", false))
    return LineLabel::ECP_CLP1;
  if (startswith(lt, "ECP-CLP2", false))
    return LineLabel::ECP_CLP2;
  if (startswith(lt, "ECP-ZLP", false))
    return LineLabel::ECP_ZLP;
  return LineLabel::Unknown;
}

void FchkReader::parse(std::istream &stream) {
  std::string line;
  while (std::getline(stream, line)) {
    switch (resolve_line(line)) {
    case LineLabel::NumElectrons: {
      auto result = scn::scan<int>(line, "Number of electrons I {}");
      m_num_electrons = result->value();
      break;
    }
    case LineLabel::SCFEnergy: {
      auto result = scn::scan<double>(line, "SCF Energy R {}");
      m_scf_energy = result->value();
      break;
    }
    case LineLabel::NumBasisFunctions: {
      auto result = scn::scan<int>(line, "Number of basis functions I {}");
      m_num_basis_functions = result->value();
      break;
    }
    case LineLabel::NumAlpha: {
      auto result = scn::scan<int>(line, "Number of alpha electrons I {}");
      m_num_alpha = result->value();
      break;
    }
    case LineLabel::NumBeta: {
      auto result = scn::scan<int>(line, "Number of beta electrons I {}");
      m_num_beta = result->value();
      break;
    }
    case LineLabel::AtomicNumbers: {
      auto result = scn::scan<size_t>(line, "Atomic numbers I N= {}");
      auto &count = result->value();
      read_matrix_block<int>(stream, m_atomic_numbers, count);
      break;
    }
    case LineLabel::NuclearCharges: {
      auto result = scn::scan<size_t>(line, "Nuclear charges R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_nuclear_charges, count);
      break;
    }
    case LineLabel::AtomicPositions: {
      auto result =
          scn::scan<size_t>(line, "Current cartesian coordinates R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_atomic_positions, count);
      break;
    }
    case LineLabel::AlphaMO: {
      auto result = scn::scan<size_t>(line, "Alpha MO coefficients R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_alpha_mos, count);
      break;
    }
    case LineLabel::BetaMO: {
      auto result = scn::scan<size_t>(line, "Beta MO coefficients R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_beta_mos, count);
      break;
    }
    case LineLabel::AlphaMOEnergies: {
      auto result = scn::scan<size_t>(line, "Alpha Orbital Energies R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_alpha_mo_energies, count);
      break;
    }
    case LineLabel::BetaMOEnergies: {
      auto result = scn::scan<size_t>(line, "Beta Orbital Energies R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_beta_mo_energies, count);
      break;
    }
    case LineLabel::NumShells: {
      auto result = scn::scan<int>(line, "Number of contracted shells I {}");
      m_basis.num_shells = result->value();
      break;
    }
    case LineLabel::NumPrimitiveShells: {
      auto result = scn::scan<int>(line, "Number of primitive shells I {}");
      m_basis.num_primitives = result->value();
      break;
    }
    case LineLabel::ShellTypes: {
      auto result = scn::scan<size_t>(line, "Shell types I N= {}");
      auto &count = result->value();
      read_matrix_block<int>(stream, m_basis.shell_types, count);
      break;
    }
    case LineLabel::PrimitivesPerShell: {
      auto result =
          scn::scan<size_t>(line, "Number of primitives per shell I N= {}");
      auto &count = result->value();
      read_matrix_block<int>(stream, m_basis.primitives_per_shell, count);
      break;
    }
    case LineLabel::ShellToAtomMap: {
      auto result = scn::scan<size_t>(line, "Shell to atom map I N= {}");
      auto &count = result->value();
      read_matrix_block<int>(stream, m_basis.shell2atom, count);
      break;
    }
    case LineLabel::PrimitiveExponents: {
      auto result = scn::scan<size_t>(line, "Primitive exponents R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_basis.primitive_exponents, count);
      break;
    }
    case LineLabel::ContractionCoefficients: {
      auto result = scn::scan<size_t>(line, "Contraction coefficients R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_basis.contraction_coefficients,
                                count);
      break;
    }
    case LineLabel::SPContractionCoefficients: {
      auto result =
          scn::scan<size_t>(line, "P(S=P) Contraction coefficients R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_basis.sp_contraction_coefficients,
                                count);
      break;
    }
    case LineLabel::ShellCoordinates: {
      auto result =
          scn::scan<size_t>(line, "Coordinates of each shell R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_basis.shell_coordinates, count);
      break;
    }
    case LineLabel::SCFDensity: {
      auto result = scn::scan<size_t>(line, "Total SCF Density R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_scf_density, count);
      break;
    }
    case LineLabel::MP2Density: {
      auto result = scn::scan<size_t>(line, "Total MP2 Density R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_mp2_density, count);
      break;
    }
    case LineLabel::PureCartesianD: {
      auto result = scn::scan<int>(line, "Pure/Cartesian d shells I {}");
      m_cartesian_d = (result->value() == 1);
      break;
    }
    case LineLabel::PureCartesianF: {
      auto result = scn::scan<int>(line, "Pure/Cartesian f shells I {}");
      m_cartesian_f = (result->value() == 1);
      break;
    }
    case LineLabel::ECP_RNFroz: {
      warn_about_ecp_reading();
      auto result = scn::scan<size_t>(line, "ECP-RNFroz R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_ecp_frozen, count);
      break;
    }
    case LineLabel::ECP_NLP: {
      warn_about_ecp_reading();
      auto result = scn::scan<size_t>(line, "ECP-NLP R N= {}");
      auto &count = result->value();
      read_matrix_block<int>(stream, m_ecp_nlp, count);
      break;
    }
    case LineLabel::ECP_CLP1: {
      warn_about_ecp_reading();
      auto result = scn::scan<size_t>(line, "ECP-CLP1 R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_ecp_clp1, count);
      break;
    }
    case LineLabel::ECP_CLP2: {
      warn_about_ecp_reading();
      auto result = scn::scan<size_t>(line, "ECP-CLP2 R N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_ecp_frozen, count);
      break;
    }
    case LineLabel::ECP_ZLP: {
      warn_about_ecp_reading();
      auto result = scn::scan<size_t>(line, "ECP-ZLP N= {}");
      auto &count = result->value();
      read_matrix_block<double>(stream, m_ecp_frozen, count);
      break;
    }
    default:
      continue;
    }
  }
}

std::vector<occ::core::Atom> FchkReader::atoms() const {
  std::vector<occ::core::Atom> atoms;
  atoms.reserve(m_atomic_numbers.size());
  for (size_t i = 0; i < m_atomic_numbers.size(); i++) {
    atoms.emplace_back(occ::core::Atom{
        m_atomic_numbers[i], m_atomic_positions[3 * i],
        m_atomic_positions[3 * i + 1], m_atomic_positions[3 * i + 2]});
  }
  return atoms;
}

occ::qm::AOBasis FchkReader::basis_set() const {
  size_t num_shells = m_basis.num_shells;
  std::vector<occ::qm::Shell> bs;
  size_t primitive_offset{0};
  constexpr int SP_SHELL{-1};
  bool any_pure = !(m_cartesian_d && m_cartesian_f);
  for (size_t i = 0; i < num_shells; i++) {
    // shell types: 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
    int shell_type = m_basis.shell_types[i];
    int l = std::abs(shell_type);
    // normally shell type < -1 will be pure
    occ::qm::Shell::Kind shell_kind = any_pure
                                          ? occ::qm::Shell::Kind::Spherical
                                          : occ::qm::Shell::Kind::Cartesian;

    size_t nprim = m_basis.primitives_per_shell[i];
    std::array<double, 3> position{
        m_basis.shell_coordinates[3 * i],
        m_basis.shell_coordinates[3 * i + 1],
        m_basis.shell_coordinates[3 * i + 2],
    };

    if (shell_type == SP_SHELL) {
      std::vector<double> alpha;
      std::vector<double> coeffs;
      std::vector<double> pcoeffs;
      for (size_t prim = 0; prim < nprim; prim++) {
        alpha.emplace_back(
            m_basis.primitive_exponents[primitive_offset + prim]);
        coeffs.emplace_back(
            m_basis.contraction_coefficients[primitive_offset + prim]);
        pcoeffs.emplace_back(
            m_basis.sp_contraction_coefficients[primitive_offset + prim]);
      }
      // sp shell
      bs.emplace_back(occ::qm::Shell(0, alpha, {coeffs}, position));
      bs.back().kind = shell_kind;
      bs.back().incorporate_shell_norm();
      bs.emplace_back(occ::qm::Shell(1, std::move(alpha), {pcoeffs}, position));
      bs.back().kind = shell_kind;
      bs.back().incorporate_shell_norm();
    } else {
      std::vector<double> alpha;
      std::vector<double> coeffs;
      for (size_t prim = 0; prim < nprim; prim++) {
        alpha.emplace_back(
            m_basis.primitive_exponents[primitive_offset + prim]);
        coeffs.emplace_back(
            m_basis.contraction_coefficients[primitive_offset + prim]);
      }
      bs.emplace_back(occ::qm::Shell(l, alpha, {coeffs}, position));
      bs.back().kind = shell_kind;
      bs.back().incorporate_shell_norm();
    }
    primitive_offset += nprim;
  }
  auto result = occ::qm::AOBasis(atoms(), bs);
  if (m_ecp_frozen.size() > 0) {
    result.ecp_electrons().clear();
    int i = 0;
    for (const auto &d : m_ecp_frozen) {
      result.ecp_electrons().push_back(static_cast<int>(d));
      occ::log::debug("ECP electrons for atom {}: {}", i, static_cast<int>(d));
      i++;
    }
  }
  result.set_pure(any_pure);
  return result;
}

void FchkReader::FchkBasis::print() const {
  size_t contraction_offset{0};
  size_t primitive_offset{0};
  for (size_t i = 0; i < num_shells; i++) {
    fmt::print("Shell {} on atom {}\n", i, shell2atom[i] - 1);
    fmt::print("Position: {:10.5f} {:10.5f} {:10.5f}\n",
               shell_coordinates[3 * i], shell_coordinates[3 * i + 1],
               shell_coordinates[3 * i + 2]);
    fmt::print("Angular momentum: {}\n", shell_types[i]);
    size_t num_primitives = primitives_per_shell[i];
    fmt::print("Primitives Gaussians: {}\n", num_primitives);
    fmt::print("Primitive exponents:");
    for (size_t i = 0; i < num_primitives; i++) {
      fmt::print(" {}", primitive_exponents[primitive_offset]);
      primitive_offset++;
    }
    fmt::print("\n");
    fmt::print("Contraction coefficients:");
    for (size_t i = 0; i < num_primitives; i++) {
      fmt::print(" {}", contraction_coefficients[contraction_offset]);
      contraction_offset++;
    }
    fmt::print("\n");
  }
}

Mat FchkReader::scf_density_matrix() const {

  Mat C_occ = alpha_mo_coefficients().leftCols(m_num_alpha);
  return C_occ * C_occ.transpose();
}

Mat FchkReader::mp2_density_matrix() const {

  size_t nbf{num_basis_functions()};
  assert(nbf * (nbf - 1) == m_mp2_density.size());
  Mat dm(nbf, nbf);
  size_t idx = 0;
  for (size_t i = 0; i < nbf; i++) {
    for (size_t j = i; j < nbf; j++) {
      if (i != j)
        dm(j, i) = m_mp2_density[idx];
      dm(i, j) = m_mp2_density[idx];
      idx++;
    }
  }
  return dm;
}

} // namespace occ::io
