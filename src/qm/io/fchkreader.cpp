#include <algorithm>
#include <cmath>
#include <fmt/ostream.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/qm/io/fchkreader.h>
#include <scn/scan.h>
#include <sstream>

namespace occ::io {

using occ::util::startswith;
using occ::util::trim_copy;

namespace {

[[noreturn]] void fail_with_error(const std::string &msg,
                                  const std::string &line) {
  throw std::runtime_error(fmt::format(
      "Unable to parse fchk file, error: {}, line = '{}'", msg, line));
}

// fchk header lines: label, type (I/R/C/L), then either a scalar value or
// "N=" followed by an array length. Gaussian writes these in fixed columns,
// but parse by tokens to tolerate hand-edited files.
struct FchkHeader {
  char type{' '};
  bool is_array{false};
  std::string value; // scalar value or array length
};

FchkHeader parse_header(const std::string &line) {
  std::istringstream ss(line);
  std::vector<std::string> tokens;
  std::string token;
  while (ss >> token)
    tokens.push_back(token);
  FchkHeader h;
  auto n = std::find(tokens.begin(), tokens.end(), "N=");
  if (n != tokens.end()) {
    if (n == tokens.begin() || n + 1 == tokens.end())
      fail_with_error("malformed array header", line);
    h.is_array = true;
    h.type = (n - 1)->front();
    h.value = *(n + 1);
  } else {
    if (tokens.size() < 2)
      fail_with_error("malformed header", line);
    h.type = tokens[tokens.size() - 2].front();
    h.value = tokens.back();
  }
  return h;
}

template <typename T>
T parse_value(const std::string &s, const std::string &line) {
  auto result = scn::scan<T>(s, "{}");
  if (!result)
    fail_with_error(fmt::format("expected a number, found '{}'", s), line);
  return result->value();
}

template <typename T>
void read_matrix_block(std::istream &stream, const std::string &header,
                       std::vector<T> &destination) {
  auto h = parse_header(header);
  if (!h.is_array)
    fail_with_error("expected an array (N=)", header);
  size_t count = parse_value<size_t>(h.value, header);
  destination.clear();
  destination.reserve(count);
  std::string line;
  while (destination.size() < count) {
    if (!std::getline(stream, line))
      fail_with_error(
          fmt::format("unexpected end of file, read {} of {} values",
                      destination.size(), count),
          header);
    // Fortran double precision exponents (1.0D+00) are not written by
    // Gaussian, but be lenient
    for (auto &c : line)
      if (c == 'D')
        c = 'E';
    auto input = scn::ranges::subrange{line};
    size_t before = destination.size();
    while (auto result = scn::scan<T>(input, "{}")) {
      destination.push_back(result->value());
      input = result->range();
    }
    if (destination.size() == before && !trim_copy(line).empty())
      fail_with_error("unable to read array values", line);
  }
  if (destination.size() != count)
    fail_with_error(
        fmt::format("expected {} values, read {}", count, destination.size()),
        header);
}

template <typename T> T read_scalar(const std::string &line) {
  auto h = parse_header(line);
  if (h.is_array)
    fail_with_error("expected a scalar value", line);
  return parse_value<T>(h.value, line);
}

} // namespace

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
  if (startswith(lt, "Number of independent functions", false))
    return LineLabel::NumIndependentFunctions;
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
  if (startswith(lt, "ECP-KFirst", false))
    return LineLabel::ECP_KFirst;
  if (startswith(lt, "ECP-KLast", false))
    return LineLabel::ECP_KLast;
  if (startswith(lt, "ECP-LMax", false))
    return LineLabel::ECP_LMax;
  if (startswith(lt, "ECP-LPSkip", false))
    return LineLabel::ECP_LPSkip;
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
    case LineLabel::NumElectrons:
      m_num_electrons = read_scalar<int>(line);
      break;
    case LineLabel::SCFEnergy:
      m_scf_energy = read_scalar<double>(line);
      break;
    case LineLabel::NumBasisFunctions:
      m_num_basis_functions = read_scalar<int>(line);
      break;
    case LineLabel::NumIndependentFunctions:
      m_num_independent_functions = read_scalar<int>(line);
      break;
    case LineLabel::NumAlpha:
      m_num_alpha = read_scalar<int>(line);
      break;
    case LineLabel::NumBeta:
      m_num_beta = read_scalar<int>(line);
      break;
    case LineLabel::AtomicNumbers:
      read_matrix_block<int>(stream, line, m_atomic_numbers);
      break;
    case LineLabel::NuclearCharges:
      read_matrix_block<double>(stream, line, m_nuclear_charges);
      break;
    case LineLabel::AtomicPositions:
      read_matrix_block<double>(stream, line, m_atomic_positions);
      break;
    case LineLabel::AlphaMO:
      read_matrix_block<double>(stream, line, m_alpha_mos);
      break;
    case LineLabel::BetaMO:
      read_matrix_block<double>(stream, line, m_beta_mos);
      break;
    case LineLabel::AlphaMOEnergies:
      read_matrix_block<double>(stream, line, m_alpha_mo_energies);
      break;
    case LineLabel::BetaMOEnergies:
      read_matrix_block<double>(stream, line, m_beta_mo_energies);
      break;
    case LineLabel::NumShells:
      m_basis.num_shells = read_scalar<int>(line);
      break;
    case LineLabel::NumPrimitiveShells:
      m_basis.num_primitives = read_scalar<int>(line);
      break;
    case LineLabel::ShellTypes:
      read_matrix_block<int>(stream, line, m_basis.shell_types);
      break;
    case LineLabel::PrimitivesPerShell:
      read_matrix_block<int>(stream, line, m_basis.primitives_per_shell);
      break;
    case LineLabel::ShellToAtomMap:
      read_matrix_block<int>(stream, line, m_basis.shell2atom);
      break;
    case LineLabel::PrimitiveExponents:
      read_matrix_block<double>(stream, line, m_basis.primitive_exponents);
      break;
    case LineLabel::ContractionCoefficients:
      read_matrix_block<double>(stream, line, m_basis.contraction_coefficients);
      break;
    case LineLabel::SPContractionCoefficients:
      read_matrix_block<double>(stream, line,
                                m_basis.sp_contraction_coefficients);
      break;
    case LineLabel::ShellCoordinates:
      read_matrix_block<double>(stream, line, m_basis.shell_coordinates);
      break;
    case LineLabel::SCFDensity:
      read_matrix_block<double>(stream, line, m_scf_density);
      break;
    case LineLabel::MP2Density:
      read_matrix_block<double>(stream, line, m_mp2_density);
      break;
    case LineLabel::PureCartesianD:
      m_cartesian_d = (read_scalar<int>(line) == 1);
      break;
    case LineLabel::PureCartesianF:
      m_cartesian_f = (read_scalar<int>(line) == 1);
      break;
    case LineLabel::ECP_RNFroz:
      read_matrix_block<double>(stream, line, m_ecp_frozen);
      break;
    case LineLabel::ECP_KFirst:
      read_matrix_block<int>(stream, line, m_ecp_kfirst);
      break;
    case LineLabel::ECP_KLast:
      read_matrix_block<int>(stream, line, m_ecp_klast);
      break;
    case LineLabel::ECP_LMax:
      read_matrix_block<int>(stream, line, m_ecp_lmax);
      break;
    case LineLabel::ECP_LPSkip:
      read_matrix_block<int>(stream, line, m_ecp_lpskip);
      break;
    case LineLabel::ECP_NLP:
      read_matrix_block<int>(stream, line, m_ecp_nlp);
      break;
    case LineLabel::ECP_CLP1:
      read_matrix_block<double>(stream, line, m_ecp_clp1);
      break;
    case LineLabel::ECP_CLP2:
      read_matrix_block<double>(stream, line, m_ecp_clp2);
      break;
    case LineLabel::ECP_ZLP:
      read_matrix_block<double>(stream, line, m_ecp_zlp);
      break;
    default:
      continue;
    }
  }
  validate();
}

void FchkReader::validate() const {
  auto fail = [](const std::string &msg) {
    throw std::runtime_error("Invalid fchk file: " + msg);
  };
  const size_t nbf = m_num_basis_functions;
  const size_t nmo = num_orbitals();
  if (m_atomic_numbers.empty())
    fail("no 'Atomic numbers' found");
  if (m_atomic_positions.size() != 3 * m_atomic_numbers.size())
    fail("'Current cartesian coordinates' does not match the number of atoms");
  if (nbf == 0)
    fail("no 'Number of basis functions' found");
  if (nmo > nbf)
    fail(fmt::format("more independent functions ({}) than basis functions "
                     "({})",
                     nmo, nbf));
  if (m_alpha_mos.size() != nbf * nmo)
    fail(fmt::format("'Alpha MO coefficients' has {} values, expected {} x {}",
                     m_alpha_mos.size(), nbf, nmo));
  if (m_alpha_mo_energies.size() != nmo)
    fail(fmt::format("'Alpha Orbital Energies' has {} values, expected {}",
                     m_alpha_mo_energies.size(), nmo));
  if (!m_beta_mos.empty() && m_beta_mos.size() != nbf * nmo)
    fail(fmt::format("'Beta MO coefficients' has {} values, expected {} x {}",
                     m_beta_mos.size(), nbf, nmo));
  if (!m_beta_mos.empty() && m_beta_mo_energies.size() != nmo)
    fail(fmt::format("'Beta Orbital Energies' has {} values, expected {}",
                     m_beta_mo_energies.size(), nmo));
  const size_t nsh = m_basis.num_shells;
  if (m_basis.shell_types.size() != nsh ||
      m_basis.primitives_per_shell.size() != nsh ||
      m_basis.shell_coordinates.size() != 3 * nsh)
    fail("inconsistent shell information");
  if (!m_ecp_frozen.empty()) {
    const size_t natom = m_atomic_numbers.size();
    const size_t nprim = m_ecp_nlp.size();
    if (m_ecp_frozen.size() != natom || m_ecp_lmax.size() != natom ||
        m_ecp_lpskip.size() != natom || m_ecp_kfirst.size() != 10 * natom ||
        m_ecp_klast.size() != 10 * natom)
      fail(fmt::format("ECP arrays don't match the {} atoms", natom));
    if (m_ecp_clp1.size() != nprim || m_ecp_zlp.size() != nprim)
      fail("ECP-NLP, ECP-CLP1 and ECP-ZLP lengths differ");
    for (size_t i = 0; i < m_ecp_kfirst.size(); i++) {
      int first = m_ecp_kfirst[i], last = m_ecp_klast[i];
      if (first == 0)
        continue;
      if (first < 1 || last < first || static_cast<size_t>(last) > nprim)
        fail(fmt::format("ECP primitive range {}-{} out of bounds ({})", first,
                         last, nprim));
    }
    for (size_t a = 0; a < natom; a++) {
      if (m_ecp_lmax[a] < 0 || m_ecp_lmax[a] > 9)
        fail(fmt::format("ECP-LMax {} out of range", m_ecp_lmax[a]));
    }
  }
}

// Gaussian's ECP layout (as used by e.g. MOKIT): for each atom there are 10
// channel slots, KFirst/KLast(atom, slot) stored column-major (natom x 10),
// giving 1-based primitive ranges in NLP (r power, r^(n-2) convention),
// ZLP (exponents) and CLP1 (coefficients). Slot 0 is the local channel
// (angular momentum LMax(atom)); slot k > 0 is the semi-local channel with
// l = k - 1. Atoms with LPSkip != 0 have no ECP. Atoms with the same ECP may
// share primitives. CLP2 holds spin-orbit terms, which occ ignores.
std::vector<occ::gto::Shell> FchkReader::ecp_shells() const {
  std::vector<occ::gto::Shell> shells;
  if (m_ecp_frozen.empty())
    return shells;
  const size_t natom = m_atomic_numbers.size();
  bool have_spin_orbit = false;
  for (double c : m_ecp_clp2)
    have_spin_orbit |= c != 0.0;
  if (have_spin_orbit)
    occ::log::warn("Ignoring spin-orbit terms (ECP-CLP2) in fchk ECPs");

  for (size_t a = 0; a < natom; a++) {
    if (m_ecp_lpskip[a] != 0)
      continue;
    std::array<double, 3> origin{m_atomic_positions[3 * a],
                                 m_atomic_positions[3 * a + 1],
                                 m_atomic_positions[3 * a + 2]};
    const int lmax = m_ecp_lmax[a];
    for (int slot = 0; slot <= lmax; slot++) {
      int first = m_ecp_kfirst[a + natom * slot];
      int last = m_ecp_klast[a + natom * slot];
      if (first == 0)
        continue;
      const int l = slot == 0 ? lmax : slot - 1;
      std::vector<double> exponents, coefficients;
      std::vector<int> powers;
      for (int p = first - 1; p < last; p++) {
        exponents.push_back(m_ecp_zlp[p]);
        coefficients.push_back(m_ecp_clp1[p]);
        powers.push_back(m_ecp_nlp[p]);
      }
      occ::gto::Shell shell(l, exponents, {coefficients}, origin);
      shell.ecp_r_exponents =
          Eigen::Map<const IVec>(powers.data(), powers.size());
      shells.push_back(std::move(shell));
    }
  }
  return shells;
}

Mat FchkReader::padded_mo_coefficients(const std::vector<double> &mos) const {
  const size_t nbf = m_num_basis_functions;
  Mat result = Mat::Zero(nbf, nbf);
  result.leftCols(num_orbitals()) =
      Eigen::Map<const Mat>(mos.data(), nbf, num_orbitals());
  return result;
}

Vec FchkReader::padded_mo_energies(const std::vector<double> &energies) const {
  Vec result = Vec::Zero(m_num_basis_functions);
  result.head(num_orbitals()) =
      Eigen::Map<const Vec>(energies.data(), num_orbitals());
  return result;
}

Mat FchkReader::alpha_mo_coefficients() const {
  return padded_mo_coefficients(m_alpha_mos);
}

Vec FchkReader::alpha_mo_energies() const {
  return padded_mo_energies(m_alpha_mo_energies);
}

// Restricted open-shell files only contain alpha MOs, shared by both spins
Mat FchkReader::beta_mo_coefficients() const {
  return padded_mo_coefficients(m_beta_mos.empty() ? m_alpha_mos : m_beta_mos);
}

Vec FchkReader::beta_mo_energies() const {
  return padded_mo_energies(m_beta_mos.empty() ? m_alpha_mo_energies
                                               : m_beta_mo_energies);
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

occ::gto::AOBasis FchkReader::basis_set() const {
  size_t num_shells = m_basis.num_shells;
  std::vector<occ::gto::Shell> bs;
  size_t primitive_offset{0};
  constexpr int SP_SHELL{-1};
  // shell types: 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f, ...
  // occ needs one kind for the whole basis
  bool any_pure = false, any_cart = false;
  for (int t : m_basis.shell_types) {
    any_pure |= t < -1;
    any_cart |= t > 1;
  }
  if (any_pure && any_cart) {
    throw std::runtime_error(
        "fchk file mixes pure and Cartesian shells (e.g. 6D 7F), which is "
        "not supported: rerun with 5D 7F or 6D 10F");
  }
  if (!any_pure && !any_cart)
    any_pure = !(m_cartesian_d || m_cartesian_f);
  const auto shell_kind = any_pure ? occ::gto::Shell::Kind::Spherical
                                   : occ::gto::Shell::Kind::Cartesian;
  for (size_t i = 0; i < num_shells; i++) {
    int shell_type = m_basis.shell_types[i];
    int l = std::abs(shell_type);

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
      bs.emplace_back(occ::gto::Shell(0, alpha, {coeffs}, position));
      bs.back().kind = shell_kind;
      bs.back().incorporate_shell_norm();
      bs.emplace_back(occ::gto::Shell(1, std::move(alpha), {pcoeffs}, position));
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
      bs.emplace_back(occ::gto::Shell(l, alpha, {coeffs}, position));
      bs.back().kind = shell_kind;
      bs.back().incorporate_shell_norm();
    }
    primitive_offset += nprim;
  }
  auto result = occ::gto::AOBasis(atoms(), bs, "", ecp_shells());
  if (m_ecp_frozen.size() > 0) {
    std::vector<int> ecp_electrons;
    for (const auto &d : m_ecp_frozen) {
      ecp_electrons.push_back(static_cast<int>(std::lround(d)));
      occ::log::debug("ECP electrons for atom {}: {}", ecp_electrons.size() - 1,
                      ecp_electrons.back());
    }
    result.set_ecp_electrons(ecp_electrons);
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

namespace {
// Gaussian stores symmetric matrices as the row-wise lower triangle
Mat unpack_lower_triangle(const std::vector<double> &values, size_t n) {
  if (values.size() != n * (n + 1) / 2)
    throw std::runtime_error(fmt::format(
        "fchk density has {} values, expected {} for {} basis functions",
        values.size(), n * (n + 1) / 2, n));
  Mat result(n, n);
  size_t idx = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j <= i; j++) {
      result(i, j) = values[idx];
      result(j, i) = values[idx];
      idx++;
    }
  }
  return result;
}
} // namespace

// occ's restricted density convention: half the total density
Mat FchkReader::scf_density_matrix() const {
  return 0.5 * unpack_lower_triangle(m_scf_density, m_num_basis_functions);
}

Mat FchkReader::mp2_density_matrix() const {
  return 0.5 * unpack_lower_triangle(m_mp2_density, m_num_basis_functions);
}

} // namespace occ::io
