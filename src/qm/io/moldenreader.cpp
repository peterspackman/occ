#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fmt/core.h>
#include <fstream>
#include <numbers>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/gto/shell_order.h>
#include <occ/qm/io/moldenreader.h>
#include <sstream>

namespace occ::io {

namespace {

[[noreturn]] void fail_with_error(const std::string &msg,
                                  const std::string &line) {
  throw std::runtime_error(fmt::format(
      "Unable to parse molden file, error: {}, line = '{}'", msg, line));
}

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

std::string trimmed(std::string s) {
  occ::util::trim(s);
  return s;
}

inline bool is_section_line(const std::string &line) {
  auto pos = line.find_first_not_of(" \t");
  return pos != std::string::npos && line[pos] == '[' &&
         line.find(']', pos) != std::string::npos;
}

// Fortran codes may write exponents as 1.0D+01
inline std::string fortran_to_c_exponents(std::string s) {
  for (auto &c : s) {
    if (c == 'D' || c == 'd')
      c = 'E';
  }
  return s;
}

std::vector<std::string> split_whitespace(const std::string &line) {
  std::vector<std::string> tokens;
  std::istringstream ss(line);
  std::string token;
  while (ss >> token)
    tokens.push_back(token);
  return tokens;
}

double parse_double(const std::string &token, const std::string &line) {
  std::string s = fortran_to_c_exponents(token);
  char *end = nullptr;
  double value = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0')
    fail_with_error(fmt::format("expected a number, found '{}'", token), line);
  return value;
}

int parse_int(const std::string &token, const std::string &line) {
  char *end = nullptr;
  long value = std::strtol(token.c_str(), &end, 10);
  if (end == token.c_str() || *end != '\0')
    fail_with_error(fmt::format("expected an integer, found '{}'", token),
                    line);
  return static_cast<int>(value);
}

int l_from_label(const std::string &label, const std::string &line) {
  static const std::string labels = "spdfghi";
  if (label.size() != 1)
    fail_with_error(fmt::format("unknown shell label '{}'", label), line);
  auto pos = labels.find(std::tolower(static_cast<unsigned char>(label[0])));
  if (pos == std::string::npos)
    fail_with_error(fmt::format("unknown shell label '{}'", label), line);
  return static_cast<int>(pos);
}

occ::gto::Shell make_shell(int l, const std::vector<double> &alpha,
                           std::vector<double> coeffs,
                           const std::array<double, 3> &position) {
  // Coefficients normally refer to normalized primitives; if the contraction
  // isn't normalized under that assumption, treat them as referring to
  // unnormalized primitives instead.
  double pi2_34 = std::pow(2 * std::numbers::pi_v<double>, 0.75);
  double norm = 0.0;
  for (size_t i = 0; i < coeffs.size(); i++) {
    size_t j;
    double a = alpha[i];
    for (j = 0; j < i; j++) {
      double b = alpha[j];
      double ab = 2 * std::sqrt(a * b) / (a + b);
      norm += 2 * coeffs[i] * coeffs[j] * std::pow(ab, l + 1.5);
    }
    norm += coeffs[i] * coeffs[j];
  }
  norm = std::sqrt(norm) * pi2_34;
  if (std::abs(pi2_34 - norm) > 1e-4) {
    occ::log::debug("Renormalizing coefficients, shell norm: {:6.3f}", norm);
    for (size_t i = 0; i < coeffs.size(); i++) {
      coeffs[i] /= std::pow(4 * alpha[i], 0.5 * l + 0.75);
      coeffs[i] = coeffs[i] * pi2_34 / norm;
    }
  }
  auto shell = occ::gto::Shell(l, alpha, {coeffs}, position);
  shell.incorporate_shell_norm();
  return shell;
}

inline int fix_orca_phase_convention(int l, int m) {
  if (l == 3 && std::abs(m) == 3) {
    // c0 c1 s1 c2 s2 c3 s3
    // +  +  +  +  +  -  -
    return -1;
  } else if (l == 4 && std::abs(m) >= 3) {
    // c0 c1 s1 c2 s2 c3 s3 c4 s4
    // +  +  +  +  +  -  -  -  -
    return -1;
  } else if (l == 5 && std::abs(m) >= 3 && std::abs(m) < 5) {
    // c0 c1 s1 c2 s2 c3 s3 c4 s4 c5 s5
    // +  +  +  +  +  -  -  -  -  +  +
    return -1;
  }
  return 1;
}

inline bool is_integer(double x, double tol = 1e-6) {
  return std::abs(x - std::round(x)) < tol;
}

} // namespace

MoldenReader::MoldenReader(const std::string &filename) : m_filename(filename) {
  occ::timing::start(occ::timing::category::io);
  std::ifstream file(filename);
  if (!file) {
    occ::timing::stop(occ::timing::category::io);
    throw std::runtime_error(
        fmt::format("Unable to open molden file: '{}'", filename));
  }
  parse(file);
  occ::timing::stop(occ::timing::category::io);
}

MoldenReader::MoldenReader(std::istream &file) {
  occ::timing::start(occ::timing::category::io);
  parse(file);
  occ::timing::stop(occ::timing::category::io);
}

void MoldenReader::parse(std::istream &stream) {
  Lines lines;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    lines.push_back(std::move(line));
  }

  size_t i = 0;
  while (i < lines.size()) {
    const std::string &current = lines[i];
    if (!is_section_line(current)) {
      i++;
      continue;
    }
    auto l = current.find('[');
    auto u = current.find(']', l);
    std::string name = upper(trimmed(current.substr(l + 1, u - l - 1)));
    std::string args = trimmed(current.substr(u + 1));
    occ::log::debug("Found molden section: [{}] {}", name, args);
    i++;
    if (name == "TITLE") {
      parse_title_section(lines, i);
    } else if (name == "ATOMS") {
      parse_atoms_section(args, lines, i);
    } else if (name == "GTO") {
      parse_gto_section(lines, i);
    } else if (name == "MO") {
      parse_mo_section(lines, i);
    } else if (name == "STO") {
      throw std::runtime_error(
          "Slater-type orbitals ([STO]) in molden files are not supported");
    } else if (name == "CORE") {
      m_have_ecp_core = true;
    } else {
      set_angular_flags(name);
    }
  }
  finalize();
}

void MoldenReader::set_angular_flags(const std::string &name) {
  // From the Molden format specification:
  //   [5D]    5D and 7F         [5D10F] 5D and 10F
  //   [5D7F]  5D and 7F         [7F]    6D and 7F
  //   [9G]    spherical G
  // PySCF and ORCA write these as separate [5d] [7f] [9g] sections, and
  // PySCF writes [6d] [10f] [15g] for Cartesian files.
  if (name.rfind("5D", 0) == 0) {
    m_pure_d = true;
    m_pure_f = name.find("10F") == std::string::npos;
  } else if (name.rfind("6D", 0) == 0) {
    m_pure_d = false;
  } else if (name.rfind("7F", 0) == 0) {
    m_pure_f = true;
  } else if (name.rfind("10F", 0) == 0) {
    m_pure_f = false;
  } else if (name.rfind("9G", 0) == 0) {
    m_pure_g = true;
  } else if (name.rfind("15G", 0) == 0) {
    m_pure_g = false;
  } else {
    occ::log::debug("Ignoring molden section [{}]", name);
  }
}

void MoldenReader::parse_title_section(const Lines &lines, size_t &i) {
  for (; i < lines.size() && !is_section_line(lines[i]); i++) {
    if (lines[i].find("orca_2mkl") != std::string::npos) {
      occ::log::debug("Detected ORCA molden file");
      m_source = Source::Orca;
    }
  }
}

void MoldenReader::parse_atoms_section(const std::string &args,
                                       const Lines &lines, size_t &i) {
  // args is e.g. "AU", "(AU)", "Angs" or "(Angs)"; Molden defaults to bohr
  double factor = 1.0;
  if (upper(args).find("ANG") != std::string::npos)
    factor = occ::units::ANGSTROM_TO_BOHR;

  for (; i < lines.size() && !is_section_line(lines[i]); i++) {
    const auto &line = lines[i];
    auto tokens = split_whitespace(line);
    if (tokens.empty())
      continue;
    if (tokens.size() < 6)
      fail_with_error("expected 'symbol index Z x y z' in [Atoms]", line);
    occ::core::Atom atom;
    atom.atomic_number = parse_int(tokens[2], line);
    atom.x = parse_double(tokens[3], line) * factor;
    atom.y = parse_double(tokens[4], line) * factor;
    atom.z = parse_double(tokens[5], line) * factor;
    m_atoms.push_back(atom);
  }
}

void MoldenReader::parse_gto_section(const Lines &lines, size_t &i) {
  if (m_atoms.empty())
    throw std::runtime_error(
        "Unable to parse molden file: [GTO] section found before [Atoms]");

  std::optional<std::array<double, 3>> position;
  while (i < lines.size() && !is_section_line(lines[i])) {
    const auto &line = lines[i++];
    auto tokens = split_whitespace(line);
    if (tokens.empty())
      continue;

    if (std::isdigit(static_cast<unsigned char>(tokens[0][0]))) {
      // atom header: "atom_index 0"
      int atom_idx = parse_int(tokens[0], line);
      if (atom_idx < 1 || atom_idx > static_cast<int>(m_atoms.size()))
        fail_with_error(fmt::format("atom index {} out of range (1-{})",
                                    atom_idx, m_atoms.size()),
                        line);
      const auto &atom = m_atoms[atom_idx - 1];
      position = std::array<double, 3>{atom.x, atom.y, atom.z};
      continue;
    }

    // shell header: "label num_primitives [scale factor]"
    if (!position)
      fail_with_error("shell found before an atom index in [GTO]", line);
    if (tokens.size() < 2)
      fail_with_error("expected 'label num_primitives' in [GTO]", line);
    std::string label = tokens[0];
    std::transform(label.begin(), label.end(), label.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    int nprim = parse_int(tokens[1], line);
    if (nprim < 1)
      fail_with_error("shell has no primitives", line);
    bool sp = label == "sp";

    std::vector<double> alpha, c0, c1;
    for (int p = 0; p < nprim; p++) {
      if (i >= lines.size())
        fail_with_error("unexpected end of file in [GTO] primitives", line);
      const auto &prim_line = lines[i++];
      auto prim = split_whitespace(prim_line);
      if (prim.size() < (sp ? 3u : 2u))
        fail_with_error("too few values for primitive", prim_line);
      alpha.push_back(parse_double(prim[0], prim_line));
      c0.push_back(parse_double(prim[1], prim_line));
      if (sp)
        c1.push_back(parse_double(prim[2], prim_line));
    }

    if (sp) {
      m_shells.push_back(make_shell(0, alpha, c0, *position));
      m_shells.push_back(make_shell(1, alpha, c1, *position));
    } else {
      m_shells.push_back(
          make_shell(l_from_label(label, line), alpha, c0, *position));
    }
  }
}

void MoldenReader::parse_mo_section(const Lines &lines, size_t &i) {
  // Each orbital is a block of keyword lines (Sym=, Ene=, Spin=, Occup=, in
  // any order, any of them optional) followed by "index coefficient" lines.
  // Coefficients may be omitted when zero.
  Orbital current;
  bool have_keywords = false;
  auto flush = [&]() {
    if (have_keywords || !current.coefficients.empty())
      m_orbitals.push_back(std::move(current));
    current = Orbital{};
    have_keywords = false;
  };

  for (; i < lines.size() && !is_section_line(lines[i]); i++) {
    const auto &line = lines[i];
    auto eq = line.find('=');
    if (eq != std::string::npos) {
      if (!current.coefficients.empty())
        flush();
      have_keywords = true;
      std::string key = upper(trimmed(line.substr(0, eq)));
      std::string value = trimmed(line.substr(eq + 1));
      if (key.rfind("ENE", 0) == 0) {
        current.energy = parse_double(value, line);
      } else if (key.rfind("SPIN", 0) == 0) {
        current.alpha = upper(value).rfind("BETA", 0) != 0;
      } else if (key.rfind("OCCUP", 0) == 0) {
        current.occupation = parse_double(value, line);
      }
      continue;
    }
    auto tokens = split_whitespace(line);
    if (tokens.empty())
      continue;
    if (tokens.size() < 2)
      fail_with_error("expected 'index coefficient' in [MO]", line);
    int idx = parse_int(tokens[0], line);
    if (idx < 1)
      fail_with_error("basis function index must be >= 1", line);
    current.coefficients.emplace_back(idx, parse_double(tokens[1], line));
  }
  flush();
}

bool MoldenReader::shell_is_pure_in_file(int l) const {
  switch (l) {
  case 0:
  case 1:
    return false;
  case 2:
    return m_pure_d;
  case 3:
    return m_pure_f;
  default:
    // the specification stops at g; treat higher shells like g
    return m_pure_g;
  }
}

Mat MoldenReader::to_internal_convention(const Mat &file_mos,
                                         bool target_pure) const {
  using occ::gto::ShellOrder;
  const Eigen::Index ncols = file_mos.cols();
  const bool orca = m_source == Source::Orca;
  Mat result = Mat::Zero(m_nbf, ncols);

  size_t file_offset = 0, our_offset = 0;
  for (const auto &shell : m_shells) {
    const int l = shell.l;
    const bool file_pure = shell_is_pure_in_file(l);
    const size_t n_sph = 2 * l + 1;
    const size_t n_cart = (l + 1) * (l + 2) / 2;

    if (l == 0) {
      result.row(our_offset) = file_mos.row(file_offset);
      file_offset += 1;
      our_offset += 1;
      continue;
    }
    if (l == 1) {
      // Molden p order is always x, y, z; occ's spherical p order is y, z, x
      if (target_pure) {
        result.row(our_offset + 0) = file_mos.row(file_offset + 1);
        result.row(our_offset + 1) = file_mos.row(file_offset + 2);
        result.row(our_offset + 2) = file_mos.row(file_offset + 0);
      } else {
        result.middleRows(our_offset, 3) = file_mos.middleRows(file_offset, 3);
      }
      file_offset += 3;
      our_offset += 3;
      continue;
    }

    if (file_pure) {
      // Molden order m = 0, +1, -1, +2, -2, ... -> occ order m = -l..l
      Mat sph(n_sph, ncols);
      size_t idx = 0;
      auto func = [&](int am, int m) {
        int theirs = occ::gto::shell_index_spherical<ShellOrder::Molden>(am, m);
        int sign = orca ? fix_orca_phase_convention(am, m) : 1;
        sph.row(idx++) = sign * file_mos.row(file_offset + theirs);
      };
      occ::gto::iterate_over_shell<false, ShellOrder::Default>(func, l);
      if (target_pure) {
        result.middleRows(our_offset, n_sph) = sph;
        our_offset += n_sph;
      } else {
        // mixed file: express this spherical shell in Cartesian functions
        Mat c2s = occ::gto::cartesian_to_spherical_transformation_matrix(l);
        result.middleRows(our_offset, n_cart) = c2s.transpose() * sph;
        our_offset += n_cart;
      }
      file_offset += n_sph;
    } else {
      // Molden Cartesian functions are each unit-normalized; occ's are not
      size_t idx = 0;
      auto func = [&](int pi, int pj, int pk, int) {
        int theirs =
            occ::gto::shell_index_cartesian<ShellOrder::Molden>(pi, pj, pk, l);
        result.row(our_offset + idx) =
            file_mos.row(file_offset + theirs) *
            occ::gto::cartesian_normalization_factor(pi, pj, pk);
        idx++;
      };
      occ::gto::iterate_over_shell<true, ShellOrder::Default>(func, l);
      file_offset += n_cart;
      our_offset += n_cart;
    }
  }
  return result;
}

void MoldenReader::finalize() {
  if (m_atoms.empty())
    throw std::runtime_error("Unable to parse molden file: no [Atoms] found");
  if (m_shells.empty())
    throw std::runtime_error("Unable to parse molden file: no [GTO] found");
  if (m_orbitals.empty())
    throw std::runtime_error("Unable to parse molden file: no [MO] found");
  if (m_have_ecp_core) {
    occ::log::warn("Molden file lists ECP core electrons, but molden files "
                   "do not store the ECPs themselves: nuclear charges and "
                   "properties that depend on them will be wrong");
  }

  // Decide the basis kind. occ needs one kind for the whole basis, so a file
  // mixing spherical and Cartesian shells is converted to Cartesian (exact,
  // since spherical functions lie in the span of Cartesian ones).
  bool any_pure = false, any_cart = false;
  size_t nbf_file = 0;
  for (const auto &shell : m_shells) {
    const int l = shell.l;
    const bool pure = shell_is_pure_in_file(l);
    nbf_file += pure ? 2 * l + 1 : (l + 1) * (l + 2) / 2;
    if (l >= 2) {
      any_pure |= pure;
      any_cart |= !pure;
    }
  }
  const bool target_pure = any_pure ? !any_cart : m_pure_d;
  if (any_pure && any_cart) {
    occ::log::info("Molden file mixes spherical and Cartesian shells, "
                   "converting to a Cartesian basis");
  }
  const auto kind = target_pure ? occ::gto::Shell::Kind::Spherical
                                : occ::gto::Shell::Kind::Cartesian;
  for (auto &shell : m_shells)
    shell.kind = kind;
  m_nbf = basis_set().nbf();
  occ::log::debug("Molden basis: {} shells, {} functions in file, {} {}",
                  m_shells.size(), nbf_file, m_nbf,
                  target_pure ? "spherical" : "Cartesian");

  // Split orbitals by spin and build the coefficient matrices in file order
  std::vector<const Orbital *> alpha, beta;
  for (const auto &orb : m_orbitals) {
    for (const auto &[idx, c] : orb.coefficients) {
      if (static_cast<size_t>(idx) > nbf_file) {
        throw std::runtime_error(fmt::format(
            "Unable to parse molden file: MO coefficient index {} exceeds the "
            "{} basis functions implied by [GTO] and the [5D]/[7F]/[9G] "
            "flags (spherical d/f/g: {}/{}/{})",
            idx, nbf_file, m_pure_d, m_pure_f, m_pure_g));
      }
    }
    (orb.alpha ? alpha : beta).push_back(&orb);
  }
  if (alpha.size() > nbf_file || beta.size() > nbf_file) {
    throw std::runtime_error(fmt::format(
        "Unable to parse molden file: more MOs ({} alpha, {} beta) than basis "
        "functions ({})",
        alpha.size(), beta.size(), nbf_file));
  }

  auto build = [&](const std::vector<const Orbital *> &orbs, Mat &coeffs,
                   Vec &energies, Vec &occupations) {
    // Files from linearly dependent bases have fewer MOs than functions;
    // the remaining columns are left as zero (unoccupied) orbitals.
    Mat file_mos = Mat::Zero(nbf_file, m_nbf);
    energies = Vec::Zero(m_nbf);
    occupations = Vec::Zero(m_nbf);
    for (size_t j = 0; j < orbs.size(); j++) {
      for (const auto &[idx, c] : orbs[j]->coefficients)
        file_mos(idx - 1, j) = c;
      energies(j) = orbs[j]->energy;
      occupations(j) = orbs[j]->occupation;
    }
    coeffs = to_internal_convention(file_mos, target_pure);
  };

  build(alpha, m_molecular_orbitals_alpha, m_energies_alpha,
        m_occupations_alpha);
  if (alpha.size() < m_nbf) {
    occ::log::debug("Molden file has {} alpha MOs for {} basis functions",
                    alpha.size(), m_nbf);
  }

  const double alpha_total = m_occupations_alpha.sum();
  if (!beta.empty()) {
    build(beta, m_molecular_orbitals_beta, m_energies_beta, m_occupations_beta);
    m_kind = occ::qm::SpinorbitalKind::Unrestricted;
    m_num_alpha = static_cast<size_t>(std::llround(alpha_total));
    m_num_beta = static_cast<size_t>(std::llround(m_occupations_beta.sum()));
    return;
  }

  // Alpha orbitals only: restricted, or restricted open-shell if the
  // occupations are all integers and some orbital is singly occupied.
  bool integral = true, singly_occupied = false;
  for (Eigen::Index j = 0; j < m_occupations_alpha.size(); j++) {
    double o = m_occupations_alpha(j);
    integral &= is_integer(o) && o > -1e-6 && o < 2 + 1e-6;
    singly_occupied |= is_integer(o) && std::llround(o) == 1;
  }
  const size_t n_electrons = static_cast<size_t>(std::llround(alpha_total));
  if (integral && singly_occupied) {
    occ::log::debug("Molden file is restricted open-shell, reading as "
                    "unrestricted");
    m_kind = occ::qm::SpinorbitalKind::Unrestricted;
    m_molecular_orbitals_beta = m_molecular_orbitals_alpha;
    m_energies_beta = m_energies_alpha;
    m_occupations_beta = (m_occupations_alpha.array() - 1.0).max(0.0).matrix();
    m_occupations_alpha = m_occupations_alpha.array().min(1.0).matrix();
    m_num_alpha = static_cast<size_t>(std::llround(m_occupations_alpha.sum()));
    m_num_beta = static_cast<size_t>(std::llround(m_occupations_beta.sum()));
    return;
  }

  m_kind = occ::qm::SpinorbitalKind::Restricted;
  m_num_beta = n_electrons / 2;
  m_num_alpha = n_electrons - m_num_beta;
  m_energies_beta = m_energies_alpha;
}

} // namespace occ::io
