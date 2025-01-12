#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/orca_json.h>

namespace occ::io {

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

Mat convert_mo_coefficients_from_orca_convention(const occ::qm::AOBasis &basis,
                                                 const Mat &mo) {
  occ::log::debug("Reordering MO coefficients from Molden ordering to "
                  "internal convention");
  auto shell2bf = basis.first_bf();
  Mat result(mo.rows(), mo.cols());
  size_t ncols = mo.cols();
  constexpr auto order = occ::gto::ShellOrder::Molden;

  for (size_t i = 0; i < basis.size(); i++) {
    const auto &shell = basis.shells()[i];
    size_t bf_first = shell2bf[i];
    int l = shell.l;
    if (l == 1) {
      // zxy -> yzx
      occ::log::debug("Swapping (l={}): z,x,y to y,z,x");
      result.block(bf_first + 0, 0, 1, ncols) =
          mo.block(bf_first + 2, 0, 1, ncols);
      result.block(bf_first + 1, 0, 1, ncols) =
          mo.block(bf_first + 0, 0, 1, ncols);
      result.block(bf_first + 2, 0, 1, ncols) =
          mo.block(bf_first + 1, 0, 1, ncols);
    } else {
      size_t idx = 0;
      auto func = [&](int am, int m) {
        int their_idx = occ::gto::shell_index_spherical<order>(am, m);
        int sign = 1;
        sign = fix_orca_phase_convention(am, m);
        result.row(bf_first + idx) = sign * mo.row(bf_first + their_idx);
        occ::log::debug("Swapping (l={}): {} <-> {}", l, idx, their_idx);
        idx++;
      };
      occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(func,
                                                                         l);
    }
  }
  return result;
}

/*
  Scaling factors for Orca

         class                           factor
    1    s                             * 1.0
         p
         d
         f(0,+1,-1,+2,-2)
         g(0,+1,-1,+2,-2)
         h(0,+1,-1,+2,-2,+5,-5)
    2    f(+3,-3)                      *-1.0
         g(+3,-3,+4,-4)
*/

double normalization_factor(double alpha, int l, int m, int n) {
  using occ::util::double_factorial;
  return std::sqrt(std::pow(4 * alpha, l + m + n) *
                   std::pow(2 * alpha / M_PI, 1.5) /
                   (double_factorial(2 * l - 1) * double_factorial(2 * m - 1) *
                    double_factorial(2 * n - 1)));
}

OrcaJSONReader::OrcaJSONReader(const std::string &filename) {
  occ::timing::start(occ::timing::category::io);
  open(filename);
  parse(m_json_file);
  occ::timing::stop(occ::timing::category::io);
}

OrcaJSONReader::OrcaJSONReader(std::istream &filehandle) {
  occ::timing::start(occ::timing::category::io);
  parse(filehandle);
  occ::timing::stop(occ::timing::category::io);
}

void OrcaJSONReader::open(const std::string &filename) {
  m_json_file.open(filename);
  if (m_json_file.fail() || m_json_file.bad()) {
    throw std::runtime_error("Unable to open fchk file: " + filename);
  }
}

void OrcaJSONReader::close() { m_json_file.close(); }

int string_to_l(const std::string &shell_label) {
  char label = shell_label[0];
  switch (label) {
  case 's':
    return 0;
  case 'S':
    return 0;
  case 'p':
    return 1;
  case 'P':
    return 1;
  case 'd':
    return 2;
  case 'D':
    return 2;
  case 'f':
    return 3;
  case 'F':
    return 3;
  case 'g':
    return 4;
  case 'G':
    return 4;
  case 'h':
    return 5;
  case 'H':
    return 5;
  case 'i':
    return 6;
  case 'I':
    return 6;
  case 'k':
    return 7;
  case 'K':
    return 7;
  default:
    return 0;
  }
}

void OrcaJSONReader::parse(std::istream &stream) {
  using occ::qm::Shell;

  auto j = nlohmann::json::parse(stream);
  const auto &mol = j["Molecule"];
  const auto &atoms_json = mol["Atoms"];

  size_t num_atoms = atoms_json.size();
  m_atomic_numbers = IVec(num_atoms);
  m_atom_labels.resize(num_atoms);
  m_atom_positions = Mat3N(3, num_atoms);
  std::vector<occ::qm::Shell> shells;

  for (const auto &atom : atoms_json) {
    size_t idx = atom["Idx"];
    m_atomic_numbers(idx) = atom["ElementNumber"];
    m_atom_labels[idx] = atom["ElementLabel"];
    const auto &pos = atom["Coords"];
    m_atom_positions(0, idx) = pos[0];
    m_atom_positions(1, idx) = pos[1];
    m_atom_positions(2, idx) = pos[2];

    std::array<double, 3> pos_array = {pos[0], pos[1], pos[2]};

    for (const auto &bf : atom["BasisFunctions"]) {
      std::vector<double> alpha;
      std::vector<double> coeffs;
      const auto &coeff = bf["Coefficients"];
      const auto &exp = bf["Exponents"];
      const auto &shell_kind = bf["Shell"];
      for (size_t i = 0; i < coeff.size(); i++) {
        coeffs.push_back(coeff[i]);
      }
      for (size_t i = 0; i < exp.size(); i++) {
        alpha.push_back(exp[i]);
      }

      int l = string_to_l(shell_kind);
      shells.emplace_back(Shell(l, alpha, {coeffs}, pos_array));
      shells.back().incorporate_shell_norm();
    }
  }
  m_basis = occ::qm::AOBasis(atoms(), shells);
  m_basis.set_pure(true); // orca only uses pure functions
  size_t nbf = m_basis.nbf();
  occ::log::debug("num atoms {}", num_atoms);

  bool unrestricted = mol["HFTyp"] == "UHF";

  if (unrestricted) {
    m_spinorbital_kind = qm::SpinorbitalKind::Unrestricted;
  }
  occ::log::debug("unrestricted: {}", unrestricted);
  occ::log::debug("nbf: {}", nbf);

  const auto &mos = mol["MolecularOrbitals"]["MOs"];
  const auto &mo_labels = mol["MolecularOrbitals"]["OrbitalLabels"];
  m_alpha_energies = Vec(nbf);
  m_alpha_coeffs = Mat(nbf, nbf);
  m_alpha_labels.reserve(nbf);
  if (unrestricted) {
    m_beta_energies = Vec(nbf);
    m_beta_coeffs = Mat(nbf, nbf);
    m_beta_labels.reserve(nbf);
  }

  size_t mo_idx = 0;
  for (const auto &mo : mos) {
    bool alpha_block = mo_idx < nbf;
    auto &e = alpha_block ? m_alpha_energies : m_beta_energies;
    auto &c = alpha_block ? m_alpha_coeffs : m_beta_coeffs;
    auto &n = alpha_block ? m_num_alpha : m_num_beta;
    auto &l = alpha_block ? m_alpha_labels : m_beta_labels;

    const auto &coeffs = mo["MOCoefficients"];
    Eigen::Index j = mo_idx % nbf;
    for (size_t i = 0; i < coeffs.size(); i++) {
      c(i, j) = coeffs[i];
    }
    e(j) = mo["OrbitalEnergy"];
    size_t occn = static_cast<size_t>(mo["Occupancy"]);
    l.push_back(mo_labels[mo_idx]);
    m_num_electrons += occn;
    n += unrestricted ? occn : occn / 2;
    mo_idx++;
  }

  if (unrestricted) {
    m_alpha_coeffs =
        convert_mo_coefficients_from_orca_convention(m_basis, m_alpha_coeffs);
    m_beta_coeffs =
        convert_mo_coefficients_from_orca_convention(m_basis, m_beta_coeffs);
  } else {
    m_num_beta = m_num_alpha;
    m_alpha_coeffs =
        convert_mo_coefficients_from_orca_convention(m_basis, m_alpha_coeffs);
  }

  occ::log::debug("Num electrons: {}", m_num_electrons);
  occ::log::debug("Num alpha electrons {}", m_num_alpha);
  occ::log::debug("Num beta electrons {}", m_num_beta);

  const auto &S = mol["S-Matrix"];
  size_t bf1 = 0;
  m_overlap = Mat(nbf, nbf);
  for (const auto &row : S) {
    size_t bf2 = 0;
    for (const auto &x : row) {
      m_overlap(bf1, bf2) = x;
      bf2++;
    }
    bf1++;
  }
}

std::vector<occ::core::Atom> OrcaJSONReader::atoms() const {
  std::vector<occ::core::Atom> atoms;
  atoms.reserve(m_atomic_numbers.size());
  for (size_t i = 0; i < m_atomic_numbers.size(); i++) {
    atoms.emplace_back(
        occ::core::Atom{m_atomic_numbers(i), m_atom_positions(0, i),
                        m_atom_positions(1, i), m_atom_positions(2, i)});
  }
  return atoms;
}

} // namespace occ::io
