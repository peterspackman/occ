#include <cmath>
#include <occ/core/constants.h>
#include <occ/core/util.h>
#include <occ/qm/basisset.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/occshell.h>

namespace detail {
/// fac[k] = k!
static constexpr std::array<int64_t, 21> fac = {{1LL,
                                                 1LL,
                                                 2LL,
                                                 6LL,
                                                 24LL,
                                                 120LL,
                                                 720LL,
                                                 5040LL,
                                                 40320LL,
                                                 362880LL,
                                                 3628800LL,
                                                 39916800LL,
                                                 479001600LL,
                                                 6227020800LL,
                                                 87178291200LL,
                                                 1307674368000LL,
                                                 20922789888000LL,
                                                 355687428096000LL,
                                                 6402373705728000LL,
                                                 121645100408832000LL,
                                                 2432902008176640000LL}};
/// df_Kminus1[k] = (k-1)!!
static constexpr std::array<int64_t, 31> df_Kminus1 = {{1LL,
                                                        1LL,
                                                        1LL,
                                                        2LL,
                                                        3LL,
                                                        8LL,
                                                        15LL,
                                                        48LL,
                                                        105LL,
                                                        384LL,
                                                        945LL,
                                                        3840LL,
                                                        10395LL,
                                                        46080LL,
                                                        135135LL,
                                                        645120LL,
                                                        2027025LL,
                                                        10321920LL,
                                                        34459425LL,
                                                        185794560LL,
                                                        654729075LL,
                                                        3715891200LL,
                                                        13749310575LL,
                                                        81749606400LL,
                                                        316234143225LL,
                                                        1961990553600LL,
                                                        7905853580625LL,
                                                        51011754393600LL,
                                                        213458046676875LL,
                                                        1428329123020800LL,
                                                        6190283353629375LL}};
} // namespace detail

namespace occ::qm {

double gint(int n, double alpha) {
    double n1_2 = 0.5 * (n + 1);
    return std::tgamma(n1_2) / (2 * std::pow(alpha, n1_2));
}

double gto_norm(int l, double alpha) {
    // to evaluate without the gamma function:
    // sqrt of the following:
    //
    // 2^{2l + 3}(l + 1)!\alpha^{l + 3/2}
    // -------------------
    // (2l + 2)! \pi^{1/2}
    //
    /*
    double nn = std::pow(2, (2 * l + 3)) * util::factorial(l + 1) *
                pow((2 * alpha), (l + 1.5)) /
                (util::factorial(2 * l + 2) * constants::sqrt_pi<double>);
    return sqrt(nn);
    */
    return 1.0 / std::sqrt(gint(l * 2 + 2, 2 * alpha));
}

void normalize_contracted_gto(int l, const Vec &alpha, Mat &coeffs) {
    Mat ee = alpha.rowwise().replicate(alpha.rows()) +
             alpha.transpose().colwise().replicate(alpha.rows());

    for (size_t i = 0; i < ee.rows(); i++) {
        for (size_t j = 0; j < ee.cols(); j++) {
            ee(i, j) = gint(l * 2 + 2, ee(i, j));
        }
    }
    Eigen::ArrayXd s1 = coeffs.transpose() * ee * coeffs;
    s1 = 1.0 / s1.sqrt();
    coeffs = coeffs * s1.matrix();
}

OccShell::OccShell(const int ang, const std::vector<double> &expo,
                   const std::vector<std::vector<double>> &contr,
                   const std::array<double, 3> &pos)
    : l(ang), origin() {
    size_t nprim = expo.size();
    size_t ncont = contr.size();
    assert(nprim > 0);
    assert(ncont > 0);
    for (const auto &c : contr) {
        assert(c.size() == nprim);
    }
    exponents = Eigen::VectorXd(nprim);
    contraction_coefficients = Eigen::MatrixXd(nprim, ncont);
    for (size_t j = 0; j < ncont; j++) {
        for (size_t i = 0; i < nprim; i++) {
            if (j == 0)
                exponents(i) = expo[i];
            contraction_coefficients(i, j) = contr[j][i];
        }
    }
    origin = {pos[0], pos[1], pos[2]};
}

// TODO remove once libint2 has been purged...
OccShell::OccShell(const libint2::Shell &sh) : l(sh.contr[0].l), origin() {
    size_t nprim = sh.nprim();
    size_t ncont = 1;
    exponents = Eigen::VectorXd(nprim);
    contraction_coefficients = Eigen::MatrixXd(nprim, 1);
    for (size_t i = 0; i < nprim; i++) {
        exponents(i) = sh.alpha[i];
        contraction_coefficients(i, 0) = sh.coeff_normalized(0, i);
    }
    kind = sh.contr[0].pure ? Kind::Spherical : Kind::Cartesian;
    origin = {sh.O[0], sh.O[1], sh.O[2]};
}

OccShell::OccShell(const occ::core::PointCharge &point_charge)
    : l(0), origin(), exponents(1), contraction_coefficients(1, 1) {
    constexpr double alpha = 1e16;
    exponents(0) = alpha;
    contraction_coefficients(0, 0) =
        -point_charge.first / (2 * constants::sqrt_pi<double> * gint(2, alpha));
    origin = {point_charge.second[0], point_charge.second[1],
              point_charge.second[2]};
}

OccShell::OccShell()
    : l(0), origin(), exponents(1), contraction_coefficients(1, 1) {
    constexpr double alpha = 1e16;
    exponents(0) = alpha;
    contraction_coefficients(0, 0) =
        -1 / (2 * constants::sqrt_pi<double> * gint(2, alpha));
    origin = {0, 0, 0};
}

bool OccShell::operator==(const OccShell &other) const {
    return &other == this ||
           (origin == other.origin && exponents == other.exponents &&
            contraction_coefficients == other.contraction_coefficients);
}

bool OccShell::operator!=(const OccShell &other) const {
    return !this->operator==(other);
}

bool OccShell::operator<(const OccShell &other) const { return l < other.l; }

size_t OccShell::num_primitives() const { return exponents.rows(); }
size_t OccShell::num_contractions() const {
    return contraction_coefficients.cols();
}

double OccShell::norm() const {
    double result = 0.0;
    for (Eigen::Index i = 0; i < contraction_coefficients.rows(); i++) {
        Eigen::Index j;
        double a = exponents(i);
        for (j = 0; j < i; j++) {
            double b = exponents(j);
            double ab = 2 * sqrt(a * b) / (a + b);
            result += 2 * contraction_coefficients(i, 0) *
                      contraction_coefficients(j, 0) * pow(ab, l + 1.5);
        }
        result +=
            contraction_coefficients(i, 0) * contraction_coefficients(j, 0);
    }
    return sqrt(result) * pi2_34;
}

double OccShell::max_exponent() const { return exponents.maxCoeff(); }
double OccShell::min_exponent() const { return exponents.minCoeff(); }

void OccShell::incorporate_shell_norm() {
    for (size_t i = 0; i < num_primitives(); i++) {
        double n = gto_norm(static_cast<int>(l), exponents(i));
        contraction_coefficients.row(i).array() *= n;
    }
    normalize_contracted_gto(l, exponents, contraction_coefficients);

    // NOTE: this is taken from libint2, and is here for compatibility
    // and consistency as the library was initially written with libint2
    // as the integral backend.
    // It's strange to treat s & p functions differently, but this yields
    // consistent results with libint2 and libint2::Shell
    /*
    {
        using detail::df_Kminus1;
        using std::pow;
        const auto sqrt_Pi_cubed = double{5.56832799683170784528481798212};
        const auto np = num_primitives();
        for (size_t i = 0; i < num_contractions(); i++) {
            auto coeff = contraction_coefficients.col(i);
            for (auto p = 0ul; p != np; ++p) {
                if (exponents(p) != 0) {
                    const auto two_alpha = 2 * exponents(p);
                    const auto two_alpha_to_am32 =
                        pow(two_alpha, l + 1) * sqrt(two_alpha);
                    const auto normalization_factor =
                        sqrt(pow(2, l) * two_alpha_to_am32 /
                             (sqrt_Pi_cubed * df_Kminus1[2 * l]));

                    coeff(p) *= normalization_factor;
                }
            }

            // need to force normalization to unity?
            if (true) {
                // compute the self-overlap of the , scale coefficients by its
                // inverse square root
                double norm{0};
                for (auto p = 0ul; p != np; ++p) {
                    for (decltype(p) q = 0ul; q <= p; ++q) {
                        auto gamma = exponents(p) + exponents(q);
                        norm += (p == q ? 1 : 2) * df_Kminus1[2 * l] *
                                sqrt_Pi_cubed * coeff(p) * coeff(q) /
                                (pow(2, l) * pow(gamma, l + 1) * sqrt(gamma));
                    }
                }
                auto normalization_factor = 1 / sqrt(norm);
                for (auto p = 0ul; p != np; ++p) {
                    coeff(p) *= normalization_factor;
                }
            }
        }
    }
    */
}

double OccShell::coeff_normalized(Eigen::Index contr_idx,
                                  Eigen::Index coeff_idx) const {
    // see NOTE in incorporate_shell_norm
    return contraction_coefficients(coeff_idx, contr_idx) /
           gto_norm(static_cast<int>(l), exponents(coeff_idx));

    /*
    {
        using detail::df_Kminus1;
        constexpr double sqrt_Pi_cubed{5.56832799683170784528481798212};
        const double two_alpha = 2 * exponents(coeff_idx);
        const double two_alpha_to_am32 =
            pow(two_alpha, l + 1) * sqrt(two_alpha);
        const double one_over_N = sqrt((sqrt_Pi_cubed * df_Kminus1[2 * l]) /
                                       (pow(2, l) * two_alpha_to_am32));
        return contraction_coefficients(contr_idx, coeff_idx) * one_over_N;
    }
    */
}

size_t OccShell::size() const {
    switch (kind) {
    case Spherical:
        return 2 * l + 1;
    default:
        return (l + 1) * (l + 2) / 2;
    }
}

char OccShell::l_to_symbol(uint_fast8_t l) {
    assert(l <= 19);
    static std::array<char, 20> l_symbols = {'s', 'p', 'd', 'f', 'g', 'h', 'i',
                                             'k', 'm', 'n', 'o', 'q', 'r', 't',
                                             'u', 'v', 'w', 'x', 'y', 'z'};
    return l_symbols[l];
}

uint_fast8_t OccShell::symbol_to_l(char symbol) {
    const char usym = ::toupper(symbol);
    switch (usym) {
    case 'S':
        return 0;
    case 'P':
        return 1;
    case 'D':
        return 2;
    case 'F':
        return 3;
    case 'G':
        return 4;
    case 'H':
        return 5;
    case 'I':
        return 6;
    case 'K':
        return 7;
    case 'M':
        return 8;
    case 'N':
        return 9;
    case 'O':
        return 10;
    case 'Q':
        return 11;
    case 'R':
        return 12;
    case 'T':
        return 13;
    case 'U':
        return 14;
    case 'V':
        return 15;
    case 'W':
        return 16;
    case 'X':
        return 17;
    case 'Y':
        return 18;
    case 'Z':
        return 19;
    default:
        throw "invalid angular momentum label";
    }
}

char OccShell::symbol() const { return l_to_symbol(l); }

OccShell OccShell::translated_copy(const Eigen::Vector3d &origin) const {
    OccShell other = *this;
    other.origin = origin;
    return other;
}

size_t OccShell::libcint_environment_size() const {
    return exponents.size() + contraction_coefficients.size();
}

int OccShell::find_atom_index(const std::vector<Atom> &atoms) const {
    double x = origin(0), y = origin(1), z = origin(2);
    auto same_site = [&x, &y, &z](const Atom &atom) {
        return atom.square_distance(x, y, z) < 1e-6;
    };
    return std::distance(begin(atoms),
                         std::find_if(begin(atoms), end(atoms), same_site));
}

bool OccShell::is_pure() const { return kind == Spherical; }

AOBasis::AOBasis(const std::vector<occ::core::Atom> &atoms,
                 const std::vector<OccShell> &shells)
    : m_atoms(atoms), m_shells(shells), m_shell_to_atom_idx(shells.size()),
      m_atom_to_shell_idxs(atoms.size()), m_bf_to_shell() {
    size_t shell_idx = 0;
    for (const auto &shell : m_shells) {
        m_kind = shell.kind;
        m_first_bf.push_back(m_nbf);
        m_nbf += shell.size();
        m_max_shell_size = std::max(m_max_shell_size, shell.size());
        int atom_idx = shell.find_atom_index(m_atoms);
        // TODO check for error
        m_shell_to_atom_idx[shell_idx] = atom_idx;
        m_atom_to_shell_idxs[atom_idx].push_back(shell_idx);
        for (int i = 0; i < shell.size(); i++) {
            m_bf_to_shell.push_back(shell_idx);
        }
        ++shell_idx;
    }
}

void AOBasis::merge(const AOBasis &rhs) {
    // TODO handle case where atoms are common
    // TODO handle case where basis sets aren't both cartesian/spherical
    size_t bf_offset = m_nbf;
    size_t shell_offset = m_shells.size();
    size_t atom_offset = m_atoms.size();

    m_nbf += rhs.m_nbf;
    m_shells.insert(m_shells.end(), rhs.m_shells.begin(), rhs.m_shells.end());
    m_first_bf.insert(m_first_bf.end(), rhs.m_first_bf.begin(),
                      rhs.m_first_bf.end());
    m_atom_to_shell_idxs.insert(m_atom_to_shell_idxs.end(),
                                rhs.m_atom_to_shell_idxs.begin(),
                                rhs.m_atom_to_shell_idxs.end());
    m_shell_to_atom_idx.insert(m_shell_to_atom_idx.end(),
                               rhs.m_shell_to_atom_idx.begin(),
                               rhs.m_shell_to_atom_idx.end());

    m_bf_to_shell.insert(m_bf_to_shell.end(), rhs.m_bf_to_shell.begin(),
                         rhs.m_bf_to_shell.end());

    // apply offsets
    for (size_t i = shell_offset; i < m_shell_to_atom_idx.size(); i++) {
        m_shell_to_atom_idx[i] += atom_offset;
        m_first_bf[i] += bf_offset;
    }

    for (auto &atom_shells : m_atom_to_shell_idxs) {
        for (auto &x : atom_shells) {
            x += shell_offset;
        }
    }
    for (size_t i = bf_offset; i < m_nbf; i++) {
        m_bf_to_shell[i] += shell_offset;
    }

    m_max_shell_size = std::max(m_max_shell_size, rhs.m_max_shell_size);
}

std::vector<OccShell> from_libint2_basis(const occ::qm::BasisSet &basis) {
    std::vector<OccShell> result;
    result.reserve(basis.size());
    for (const auto &sh : basis) {
        result.emplace_back(occ::qm::OccShell(sh));
        result.back().incorporate_shell_norm();
    }
    return result;
}

std::ostream &operator<<(std::ostream &stream, const OccShell &shell) {
    stream << shell.symbol() << " (" << shell.origin(0) << ","
           << shell.origin(1) << ", " << shell.origin(2) << ")\n";
    stream << "exp   contr\n";
    for (int i = 0; i < shell.num_primitives(); i++) {
        stream << " " << shell.exponents(i);
        for (int j = 0; j < shell.num_contractions(); j++) {
            stream << " " << shell.contraction_coefficients(i, j);
        }
        stream << "\n";
    }
    return stream;
}

} // namespace occ::qm
