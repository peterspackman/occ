#include <cmath>
#include <filesystem>
#include <occ/core/constants.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/json_basis.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/shell.h>

namespace fs = std::filesystem;

namespace occ::qm {

namespace impl {
static std::string basis_set_directory_override{""};
}

void override_basis_set_directory(const std::string &s) {
    impl::basis_set_directory_override = s;
}


double gint(int n, double alpha) {
    double n1_2 = 0.5 * (n + 1);
    return std::tgamma(n1_2) / (2 * std::pow(alpha, n1_2));
}

double psi4_primitive_normalization(int n, double alpha) {
    double tmp1 = n + 1.5;
    double g = 2.0 * alpha;
    double z = std::pow(g, tmp1);
    double normg =
        std::sqrt((pow(2.0, n) * z) / (M_PI * std::sqrt(M_PI) *
                                       occ::util::double_factorial(2 * n)));
    return normg;
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

Shell::Shell(const int ang, const std::vector<double> &expo,
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
    u_coefficients = contraction_coefficients;
    origin = {pos[0], pos[1], pos[2]};
}

Shell::Shell(const occ::core::PointCharge &point_charge)
    : l(0), origin(), exponents(1), contraction_coefficients(1, 1) {
    constexpr double alpha = 1e16;
    exponents(0) = alpha;
    contraction_coefficients(0, 0) =
        -point_charge.first / (2 * constants::sqrt_pi<double> * gint(2, alpha));
    u_coefficients = contraction_coefficients;
    origin = {point_charge.second[0], point_charge.second[1],
              point_charge.second[2]};
}

Shell::Shell() : l(0), origin(), exponents(1), contraction_coefficients(1, 1) {
    constexpr double alpha = 1e16;
    exponents(0) = alpha;
    contraction_coefficients(0, 0) =
        -1 / (2 * constants::sqrt_pi<double> * gint(2, alpha));
    u_coefficients = contraction_coefficients;
    origin = {0, 0, 0};
}

bool Shell::operator==(const Shell &other) const {
    return &other == this ||
           (origin == other.origin && exponents == other.exponents &&
            contraction_coefficients == other.contraction_coefficients);
}

bool Shell::operator!=(const Shell &other) const {
    return !this->operator==(other);
}

bool Shell::operator<(const Shell &other) const { return l < other.l; }

size_t Shell::num_primitives() const { return exponents.rows(); }
size_t Shell::num_contractions() const {
    return contraction_coefficients.cols();
}

double Shell::norm() const {
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

double Shell::max_exponent() const { return exponents.maxCoeff(); }
double Shell::min_exponent() const { return exponents.minCoeff(); }

Mat Shell::coeffs_normalized_for_libecpint() const {
    Mat result = contraction_coefficients;
    for (int i = 0; i < num_primitives(); ++i) {
        double normalization = psi4_primitive_normalization(l, exponents(i));
        result(i, 0) *= normalization;
    }
    double e_sum = 0.0, g, z;

    for (int i = 0; i < num_primitives(); ++i) {
        for (int j = 0; j < num_primitives(); ++j) {
            g = exponents(i) + exponents(j);
            z = std::pow(g, l + 1.5);
            e_sum += result(i, 0) * result(j, 0) / z;
        }
    }

    double tmp =
        ((2.0 * M_PI / M_2_SQRTPI) * occ::util::double_factorial(2 * l)) /
        std::pow(2.0, l);
    double norm = std::sqrt(1.0 / (tmp * e_sum));

    // Set the normalization
    for (int i = 0; i < num_primitives(); ++i) {
        result(i, 0) *= norm;
    }

    if (norm != norm) {
        for (int i = 0; i < num_primitives(); ++i)
            result(i, 0) = 1.0;
    }
    return result;
}

void Shell::incorporate_shell_norm() {
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

double Shell::coeff_normalized(Eigen::Index contr_idx,
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

size_t Shell::size() const {
    switch (kind) {
    case Spherical:
        return 2 * l + 1;
    default:
        return (l + 1) * (l + 2) / 2;
    }
}

char Shell::l_to_symbol(uint_fast8_t l) {
    assert(l <= 19);
    static std::array<char, 20> l_symbols = {'s', 'p', 'd', 'f', 'g', 'h', 'i',
                                             'k', 'm', 'n', 'o', 'q', 'r', 't',
                                             'u', 'v', 'w', 'x', 'y', 'z'};
    return l_symbols[l];
}

uint_fast8_t Shell::symbol_to_l(char symbol) {
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

char Shell::symbol() const { return l_to_symbol(l); }

Shell Shell::translated_copy(const Eigen::Vector3d &origin) const {
    Shell other = *this;
    other.origin = origin;
    return other;
}

size_t Shell::libcint_environment_size() const {
    return exponents.size() + contraction_coefficients.size();
}

int Shell::find_atom_index(const std::vector<Atom> &atoms) const {
    double x = origin(0), y = origin(1), z = origin(2);
    auto same_site = [&x, &y, &z](const Atom &atom) {
        return atom.square_distance(x, y, z) < 1e-6;
    };
    return std::distance(begin(atoms),
                         std::find_if(begin(atoms), end(atoms), same_site));
}

bool Shell::is_pure() const { return kind == Spherical; }

AOBasis::AOBasis(const std::vector<occ::core::Atom> &atoms,
                 const std::vector<Shell> &shells, const std::string &name,
                 const ShellList &ecp_shells)
    : m_basis_name(name), m_atoms(atoms),
      m_shells(shells),  m_ecp_shells(ecp_shells),
      m_shell_to_atom_idx(shells.size()),
      m_ecp_shell_to_atom_idx(ecp_shells.size()),
      m_bf_to_shell(), m_bf_to_atom(),
      m_atom_to_shell_idxs(atoms.size()),
      m_atom_to_ecp_shell_idxs(atoms.size()),
      m_ecp_electrons(atoms.size(), 0)  {

    size_t shell_idx = 0;
    for (const auto &shell : m_shells) {
        m_kind = shell.kind;
        m_first_bf.push_back(m_nbf);
        m_nbf += shell.size();
        m_max_shell_size = std::max(m_max_shell_size, shell.size());
        int atom_idx = shell.find_atom_index(m_atoms);
        // TODO check for error
        if (atom_idx >= m_atom_to_shell_idxs.size() || atom_idx < 0) {
            throw std::runtime_error("Unable to map shell to atoms in AOBasis");
        }
        m_shell_to_atom_idx[shell_idx] = atom_idx;
        m_atom_to_shell_idxs[atom_idx].push_back(shell_idx);
        for (int i = 0; i < shell.size(); i++) {
            m_bf_to_shell.push_back(shell_idx);
            m_bf_to_atom.push_back(atom_idx);
        }
        ++shell_idx;
    }
    size_t ecp_shell_idx = 0;
    for (const auto &shell : m_ecp_shells) {
        m_max_ecp_shell_size = std::max(m_max_ecp_shell_size, shell.size());
        int atom_idx = shell.find_atom_index(m_atoms);
        if (atom_idx >= m_atom_to_shell_idxs.size() || atom_idx < 0) {
            throw std::runtime_error(
                "Unable to map ECP shell to atoms in AOBasis");
        }
        m_ecp_shell_to_atom_idx[ecp_shell_idx] = atom_idx;
        m_atom_to_ecp_shell_idxs[atom_idx].push_back(ecp_shell_idx);
        ++ecp_shell_idx;
    }
}

void AOBasis::calculate_shell_cutoffs() {
    Vec extents = occ::gto::evaluate_decay_cutoff(*this);
    for (size_t i = 0; i < m_shells.size(); i++) {
        m_shells[i].extent = extents(i);
    }
}

void AOBasis::update_bf_maps() {
    m_first_bf.clear();
    m_nbf = 0;
    m_max_shell_size = 0;
    m_bf_to_shell.clear();
    m_bf_to_atom.clear();
    size_t shell_idx = 0;
    for (const auto &shell : m_shells) {
        m_first_bf.push_back(m_nbf);
        m_nbf += shell.size();
        m_max_shell_size = std::max(m_max_shell_size, shell.size());
        int atom_idx = m_shell_to_atom_idx[shell_idx];
        for (int i = 0; i < shell.size(); i++) {
            m_bf_to_shell.push_back(shell_idx);
            m_bf_to_atom.push_back(atom_idx);
        }
        ++shell_idx;
    }
}

void AOBasis::merge(const AOBasis &rhs) {
    // TODO handle case where atoms are common
    // TODO handle case where basis sets aren't both cartesian/spherical
    size_t bf_offset = m_nbf;
    size_t shell_offset = m_shells.size();
    size_t ecp_shell_offset = m_shells.size();
    size_t atom_offset = m_atoms.size();

    m_nbf += rhs.m_nbf;
    auto append = [](auto &v1, auto &v2) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    };

    append(m_shells, rhs.m_shells);
    append(m_ecp_shells, rhs.m_ecp_shells);

    append(m_first_bf, rhs.m_first_bf);

    append(m_atom_to_shell_idxs, rhs.m_atom_to_shell_idxs);
    append(m_atom_to_ecp_shell_idxs, rhs.m_atom_to_ecp_shell_idxs);

    append(m_shell_to_atom_idx, rhs.m_shell_to_atom_idx);
    append(m_ecp_shell_to_atom_idx, rhs.m_ecp_shell_to_atom_idx);

    append(m_bf_to_shell, rhs.m_bf_to_shell);
    append(m_bf_to_atom, rhs.m_bf_to_atom);
    append(m_atoms, rhs.m_atoms);
    append(m_ecp_electrons, rhs.m_ecp_electrons);

    // apply offsets
    for (size_t i = shell_offset; i < m_shell_to_atom_idx.size(); i++) {
        m_shell_to_atom_idx[i] += atom_offset;
        m_first_bf[i] += bf_offset;
    }

    for (size_t i = ecp_shell_offset; i < m_ecp_shell_to_atom_idx.size(); i++) {
        m_ecp_shell_to_atom_idx[i] += atom_offset;
    }

    for (size_t i = atom_offset; i < m_atom_to_shell_idxs.size(); i++) {
        for (auto &x : m_atom_to_shell_idxs[i]) {
            x += shell_offset;
        }
    }

    for (size_t i = bf_offset; i < m_nbf; i++) {
        m_bf_to_shell[i] += shell_offset;
        m_bf_to_atom[i] += atom_offset;
    }

    m_max_shell_size = std::max(m_max_shell_size, rhs.m_max_shell_size);
}


std::string canonicalize_name(const std::string &name) {
    auto result = name;
    std::transform(name.begin(), name.end(), result.begin(), [](auto &c) {
        char cc = ::tolower(c);
        switch (cc) {
        case '/':
            cc = 'I';
            break;
        }
        return cc;
    });

    if (result == "6-311g**") {
        result = "6-311g(d,p)";
    } else if (result == "6-31g**") {
        result = "6-31g(d,p)";
    } else if (result == "6-31g*") {
        result = "6-31g(d)";
    }

    return result;
}

std::string data_path() {
    std::string path{"."};
    const char *data_path_env = 
	impl::basis_set_directory_override.empty() ? getenv("OCC_DATA_PATH") : impl::basis_set_directory_override.c_str();
    if (data_path_env) {
        path = data_path_env;
    } else {
#if defined(DATADIR)
        path = std::string{DATADIR};
#elif defined(SRCDATADIR)
        path = std::string{SRCDATADIR};
#endif
    }
    // validate basis_path = path + "/basis"
    std::string basis_path = path + std::string("/basis");
    bool path_exists = fs::exists(basis_path);
    std::string errmsg;
    if (!path_exists) { // try without "/basis"
        occ::log::warn("There is a problem with the basis set directory, the "
                       "path '{}' is not valid (does not exist)",
                       basis_path);
        basis_path = fs::current_path().string();
    } else if (!fs::is_directory(basis_path)) {
        occ::log::warn("There is a problem with the basis set directory, the "
                       "path '{}' is not valid (not a directory)",
                       basis_path);
        basis_path = fs::current_path().string();
    }
    return basis_path;
}

AOBasis AOBasis::load(const AtomList &atoms, const std::string &name) {
    std::string basis_lib_path = data_path();

    auto canonical_name = canonicalize_name(name);

    std::string json_filepath = canonical_name + ".json";
    if (!fs::exists(canonical_name + ".json")) {
        json_filepath = basis_lib_path + "/" + json_filepath;
    }
    occ::io::JsonBasisReader parser(json_filepath);
    auto element_map = parser.element_map();

    std::vector<Shell> shells;
    std::vector<Shell> ecp_shells;

    std::vector<int> ecp_electrons(atoms.size(), 0);
    int nsh = 0;
    int nsh_ecp = 0;

    for (size_t a = 0; a < atoms.size(); ++a) {
        std::array<double, 3> origin = {atoms[a].x, atoms[a].y, atoms[a].z};
        const std::size_t Z = atoms[a].atomic_number;
        if (!element_map.contains(Z)) {
            throw std::runtime_error(
                fmt::format("element {} not found in basis", Z));
        }
        const occ::io::ElementBasis element_basis = element_map.at(Z);
        if (!element_basis.electron_shells.empty()) {
            for (const auto &s : element_basis.electron_shells) {
                // handle general contractions by splitting
                for (int i = 0; i < s.coefficients.size(); i++) {
                    shells.push_back(
                        Shell(s.angular_momentum[i % s.angular_momentum.size()],
                              s.exponents, {s.coefficients[i]}, origin));
                    shells[nsh].incorporate_shell_norm();
                    nsh++;
                }
            }
        } else {
            std::string errmsg = fmt::format(
                "No matching basis for element (z={}) in {}", Z, json_filepath);
            throw std::logic_error(errmsg);
        }
        if (element_basis.ecp_electrons > 0) {
            occ::log::debug("Setting ECPs on atom {}", a);
            if (element_basis.ecp_shells.size() < 1) {
                std::string errmsg = fmt::format(
                    "Element (z={}) in basis '{}' has ECP electrons but "
                    "no defined ECP shells",
                    Z, json_filepath);
                throw std::runtime_error(errmsg);
            }
            for (const auto &s : element_basis.ecp_shells) {
                // handle general contractions by splitting
                for (int i = 0; i < s.angular_momentum.size(); i++) {
                    ecp_shells.push_back(Shell(s.angular_momentum[i],
                                               s.exponents, {s.coefficients[i]},
                                               origin));
                    const auto &n = s.r_exponents;
                    auto &shell = ecp_shells[nsh_ecp];
                    shell.ecp_r_exponents =
                        Eigen::Map<const IVec>(n.data(), n.size());
                    nsh_ecp++;
                }
            }
            ecp_electrons[a] = element_basis.ecp_electrons;
        }
    }
    AOBasis result(atoms, shells, name, ecp_shells);
    result.set_ecp_electrons(ecp_electrons);
    return result;
}

bool AOBasis::operator==(const AOBasis &rhs) const {
    return (m_ecp_electrons == rhs.m_ecp_electrons) &&
           (m_shells == rhs.m_shells) && (m_ecp_shells == rhs.m_ecp_shells);
}

void AOBasis::rotate(const occ::Mat3 &rotation) {
    int shell_idx = 0;
    for (auto &shell : m_shells) {
        auto rot_pos = rotation * shell.origin;
        shell.origin = rot_pos;
        int atom_idx = m_shell_to_atom_idx[shell_idx];
        auto &atom = m_atoms[atom_idx];
        atom.x = shell.origin(0);
        atom.y = shell.origin(1);
        atom.z = shell.origin(2);
        shell_idx++;
    }

    // nothing needs to happen to ECPs besides their
    for (auto &shell : m_ecp_shells) {
        auto rot_pos = rotation * shell.origin;
        shell.origin = rot_pos;
    }
}

void AOBasis::translate(const occ::Vec3 &translation) {
    int shell_idx = 0;
    for (auto &shell : m_shells) {
        auto t_pos = translation + shell.origin;
        shell.origin = t_pos;
        auto &atom = m_atoms[m_shell_to_atom_idx[shell_idx]];
        atom.x = shell.origin(0);
        atom.y = shell.origin(1);
        atom.z = shell.origin(2);
        shell_idx++;
    }

    for (auto &shell : m_ecp_shells) {
        auto t_pos = translation + shell.origin;
        shell.origin = t_pos;
    }
}

uint_fast8_t AOBasis::l_max() const {
    uint_fast8_t l_max = 0;
    for (const auto &sh : m_shells) {
        l_max = std::max(l_max, sh.l);
    }
    return l_max;
}

uint_fast8_t AOBasis::ecp_l_max() const {
    uint_fast8_t l_max = 0;
    for (const auto &sh : m_shells) {
        l_max = std::max(l_max, sh.l);
    }
    return l_max;
}

} // namespace occ::qm

auto fmt::formatter<occ::qm::Shell>::format(const occ::qm::Shell& shell, format_context& ctx) const -> decltype(ctx.out()) {
    auto out = ctx.out();
    out = fmt::format_to(out, "{} [{}, {}, {}]\n  exponents  | contraction coefficients\n", shell.symbol(), 
		nested(shell.origin(0)), nested(shell.origin(1)), nested(shell.origin(2)));
    
    for (int i = 0; i < shell.num_primitives(); i++) {
	out = fmt::format_to(out, "{} |", nested(shell.exponents(i)));
	for (int j = 0; j < shell.num_contractions(); j++) {
	    out = fmt::format_to(out, " {}", nested(shell.contraction_coefficients(i, j)));
	}
	out = fmt::format_to(out, "\n");
    }
    
    return out;
}

