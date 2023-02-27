#include <fmt/ostream.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/moldenreader.h>
#include <scn/scn.h>

constexpr size_t expected_max_line_length{1024};

namespace occ::io {
using occ::util::startswith;
using occ::util::to_lower;
using occ::util::trim;

inline bool is_section_line(const std::string &line) {
    return line.find('[') != std::string::npos;
}

std::string parse_section_name(const std::string &line) {
    auto l = line.find('[');
    auto u = line.find(']');
    return line.substr(l + 1, u - l - 1);
}

std::optional<std::string> extract_section_args(const std::string &line) {
    auto u = line.find(']');
    if (u != std::string::npos && (u + 1) < line.size())
        return std::make_optional(line.substr(u + 1));
    return std::nullopt;
}

MoldenReader::MoldenReader(const std::string &filename) : m_filename(filename) {
    m_current_line.reserve(expected_max_line_length);
    occ::timing::start(occ::timing::category::io);
    std::ifstream file(filename);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

MoldenReader::MoldenReader(std::istream &file) {
    m_current_line.reserve(expected_max_line_length);
    occ::timing::start(occ::timing::category::io);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

void MoldenReader::parse(std::istream &stream) {
    while (std::getline(stream, m_current_line)) {
        if (is_section_line(m_current_line)) {
            auto section_name = parse_section_name(m_current_line);
            auto section_args = extract_section_args(m_current_line);
            occ::log::debug("Found section: {}", section_name);
            parse_section(section_name, section_args, stream);
        }
    }
}

void MoldenReader::parse_section(const std::string &section_name,
                                 const std::optional<std::string> &args,
                                 std::istream &stream) {
    if (section_name == "Title") {
        parse_title_section(args, stream);
    } else if (section_name == "Atoms") {
        parse_atoms_section(args, stream);
    } else if (section_name == "GTO") {
        parse_gto_section(args, stream);
    } else if (section_name == "MO") {
        parse_mo_section(args, stream);
    } else if (section_name == "5D") {
        m_pure = true;
        for (auto &shell : m_shells) {
            shell.kind = qm::Shell::Kind::Spherical;
        }
        occ::log::debug("Basis uses pure spherical harmonics");
    }
}

void MoldenReader::parse_atoms_section(const std::optional<std::string> &args,
                                       std::istream &stream) {
    occ::log::debug("Parsing Atoms section");
    auto pos = stream.tellg();
    std::vector<int> idx;
    std::string unit = args.value_or("bohr");
    trim(unit);
    to_lower(unit);
    double factor = 1.0;
    if (startswith(unit, "angs", false))
        factor = 0.5291772108;
    while (std::getline(stream, m_current_line)) {
        if (is_section_line(m_current_line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        std::string symbol;
        int idx;
        occ::core::Atom atom;
        auto scan_result =
            scn::scan_default(m_current_line, symbol, idx, atom.atomic_number,
                              atom.x, atom.y, atom.z);
        if (factor != 1.0) {
            atom.x *= factor;
            atom.y *= factor;
            atom.z *= factor;
        }
        m_atoms.push_back(atom);
    }
}

void MoldenReader::parse_title_section(const std::optional<std::string> &args,
                                       std::istream &stream) {
    occ::log::debug("Parsing Title section");
    auto pos = stream.tellg();
    while (std::getline(stream, m_current_line)) {
        if (is_section_line(m_current_line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        if (m_current_line.find("orca_2mkl") != std::string::npos) {
            occ::log::debug("Detected ORCA molden file");
            source = Source::Orca;
        }
    }
}

inline int l_from_char(const char c) {
    switch (c) {
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
    default:
        return 0;
    }
}

inline occ::qm::Shell parse_molden_shell(const std::array<double, 3> &position,
                                         bool pure, std::istream &stream,
                                         std::string &line_buffer) {
    std::getline(stream, line_buffer);
    using occ::util::double_factorial;
    occ::qm::Shell::Kind shell_kind = pure ? occ::qm::Shell::Kind::Spherical
                                           : occ::qm::Shell::Kind::Cartesian;

    std::string shell_type;
    int num_primitives, second;
    auto result = scn::make_result(line_buffer);
    if (!(result = scn::scan_default(result.range(), shell_type, num_primitives,
                                     second))) {
        fmt::print("Error: {}\n", result.error().msg());
        throw std::runtime_error(fmt::format(
            "Unable to parse molden file, error: {}", result.error().msg()));
    }
    std::vector<double> alpha, coeffs;
    alpha.reserve(num_primitives), coeffs.reserve(num_primitives);
    int l = l_from_char(shell_type[0]);
    for (int i = 0; i < num_primitives; i++) {
        std::getline(stream, line_buffer);
        double e, c;
        auto scan_result = scn::scan_default(line_buffer, e, c);
        alpha.push_back(e);
        coeffs.push_back(c);
    }

    {
        double pi2_34 = pow(2 * M_PI, 0.75);
        double norm = 0.0;
        for (size_t i = 0; i < coeffs.size(); i++) {
            size_t j;
            double a = alpha[i];
            for (j = 0; j < i; j++) {
                double b = alpha[j];
                double ab = 2 * sqrt(a * b) / (a + b);
                norm += 2 * coeffs[i] * coeffs[j] * pow(ab, l + 1.5);
            }
            norm += coeffs[i] * coeffs[j];
        }
        norm = sqrt(norm) * pi2_34;
        if (std::abs(pi2_34 - norm) > 1e-4) {
            occ::log::debug("Renormalizing coefficients, shell norm: {:6.3f}",
                            norm);
            for (size_t i = 0; i < coeffs.size(); i++) {
                coeffs[i] /= pow(4 * alpha[i], 0.5 * l + 0.75);
                coeffs[i] = coeffs[i] * pi2_34 / norm;
            }
        }
    }
    auto shell = occ::qm::Shell(l, alpha, {coeffs}, position);
    shell.kind = shell_kind;
    shell.incorporate_shell_norm();
    return shell;
}

void MoldenReader::parse_gto_section(const std::optional<std::string> &args,
                                     std::istream &stream) {
    occ::log::debug("Parsing GTO section");
    auto pos = stream.tellg();
    while (std::getline(stream, m_current_line)) {
        if (is_section_line(m_current_line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        int atom_idx, second;
        auto scan_result = scn::scan_default(m_current_line, atom_idx, second);
        assert(atom_idx <= m_atoms.size());
        std::array<double, 3> position{m_atoms[atom_idx - 1].x,
                                       m_atoms[atom_idx - 1].y,
                                       m_atoms[atom_idx - 1].z};
        while (std::getline(stream, m_current_line)) {
            trim(m_current_line);
            if (m_current_line.empty()) {
                break;
            }
            stream.seekg(pos, std::ios_base::beg);
            m_shells.push_back(
                parse_molden_shell(position, m_pure, stream, m_current_line));
            pos = stream.tellg();
        }
    }
}

inline void fail_with_error(const std::string &msg, const std::string &line) {
    throw std::runtime_error(fmt::format(
        "Unable to parse molden file, error: {}, line = '{}'", msg, line));
}

void MoldenReader::parse_mo(size_t &mo_a, size_t &mo_b, std::istream &stream) {
    double energy{0.0};
    bool alpha{false};
    double occupation{0.0};
    while (std::getline(stream, m_current_line)) {
        trim(m_current_line);
        auto result = scn::make_result(m_current_line);
        if (startswith(m_current_line, "Sym", true)) {

        } else if (startswith(m_current_line, "Ene", true)) {
            if (!(result = scn::scan(result.range(), "Ene= {}", energy))) {
                fail_with_error(result.error().msg(), m_current_line);
            }
        } else if (startswith(m_current_line, "Spin", true)) {
            std::string spin;
            if (!(result = scn::scan(result.range(), "Spin= {}", spin))) {
                fail_with_error(result.error().msg(), m_current_line);
            }
            to_lower(spin);
            alpha = (spin == "alpha");
        } else if (startswith(m_current_line, "Occup", true)) {
            if (!(result =
                      scn::scan(result.range(), "Occup= {}", occupation))) {
                fail_with_error(result.error().msg(), m_current_line);
            }
            m_num_electrons += occupation;
        } else {
            for (size_t i = 0; i < nbf(); i++) {
                if (i > 0)
                    std::getline(stream, m_current_line);
                int idx;
                double coeff;
                auto scan_result = scn::make_result(m_current_line);
                if (!(scan_result =
                          scn::scan_default(scan_result.range(), idx, coeff))) {
                    fail_with_error(scan_result.error().msg(), m_current_line);
                }
                if (alpha) {
                    m_molecular_orbitals_alpha(idx - 1, mo_a) = coeff;
                    m_energies_alpha(mo_a) = energy;
                } else {
                    m_molecular_orbitals_beta(idx - 1, mo_b) = coeff;
                    m_energies_beta(mo_b) = energy;
                }
            }
            break;
        }
    }
    if (alpha) {
        m_total_alpha_occupation += occupation;
        mo_a++;
    } else {
        m_total_beta_occupation += occupation;
        mo_b++;
    }
}

void MoldenReader::parse_mo_section(const std::optional<std::string> &args,
                                    std::istream &stream) {
    occ::log::debug("Parsing MO section");
    auto pos = stream.tellg();
    m_energies_alpha = occ::Vec(nbf());
    m_energies_beta = occ::Vec(nbf());
    m_molecular_orbitals_alpha = Mat(nbf(), nbf());
    m_molecular_orbitals_beta = Mat(nbf(), nbf());
    size_t num_alpha = 0, num_beta = 0;
    while (std::getline(stream, m_current_line)) {
        if (is_section_line(m_current_line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        parse_mo(num_alpha, num_beta, stream);
    }
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

Mat MoldenReader::convert_mo_coefficients_from_molden_convention(
    const occ::qm::AOBasis &basis, const Mat &mo) const {

    if (basis.l_max() < 1)
        return mo;

    occ::log::debug("Reordering MO coefficients from Molden ordering to "
                    "internal convention");
    auto shell2bf = basis.first_bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    bool orca = source == Source::Orca;
    if (orca) {
        occ::log::debug("Detected ORCA wavefunction source, will fix phase "
                        "convention for l >= 3...");
    }
    if (basis.l_max() < 1)
        return mo;
    constexpr auto order = occ::gto::ShellOrder::Molden;

    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis.shells()[i];
        size_t bf_first = shell2bf[i];
        int l = shell.l;
        if (l == 1) {
            // xyz -> yzx
            occ::log::debug("Swapping (l={}): (2, 0, 1) <-> (0, 1, 2)", l);
            result.block(bf_first + 0, 0, 1, ncols) =
                mo.block(bf_first + 1, 0, 1, ncols);
            result.block(bf_first + 1, 0, 1, ncols) =
                mo.block(bf_first + 2, 0, 1, ncols);
            result.block(bf_first + 2, 0, 1, ncols) =
                mo.block(bf_first + 0, 0, 1, ncols);
        } else {
            size_t idx = 0;
            auto func = [&](int am, int m) {
                int their_idx = occ::gto::shell_index_spherical<order>(am, m);
                int sign = 1;
                if (orca)
                    sign = fix_orca_phase_convention(am, m);
                result.row(bf_first + idx) =
                    sign * mo.row(bf_first + their_idx);
                occ::log::debug("Swapping (l={}): {} <-> {}", l, idx,
                                their_idx);
                idx++;
            };
            occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(
                func, l);
        }
    }
    return result;
}
} // namespace occ::io
