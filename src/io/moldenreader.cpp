#include <fmt/ostream.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/moldenreader.h>
#include <scn/scn.h>

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
    occ::timing::start(occ::timing::category::io);
    std::ifstream file(filename);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

MoldenReader::MoldenReader(std::istream &file) {
    occ::timing::start(occ::timing::category::io);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

void MoldenReader::parse(std::istream &stream) {
    std::string line;
    while (std::getline(stream, line)) {
        if (is_section_line(line)) {
            auto section_name = parse_section_name(line);
            auto section_args = extract_section_args(line);
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
    }
    else if (section_name == "Atoms") {
        parse_atoms_section(args, stream);
    } else if (section_name == "GTO") {
        parse_gto_section(args, stream);
    } else if (section_name == "MO") {
        parse_mo_section(args, stream);
    } else if (section_name == "5D") {
        m_basis.set_pure(true);
        m_pure = true;
	occ::log::debug("Basis uses pure spherical harmonics");
    }
}

void MoldenReader::parse_atoms_section(const std::optional<std::string> &args,
                                       std::istream &stream) {
    occ::log::debug("Parsing Atoms section");
    auto pos = stream.tellg();
    std::string line;
    std::vector<int> idx;
    std::string unit = args.value_or("bohr");
    trim(unit);
    to_lower(unit);
    double factor = 1.0;
    if (startswith(unit, "angs", false))
        factor = 0.5291772108;
    while (std::getline(stream, line)) {
        if (is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        std::string symbol;
        int idx;
        occ::core::Atom atom;
        scn::scan_default(line, symbol, idx, atom.atomic_number, atom.x, atom.y,
                          atom.z);
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
    std::string line;
    while (std::getline(stream, line)) {
        if (is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        if(line.find("orca_2mkl") != std::string::npos) {
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

inline libint2::Shell parse_molden_shell(const std::array<double, 3> &position,
                                         bool pure, std::istream &stream) {
    std::string line;
    std::getline(stream, line);
    using occ::util::double_factorial;
    char shell_type;
    int num_primitives, second;
    scn::scan_default(line, shell_type, num_primitives, second);
    libint2::svector<double> alpha, coeffs;
    alpha.reserve(num_primitives), coeffs.reserve(num_primitives);
    int l = l_from_char(shell_type);
    for (int i = 0; i < num_primitives; i++) {
        std::getline(stream, line);
        double e, c;
        scn::scan_default(line, e, c);
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
	    occ::log::debug("Renormalizing coefficients, shell norm: {:6.3f}", norm);
            for (size_t i = 0; i < coeffs.size(); i++) {
                coeffs[i] /= pow(4 * alpha[i], 0.5 * l + 0.75);
                coeffs[i] = coeffs[i] * pi2_34 / norm;
            }
        }
    }
    auto shell = libint2::Shell{std::move(alpha),
                                {
                                    {l, pure, std::move(coeffs)},
                                },
                                position};
    return shell;
}

void MoldenReader::parse_gto_section(const std::optional<std::string> &args,
                                     std::istream &stream) {
    occ::log::debug("Parsing GTO section");
    auto pos = stream.tellg();
    std::string line;
    while (std::getline(stream, line)) {
        if (is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        int atom_idx, second;
        scn::scan_default(line, atom_idx, second);
        assert(atom_idx <= m_atoms.size());
        std::array<double, 3> position{m_atoms[atom_idx - 1].x,
                                       m_atoms[atom_idx - 1].y,
                                       m_atoms[atom_idx - 1].z};
        while (std::getline(stream, line)) {
            trim(line);
            if (line.empty()) {
                break;
            }
            stream.seekg(pos, std::ios_base::beg);
            m_basis.push_back(parse_molden_shell(position, m_pure, stream));
            pos = stream.tellg();
        }
    }
    m_basis.update();
}

void MoldenReader::parse_mo(size_t &mo_a, size_t &mo_b, std::istream &stream) {
    double energy;
    bool alpha;
    double occupation{0.0};
    std::string line;
    while (std::getline(stream, line)) {
        if (startswith(line, "Sym", true)) {

        } else if (startswith(line, "Ene", true)) {
            scn::scan(line, "Ene= {}", energy);
        } else if (startswith(line, "Spin", true)) {
            std::string spin;
            scn::scan(line, "Spin= {}", spin);
            to_lower(spin);
            alpha = (spin == "alpha");
        } else if (startswith(line, "Occup", true)) {
            scn::scan(line, "Occup= {}", occupation);
            m_num_electrons += occupation;
        } else {
            for (size_t i = 0; i < nbf(); i++) {
                if (i > 0)
                    std::getline(stream, line);
                int idx;
                double coeff;
                scn::scan_default(line, idx, coeff);
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
    std::string line;
    m_energies_alpha = occ::Vec(nbf());
    m_energies_beta = occ::Vec(nbf());
    m_molecular_orbitals_alpha = Mat(nbf(), nbf());
    m_molecular_orbitals_beta = Mat(nbf(), nbf());
    size_t num_alpha = 0, num_beta = 0;
    while (std::getline(stream, line)) {
        if (is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        parse_mo(num_alpha, num_beta, stream);
    }
}

Mat MoldenReader::convert_mo_coefficients_from_molden_convention(
    const occ::qm::BasisSet &basis, const Mat &mo) const {

    if (occ::qm::max_l(basis) < 1)
        return mo;

    occ::log::debug("Reordering MO coefficients from Molden ordering to "
                    "internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    bool orca = source == Source::Orca;
    if(orca) occ::log::debug("ORCA phase convention...");
    if (occ::qm::max_l(basis) < 1)
        return mo;
    constexpr auto order = occ::gto::ShellOrder::Molden;

    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis[i];
        size_t bf_first = shell2bf[i];
        int l = shell.contr[0].l;
	if(l == 1) {
	    // xyz -> yzx
	    occ::log::debug("Swapping (l={}): (2, 0, 1) <-> (0, 1, 2)", l); 
	    result.block(bf_first, 0, 1, ncols) = mo.block(bf_first + 1, 0, 1, ncols);
	    result.block(bf_first + 1, 0, 1, ncols) = mo.block(bf_first + 2, 0, 1, ncols);
	    result.block(bf_first + 2, 0, 1, ncols) = mo.block(bf_first, 0, 1, ncols);
	}
	else {
	    size_t idx = 0;
	    auto func = [&](int am, int m) {
		int their_idx = occ::gto::shell_index_spherical<order>(am, m);
		result.row(bf_first + idx)= mo.row(bf_first + their_idx);
		occ::log::debug("Swapping (l={}): {} <-> {}", l, idx, their_idx);
		idx++;
	    };
	    occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(func, l);
	}
    }

    /*
        case 3:
	    // c0 c1 s1 c2 s2 c3 s3
	    // +  +  +  +  +  -  -
	    // orca has modified phase
	    for(size_t i = 0; i < shell_size; i++) {
		if(orca && (i > 4)) phase.emplace_back(-1.0);
		else phase.emplace_back(1.0);
	    }
	    break;
	case 4:
	    // c0 c1 s1 c2 s2 c3 s3 c4 s4
	    // +  +  +  +  +  -  -  -  -
	    // orca has modified phase
	    for(size_t i = 0; i < shell_size; i++) {
		if(orca && (i > 4)) phase.emplace_back(-1.0);
		else phase.emplace_back(1.0);
	    }
	    break;
	case 5:
	    // c0 c1 s1 c2 s2 c3 s3 c4 s4 c5 s5
	    // +  +  +  +  +  -  -  -  -  +  + 
	    // orca has modified phase (weird)
	    for(size_t i = 0; i < shell_size; i++) {
		if(orca && ((i > 4) && (i < 9))) phase.emplace_back(-1.0);
		else phase.emplace_back(1.0);
	    }
	    break;
	    */
    return result;
}
} // namespace occ::io
