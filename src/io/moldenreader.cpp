#include <occ/io/moldenreader.h>
#include <scn/scn.h>
#include <fmt/ostream.h>
#include <occ/core/logger.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/core/timings.h>

namespace occ::io {
using occ::util::trim;
using occ::util::startswith;
using occ::util::to_lower;

inline bool is_section_line(const std::string &line)
{
    return line.find('[') != std::string::npos;
}

std::string parse_section_name(const std::string &line)
{
    auto l = line.find('[');
    auto u = line.find(']');
    return line.substr(l + 1, u - l - 1);
}

std::optional<std::string> extract_section_args(const std::string &line)
{
    auto u = line.find(']');
    if (u != std::string::npos && (u + 1) < line.size()) return std::make_optional(line.substr(u + 1));
    return std::nullopt;
}

MoldenReader::MoldenReader(const std::string &filename) : m_filename(filename)
{
    occ::timing::start(occ::timing::category::io);
    std::ifstream file(filename);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

MoldenReader::MoldenReader(std::istream &file)
{
    occ::timing::start(occ::timing::category::io);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

void MoldenReader::parse(std::istream &stream)
{
    std::string line;
    while(std::getline(stream, line))
    {
        if (is_section_line(line))
        {
            auto section_name = parse_section_name(line);
            auto section_args = extract_section_args(line);
            occ::log::debug("Found section: {}", section_name);
            parse_section(section_name, section_args, stream);
        }
    }
}

void MoldenReader::parse_section(const std::string &section_name, const std::optional<std::string> &args, std::istream &stream)
{
    if(section_name == "Atoms") {
        parse_atoms_section(args, stream);
    }
    else if (section_name == "GTO")
    {
        parse_gto_section(args, stream);
    }
    else if (section_name == "MO")
    {
        parse_mo_section(args, stream);
    }
    else if (section_name == "5D") {
        m_basis.set_pure(true);
        m_pure = true;
    }

}

void MoldenReader::parse_atoms_section(const std::optional<std::string> &args, std::istream& stream)
{
    occ::log::debug("Parsing Atoms section");
    auto pos = stream.tellg();
    std::string line;
    std::vector<int> idx;
    std::string unit = args.value_or("bohr");
    trim(unit);
    to_lower(unit);
    double factor = 1.0;
    if(startswith(unit, "angs", false)) factor = 0.5291772108;
    while(std::getline(stream, line))
    {
        if(is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        std::string symbol;
        int idx;
        occ::core::Atom atom;
        scn::scan_default(line, symbol, idx, atom.atomic_number, atom.x, atom.y, atom.z);
        if(factor != 1.0)
        {
            atom.x *= factor;
            atom.y *= factor;
            atom.z *= factor;
        }
        m_atoms.push_back(atom);
    }
}

inline int l_from_char(const char c)
{
    switch(c)
    {
    case 's': return 0;
    case 'S': return 0;
    case 'p': return 1;
    case 'P': return 1;
    case 'd': return 2;
    case 'D': return 2;
    case 'f': return 3;
    case 'F': return 3;
    case 'g': return 4;
    case 'G': return 4;
    case 'h': return 5;
    case 'H': return 5;
    case 'i': return 6;
    case 'I': return 6;
    default:
        return 0;
    }
}

inline libint2::Shell parse_molden_shell(const std::array<double, 3>& position, bool pure, std::istream &stream)
{
    std::string line;
    std::getline(stream, line);
    using occ::util::double_factorial;
    char shell_type;
    int num_primitives, second;
    scn::scan_default(line, shell_type, num_primitives, second);
    libint2::svector<double> alpha, coeffs;
    alpha.reserve(num_primitives), coeffs.reserve(num_primitives);
    int l = l_from_char(shell_type);
    for(int i = 0; i < num_primitives; i++)
    {
        std::getline(stream, line);
        double e, c;
        scn::scan_default(line, e, c);
        alpha.push_back(e);
        coeffs.push_back(c);
    }

    {
        double pi2_34 = pow(2*M_PI, 0.75);
        double norm = 0.0;
        for(size_t i = 0; i < coeffs.size(); i++)
        {
            size_t j;
            double a = alpha[i];
            for(j = 0; j < i; j++)
            {
                double b = alpha[j];
                double ab = 2 * sqrt(a*b)/(a + b);
                norm += 2 * coeffs[i] * coeffs[j] * pow(ab, l + 1.5);
            }
            norm += coeffs[i] * coeffs[j];
        }
        norm = sqrt(norm) * pi2_34;
        if(std::abs(pi2_34 - norm) > 1e-4) {
            for(size_t i = 0; i < coeffs.size(); i++)
            {
                occ::log::debug("Renormalizing coefficients: {} != (2*pi)^(3/4)\n", norm);
                coeffs[i] /= pow(4*alpha[i], 0.5 * l + 0.75);
                coeffs[i] = coeffs[i] * pi2_34 / norm;
            }
        }
    }
    auto shell = libint2::Shell{
        std::move(alpha),
        {{l, pure, std::move(coeffs)},},
        position
    };
    return shell;
}

void MoldenReader::parse_gto_section(const std::optional<std::string> &args, std::istream &stream)
{
    occ::log::debug("Parsing GTO section");
    auto pos = stream.tellg();
    std::string line;
    while(std::getline(stream, line))
    {
        if(is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        int atom_idx, second;
        scn::scan_default(line, atom_idx, second);
        assert(atom_idx <= m_atoms.size());
        std::array<double, 3> position{m_atoms[atom_idx - 1].x, m_atoms[atom_idx - 1].y, m_atoms[atom_idx - 1].z};
        while(std::getline(stream, line))
        {
            trim(line);
            if(line.empty()) {
                break;
            }
            stream.seekg(pos, std::ios_base::beg);
            m_basis.push_back(parse_molden_shell(position, m_pure, stream));
            pos = stream.tellg();
        }
    }
    m_basis.update();
}

void MoldenReader::parse_mo(size_t &mo_a, size_t &mo_b, std::istream &stream)
{
    double energy;
    bool alpha;
    double occupation{0.0};
    std::string line;
    while(std::getline(stream, line))
    {
        if(startswith(line, "Sym", false)) {

        }
        else if(startswith(line, "Ene", false)) {
            scn::scan(line, "Ene= {}", energy);
        }
        else if(startswith(line, "Spin", false)) {
            std::string spin;
            scn::scan(line, "Spin= {}", spin);
            to_lower(spin);
            alpha = (spin == "alpha");
        }
        else if(startswith(line, "Occup", false))
        {
            scn::scan(line, "Occup= {}", occupation);
            m_num_electrons += occupation;
        }
        else {
            for(size_t i = 0; i < nbf(); i++)
            {
                if(i > 0) std::getline(stream, line);
                int idx;
                double coeff;
                scn::scan_default(line, idx, coeff);
                if(alpha) {
                    m_molecular_orbitals_alpha(idx - 1, mo_a) = coeff;
                    m_energies_alpha(mo_a) = energy;
                }
                else {
                    m_molecular_orbitals_beta(idx - 1, mo_b) = coeff;
                    m_energies_beta(mo_b) = energy;
                }
            }
            break;
        }
    }
    if(alpha) {
        m_total_alpha_occupation += occupation;
        mo_a++;
    }
    else {
        m_total_beta_occupation += occupation;
        mo_b++;
    }
}


void MoldenReader::parse_mo_section(const std::optional<std::string> &args, std::istream &stream)
{
    occ::log::debug("Parsing MO section");
    auto pos = stream.tellg();
    std::string line;
    m_energies_alpha = occ::Vec(nbf());
    m_energies_beta = occ::Vec(nbf());
    m_molecular_orbitals_alpha = Mat(nbf(), nbf());
    m_molecular_orbitals_beta = Mat(nbf(), nbf());
    size_t num_alpha = 0, num_beta = 0;
    while(std::getline(stream, line))
    {
        if(is_section_line(line)) {
            stream.seekg(pos, std::ios_base::beg);
            break;
        }
        pos = stream.tellg();
        parse_mo(num_alpha, num_beta, stream);
    }
}


Mat MoldenReader::convert_mo_coefficients_from_molden_convention(const occ::qm::BasisSet& basis, const Mat& mo) const
{
    using occ::util::index_of;
    // no reordering should occur unless there are d, f, g, h etc. functions
    if(occ::qm::max_l(basis) < 2) return mo;
    struct xyz {
        uint_fast8_t x{0};
        uint_fast8_t y{0};
        uint_fast8_t z{0};
        bool operator ==(const xyz& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    };

    occ::log::debug("Reordering MO coefficients from Molden ordering to internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for(size_t i = 0; i < basis.size(); i++)
    {
        const auto& shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if(l < 2) {
            result.block(bf_first, 0, shell_size, ncols) = mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> molden_order;
        switch(l) {
        case 2:
            molden_order = {
                // xx, yy, zz, xy, xz, yz
                {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                {1, 1, 0}, {1, 0, 1}, {0, 1, 1}
            };
            break;
        case 3:
            molden_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3},
                {1, 2, 0}, {2, 1, 0}, {2, 0, 1},
                {1, 0, 2}, {0, 1, 2}, {0, 2, 1},
                {1, 1, 1}
            };
            break;
        }
        if (molden_order.size() == 0) {
            occ::log::warn("Unknown Molden ordering for shell with angular momentum {}, not reordering", l);
            continue;
        }

        int xp, yp, zp;
        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l)
        {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp), static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, molden_order);
            occ::log::debug("Setting row {} <- row {}", our_idx, gaussian_idx);
            result.row(bf_first + our_idx) = mo.row(bf_first + gaussian_idx);
            double normalization_factor = occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + our_idx) *= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}
}
