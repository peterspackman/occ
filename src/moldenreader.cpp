#include "moldenreader.h"
#include <scn/scn.h>
#include <fmt/ostream.h>
#include "logger.h"
#include "util.h"


namespace tonto::io {
using tonto::util::trim;
using tonto::util::startswith;
using tonto::util::to_lower;

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
    std::ifstream file(filename);
    parse(file);
}

MoldenReader::MoldenReader(std::istream &file)
{
    parse(file);
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
            fmt::print("Section: {}\n", section_name);
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
    tonto::log::info("Parsing Atoms section");
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
        libint2::Atom atom;
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
    char shell_type;
    int num_primitives, second;
    scn::scan_default(line, shell_type, num_primitives, second);
    libint2::svector<double> alpha, coeffs;
    alpha.reserve(num_primitives), coeffs.reserve(num_primitives);
    for(int i = 0; i < num_primitives; i++)
    {
        std::getline(stream, line);
        double e, c;
        scn::scan_default(line, e, c);
        alpha.push_back(e);
        coeffs.push_back(c);
    }
    return libint2::Shell{
        std::move(alpha),
        {
            {l_from_char(shell_type), pure, std::move(coeffs)},
        },
        {position}
    };
}

void MoldenReader::parse_gto_section(const std::optional<std::string> &args, std::istream &stream)
{
    tonto::log::info("Parsing GTO section");
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
        bool pure = true;
        while(std::getline(stream, line))
        {
            trim(line);
            if(line.empty()) {
                break;
            }
            stream.seekg(pos, std::ios_base::beg);
            m_basis.push_back(parse_molden_shell(position, pure, stream));
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
    tonto::log::info("Parsing MO section");
    auto pos = stream.tellg();
    std::string line;
    m_energies_alpha = tonto::Vec(nbf());
    m_energies_beta = tonto::Vec(nbf());
    m_molecular_orbitals_alpha = tonto::MatRM(nbf(), nbf());
    m_molecular_orbitals_beta = tonto::MatRM(nbf(), nbf());
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

}
