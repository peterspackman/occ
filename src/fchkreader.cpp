#include "fchkreader.h"
#include "util.h"
#include <scn/scn.h>
#include <fmt/core.h>

namespace tonto::io {

using tonto::util::trim_copy;


template<typename T>
void read_matrix_block(std::istream& stream, std::vector<T>& destination, size_t count, size_t values_per_line = 5)
{
    destination.reserve(count);
    std::string line;
    for(size_t i = 0; i < count; i+= values_per_line)
    {
        std::getline(stream, line);
        scn::scan_list(line, destination);
    }
}

inline bool startswith(const std::string& h, const std::string& prefix, bool trimmed = true)
{
    if(trimmed) {
        auto trimmed_str = trim_copy(h);
        return trimmed_str.rfind(prefix, 0) == 0;
    }
    return h.rfind(prefix, 0) == 0;
}

FchkReader::FchkReader(const std::string& filename)
{
    open(filename);
    parse(m_fchk_file);
}

FchkReader::FchkReader(std::istream& filehandle)
{
    parse(filehandle);
}

void FchkReader::open(const std::string& filename)
{
    m_fchk_file.open(filename);
    if(m_fchk_file.fail() | m_fchk_file.bad())
    {
        throw std::runtime_error("Unable to open fchk file: " + filename);
    }
}

void FchkReader::close()
{
    m_fchk_file.close();
}


FchkReader::LineLabel FchkReader::resolve_line(const std::string& line) const
{
    std::string lt = trim_copy(line);
    if(startswith(lt, "Number of electrons", false)) return NumElectrons;
    if(startswith(lt, "Atomic numbers", false)) return AtomicNumbers;
    if(startswith(lt, "Current cartesian coordinates", false)) return AtomicPositions;
    if(startswith(lt, "Number of basis functions", false)) return NumBasisFunctions;
    if(startswith(lt, "Number of alpha electrons", false)) return NumAlpha;
    if(startswith(lt, "Number of beta electrons", false)) return NumBeta;
    if(startswith(lt, "Alpha MO coefficients", false)) return AlphaMO;
    if(startswith(lt, "Beta MO coefficients", false)) return BetaMO;
    if(startswith(lt, "Alpha Orbital Energies", false)) return AlphaMOEnergies;
    if(startswith(lt, "Alpha Orbital Energies", false)) return BetaMOEnergies;
    if(startswith(lt, "Number of contracted shells", false)) return NumShells;
    if(startswith(lt, "Number of primitive shells", false)) return NumPrimitiveShells;
    if(startswith(lt, "Shell types", false)) return ShellTypes;
    if(startswith(lt, "Number of primitives per shell", false)) return PrimitivesPerShell;
    if(startswith(lt, "Shell to atom map", false)) return ShellToAtomMap;
    if(startswith(lt, "Primitive exponents", false)) return PrimitiveExponents;
    if(startswith(lt, "Contraction coefficients", false)) return ContractionCoefficients;
    if(startswith(lt, "Coordinates of each shell", false)) return ShellCoordinates;
    return Unknown;
}

void FchkReader::parse(std::istream& stream)
{
    std::string line;
    size_t count;
    while(std::getline(stream, line))
    {
        switch(resolve_line(line))
        {
        case NumElectrons:
            scn::scan(line, "Number of electrons I {}", m_num_electrons);
            break;
        case NumBasisFunctions:
            scn::scan(line, "Number of basis functions I {}", m_num_basis_functions);
            break;
        case NumAlpha:
            scn::scan(line, "Number of alpha electrons I {}", m_num_alpha);
            break;
        case NumBeta:
            scn::scan(line, "Number of beta electrons I {}", m_num_beta);
            break;
        case AtomicNumbers:
            scn::scan(line, "Atomic numbers I N= {}", count);
            read_matrix_block<int>(stream, m_atomic_numbers, count);
            break;
        case AtomicPositions:
            scn::scan(line, "Current cartesian coordinates R N= {}", count);
            read_matrix_block<double>(stream, m_atomic_positions, count);
            break;
        case AlphaMO:
            scn::scan(line, "Alpha MO coefficients R N= {}", count);
            read_matrix_block<double>(stream, m_alpha_mos, count);
            break;
        case BetaMO:
            scn::scan(line, "Beta MO coefficients R N= {}", count);
            read_matrix_block<double>(stream, m_beta_mos, count);
            break;
        case AlphaMOEnergies:
            scn::scan(line, "Alpha Orbital Energies R N= {}", count);
            read_matrix_block<double>(stream, m_alpha_mo_energies, count);
            break;
        case BetaMOEnergies:
            scn::scan(line, "Alpha Orbital Energies R N= {}", count);
            read_matrix_block<double>(stream, m_beta_mo_energies, count);
            break;
        case NumShells:
            scn::scan(line, "Number of contracted shells I {}", m_basis.num_shells);
            break;
        case NumPrimitiveShells:
            scn::scan(line, "Number of primitive shells I {}", m_basis.num_primitives);
            break;
        case ShellTypes:
            scn::scan(line, "Shell types I N= {}", count);
            read_matrix_block<int>(stream, m_basis.shell_types, count);
            break;
        case PrimitivesPerShell:
            scn::scan(line, "Number of primitives per shell I N= {}", count);
            read_matrix_block<int>(stream, m_basis.primitives_per_shell, count);
            break;
        case ShellToAtomMap:
            scn::scan(line, "Shell to atom map I N= {}", count);
            read_matrix_block<int>(stream, m_basis.shell2atom, count);
            break;
        case PrimitiveExponents:
            scn::scan(line, "Primitive exponents R N= {}", count);
            read_matrix_block<double>(stream, m_basis.primitive_exponents, count);
            break;
        case ContractionCoefficients:
            scn::scan(line, "Contraction coefficients R N= {}", count);
            read_matrix_block<double>(stream, m_basis.contraction_coefficients, count);
            break;
        case ShellCoordinates:
            scn::scan(line, "Coordinates of each shell R N= {}", count);
            read_matrix_block<double>(stream, m_basis.shell_coordinates, count);
            break;
        default: continue;
        }
    }
}

std::vector<libint2::Atom> FchkReader::atoms() const {
    std::vector<libint2::Atom> atoms;
    atoms.reserve(m_atomic_numbers.size());
    for(size_t i = 0; i < m_atomic_numbers.size(); i++)
    {
        atoms.emplace_back(libint2::Atom{m_atomic_numbers[i], m_atomic_positions[3*i], m_atomic_positions[3*i + 1], m_atomic_positions[3*i + 2]});
    }
    return atoms;
}

libint2::BasisSet FchkReader::libint_basis() const {
    size_t num_shells = m_basis.num_shells;
    libint2::BasisSet basis_set;
    size_t primitive_offset{0};
    for(size_t i = 0; i < num_shells; i++) {
        int l = m_basis.shell_types[i];
        bool pure = true;
        size_t nprim = m_basis.primitives_per_shell[i];
        libint2::svector<double> alpha(nprim);
        libint2::svector<double> coeffs(nprim);
        std::array<double, 3> position;
        std::copy(m_basis.contraction_coefficients.begin() + primitive_offset, m_basis.contraction_coefficients.begin() + primitive_offset + nprim, coeffs.begin());
        std::copy(m_basis.primitive_exponents.begin() + primitive_offset, m_basis.primitive_exponents.begin() + primitive_offset + nprim, alpha.begin());
        std::copy(m_basis.shell_coordinates.begin() + 3 * i, m_basis.shell_coordinates.begin() + 3 * (i + 1), position.begin());
        libint2::Shell::Contraction c{l, pure, coeffs};
        basis_set.emplace_back(libint2::Shell(alpha, {c}, position));
        primitive_offset += nprim;
    }
    return basis_set;
}

void FchkReader::FchkBasis::print()
{
    size_t contraction_offset{0};
    size_t primitive_offset{0};
    for(size_t i = 0; i < num_shells; i++) {
        fmt::print("Shell {} on atom {}\n", i, shell2atom[i] - 1);
        fmt::print("Position: {:10.5f} {:10.5f} {:10.5f}\n", shell_coordinates[3 * i], shell_coordinates[3 * i + 1], shell_coordinates[3 * i + 2]);
        fmt::print("Angular momentum: {}\n", shell_types[i]);
        size_t num_primitives = primitives_per_shell[i];
        fmt::print("Primitives Gaussians: {}\n", num_primitives);
        fmt::print("Primitive exponents:");
        for(size_t i = 0; i < num_primitives; i++) {
            fmt::print(" {}", primitive_exponents[primitive_offset]);
            primitive_offset++;
        }
        fmt::print("\n");
        fmt::print("Contraction coefficients:");
        for(size_t i = 0; i < num_primitives; i++) {
            fmt::print(" {}", contraction_coefficients[contraction_offset]);
            contraction_offset++;
        }
        fmt::print("\n");
    }
}

}
