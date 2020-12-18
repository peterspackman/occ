#include <tonto/core/util.h>
#include <tonto/core/logger.h>
#include <tonto/io/fchkreader.h>
#include <scn/scn.h>
#include <fmt/ostream.h>
#include <libint2/cgshell_ordering.h>
#include <tonto/gto/gto.h>

namespace tonto::io {

using tonto::util::trim_copy;
using tonto::util::startswith;
using tonto::qm::BasisSet;


template<typename T>
void read_matrix_block(std::istream& stream, std::vector<T>& destination, size_t count)
{
    destination.reserve(count);
    std::string line;
    while(destination.size() < count)
    {
        std::getline(stream, line);
        scn::scan_list(line, destination);
    }
}

FchkReader::FchkReader(const std::string& filename)
{
    tonto::timing::start(tonto::timing::category::io);
    open(filename);
    parse(m_fchk_file);
    tonto::timing::stop(tonto::timing::category::io);
}

FchkReader::FchkReader(std::istream& filehandle)
{
    tonto::timing::start(tonto::timing::category::io);
    parse(filehandle);
    tonto::timing::stop(tonto::timing::category::io);
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
    if(startswith(lt, "Number of electrons", false)) return NumElectrons;
    if(startswith(lt, "Number of alpha electrons", false)) return NumAlpha;
    if(startswith(lt, "Number of beta electrons", false)) return NumBeta;
    if(startswith(lt, "Alpha MO coefficients", false)) return AlphaMO;
    if(startswith(lt, "Beta MO coefficients", false)) return BetaMO;
    if(startswith(lt, "Alpha Orbital Energies", false)) return AlphaMOEnergies;
    if(startswith(lt, "Beta Orbital Energies", false)) return BetaMOEnergies;
    if(startswith(lt, "Number of contracted shells", false)) return NumShells;
    if(startswith(lt, "Number of primitive shells", false)) return NumPrimitiveShells;
    if(startswith(lt, "Shell types", false)) return ShellTypes;
    if(startswith(lt, "Number of primitives per shell", false)) return PrimitivesPerShell;
    if(startswith(lt, "Shell to atom map", false)) return ShellToAtomMap;
    if(startswith(lt, "Primitive exponents", false)) return PrimitiveExponents;
    if(startswith(lt, "Contraction coefficients", false)) return ContractionCoefficients;
    if(startswith(lt, "P(S=P) Contraction coefficients", false)) return SPContractionCoefficients;
    if(startswith(lt, "Coordinates of each shell", false)) return ShellCoordinates;
    if(startswith(lt, "Total SCF Density", false)) return SCFDensity;
    if(startswith(lt, "Total MP2 Density", false)) return MP2Density;

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
            scn::scan(line, "Beta Orbital Energies R N= {}", count);
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
        case SPContractionCoefficients:
            scn::scan(line, "P(S=P) Contraction coefficients R N= {}", count);
            read_matrix_block<double>(stream, m_basis.sp_contraction_coefficients, count);
            break;
        case ShellCoordinates:
            scn::scan(line, "Coordinates of each shell R N= {}", count);
            read_matrix_block<double>(stream, m_basis.shell_coordinates, count);
            break;
        case SCFDensity:
            scn::scan(line, "Total SCF Density R N= {}", count);
            read_matrix_block<double>(stream, m_scf_density, count);
            break;
        case MP2Density:
            scn::scan(line, "Total MP2 Density R N= {}", count);
            read_matrix_block<double>(stream, m_mp2_density, count);
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

BasisSet FchkReader::basis_set() const {
    size_t num_shells = m_basis.num_shells;
    BasisSet bs;
    size_t primitive_offset{0};
    constexpr int SP_SHELL{-1};
    for(size_t i = 0; i < num_shells; i++) {
        // shell types: 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
        int shell_type = m_basis.shell_types[i];
        int l = std::abs(shell_type);
        // normally shell type < -1 will be pure
        auto pure = false;

        size_t nprim = m_basis.primitives_per_shell[i];
        std::array<double, 3> position{
            m_basis.shell_coordinates[3 * i],
            m_basis.shell_coordinates[3 * i + 1],
            m_basis.shell_coordinates[3 * i + 2],
        };

        if(shell_type == SP_SHELL) {
            libint2::svector<double> alpha;
            libint2::svector<double> coeffs;
            libint2::svector<double> pcoeffs;
            for(size_t prim = 0; prim < nprim; prim++) {
                alpha.emplace_back(m_basis.primitive_exponents[primitive_offset + prim]);
                coeffs.emplace_back(m_basis.contraction_coefficients[primitive_offset + prim]);
                pcoeffs.emplace_back(m_basis.sp_contraction_coefficients[primitive_offset + prim]);
            }
            //sp shell
            bs.emplace_back(libint2::Shell{
                                alpha,
                                {
                                    {0, pure, std::move(coeffs)},
                                },
                                {position}
                            });
            bs.emplace_back(libint2::Shell{
                                std::move(alpha),
                                {
                                    {1, pure, std::move(pcoeffs)},
                                },
                                {std::move(position)}
                            });
        }
        else {
            libint2::svector<double> alpha;
            libint2::svector<double> coeffs;
            for(size_t prim = 0; prim < nprim; prim++) {
                alpha.emplace_back(m_basis.primitive_exponents[primitive_offset + prim]);
                coeffs.emplace_back(m_basis.contraction_coefficients[primitive_offset + prim]);
            }
            bs.emplace_back(libint2::Shell{
                                std::move(alpha),
                                {
                                    {l, pure, std::move(coeffs)}
                                },
                                {std::move(position)}
                            });
        }
        primitive_offset += nprim;
    }
    bs.update();
    bs.set_pure(false);
    return bs;
}

void FchkReader::FchkBasis::print() const
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

tonto::MatRM FchkReader::scf_density_matrix() const
{

    tonto::MatRM C_occ = alpha_mo_coefficients().leftCols(m_num_alpha);
    return C_occ * C_occ.transpose();
}

tonto::MatRM FchkReader::mp2_density_matrix() const
{

    size_t nbf{num_basis_functions()};
    assert(nbf * (nbf - 1) == m_mp2_density.size());
    tonto::MatRM dm(nbf, nbf);
    size_t idx = 0;
    for(size_t i = 0; i < nbf; i++) {
        for(size_t j = i; j < nbf; j++) {
            if(i != j) dm(j, i) = m_mp2_density[idx];
            dm(i, j) = m_mp2_density[idx];
            idx++;
        }
    }
    return dm;
}

}
