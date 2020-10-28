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
        default: continue;
        }
    }
}

}
