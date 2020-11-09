#pragma once
#include <istream>
#include <fstream>
#include <vector>
#include "linear_algebra.h"
#include "basisset.h"
#include "spinorbital.h"

namespace tonto::io {

class FchkReader
{
public:

    struct FchkBasis {
        size_t num_shells;
        size_t num_primitives;
        std::vector<int> shell_types;
        std::vector<int> primitives_per_shell;
        std::vector<int> shell2atom;
        std::vector<double> primitive_exponents;
        std::vector<double> contraction_coefficients;
        std::vector<double> sp_contraction_coefficients;
        std::vector<double> shell_coordinates;
        void print() const;
    };

    enum LineLabel {
        Unknown,
        NumElectrons,
        AtomicNumbers,
        AtomicPositions,
        NumBasisFunctions,
        NumAlpha,
        NumBeta,
        AlphaMO,
        BetaMO,
        AlphaMOEnergies,
        BetaMOEnergies,
        ShellToAtomMap,
        PrimitiveExponents,
        ContractionCoefficients,
        SPContractionCoefficients,
        ShellCoordinates,
        PrimitivesPerShell,
        NumShells,
        NumPrimitiveShells,
        ShellTypes,
        SCFDensity,
        MP2Density,
    };

    FchkReader(const std::string& filename);
    FchkReader(std::istream&);

    inline auto num_basis_functions() const { return m_num_basis_functions; }
    inline auto num_orbitals() const { return m_num_basis_functions; }
    inline auto num_electrons() const { return m_num_alpha; }
    inline auto num_alpha() const { return m_num_alpha; }
    inline auto num_beta() const { return m_num_beta; }

    inline auto spinorbital_kind() const {
        if((m_num_alpha != m_num_beta) ||
            (m_beta_mos.size() != 0) ) return tonto::qm::SpinorbitalKind::Unrestricted;
        return tonto::qm::SpinorbitalKind::Restricted;
    }

    inline const auto& basis() const { return m_basis; }

    inline auto atomic_numbers() const {
        return Eigen::Map<const tonto::IVec, 0>(m_atomic_numbers.data(), m_atomic_numbers.size());
    }

    inline auto atomic_positions() const {
        return Eigen::Map<const tonto::Mat, 0>(m_atomic_positions.data(), 3, m_atomic_positions.size() / 3);
    }

    inline auto alpha_mo_coefficients() const {
        return Eigen::Map<const tonto::Mat, 0>(m_alpha_mos.data(), m_num_basis_functions, m_num_basis_functions);
    }

    inline auto alpha_mo_energies() const {
        return Eigen::Map<const tonto::Vec, 0>(m_alpha_mo_energies.data(), m_alpha_mo_energies.size());
    }

    inline auto beta_mo_coefficients() const {
        return Eigen::Map<const tonto::Mat, 0>(m_beta_mos.data(), m_num_basis_functions, m_num_basis_functions);
    }

    inline auto beta_mo_energies() const {
        return Eigen::Map<const tonto::Vec, 0>(m_beta_mo_energies.data(), m_beta_mo_energies.size());
    }

    tonto::MatRM scf_density_matrix() const;
    tonto::MatRM mp2_density_matrix() const;

    std::vector<libint2::Atom> atoms() const;

    tonto::qm::BasisSet basis_set() const;
    tonto::MatRM convert_mo_coefficients_from_g09_convention(const tonto::qm::BasisSet&, const tonto::MatRM&) const;
private:
    void parse(std::istream&);
    void open(const std::string& filename);
    void close();
    LineLabel resolve_line(const std::string&) const;


    std::ifstream m_fchk_file;
    size_t m_num_electrons{0};
    size_t m_num_basis_functions{0};
    size_t m_num_alpha{0};
    size_t m_num_beta{0};

    std::vector<int> m_atomic_numbers;
    std::vector<double> m_atomic_positions;
    std::vector<double> m_alpha_mos;
    std::vector<double> m_alpha_mo_energies;
    std::vector<double> m_beta_mos;
    std::vector<double> m_beta_mo_energies;
    std::vector<double> m_scf_density;
    std::vector<double> m_mp2_density;
    FchkBasis m_basis;
};

}
