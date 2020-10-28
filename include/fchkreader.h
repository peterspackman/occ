#pragma once
#include <istream>
#include <fstream>
#include <vector>
#include "linear_algebra.h"

namespace tonto::io {

class FchkReader
{
public:
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

    };

    FchkReader(const std::string& filename);
    FchkReader(std::istream&);

    inline auto num_basis_functions() const { return m_num_basis_functions; }
    inline auto num_orbitals() const { return m_num_basis_functions; }
    inline auto num_alpha() const { return m_num_alpha; }
    inline auto num_beta() const { return m_num_beta; }

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
};

}
