#pragma once
#include <istream>
#include <fstream>
#include <vector>

namespace tonto::io {

class FchkReader
{
public:
    enum LineLabel {
      Unknown,
      NumBasisFunctions,
      NumAlpha,
      NumBeta,
      AlphaMO,
      BetaMO,
    };

    FchkReader(const std::string& filename);
    FchkReader(std::istream&);

    inline auto num_basis_functions() const { return m_num_basis_functions; }
    inline auto num_orbitals() const { return m_num_basis_functions; }
    inline auto num_alpha() const { return m_num_alpha; }
    inline auto num_beta() const { return m_num_beta; }

private:
    void parse(std::istream&);
    void open(const std::string& filename);
    void close();
    LineLabel resolve_line(const std::string&) const;


    std::ifstream m_fchk_file;
    uint_fast32_t m_num_basis_functions{0};
    uint_fast32_t m_num_alpha{0};
    uint_fast32_t m_num_beta{0};
    std::vector<double> m_alpha_mos;
    std::vector<double> m_beta_mos;
};

}
