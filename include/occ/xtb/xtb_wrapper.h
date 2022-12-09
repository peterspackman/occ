#pragma once
#include <occ/core/molecule.h>

extern "C" {
#include "tblite/calculator.h"
#include "tblite/error.h"
#include "tblite/structure.h"
#include "tblite/version.h"
}

namespace occ::xtb {

class XTBCalculator {
  public:
    enum class Method { GFN1, GFN2 };
    XTBCalculator(const occ::core::Molecule &mol);
    XTBCalculator(const occ::core::Molecule &mol, Method method);
    double single_point_energy();
    inline const auto &gradients() const { return m_gradients; }

    ~XTBCalculator();

  private:
    void initialize_context();
    void initialize_method();
    void initialize_structure();

    Mat3N m_positions_bohr;
    Mat3N m_gradients;
    IVec m_atomic_numbers;
    Method m_method{Method::GFN2};
    tblite_error m_tb_error{nullptr};
    tblite_context m_tb_ctx{nullptr};
    tblite_result m_tb_result{nullptr};
    tblite_structure m_tb_structure{nullptr};
    tblite_calculator m_tb_calc{nullptr};
};

std::string tblite_version();

} // namespace occ::xtb
