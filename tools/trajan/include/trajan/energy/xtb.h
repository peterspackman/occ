#pragma once
#include <occ/crystal/unitcell.h>
#include <trajan/core/frame.h>
##include <xtb.h>

namespace trajan::energy {

struct SinglePoint {
  double energy{0};
  occ::Mat3N grads;
  occ::Mat3 virial;
};

class XTBModel {
public:
  enum Type { GFN0xTB, GFN1xTB, GFN2xTB, GFNFF };
  enum Calc { Singlepoint, Hessian };

  XTBModel();
  ~XTBModel();
  void set_verbosity(int v);

  SinglePoint single_point(const core::Frame &frame, Type model = Type::GFNFF);
  occ::Mat hessian(const core::Frame &frame, Type model = Type::GFNFF);

private:
  void check_errors();
  void free_all();
  void initialise();
  void initialise_molecule(const core::Frame &frame, Type model);
  xtb_TEnvironment m_env;
  xtb_TCalculator m_calc;
  xtb_TResults m_res;
  xtb_TMolecule m_mol;
  bool m_has_initialised_molecule{false};
  Calc m_previous_calc = Calc::Singlepoint;
};

} // namespace trajan::energy
