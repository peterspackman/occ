#pragma once
#include <occ/numint/grid_types.h>
#include <occ/qm/mo.h>
#include <occ/gto/shell.h>

namespace occ::dft {

class NonLocalCorrelationFunctional {
public:
  enum class Kind {
    VV10,
  };

  struct Parameters {
    Kind kind;
    double vv10_b{6.0};
    double vv10_C{0.01};
  };

  struct Result {
    double energy;
    Mat Vxc;
  };

  // WARNING: VV10/NLC gradients are experimental and not fully tested.
  // Current implementation assumes post-SCF correction (not self-consistent)
  // and may not match programs that include VV10 in the SCF.
  struct GradientResult {
    double energy;
    Mat Vxc;
    Mat3N gradient;  // Nuclear gradient contributions
  };

  NonLocalCorrelationFunctional();
  void set_parameters(const Parameters &params);
  void set_integration_grid(const gto::AOBasis &basis,
                            const GridSettings &settings = {
                                110, 50, 50, 1e-7,
                                false}); // don't reduce H by default

  Result operator()(const gto::AOBasis &, const qm::MolecularOrbitals &);
  GradientResult compute_gradient(const gto::AOBasis &, const qm::MolecularOrbitals &) const;

private:
  Result vv10(const gto::AOBasis &, const qm::MolecularOrbitals &mo);
  GradientResult vv10_gradient(const gto::AOBasis &, const qm::MolecularOrbitals &mo) const;
  Parameters m_params;
  std::vector<AtomGrid> m_nlc_atom_grids;
};
} // namespace occ::dft
