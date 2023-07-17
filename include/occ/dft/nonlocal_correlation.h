#pragma once
#include <occ/dft/grid.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>

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

    NonLocalCorrelationFunctional();
    void set_parameters(const Parameters &params);
    void set_integration_grid(const qm::AOBasis &basis,
                              const BeckeGridSettings &settings = {
                                  110, 50, 50, 1e-7,
                                  false}); // don't reduce H by default

    Result operator()(const qm::AOBasis &, const qm::MolecularOrbitals &);

  private:
    Result vv10(const qm::AOBasis &, const qm::MolecularOrbitals &mo);
    Parameters m_params;
    std::vector<AtomGrid> m_nlc_atom_grids;
};
} // namespace occ::dft
