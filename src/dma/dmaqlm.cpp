#include <occ/dma/dmaqlm.h>
#include <occ/dma/multipole_calculator.h>

namespace occ::dma {

std::vector<Mult> dmaqlm(const qm::AOBasis &basis,
                         const qm::MolecularOrbitals &mo, const DMASites &sites,
                         const DMASettings &settings) {
  MultipoleCalculator calculator(basis, mo, sites, settings);
  return calculator.calculate();
}

} // namespace occ::dma