#include <occ/dma/dmaql0.h>
#include <occ/dma/linear_multipole_calculator.h>

namespace occ::dma {

std::vector<Mult> dmaql0(const occ::qm::Wavefunction &wfn, int max_rank,
                         bool include_nuclei, bool use_slices) {
  
  // Setup settings for the linear calculator
  LinearDMASettings settings;
  settings.max_rank = max_rank;
  settings.include_nuclei = include_nuclei;
  settings.use_slices = use_slices;
  
  // Create and use the linear multipole calculator
  LinearMultipoleCalculator calculator(wfn, settings);
  return calculator.calculate();
}
} // namespace occ::dma
