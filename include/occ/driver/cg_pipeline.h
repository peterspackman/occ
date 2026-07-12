#pragma once
#include <occ/cg/result_types.h>
#include <occ/crystal/crystal.h>
#include <occ/driver/cg_runner.h>
#include <occ/driver/crystal_growth.h>
#include <vector>

namespace occ::driver {

/// Result of loading and configuring a cg run: the crystal, the calculator
/// options and any per-molecule charges. Lets calculators defined outside
/// occ_driver (e.g. the DMA+exp-6 model in occ_mults) reuse the shared
/// pipeline below.
struct CGPreparation {
  crystal::Crystal crystal;
  CrystalGrowthCalculatorOptions opts;
  std::vector<int> charges;
};

/// Load the crystal and build calculator options + charges from \p config.
CGPreparation prepare_cg(CGConfig const &config);

/// Run the full crystal-growth pipeline on an already-constructed calculator:
/// monomer energies -> lattice convergence -> molecular surroundings ->
/// surface energies / morphology -> JSON serialization.
occ::cg::CrystalGrowthResult
run_cg_pipeline(CrystalGrowthCalculator &calc,
                const CrystalGrowthCalculatorOptions &opts,
                CGConfig const &config);

} // namespace occ::driver
