#pragma once
#include <occ/cg/result_types.h>
#include <occ/driver/cg_runner.h>
#include <occ/driver/crystal_growth.h>
#include <occ/mults/dma_energy_model.h>
#include <vector>

namespace occ::mults {

/// Crystal-growth calculator using a DMA-multipole + exp-6 (Williams)
/// interaction model. Overrides the monomer setup and lattice-energy
/// convergence of CEModelCrystalGrowthCalculator; solvation is a fixed
/// (unpolarized) continuum from the gas-phase DMA multipoles on a CPCM cavity.
class DMACrystalGrowthCalculator
    : public occ::driver::CEModelCrystalGrowthCalculator {
public:
  DMACrystalGrowthCalculator(
      const occ::crystal::Crystal &crystal,
      const occ::driver::CrystalGrowthCalculatorOptions &options);

  void init_monomer_energies() override;
  void converge_lattice_energy() override;

  /// Set the reference QM level used to generate the monomer multipoles
  /// (defaults to ce-b3lyp). Call before init_monomer_energies().
  void set_reference_level(const occ::driver::DMAReferenceLevel &ref) {
    m_reference = ref;
  }

private:
  /// Reference level resolved from m_reference; used for both the monomer
  /// wavefunctions and the dimer-cache key so the two cannot diverge.
  struct ResolvedLevel {
    std::string method, basis, label;
  };
  ResolvedLevel resolved_reference_level() const;

  std::vector<DMAMonomer> m_monomers;
  occ::driver::DMAReferenceLevel m_reference;
};

/// True if \p model_name selects the DMA+exp-6 interaction model.
bool model_name_is_dma(const std::string &model_name);

/// Run a crystal-growth calculation with the DMA+exp-6 model (reuses the
/// shared occ_driver cg pipeline).
occ::cg::CrystalGrowthResult run_cg_dma(const occ::driver::CGConfig &config);

/// Unified crystal-growth entry point: dispatches DMA+exp-6 models to
/// run_cg_dma() and everything else to occ::driver::run_cg(). Front-ends
/// (CLI, bindings) should call this. The dispatch lives here because
/// occ_driver must not depend on occ_mults.
occ::cg::CrystalGrowthResult run_crystal_growth(const occ::driver::CGConfig &config);

} // namespace occ::mults
