#pragma once
#include <occ/dma/dma.h>
#include <occ/dma/mult.h>
#include <occ/io/structure_format.h>
#include <string>
#include <vector>

namespace occ::mults {

struct DMAForceFieldOptions {
  /// Name or alias from short_range_model_registry(), declared in
  /// occ/mults/dimer_interaction.h.
  std::string force_field{"w99"};
  std::string molecule_name{"mol"};
  /// Translate sites so the molecular centre of mass is at the origin, which
  /// is the body-frame convention CSP programs expect.
  bool center_on_com{true};
};

/**
 * @brief Build the force-field description that `occ dma --write-csp-input`
 *        serializes.
 *
 * Assigns NEIGHCRYS atom types (when the chosen set is typed), converts the
 * distributed multipoles into the structure-format layout, and emits pair
 * potentials for exactly the type pairs present in this molecule.
 *
 * @param sites      DMA sites, positions in Bohr
 * @param multipoles one entry per site, in the same order
 */
occ::io::Basis
build_dma_force_field_basis(const occ::dma::DMASites &sites,
                            const std::vector<occ::dma::Mult> &multipoles,
                            const DMAForceFieldOptions &options = {});

} // namespace occ::mults
