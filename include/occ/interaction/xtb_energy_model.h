#pragma once
#include <occ/crystal/crystal.h>
#include <occ/interaction/energy_model_base.h>

namespace occ::interaction {

class XTBEnergyModel : public EnergyModelBase {
public:
  explicit XTBEnergyModel(const crystal::Crystal &crystal);

  CEEnergyComponents compute_energy(const core::Dimer &dimer) override;
  Mat3N compute_electric_field(const core::Dimer &dimer) override;
  const std::vector<Vec> &partial_charges() const override;
  double coulomb_scale_factor() const override { return 1.0; }

private:
  crystal::Crystal m_crystal;
  std::vector<double> m_monomer_energies;
  std::vector<Vec> m_partial_charges;
};

} // namespace occ::interaction
