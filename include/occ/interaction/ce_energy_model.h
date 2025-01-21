#pragma once
#include <occ/crystal/crystal.h>
#include <occ/interaction/energy_model_base.h>

namespace occ::interaction {

class CEEnergyModel : public EnergyModelBase {
public:
  CEEnergyModel(const crystal::Crystal &crystal,
                const std::vector<Wavefunction> &wfns_a,
                const std::vector<Wavefunction> &wfns_b = {});

  void set_model_name(const std::string &model_name) {
    m_model_name = model_name;
  }

  CEEnergyComponents compute_energy(const core::Dimer &dimer) override;
  Mat3N compute_electric_field(const core::Dimer &dimer) override;
  const std::vector<Vec> &partial_charges() const override;
  double coulomb_scale_factor() const override;

private:
  Wavefunction prepare_wavefunction(const core::Molecule &mol,
                                    const Wavefunction &wfn) const;

  crystal::Crystal m_crystal;
  std::string m_model_name{"ce-b3lyp"};
  std::vector<Wavefunction> m_wavefunctions_a;
  std::vector<Wavefunction> m_wavefunctions_b;
  mutable std::vector<Vec> m_partial_charges;
};

} // namespace occ::interaction
