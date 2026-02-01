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
    m_model_params = ce_model_from_string(m_model_name);
  }

  CEEnergyComponents compute_energy(const core::Dimer &dimer) override;
  Mat3N compute_electric_field(const core::Dimer &dimer) override;
  const std::vector<Vec> &partial_charges() const override;
  double coulomb_scale_factor() const override;
  double polarization_scale_factor() const { return m_model_params.polarization; }
  void compute_total_energy(CEEnergyComponents &components) const override;

  Mat3N compute_total_electric_field_from_neighbors(
    const core::Molecule &target_molecule,
    const std::vector<core::Dimer> &neighbor_dimers);

  double compute_crystal_field_polarization_energy(
    const core::Molecule &molecule,
    const Mat3N &crystal_field) const;

  Vec get_polarizabilities(const core::Molecule &molecule) const;


private:
  Wavefunction prepare_wavefunction(const core::Molecule &mol,
                                    const Wavefunction &wfn) const;

  crystal::Crystal m_crystal;
  std::string m_model_name{"ce-b3lyp"};
  CEParameterizedModel m_model_params;
  std::vector<Wavefunction> m_wavefunctions_a;
  std::vector<Wavefunction> m_wavefunctions_b;
  mutable std::vector<Vec> m_partial_charges;
};

} // namespace occ::interaction
