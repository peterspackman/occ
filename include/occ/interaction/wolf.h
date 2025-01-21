#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/energy_model_base.h>

namespace occ::interaction {

struct WolfParameters {
  double cutoff{16.0};
  double alpha{0.2};
};

double wolf_coulomb_energy(double qi, const Vec3 &pi, Eigen::Ref<const Vec> qj,
                           Eigen::Ref<const Mat3N> pj,
                           const WolfParameters &params);

class WolfSum {
public:
  WolfSum(WolfParameters params = {});

  void initialize(const crystal::Crystal &crystal,
                  const EnergyModelBase &energy_model);

  double compute_correction(
      const std::vector<double> &charge_energies,
      const std::vector<CEEnergyComponents> &model_energies) const;

  Mat3N &electric_field_for_molecule(size_t idx) {
    return m_electric_field_values[idx];
  }

  inline const auto &asymmetric_charges() const { return m_asym_charges; }

private:
  void compute_self_energies(const crystal::Crystal &crystal);
  void compute_wolf_energies(const crystal::Crystal &crystal);

  WolfParameters m_params;
  Vec m_asym_charges;
  std::vector<double> m_charge_self_energies;
  std::vector<Mat3N> m_electric_field_values;
  double m_total_energy{0.0};
};

} // namespace occ::interaction
