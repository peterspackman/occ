#pragma once
#include <occ/core/dimer.h>
#include <occ/interaction/pair_energy.h>

namespace occ::interaction {

class EnergyModelBase {
public:
  virtual ~EnergyModelBase() = default;
  virtual CEEnergyComponents compute_energy(const core::Dimer &dimer) = 0;
  virtual Mat3N compute_electric_field(const core::Dimer &dimer) = 0;
  virtual const std::vector<Vec> &partial_charges() const = 0;
  virtual double coulomb_scale_factor() const = 0;
};

} // namespace occ::interaction
