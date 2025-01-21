#pragma once
#include <occ/core/dimer.h>
#include <occ/crystal/crystal.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>

namespace occ::interaction {

struct PairEnergy {
  struct Monomer {
    occ::qm::Wavefunction wfn;
    occ::Mat3 rotation{occ::Mat3::Identity()};
    occ::Vec3 translation{occ::Vec3::Zero()};
  };

  PairEnergy(const occ::io::OccInput &input);

  CEEnergyComponents compute();
  Monomer a;
  Monomer b;
  CEParameterizedModel model{occ::interaction::CE_B3LYP_631Gdp};
  CEEnergyComponents energy;
};

Wavefunction load_wavefunction(const std::string &filename);

} // namespace occ::interaction
