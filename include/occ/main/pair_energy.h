#pragma once
#include <occ/core/dimer.h>
#include <occ/crystal/crystal.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/io/occ_input.h>
#include <occ/main/pair_energy.h>
#include <occ/qm/wavefunction.h>
#include <optional>

namespace occ::main {

using occ::core::Dimer;
using occ::crystal::Crystal;
using occ::interaction::CEEnergyComponents;
using occ::interaction::CEParameterizedModel;
using occ::qm::Wavefunction;

struct PairEnergy {
    struct Monomer {
        occ::qm::Wavefunction wfn;
        occ::Mat3 rotation{occ::Mat3::Identity()};
        occ::Vec3 translation{occ::Vec3::Zero()};
    };

    PairEnergy(const occ::io::OccInput &input);

    void compute();
    Monomer a;
    Monomer b;
    CEParameterizedModel model{occ::interaction::CE_B3LYP_631Gdp};
    CEEnergyComponents energy;
};

bool load_dimer_energy(const std::string &, CEEnergyComponents &);
bool write_xyz_dimer(const std::string &, const Dimer &,
                     std::optional<CEEnergyComponents> energies = {});

CEEnergyComponents ce_model_energy(const Dimer &dimer,
                                   const std::vector<Wavefunction> &wfns_a,
                                   const std::vector<Wavefunction> &wfns_b,
                                   const Crystal &crystal);

std::vector<CEEnergyComponents>
ce_model_energies(const Crystal &crystal, const std::vector<Dimer> &dimers,
                  const std::vector<Wavefunction> &wfns_a,
                  const std::vector<Wavefunction> &wfns_b,
                  const std::string &basename = "dimer");

struct LatticeConvergenceSettings {
    double min_radius{3.8};       // angstroms
    double max_radius{30.0};      // angstroms
    double radius_increment{3.8}; // angstroms
    double energy_tolerance{1.0}; // kj/mol
};

std::pair<occ::crystal::CrystalDimers, std::vector<CEEnergyComponents>>
converged_lattice_energies(const Crystal &crystal,
                           const std::vector<Wavefunction> &wfns_a,
                           const std::vector<Wavefunction> &wfns_b,
                           const std::string &basename = "crystal_dimer",
                           const LatticeConvergenceSettings conv = {});

std::pair<occ::crystal::CrystalDimers, std::vector<CEEnergyComponents>>
converged_xtb_lattice_energies(const Crystal &crystal,
                               const std::string &basename = "crystal_dimer",
                               const LatticeConvergenceSettings conv = {});

} // namespace occ::main
