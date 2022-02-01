#pragma once
#include <occ/main/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/core/dimer.h>
#include <occ/crystal/crystal.h>
#include <optional>

namespace occ::main {

using EnergyComponentsCE = occ::interaction::CEModelInteraction::EnergyComponents;
using occ::core::Dimer;
using occ::qm::Wavefunction;
using occ::crystal::Crystal;

bool load_dimer_energy(const std::string &, EnergyComponentsCE&);
bool write_xyz_dimer(const std::string&, const Dimer &,
                     std::optional<EnergyComponentsCE> energies = {});

EnergyComponentsCE ce_model_energy(const Dimer &dimer,
                                   const std::vector<Wavefunction> &wfns_a,
                                   const std::vector<Wavefunction> &wfns_b,
                                   const Crystal &crystal);

std::vector<EnergyComponentsCE> ce_model_energies(
                              const Crystal &crystal, 
                              const std::vector<Dimer> &dimers,
                              const std::vector<Wavefunction> &wfns_a,
                              const std::vector<Wavefunction> &wfns_b,
                              const std::string &basename = "dimer");


struct LatticeConvergenceSettings {
    double min_radius{3.8}; // angstroms
    double max_radius{30.0}; // angstroms
    double radius_increment{3.8}; // angstroms
    double energy_tolerance{1.0}; // kj/mol
};

std::pair<occ::crystal::CrystalDimers, std::vector<EnergyComponentsCE>>
converged_lattice_energies(const Crystal &crystal,
    const std::vector<Wavefunction> &wfns_a,
    const std::vector<Wavefunction> &wfns_b,
    const std::string &basename = "crystal_dimer",
    const LatticeConvergenceSettings conv = {});
}
