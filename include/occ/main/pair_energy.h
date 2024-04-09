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

    CEEnergyComponents compute();
    Monomer a;
    Monomer b;
    CEParameterizedModel model{occ::interaction::CE_B3LYP_631Gdp};
    CEEnergyComponents energy;
};


struct PairEnergyStore {
  enum class Kind {JSON, Xyz, Memory};

  bool save(int id, const Dimer &d, const CEEnergyComponents &);
  bool load(int id, const Dimer &d, CEEnergyComponents &);
  std::string dimer_filename(int id, const Dimer &d);

  Kind kind{Kind::JSON};
  std::string name;
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
    bool wolf_sum{false};
    bool crystal_field_polarization{false};
    std::string model_name{"ce-b3lyp"};
    std::string crystal_filename{""};
    std::string output_json_filename{""};
    bool spherical_basis{false};
};

struct LatticeEnergyResult {
    double lattice_energy{0.0};
    occ::crystal::CrystalDimers dimers;
    std::vector<CEEnergyComponents> energy_components;
};

LatticeEnergyResult
converged_lattice_energies(const Crystal &crystal,
                           const std::vector<Wavefunction> &wfns_a,
                           const std::vector<Wavefunction> &wfns_b,
                           const std::string &basename = "crystal_dimer",
                           const LatticeConvergenceSettings conv = {});

LatticeEnergyResult
converged_xtb_lattice_energies(const Crystal &crystal,
                               const std::string &basename = "crystal_dimer",
                               const LatticeConvergenceSettings conv = {});

Wavefunction load_wavefunction(const std::string &filename);

} // namespace occ::main
