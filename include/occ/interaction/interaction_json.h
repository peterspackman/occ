#pragma once
#include <nlohmann/json.hpp>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/polarization_partitioning.h>
#include <occ/crystal/crystal.h>

namespace occ::interaction {

void to_json(nlohmann::json &j, const CEEnergyComponents &);
void from_json(const nlohmann::json &j, CEEnergyComponents &);

void to_json(nlohmann::json &j, const polarization_partitioning::CouplingTerm &);
void from_json(const nlohmann::json &j, polarization_partitioning::CouplingTerm &);

void to_json(nlohmann::json &j, const polarization_partitioning::MoleculeCouplingResults &);
void from_json(const nlohmann::json &j, polarization_partitioning::MoleculeCouplingResults &);

struct ElatResults {
    occ::crystal::Crystal crystal;
    LatticeEnergyResult lattice_energy_result;
    std::string title;
    std::string model;
};

/**
 * \brief Write elat JSON results in compact format
 *
 * Writes crystal, dimers, and energy data to JSON file
 *
 * \param filename Output filename
 * \param results ElatResults to write
 */
void write_elat_json(const std::string& filename, const ElatResults& results);

/**
 * \brief Read elat JSON results and reconstruct data structures
 *
 * Loads crystal and reconstructs LatticeEnergyResult with energies
 * mapped from the compact JSON format produced by write_elat_json()
 *
 * \param filename Path to elat JSON results file
 * \return ElatResults containing crystal and lattice energy data
 */
ElatResults read_elat_json(const std::string& filename);

} // namespace occ::interaction
