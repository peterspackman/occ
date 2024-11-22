#include <nlohmann/json.hpp>
#include <occ/core/combinations.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/io/core_json.h>
#include <occ/io/eigen_json.h>
#include <occ/io/kmcpp.h>

namespace occ::io::kmcpp {

InputWriter::InputWriter(const std::string &filename)
    : m_owned_destination(filename), m_dest(m_owned_destination) {}

InputWriter::InputWriter(std::ostream &stream) : m_dest(stream) {}

void InputWriter::write(const occ::crystal::Crystal &crystal,
                        const occ::crystal::CrystalDimers &uc_dimers,
                        const std::vector<double> &solution_term) {

  nlohmann::json j;
  j["title"] = "title";
  const auto &uc_molecules = crystal.unit_cell_molecules();
  j["unique_sites"] = uc_molecules.size();
  const auto &neighbors = uc_dimers.molecule_neighbors;
  nlohmann::json repr;
  repr["kind"] = "atoms";
  j["lattice_vectors"] = crystal.unit_cell().direct();
  repr["elements"] = {};
  repr["positions"] = {};
  j["solution term"] = solution_term;
  j["neighbor_offsets"] = {};
  size_t uc_idx_a = 0;
  j["neighbor_energies"] = {};
  for (const auto &mol : uc_molecules) {
    nlohmann::json molj = mol;
    repr["elements"].push_back(molj["elements"]);
    repr["positions"].push_back(molj["positions"]);
    j["neighbor_energies"].push_back(nlohmann::json::array({}));
    std::vector<std::vector<int>> shifts;
    for (const auto &[n, unique_index] : neighbors[uc_idx_a]) {
      const auto uc_shift = n.b().cell_shift();
      const auto uc_idx_b = n.b().unit_cell_molecule_idx();
      shifts.push_back({uc_shift[0], uc_shift[1], uc_shift[2], uc_idx_b});
    }
    j["neighbor_offsets"][uc_idx_a] = shifts;
    const auto &neighbors_a = neighbors[uc_idx_a];
    occ::log::debug("Generating all combinations for {} neighbors",
                    neighbors_a.size());
    for (const auto &[n, unique_index] : neighbors_a) {
      j["neighbor_energies"][uc_idx_a].push_back(n.interaction_energy());
    }
    uc_idx_a++;
  }
  j["representation"] = repr;

  m_dest << j.dump(2);
}

} // namespace occ::io::kmcpp
