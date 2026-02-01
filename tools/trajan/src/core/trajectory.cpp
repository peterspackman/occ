#include "occ/core/bondgraph.h"
#include "occ/core/units.h"
#include <algorithm>
#include <cmath>
#include <mach/message.h>
#include <occ/core/element.h>
#include <occ/crystal/unitcell.h>
#include <optional>
#include <stdexcept>
#include <trajan/core/atom.h>
#include <trajan/core/atomgraph.h>
#include <trajan/core/log.h>
#include <trajan/core/molecule.h>
#include <trajan/core/neigh.h>
#include <trajan/core/topology.h>
#include <trajan/core/trajectory.h>
#include <trajan/io/file_handler.h>
#include <trajan/io/selection.h>
#include <vector>

namespace trajan::core {

using trajan::core::AtomGraph;
using Bond = trajan::core::BondEdge;

Trajectory::~Trajectory() {
  for (auto &handler : m_handlers) {
    if (handler) {
      handler->finalise();
    }
  }
  if (m_output_handler) {
    m_output_handler->finalise();
  }
}

void Trajectory::load_files(const std::vector<fs::path> &files) {
  m_handlers = io::read_input_files(files);
  m_files = files;

  if (m_handlers.empty()) {
    throw std::runtime_error("Could not load files.");
  }

  m_handlers[0]->initialise(io::FileHandler::Mode::Read);
}

void Trajectory::load_files_into_memory(const std::vector<fs::path> &files) {
  this->load_files(files);

  while (m_current_handler_index < m_handlers.size()) {
    if (m_handlers[m_current_handler_index]->initialise(
            io::FileHandler::Mode::Read)) {
      Frame frame(m_frame);
      frame.set_index(m_frames.size());
      if (m_handlers[m_current_handler_index]->read_frame(frame)) {
        m_frames.push_back(frame);
        m_frame = frame;
        continue;
      }
    }
    m_handlers[m_current_handler_index]->finalise();
    m_current_handler_index++;
  }

  m_frames_in_memory = true;

  trajan::log::debug("Loaded {} frames into memory.", m_frames.size());
}

void Trajectory::set_output_file(const fs::path &file) {
  m_output_handler = io::write_output_file(file);
  m_output_handler->initialise(io::FileHandler::Mode::Write);
}

bool Trajectory::_next_frame() {
  if (m_frames_in_memory) {
    if (m_current_frame_index >= m_frames.size()) {
      m_frame_loaded = false;
      return false;
    }
    m_frame = m_frames[m_current_frame_index];
    m_frame_loaded = true;
    m_current_frame_index++;
    return true;
  }

  if (m_handlers.empty()) {
    return false;
  }

  if (m_current_handler_index >= m_handlers.size()) {
    throw std::runtime_error("Incorrect index.");
  }

  while (m_current_handler_index < m_handlers.size()) {
    if (m_handlers[m_current_handler_index]->initialise(
            io::FileHandler::Mode::Read)) {
      if (m_handlers[m_current_handler_index]->read_frame(m_frame)) {
        m_frame_loaded = true;
        m_frame.set_index(m_current_frame_index);
        m_current_frame_index++;
        return true;
      }
    }
    m_handlers[m_current_handler_index]->finalise();
    m_current_handler_index++;
  }
  m_frame_loaded = false;

  return false;
}

bool Trajectory::next_frame() {
  bool next = this->_next_frame();
  if (next) {
    return next;
  }
  this->reset();
  if (m_frames_in_memory) {
    m_frame_loaded = true;
  }
  this->load_files(m_files);
  return next;
}

void Trajectory::write_frame() {
  if (!m_output_handler) {
    // trajan::log::warn("Output file not set. Writing to {}", "PLACEHOLDER");
    // // TODO: set default like extended xyz or something.
    throw std::runtime_error("Output file not set. ");
  }
  if (!m_frame_loaded) {
    throw std::runtime_error("No frame loaded to write.");
  }
  m_output_handler->write_frame(m_frame);
}

void Trajectory::reset() {
  for (auto &handler : m_handlers) {
    if (handler) {
      handler->finalise();
    }
  }

  m_current_handler_index = 0;
  m_current_frame_index = 0;
  m_frame_loaded = false;
}

const std::vector<EntityVariant>
Trajectory::get_entities(const io::SelectionCriteria &selection) {
  const std::vector<Atom> atoms = this->atoms();
  std::vector<Molecule> molecules;
  std::vector<EntityVariant> entities;
  entities.reserve(atoms.size());

  trajan::log::debug("Processing selection {}", selection.index());

  if (std::holds_alternative<io::MoleculeIndexSelection>(selection) ||
      std::holds_alternative<io::MoleculeTypeSelection>(selection)) {
    // build molecules
    molecules = this->get_molecules();
  }

  entities = std::visit(
      [&](const auto &sel) {
        using ActualType = std::decay_t<decltype(sel)>;
        return io::process_selection<ActualType>(sel, atoms, molecules,
                                                 entities);
      },
      selection);

  size_t num_entities = entities.size();
  if (num_entities == 0) {
    // FIXME: Fix the selection names for clarity.
    throw std::runtime_error("No entities found in selection.");
  }
  trajan::log::debug("Identified {} entities in selection", num_entities);

  return entities;
}

const std::vector<EntityVariant>
Trajectory::get_entities(const std::vector<io::SelectionCriteria> &selections) {
  std::vector<EntityVariant> entities;
  for (const io::SelectionCriteria &sel : selections) {
    std::vector<EntityVariant> e2 = this->get_entities(sel);
    entities.reserve(entities.size() + e2.size());
    entities.insert(entities.end(), e2.begin(), e2.end());
  }
  return entities;
}

const std::vector<Atom>
Trajectory::get_atoms(const io::SelectionCriteria &selection) {
  const std::vector<EntityVariant> entities = this->get_entities(selection);
  return core::get_entities_of_type<Atom>(entities);
}

const std::vector<Atom>
Trajectory::get_atoms(const std::vector<io::SelectionCriteria> &selections) {
  const std::vector<EntityVariant> entities = this->get_entities(selections);
  return core::get_entities_of_type<Atom>(entities);
}

const std::vector<Molecule>
Trajectory::get_molecules(const io::SelectionCriteria &selection) {
  const std::vector<EntityVariant> entities = this->get_entities(selection);
  return core::get_entities_of_type<Molecule>(entities);
}

const std::vector<Molecule> Trajectory::get_molecules(
    const std::vector<io::SelectionCriteria> &selections) {
  const std::vector<EntityVariant> entities = this->get_entities(selections);
  return core::get_entities_of_type<Molecule>(entities);
}

const Topology &Trajectory::get_topology(
    const std::optional<TopologyUpdateSettings> &opt_settings) {
  if (m_topology_needs_update ||
      m_topology_update_frequency == m_current_frame_index) {
    this->update_topology(opt_settings);
    m_topology_needs_update = false;
  }
  return m_topology;
}

void Trajectory::update_topology(
    const std::optional<TopologyUpdateSettings> &opt_settings) {
  const std::vector<Atom> &atoms = this->atoms();

  if (atoms.empty()) {
    trajan::log::error("No atoms available for topology update");
    return;
  }

  TopologyUpdateSettings default_settings;
  default_settings.bond_tolerance = occ::core::get_bond_tolerance();

  auto final_settings = opt_settings.value_or(default_settings);

  trajan::log::debug("Creating atom graph");
  AtomGraph atom_graph;
  std::vector<AtomGraph::VertexDescriptor> atom_graph_vertices;
  for (int i = 0; i < atoms.size(); i++) {
    atom_graph.add_vertex(trajan::core::AtomVertex{i});
  }
  trajan::log::debug("Atom graph created successfully");
  double max_cov_radii = occ::core::max_covalent_radius();
  double max_cov_cutoff = max_cov_radii + final_settings.bond_tolerance;
  trajan::log::debug(
      "Creating NeighbourList with cutoff {:.3f} ({:.3f} + {:.3f}) Angstroms",
      max_cov_cutoff, max_cov_radii, final_settings.bond_tolerance);
  NeighbourList nl(max_cov_cutoff);
  trajan::log::debug("NeighbourList created, updating with atoms");

  nl.update(atoms, this->unit_cell());
  trajan::log::debug("NeighbourList updated successfully");

  size_t bond_count = 0;

  NeighbourCallback func = [&](const Entity &ent1, const Entity &ent2,
                               double rsq) {
    const Atom &atom1 = atoms[ent1.index];
    const Atom &atom2 = atoms[ent2.index];
    if (std::binary_search(final_settings.no_bonds.begin(),
                           final_settings.no_bonds.end(), atom1.index) ||
        std::binary_search(final_settings.no_bonds.begin(),
                           final_settings.no_bonds.end(), atom2.index)) {
      trajan::log::debug("Atom pair {} {} skipped due to --no-bond flag",
                         atom1.index, atom2.index); // TODO: elaborate output
      return;
    }
    std::optional<Bond> bond =
        atom1.is_bonded_with_rsq(atom2, rsq, final_settings.bond_tolerance);
    if (!bond.has_value()) {
      return;
    }
    for (const auto &bc : final_settings.bond_cutoffs) {
      double bc_cutoffsq = bc.threshold * bc.threshold;
      if (!(std::binary_search(bc.atom_indices1.begin(), bc.atom_indices1.end(),
                               atom1.index) &&
            std::binary_search(bc.atom_indices2.begin(), bc.atom_indices2.end(),
                               atom2.index))) {
        continue;
      }
      switch (bc.op) {
      case BondCutoff::ComparisonOp::LessThan:
        if (rsq > bc_cutoffsq) {
          trajan::log::debug("Atom pair {} and {} with distance {:.3f} skipped "
                             "due to --bond-critera.",
                             atom1.index, atom2.index, std::sqrt(rsq));
          return;
        }
        break;
      case BondCutoff::ComparisonOp::GreaterThan:
        if (rsq < bc_cutoffsq) {
          trajan::log::debug("Atom pair {} and {} with distance {:.3f} skipped "
                             "due to --bond-critera.",
                             atom1.index, atom2.index, std::sqrt(rsq));
          return;
        }
        break;
      default:
        throw std::runtime_error(
            "Comparison operator needed. Should not have happened");
      }
    }
    Bond bv = bond.value();
    bv.indices = {ent1.index, ent2.index};
    atom_graph.add_edge(ent1.index, ent2.index, bv, true);
    bond_count++;
  };

  trajan::log::debug("Starting neighbour iteration...");
  nl.iterate_neighbours(func);
  trajan::log::debug("Neighbour iteration complete, found {} bonds",
                     bond_count);
  m_topology = Topology(atoms, atom_graph);
}

const std::vector<Molecule> Trajectory::get_molecules() {
  if (m_topology_needs_update) {
    this->update_topology();
  }
  return m_topology.extract_molecules();
}

} // namespace trajan::core
