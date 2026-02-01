#pragma once
#include <filesystem>
#include <occ/crystal/unitcell.h>
#include <trajan/core/frame.h>
#include <trajan/core/neigh.h>
#include <trajan/core/topology.h>
#include <trajan/io/file_handler.h>
#include <trajan/io/selection.h>
#include <vector>

namespace trajan::core {

namespace fs = std::filesystem;

using occ::crystal::UnitCell;
using Atom = trajan::core::EnhancedAtom;
using Molecule = trajan::core::EnhancedMolecule;

struct BondCutoff {
  std::vector<int> atom_indices1;
  std::vector<int> atom_indices2;
  enum class ComparisonOp { LessThan, GreaterThan } op;
  double threshold;
};

struct TopologyUpdateSettings {
  bool top_auto{true};
  int update_frequency{0};
  std::vector<BondCutoff> bond_cutoffs{};
  std::vector<int> no_bonds{};
  double bond_tolerance;
};

class Trajectory {

public:
  Trajectory(const Trajectory &) = delete;
  Trajectory &operator=(const Trajectory &) = delete;

  Trajectory(Trajectory &&) = default;
  Trajectory &operator=(Trajectory &&) = default;

  Trajectory() = default;
  ~Trajectory();

  void load_files(const std::vector<fs::path> &files);
  void load_files_into_memory(const std::vector<fs::path> &files);
  void set_output_file(const fs::path &file);

  bool next_frame();
  void write_frame();

  void reset();

  inline bool has_frames() const {
    return !m_handlers.empty() && m_current_handler_index < m_handlers.size();
  }

  inline size_t current_frame_index() const { return m_current_frame_index; }

  inline Frame &frame() { return m_frame; }

  inline size_t num_atoms() const { return m_frame.num_atoms(); }

  inline const std::optional<UnitCell> &unit_cell() const {
    return m_frame.unit_cell();
  }

  const std::vector<EntityVariant>
  get_entities(const io::SelectionCriteria &selection);
  const std::vector<EntityVariant>
  get_entities(const std::vector<io::SelectionCriteria> &selections);
  inline const std::vector<Atom> &atoms() const { return m_frame.atoms(); }
  inline const std::vector<Atom> &get_atoms() const { return m_frame.atoms(); }
  const std::vector<Atom> get_atoms(const io::SelectionCriteria &selection);
  const std::vector<Atom>
  get_atoms(const std::vector<io::SelectionCriteria> &selections);
  const std::vector<Molecule> get_molecules();
  const std::vector<Molecule>
  get_molecules(const io::SelectionCriteria &selection);
  const std::vector<Molecule>
  get_molecules(const std::vector<io::SelectionCriteria> &selections);

  const Topology &get_topology(
      const std::optional<TopologyUpdateSettings> &settings = std::nullopt);
  void update_topology(
      const std::optional<TopologyUpdateSettings> &settings = std::nullopt);

  inline void set_topology_update_frequency(int freq) {
    m_topology_update_frequency = 0;
  }

private:
  bool _next_frame();
  bool m_guess_connectivity{true};

  std::vector<fs::path> m_files{};
  size_t m_current_handler_index{0};
  size_t m_current_frame_index{0};
  bool m_frame_loaded{false};
  std::vector<io::FileHandlerPtr> m_handlers;
  io::FileHandlerPtr m_output_handler;
  Frame m_frame;
  std::vector<Frame> m_frames;
  bool m_frames_in_memory{false};
  int m_topology_update_frequency{0};

  mutable Topology m_topology;
  mutable std::vector<Molecule> m_molecules{};
  mutable bool m_topology_needs_update{true};
};

}; // namespace trajan::core
