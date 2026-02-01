#pragma once
#include <CLI/CLI.hpp>
#include <trajan/core/trajectory.h>
#include <trajan/core/util.h>
#include <trajan/io/selection.h>

namespace trajan::main {

using trajan::core::Trajectory;

struct BondCriteria {
  std::vector<io::SelectionCriteria> sel1;
  std::vector<io::SelectionCriteria> sel2;
  enum class ComparisonOp { LessThan, GreaterThan } op;
  double threshold;
};

struct TopologyOpts {
  bool top_auto{true};
  int update_frequency{0};
  std::string nb_raw_sel;
  std::vector<io::SelectionCriteria> nb_parsed_sel;
  std::vector<BondCriteria> bond_criterias;
  double bond_tolerance;
};

auto bond_criteria_validator(std::vector<BondCriteria> &bond_criteria,
                             std::optional<std::vector<char>> restrictions);

void run_topology_subcommand(const TopologyOpts &opts, Trajectory &traj);
CLI::App *add_topology_subcommand(CLI::App &app, Trajectory &traj);

} // namespace trajan::main
