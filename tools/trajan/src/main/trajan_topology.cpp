#include <CLI/CLI.hpp>
#include <occ/core/bondgraph.h>
#include <trajan/core/neigh.h>
#include <trajan/core/trajectory.h>
#include <trajan/io/selection.h>
#include <trajan/main/trajan_topology.h>

namespace trajan::main {

using trajan::core::Atom;

auto bond_criteria_validator(std::vector<BondCriteria> &bond_criterias,
                             std::optional<std::vector<char>> restrictions) {
  auto f = [&](const std::string &input) {
    // parse format: "sel1,sel2</>num"
    size_t op_pos = input.find_first_of("<>");
    if (op_pos == std::string::npos) {
      return std::string("Missing comparison operator (< or >)");
    }
    const char op_char = input[op_pos];
    std::string left = input.substr(0, op_pos);
    std::string right = input.substr(op_pos + 1);

    size_t and_pos = left.find('&');
    if (and_pos == std::string::npos) {
      return std::string("Missing comma separator between selections");
    }
    std::string sel1_str = left.substr(0, and_pos);
    std::string sel2_str = left.substr(and_pos + 1);

    double threshold;
    try {
      threshold = std::stod(right);
    } catch (const std::exception &e) {
      return std::string(
          fmt::format("Invalid threshold value: {} ({})", right, e.what()));
    }

    BondCriteria bc;

    std::string error1 =
        io::selection_validator(bc.sel1, restrictions)(sel1_str);
    if (!error1.empty()) {
      return error1;
    }
    std::string error2 =
        io::selection_validator(bc.sel2, restrictions)(sel2_str);
    if (!error2.empty()) {
      return error2;
    }
    bc.threshold = threshold;
    bc.op = (op_char == '<') ? BondCriteria::ComparisonOp::LessThan
                             : BondCriteria::ComparisonOp::GreaterThan;
    bond_criterias.push_back(bc);

    return std::string();
  };

  return f;
}

void run_topology_subcommand(const TopologyOpts &opts, Trajectory &traj) {
  core::TopologyUpdateSettings settings;
  for (const auto &bond_criteria : opts.bond_criterias) {
    core::BondCutoff bond_cutoff;
    auto atoms1 = traj.get_atoms(bond_criteria.sel1);
    bond_cutoff.atom_indices1.reserve(atoms1.size());
    for (const auto &a : atoms1) {
      bond_cutoff.atom_indices1.push_back(a.index);
    }
    std::sort(bond_cutoff.atom_indices1.begin(),
              bond_cutoff.atom_indices1.end());
    auto atoms2 = traj.get_atoms(bond_criteria.sel2);
    bond_cutoff.atom_indices2.reserve(atoms2.size());
    for (const auto &a : atoms2) {
      bond_cutoff.atom_indices2.push_back(a.index);
    }
    std::sort(bond_cutoff.atom_indices2.begin(),
              bond_cutoff.atom_indices2.end());
    bond_cutoff.op = (bond_criteria.op == BondCriteria::ComparisonOp::LessThan)
                         ? core::BondCutoff::ComparisonOp::LessThan
                         : core::BondCutoff::ComparisonOp::GreaterThan;
    bond_cutoff.threshold = bond_criteria.threshold;
    settings.bond_cutoffs.push_back(bond_cutoff);
  }
  auto nbatoms = traj.get_atoms(opts.nb_parsed_sel);
  settings.no_bonds.reserve(nbatoms.size());
  for (const auto &a : nbatoms) {
    settings.no_bonds.push_back(a.index);
  }
  std::sort(settings.no_bonds.begin(), settings.no_bonds.end());
  settings.top_auto = opts.top_auto;
  settings.bond_tolerance = opts.bond_tolerance;
  occ::core::set_bond_tolerance(opts.bond_tolerance);
  traj.update_topology(settings);
  traj.set_topology_update_frequency(opts.update_frequency);
}

CLI::App *add_topology_subcommand(CLI::App &app, Trajectory &traj) {
  CLI::App *top =
      app.add_subcommand("top,topology", "Compute the molecular topology.");
  auto opts = std::make_shared<TopologyOpts>();
  top->add_flag("-a,--auto", opts->top_auto,
                "Generate topology automatically from atom covalent radii.");
  top->add_option("-nb,--no-bond", opts->nb_raw_sel,
                  "Selected atoms will not be allowed to bond to other atoms\n"
                  "First selection (prefix: i=atom indices, a=atom types)\n"
                  "Examples:\n"
                  "  i1,2,3-5    (atom indices 1,2,3,4,5)\n"
                  "  aC,N,O      (atom types C, N, O)\n"
                  "  j/m not allowed\n")
      ->check(io::selection_validator(
          opts->nb_parsed_sel,
          std::make_optional<std::vector<char>>({'j', 'm'})));
  top->add_option(
         "-bc,--bond-criteria",
         "Selected pairs of atoms will only be allowed to form a bond if the "
         "distance criteria is less than or greater than the input "
         "threshold.\n Selection one and two should be separated by '&'.\n "
         "Examples:\n  i1,2,3-5&aC,N,O<1.4 (bonds only formed between atom "
         "indices 1,2,3,4,5 and atom types C,N,O when separated by less than "
         "1.4 Angstroms.)\n  i1,2&aN,H>2.5\n")
      ->type_size(0, -1)
      ->each(bond_criteria_validator(
          opts->bond_criterias,
          std::make_optional<std::vector<char>>({'j', 'm'})));
  opts->bond_tolerance = occ::core::get_bond_tolerance();
  top->add_option("-bt,--bond-tolerance", opts->bond_tolerance,
                  "Bond tolerance in Angstroms to use when deciding if two "
                  "atoms are bonded. This number is added to the sum of the "
                  "covalent radii.\n");
  top->add_option(
      "-uf,--update-frequency", opts->update_frequency,
      "How often to recompute the topology. Default is set to 0 which means to "
      "only compute the topology once at the start of the trajectory analysis. "
      "This means the topology is assumed to unchage throughout. Setting this "
      "to anything other than 0 will slow down the trajectory analysis, but it "
      "can be useful when analysing non-classical or reactive-classical "
      "simulations.\n");
  top->callback([opts, &traj]() { run_topology_subcommand(*opts, traj); });
  return top;
}

} // namespace trajan::main
