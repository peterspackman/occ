#include <ankerl/unordered_dense.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/descriptors/steinhardt.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_describe.h>

using occ::core::Molecule;

namespace occ::main {

std::string to_string(DescribeConfig::Descriptor desc) {
  switch (desc) {
  case DescribeConfig::Descriptor::Steinhardt:
    return "steinhardt";
  default:
    return "unknown descriptor";
  }
}

std::vector<DescribeConfig::Descriptor> DescribeConfig::descriptors() const {
  std::vector<DescribeConfig::Descriptor> desc{
      DescribeConfig::Descriptor::Steinhardt,
  };

  ankerl::unordered_dense::set<DescribeConfig::Descriptor> result;

  ankerl::unordered_dense::map<std::string, DescribeConfig::Descriptor>
      name2desc;

  for (const auto &d : desc) {
    name2desc.insert({to_string(d), d});
  }

  for (const auto &d : descriptor_strings) {
    auto s = occ::util::to_lower_copy(d);
    auto loc = name2desc.find(s);
    if (loc != name2desc.end()) {
      result.insert(loc->second);
    } else {
      occ::log::warn("Unknown descriptor: {}, ignoring", d);
    }
  }

  return std::vector<DescribeConfig::Descriptor>(result.begin(), result.end());
}

CLI::App *add_describe_subcommand(CLI::App &app) {
  CLI::App *desc =
      app.add_subcommand("describe", "compute atomic/molecular descriptors");
  auto config = std::make_shared<DescribeConfig>();

  desc->add_option("geometry", config->geometry_filename,
                   "input geometry file (xyz)")
      ->required();

  desc->add_option("--descriptor", config->descriptor_strings,
                   "Descriptors to compute");

  desc->fallthrough();
  desc->callback([config]() { run_describe_subcommand(*config); });
  return desc;
}

void run_describe_subcommand(DescribeConfig const &config) {

  const auto descriptors_to_compute = config.descriptors();
  if (descriptors_to_compute.size() > 0) {
    occ::log::info("Descriptors to compute:");
    for (const auto &desc : descriptors_to_compute) {
      occ::log::info("{}", to_string(desc));
    }
  }

  Molecule m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);

  occ::log::info("Found {} atoms\n", m1.size());

  occ::log::info("Steinhardt Q parameters");
  occ::descriptors::Steinhardt s(6);
  auto q = s.compute_averaged_q(m1.positions());
  for (int l = 0; l < q.rows(); l++) {
    occ::log::warn("Q({}): {:12.6f}", l, q(l));
  }

  occ::log::info("Steinhardt W parameters");
  auto w = s.compute_averaged_w(m1.positions());
  for (int l = 0; l < w.rows(); l++) {
    occ::log::warn("W({}): {:12.6f}", l, w(l));
  }
}

} // namespace occ::main
