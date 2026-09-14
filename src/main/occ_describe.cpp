#include <ankerl/unordered_dense.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/crystal/crystal.h>
#include <occ/descriptors/rinse.h>
#include <occ/descriptors/steinhardt.h>
#include <occ/io/cifparser.h>
#include <occ/io/load_geometry.h>
#include <occ/io/shelxfile.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_describe.h>

using occ::core::Molecule;
using occ::crystal::Crystal;

namespace occ::main {

std::string to_string(DescribeConfig::Descriptor desc) {
  switch (desc) {
  case DescribeConfig::Descriptor::Steinhardt:
    return "steinhardt";
  case DescribeConfig::Descriptor::Rinse:
    return "rinse";
  default:
    return "unknown descriptor";
  }
}

std::vector<DescribeConfig::Descriptor> DescribeConfig::descriptors() const {
  std::vector<DescribeConfig::Descriptor> desc{
      DescribeConfig::Descriptor::Steinhardt,
      DescribeConfig::Descriptor::Rinse,
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
                   "input structure file (xyz for molecules, cif or "
                   "res/ins for crystals)")
      ->required();

  desc->add_option("--descriptor", config->descriptor_strings,
                   "Descriptors to compute (steinhardt, rinse)");

  desc->add_option("--hash-words", config->hash_words,
                   "Words in the RINSE hash; each carries 16 bits");

  desc->add_option_function<double>(
          "--fixed-uiso", [config](const double &u) { config->fixed_uiso = u; },
          "Give every atom this isotropic U (A^2) in place of its ADPs for "
          "RINSE, so structures with and without ADPs can be compared")
      ->check(CLI::NonNegativeNumber);

  desc->fallthrough();
  desc->callback([config]() { run_describe_subcommand(*config); });
  return desc;
}

namespace {

void describe_molecule(const std::string &filename) {
  Molecule molecule = occ::io::molecule_from_xyz_file(filename);
  occ::log::info("Found {} atoms\n", molecule.size());

  occ::descriptors::Steinhardt steinhardt(6);

  occ::log::info("Steinhardt Q parameters");
  const auto q = steinhardt.compute_averaged_q(molecule.positions());
  for (int l = 0; l < q.rows(); l++)
    occ::log::info("Q({}): {:12.6f}", l, q(l));

  occ::log::info("Steinhardt W parameters");
  const auto w = steinhardt.compute_averaged_w(molecule.positions());
  for (int l = 0; l < w.rows(); l++)
    occ::log::info("W({}): {:12.6f}", l, w(l));
}

void describe_crystal(const std::string &filename, int hash_words,
                      std::optional<double> fixed_uiso) {
  const Crystal crystal = occ::io::load_crystal(filename);
  occ::log::info("Space group {}, {} atoms in the unit cell\n",
                 crystal.space_group().symbol(),
                 crystal.unit_cell_atoms().size());

  occ::descriptors::RinseParams rinse_params{};
  rinse_params.fixed_uiso = fixed_uiso;
  const occ::descriptors::Rinse rinse(rinse_params);
  const auto &params = rinse.parameters();
  if (params.fixed_uiso)
    occ::log::info("Every atom given U = {:.4f} A^2 in place of its ADPs",
                   *params.fixed_uiso);
  const occ::Mat spectrum = rinse(crystal);

  occ::log::info("RINSE power spectrum, sin(theta)/lambda <= {:.3f} A^-1",
                 params.sin_theta_over_lambda_max());
  std::string header = fmt::format("{:>4}", "n");
  for (const int l : params.l_values())
    header += fmt::format(" {:>10}", fmt::format("l={}", l));
  occ::log::info("{}", header);
  for (int n = 0; n < spectrum.rows(); n++) {
    std::string row = fmt::format("{:>4}", n);
    for (int k = 0; k < spectrum.cols(); k++)
      row += fmt::format(" {:10.6f}", spectrum(n, k));
    occ::log::info("{}", row);
  }
  occ::log::info("\nRINSE hash: {}",
                 occ::descriptors::rinse_hash(
                     occ::descriptors::Rinse::flatten(spectrum), hash_words));
}

} // namespace

void run_describe_subcommand(DescribeConfig const &config) {
  // The file decides what can be computed: RINSE needs a lattice, the
  // Steinhardt parameters need neighbours around a point.
  const bool is_crystal =
      occ::io::CifParser::is_likely_cif_filename(config.geometry_filename) ||
      occ::io::ShelxFile::is_likely_shelx_filename(config.geometry_filename);
  const auto available = is_crystal ? DescribeConfig::Descriptor::Rinse
                                    : DescribeConfig::Descriptor::Steinhardt;

  // Without --descriptor, compute whatever the input allows; with it, only
  // what was asked for.
  bool wanted = config.descriptor_strings.empty();
  for (const auto &desc : config.descriptors()) {
    if (desc == available)
      wanted = true;
    else
      occ::log::warn("Skipping {}: it needs {} input", to_string(desc),
                     is_crystal ? "molecular" : "crystal");
  }
  if (!wanted)
    throw std::runtime_error(
        fmt::format("None of the requested descriptors can be computed for {}",
                    config.geometry_filename));

  if (is_crystal) {
    describe_crystal(config.geometry_filename, config.hash_words,
                     config.fixed_uiso);
  } else {
    if (config.fixed_uiso)
      occ::log::warn("--fixed-uiso only affects RINSE, ignoring it");
    describe_molecule(config.geometry_filename);
  }
}

} // namespace occ::main
