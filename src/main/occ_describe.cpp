#include <algorithm>
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

  desc->add_option("structures", config->structure_filenames,
                   "input structure files: xyz for molecules, cif or res/ins "
                   "for crystals")
      ->required();

  // Not greedy: with several structures on the command line, a space-separated
  // list would swallow the file names that follow it.
  desc->add_option("--descriptor", config->descriptor_strings,
                   "Descriptors to compute, comma separated (steinhardt, "
                   "rinse)")
      ->delimiter(',')
      ->allow_extra_args(false);

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

bool is_crystal_file(const std::string &filename) {
  return occ::io::CifParser::is_likely_cif_filename(filename) ||
         occ::io::ShelxFile::is_likely_shelx_filename(filename);
}

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

/// Log the RINSE power spectrum of a crystal, and return its hash.
std::string describe_crystal(const std::string &filename,
                             const occ::descriptors::Rinse &rinse,
                             int hash_words) {
  const Crystal crystal = occ::io::load_crystal(filename);
  occ::log::info("Space group {}, {} atoms in the unit cell\n",
                 crystal.space_group().symbol(),
                 crystal.unit_cell_atoms().size());

  const auto &params = rinse.parameters();
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

  const std::string hash = occ::descriptors::rinse_hash(
      occ::descriptors::Rinse::flatten(spectrum), hash_words);
  occ::log::info("\nRINSE hash: {}", hash);
  return hash;
}

} // namespace

std::vector<DescribeOutcome> describe_structures(DescribeConfig const &config) {
  using Descriptor = DescribeConfig::Descriptor;
  using Status = DescribeOutcome::Status;

  // Without --descriptor, compute whatever each input allows; with it, only
  // what was asked for. The file decides what can be computed: RINSE needs a
  // lattice, the Steinhardt parameters need neighbours around a point.
  const auto requested = config.descriptors();
  const auto wanted = [&](Descriptor descriptor) {
    return config.descriptor_strings.empty() ||
           std::find(requested.begin(), requested.end(), descriptor) !=
               requested.end();
  };

  const auto &filenames = config.structure_filenames;
  const bool any_crystal =
      std::any_of(filenames.begin(), filenames.end(), is_crystal_file);
  if (config.fixed_uiso && !any_crystal)
    occ::log::warn("--fixed-uiso only affects RINSE, ignoring it");

  // One instance serves every crystal: constructing it is the part worth
  // sharing.
  occ::descriptors::RinseParams rinse_params{};
  rinse_params.fixed_uiso = config.fixed_uiso;
  const occ::descriptors::Rinse rinse(rinse_params);
  if (config.fixed_uiso && any_crystal)
    occ::log::info("Every atom given U = {:.4f} A^2 in place of its ADPs",
                   *config.fixed_uiso);

  std::vector<DescribeOutcome> outcomes;
  for (const auto &filename : filenames) {
    const bool crystal = is_crystal_file(filename);
    const Descriptor descriptor =
        crystal ? Descriptor::Rinse : Descriptor::Steinhardt;
    if (!wanted(descriptor)) {
      std::string reason =
          fmt::format("{} input only supports {}",
                      crystal ? "crystal" : "molecular", to_string(descriptor));
      occ::log::warn("Skipping {}: {}", filename, reason);
      outcomes.push_back({filename, Status::Skipped, std::move(reason)});
      continue;
    }

    if (filenames.size() > 1)
      occ::log::info("\n{}", filename);
    try {
      if (crystal) {
        outcomes.push_back(
            {filename, Status::Described,
             describe_crystal(filename, rinse, config.hash_words)});
      } else {
        describe_molecule(filename);
        outcomes.push_back({filename, Status::Described, {}});
      }
    } catch (const std::exception &e) {
      occ::log::error("Could not describe {}: {}", filename, e.what());
      outcomes.push_back({filename, Status::Failed, e.what()});
    }
  }
  return outcomes;
}

void run_describe_subcommand(DescribeConfig const &config) {
  using Status = DescribeOutcome::Status;
  const auto outcomes = describe_structures(config);

  if (outcomes.size() > 1) {
    size_t width = 0;
    for (const auto &outcome : outcomes)
      width = std::max(width, outcome.filename.size());
    occ::log::info("\nSummary");
    for (const auto &outcome : outcomes) {
      std::string result;
      switch (outcome.status) {
      case Status::Described:
        result = outcome.detail.empty() ? "described" : outcome.detail;
        break;
      case Status::Skipped:
        result = fmt::format("skipped: {}", outcome.detail);
        break;
      case Status::Failed:
        result = fmt::format("failed: {}", outcome.detail);
        break;
      }
      occ::log::info("{:<{}}  {}", outcome.filename, width, result);
    }
  }

  const auto count = [&outcomes](Status status) {
    return std::count_if(
        outcomes.begin(), outcomes.end(),
        [status](const DescribeOutcome &o) { return o.status == status; });
  };
  if (const auto failed = count(Status::Failed); failed > 0)
    throw std::runtime_error(fmt::format(
        "{} of {} structures could not be described", failed, outcomes.size()));
  if (count(Status::Described) == 0)
    throw std::runtime_error(
        "None of the requested descriptors apply to the input structures");
}

} // namespace occ::main
