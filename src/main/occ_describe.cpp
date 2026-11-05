#include <algorithm>
#include <filesystem>
#include <ankerl/unordered_dense.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/core/version.h>
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

  desc->add_option("--json", config->json_filename,
                   "Write the RINSE descriptors and hashes to this JSON file, "
                   "one record per structure (each data block of a CIF); a "
                   "structure that fails is a record with an error");
  desc->add_option("--jsonl", config->jsonl_filename,
                   "As --json, streamed as JSON Lines: a header line, then one "
                   "record per line as each structure is described");

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

/// The RINSE descriptor of a crystal, logging its power spectrum when
/// `log_spectrum` is set.
occ::Vec describe_crystal(const Crystal &crystal,
                          const occ::descriptors::Rinse &rinse,
                          bool log_spectrum) {
  const auto &params = rinse.parameters();
  const occ::Mat spectrum = rinse(crystal);
  if (!log_spectrum)
    return occ::descriptors::Rinse::flatten(spectrum);

  occ::log::info("Space group {}, {} atoms in the unit cell\n",
                 crystal.space_group().symbol(),
                 crystal.unit_cell_atoms().size());
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
  return occ::descriptors::Rinse::flatten(spectrum);
}

std::string stem(const std::string &filename) {
  return std::filesystem::path(filename).stem().string();
}

/// Everything about a run of RINSE records but the records: the JSON header,
/// and the top level of the --json document.
nlohmann::json rinse_header(const occ::descriptors::RinseParams &params,
                            int hash_words) {
  const char *basis =
      params.radial_basis == occ::descriptors::RinseRadialBasis::CVGaussian
          ? "cv_gaussian"
          : "lin_gaussian";
  nlohmann::json p = {
      {"n_max", params.n_max},
      {"l_min", params.l_min},
      {"l_max", params.l_max},
      {"l", params.l_values()},
      {"radial_scale", params.radial_scale},
      {"qmax_factor", params.qmax_factor},
      {"radial_basis", basis},
      {"monopole_normalisation", params.monopole_normalisation},
      {"log1p", params.log1p},
      {"normalisation", params.l2 ? "l2" : "none"},
      {"fixed_uiso", params.fixed_uiso ? nlohmann::json(*params.fixed_uiso)
                                       : nlohmann::json(nullptr)},
      {"hash_words", hash_words},
      {"distance", "euclidean"},
  };
  return {
      {"occ_version", std::string(occ::version())},
      {"descriptor", "rinse"},
      {"rinse_version", occ::descriptors::rinse_reference_version},
      {"params", std::move(p)},
      {"shape", {params.n_max, params.num_l_levels()}},
      {"layout", "row-major [radial][angular]"},
  };
}

nlohmann::json rinse_record(const DescribeOutcome &outcome) {
  using Status = DescribeOutcome::Status;
  const bool described = outcome.status == Status::Described;
  nlohmann::json record = {
      {"file", outcome.filename},
      {"block", outcome.block},
      {"index", outcome.index},
      {"values", nullptr},
      {"hash", nullptr},
      {"error", nullptr},
  };
  if (described) {
    record["values"] = std::vector<double>(
        outcome.values.data(), outcome.values.data() + outcome.values.size());
    record["hash"] = outcome.detail;
  } else {
    record["error"] = {{"message", outcome.detail}};
  }
  return record;
}

} // namespace

std::vector<DescribeOutcome> describe_structures(
    DescribeConfig const &config,
    const std::function<void(const DescribeOutcome &)> &on_outcome) {
  using Descriptor = DescribeConfig::Descriptor;
  using Status = DescribeOutcome::Status;

  // Without --descriptor, compute whatever each input allows; with it, only
  // what was asked for. The file decides what can be computed: RINSE needs a
  // lattice, the Steinhardt parameters need neighbours around a point.
  // Records written to a file are RINSE records.
  const auto requested = config.descriptors();
  const auto wanted = [&](Descriptor descriptor) {
    if (config.writes_records())
      return descriptor == Descriptor::Rinse;
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
  const auto record = [&](DescribeOutcome outcome) {
    if (outcome.status == Status::Failed)
      occ::log::error("Could not describe {} [{}]: {}", outcome.filename,
                      outcome.block, outcome.detail);
    if (on_outcome)
      on_outcome(outcome);
    outcomes.push_back(std::move(outcome));
  };

  for (const auto &filename : filenames) {
    const bool crystal = is_crystal_file(filename);
    const Descriptor descriptor =
        crystal ? Descriptor::Rinse : Descriptor::Steinhardt;
    if (!wanted(descriptor)) {
      std::string reason =
          fmt::format("{} input only supports {}",
                      crystal ? "crystal" : "molecular", to_string(descriptor));
      occ::log::warn("Skipping {}: {}", filename, reason);
      record({filename, stem(filename), 0, Status::Skipped, std::move(reason)});
      continue;
    }

    if (!crystal) {
      if (filenames.size() > 1)
        occ::log::info("\n{}", filename);
      try {
        describe_molecule(filename);
        record({filename, stem(filename), 0, Status::Described, {}});
      } catch (const std::exception &e) {
        record({filename, stem(filename), 0, Status::Failed, e.what()});
      }
      continue;
    }

    std::vector<occ::io::CrystalBlock> blocks;
    try {
      blocks = occ::io::load_crystal_blocks(filename);
    } catch (const std::exception &e) {
      record({filename, stem(filename), 0, Status::Failed, e.what()});
      continue;
    }
    // The full power spectrum for a lone structure; with many, a line each.
    const bool lone = filenames.size() == 1 && blocks.size() == 1;
    for (auto &block : blocks) {
      DescribeOutcome outcome{filename, block.name, block.index};
      if (!block.crystal) {
        outcome.status = Status::Failed;
        outcome.detail = block.error;
        record(std::move(outcome));
        continue;
      }
      try {
        outcome.values = describe_crystal(*block.crystal, rinse, lone);
        outcome.detail =
            occ::descriptors::rinse_hash(outcome.values, config.hash_words);
        if (lone)
          occ::log::info("\nRINSE hash: {}", outcome.detail);
        else
          occ::log::info("{}  {}", block.name, outcome.detail);
      } catch (const std::exception &e) {
        outcome.status = Status::Failed;
        outcome.detail = e.what();
      }
      record(std::move(outcome));
    }
  }
  return outcomes;
}

void run_describe_subcommand(DescribeConfig const &config) {
  using Status = DescribeOutcome::Status;

  occ::descriptors::RinseParams params{};
  params.fixed_uiso = config.fixed_uiso;
  const nlohmann::json header = rinse_header(params, config.hash_words);

  std::ofstream jsonl;
  if (!config.jsonl_filename.empty()) {
    jsonl.open(config.jsonl_filename);
    if (!jsonl)
      throw std::runtime_error(fmt::format("Could not open {} for writing",
                                           config.jsonl_filename));
    jsonl << header.dump() << '\n';
  }
  nlohmann::json records = nlohmann::json::array();
  const auto outcomes =
      describe_structures(config, [&](const DescribeOutcome &outcome) {
        if (!config.writes_records())
          return;
        nlohmann::json record = rinse_record(outcome);
        if (jsonl.is_open())
          jsonl << record.dump() << std::endl;
        if (!config.json_filename.empty())
          records.push_back(std::move(record));
      });

  if (!config.json_filename.empty()) {
    nlohmann::json document = header;
    document["records"] = std::move(records);
    std::ofstream out(config.json_filename);
    if (!out)
      throw std::runtime_error(fmt::format("Could not open {} for writing",
                                           config.json_filename));
    out << document.dump() << '\n';
  }

  const auto count = [&outcomes](Status status) {
    return std::count_if(
        outcomes.begin(), outcomes.end(),
        [status](const DescribeOutcome &o) { return o.status == status; });
  };
  const auto described = count(Status::Described);
  const auto skipped = count(Status::Skipped);
  const auto failed = count(Status::Failed);
  if (outcomes.size() > 1) {
    occ::log::info("\n{} structures: {} described, {} skipped, {} failed",
                   outcomes.size(), described, skipped, failed);
    for (const auto &outcome : outcomes)
      if (outcome.status != Status::Described)
        occ::log::info("  {} [{}]: {}", outcome.filename, outcome.block,
                       outcome.detail);
  }
  if (!config.json_filename.empty())
    occ::log::info("RINSE records written to {}", config.json_filename);
  if (!config.jsonl_filename.empty())
    occ::log::info("RINSE records written to {}", config.jsonl_filename);

  if (described == 0)
    throw std::runtime_error(
        failed > 0 ? "No structure could be described"
                   : "None of the requested descriptors apply to the input "
                     "structures");
  if (failed > 0 && !config.writes_records())
    throw std::runtime_error(fmt::format(
        "{} of {} structures could not be described", failed, outcomes.size()));
}

} // namespace occ::main
