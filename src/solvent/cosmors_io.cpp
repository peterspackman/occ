#include <algorithm>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/os.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/solvent/cosmors_io.h>
#include <occ/solvent/parameters.h>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace occ::solvent::cosmors {

namespace {

constexpr const char *json_extension = ".json";
constexpr const char *text_extension = ".rsseg";

/// Candidate filenames for a solvent, in preference order: occ's own JSON
/// layout first, then the original text one.
std::vector<std::string> segment_filenames(const std::string &name) {
  const auto stem = occ::util::to_lower_copy(name);
  return {stem + json_extension, stem + text_extension};
}

bool is_json_path(const std::string &path) {
  return fs::path(path).extension() == json_extension;
}

/// Everything about the ensemble that is not a per-segment column. Shared by
/// both writers so the two layouts cannot drift apart.
nlohmann::json build_meta(const std::string &name, const Component &component,
                          const Parameters &params, const std::string &method,
                          const std::string &basis) {
  nlohmann::json meta{
      {"name", name},
      {"area [A^2]", component.total_area()},
      {"volume [A^3]", component.volume},
      {"r_av [A]", params.r_av},
      {"r_corr [A]", params.r_corr},
      {"sigma_orth_factor", params.sigma_orth_factor},
      {"segments", component.size()},
      {"generator", "occ"},
  };
  if (!method.empty())
    meta["method"] = method;
  if (!basis.empty())
    meta["basis"] = basis;
  return meta;
}

/// Assemble a ComponentFile from the metadata and the four per-segment
/// columns, whichever layout they were read from.
ComponentFile assemble(const nlohmann::json &meta, std::vector<double> &sigma,
                       std::vector<double> &sigma_orth,
                       std::vector<double> &area,
                       std::vector<int> &atomic_numbers,
                       const std::string &path) {
  if (sigma.empty())
    throw std::runtime_error(
        fmt::format("read_segments: no segments in '{}'", path));
  if (sigma_orth.size() != sigma.size() || area.size() != sigma.size() ||
      atomic_numbers.size() != sigma.size())
    throw std::runtime_error(fmt::format(
        "read_segments: columns have different lengths in '{}'", path));

  const Eigen::Index n = static_cast<Eigen::Index>(sigma.size());
  ComponentFile out;
  out.name = meta.value("name", std::string{});
  out.r_av = meta.value("r_av [A]", 0.0);
  out.r_corr = meta.value("r_corr [A]", 0.0);
  out.sigma_orth_factor = meta.value("sigma_orth_factor", 0.0);
  out.component.sigma = Eigen::Map<Vec>(sigma.data(), n);
  out.component.sigma_orth = Eigen::Map<Vec>(sigma_orth.data(), n);
  out.component.area = Eigen::Map<Vec>(area.data(), n);
  out.component.volume = meta.value("volume [A^3]", 0.0);
  out.component.cavity_area = meta.value("area [A^2]", 0.0);
  out.component.atomic_number = Eigen::Map<IVec>(atomic_numbers.data(), n);
  out.method = meta.value("method", std::string{});
  out.basis = meta.value("basis", std::string{});
  return out;
}

ComponentFile read_json_segments(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error(
        fmt::format("read_segments: cannot open '{}'", path));
  nlohmann::json doc;
  try {
    input >> doc;
  } catch (const nlohmann::json::exception &e) {
    throw std::runtime_error(
        fmt::format("read_segments: '{}' is not valid JSON: {}", path,
                    e.what()));
  }
  for (const char *key : {"sigma", "sigma_orth", "area", "atomic_number"})
    if (!doc.contains(key))
      throw std::runtime_error(fmt::format(
          "read_segments: '{}' has no '{}' column", path, key));

  auto sigma = doc.at("sigma").get<std::vector<double>>();
  auto sigma_orth = doc.at("sigma_orth").get<std::vector<double>>();
  auto area = doc.at("area").get<std::vector<double>>();
  auto atomic_numbers = doc.at("atomic_number").get<std::vector<int>>();
  return assemble(doc, sigma, sigma_orth, area, atomic_numbers, path);
}

ComponentFile read_text_segments(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error(
        fmt::format("read_segments: cannot open '{}'", path));

  nlohmann::json meta;
  std::vector<double> sigma, sigma_orth, area;
  std::vector<int> atomic_numbers;

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty())
      continue;
    if (line.front() == '#') {
      const auto marker = line.find("# meta:");
      if (marker != std::string::npos)
        meta = nlohmann::json::parse(line.substr(marker + 7));
      continue;
    }
    std::istringstream row(line);
    double s = 0.0, so = 0.0, a = 0.0;
    int z = 0;
    if (!(row >> s >> so >> a >> z))
      throw std::runtime_error(
          fmt::format("read_segments: bad row in '{}': {}", path, line));
    sigma.push_back(s);
    sigma_orth.push_back(so);
    area.push_back(a);
    atomic_numbers.push_back(z);
  }
  return assemble(meta, sigma, sigma_orth, area, atomic_numbers, path);
}

} // namespace

ComponentFile read_segments(const std::string &path) {
  return is_json_path(path) ? read_json_segments(path)
                            : read_text_segments(path);
}

void write_segments(const std::string &path, const std::string &name,
                    const Component &component, const Parameters &params,
                    const std::string &method, const std::string &basis) {
  if (component.sigma_orth.size() != component.size())
    throw std::runtime_error(
        "write_segments: sigma_orth has not been computed");
  if (component.atomic_number.size() != component.size())
    throw std::runtime_error(
        "write_segments: one atomic number per segment is required");

  auto meta = build_meta(name, component, params, method, basis);

  if (is_json_path(path)) {
    // Columns rather than a row per segment: they map straight onto the
    // Eigen vectors, and the descriptors carry about eight significant
    // figures of real information, so writing more is noise.
    meta["sigma"] = std::vector<double>(
        component.sigma.data(), component.sigma.data() + component.size());
    meta["sigma_orth"] =
        std::vector<double>(component.sigma_orth.data(),
                            component.sigma_orth.data() + component.size());
    meta["area"] = std::vector<double>(
        component.area.data(), component.area.data() + component.size());
    meta["atomic_number"] =
        std::vector<int>(component.atomic_number.data(),
                         component.atomic_number.data() + component.size());
    std::ofstream output(path);
    if (!output)
      throw std::runtime_error(
          fmt::format("write_segments: cannot open '{}'", path));
    output << meta.dump(2) << '\n';
    return;
  }

  auto output =
      fmt::output_file(path, fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
  output.print("# meta: {}\n", meta.dump());
  output.print("# Rows are: sigma [e/A^2], sigma_orth [e/A^2], area [A^2], "
               "atomic number\n");
  for (Eigen::Index i = 0; i < component.size(); i++)
    output.print("{:.14e} {:.14e} {:.14e} {:d}\n", component.sigma(i),
                 component.sigma_orth(i), component.area(i),
                 component.atomic_number(i));
}

SegmentStore::SegmentStore(std::vector<std::string> search_paths)
    : m_search_paths(std::move(search_paths)) {}

SegmentStore SegmentStore::standard() {
  return SegmentStore({(fs::path(solvent_data_path()) / "cosmors").string(),
                       fs::current_path().string()});
}

bool SegmentStore::contains(const std::string &name) const {
  const auto filenames = segment_filenames(name);
  return std::any_of(m_search_paths.begin(), m_search_paths.end(),
                     [&](const std::string &directory) {
                       return std::any_of(
                           filenames.begin(), filenames.end(),
                           [&](const std::string &filename) {
                             return fs::exists(fs::path(directory) / filename);
                           });
                     });
}

ComponentFile SegmentStore::get(const std::string &name) const {
  const auto filenames = segment_filenames(name);
  // Directory first, then format: a JSON ensemble beside a .rsseg one wins,
  // but a .rsseg earlier in the search path still beats a JSON one later.
  for (const auto &directory : m_search_paths) {
    for (const auto &filename : filenames) {
      const auto path = fs::path(directory) / filename;
      if (fs::exists(path))
        return read_segments(path.string());
    }
  }
  throw std::runtime_error(fmt::format(
      "no openCOSMO-RS segments for solvent '{}' (looked for {} in: {}). "
      "Generate them with: occ cosmo-rs {}.xyz --write-segments {}",
      name, occ::util::join(filenames, " or "),
      occ::util::join(m_search_paths, ", "), occ::util::to_lower_copy(name),
      filenames.front()));
}

std::vector<std::string> SegmentStore::available() const {
  std::vector<std::string> names;
  for (const auto &directory : m_search_paths) {
    if (!fs::is_directory(directory))
      continue;
    for (const auto &entry : fs::directory_iterator(directory)) {
      const auto extension = entry.path().extension();
      if (extension != json_extension && extension != text_extension)
        continue;
      auto stem = entry.path().stem().string();
      if (std::find(names.begin(), names.end(), stem) == names.end())
        names.push_back(std::move(stem));
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

ComponentFile load_solvent(const SegmentStore &store, const std::string &name,
                           const Parameters &params, const std::string &method,
                           const std::string &basis) {
  auto file = store.get(name);
  if (file.r_av > 0.0 && std::abs(file.r_av - params.r_av) > 1e-12)
    throw std::runtime_error(fmt::format(
        "solvent '{}' was averaged on r_av = {} but the parameters in use "
        "specify {}; the descriptors are not comparable",
        name, file.r_av, params.r_av));
  if (!basis.empty() && !file.basis.empty() && file.basis != basis)
    occ::log::warn("solvent '{}' segments were computed with {}/{} but this "
                   "run uses {}/{}; the descriptors are not comparable",
                   name, file.method, file.basis, method, basis);
  return file;
}

} // namespace occ::solvent::cosmors
