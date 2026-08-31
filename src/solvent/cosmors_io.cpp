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

std::string segments_filename(const std::string &name) {
  return occ::util::to_lower_copy(name) + ".rsseg";
}

} // namespace

ComponentFile read_segments(const std::string &path) {
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

  if (sigma.empty())
    throw std::runtime_error(
        fmt::format("read_segments: no data rows in '{}'", path));

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

void write_segments(const std::string &path, const std::string &name,
                    const Component &component, const Parameters &params,
                    const std::string &method, const std::string &basis) {
  if (component.sigma_orth.size() != component.size())
    throw std::runtime_error(
        "write_segments: sigma_orth has not been computed");
  if (component.atomic_number.size() != component.size())
    throw std::runtime_error(
        "write_segments: one atomic number per segment is required");

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
  const auto filename = segments_filename(name);
  return std::any_of(m_search_paths.begin(), m_search_paths.end(),
                     [&](const std::string &directory) {
                       return fs::exists(fs::path(directory) / filename);
                     });
}

ComponentFile SegmentStore::get(const std::string &name) const {
  const auto filename = segments_filename(name);
  for (const auto &directory : m_search_paths) {
    const auto path = fs::path(directory) / filename;
    if (fs::exists(path))
      return read_segments(path.string());
  }
  throw std::runtime_error(fmt::format(
      "no openCOSMO-RS segments for solvent '{}' (looked for {} in: {}). "
      "Generate them with: occ sigma {}.xyz --write-segments {}",
      name, filename, occ::util::join(m_search_paths, ", "),
      occ::util::to_lower_copy(name), filename));
}

std::vector<std::string> SegmentStore::available() const {
  std::vector<std::string> names;
  for (const auto &directory : m_search_paths) {
    if (!fs::is_directory(directory))
      continue;
    for (const auto &entry : fs::directory_iterator(directory)) {
      if (entry.path().extension() != ".rsseg")
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
