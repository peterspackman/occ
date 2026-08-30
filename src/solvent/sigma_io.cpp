#include <fmt/os.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/solvent/sigma_io.h>
#include <sstream>
#include <stdexcept>

namespace occ::solvent::sigma {

ProfileFile read_sigma_profile(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error(
        fmt::format("read_sigma_profile: cannot open '{}'", path));

  nlohmann::json meta;
  std::vector<double> sigma_values, psigma_values;
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
    double s = 0.0, p = 0.0;
    if (!(row >> s >> p))
      throw std::runtime_error(
          fmt::format("read_sigma_profile: bad row in '{}': {}", path, line));
    sigma_values.push_back(s);
    psigma_values.push_back(p);
  }

  if (sigma_values.empty())
    throw std::runtime_error(
        fmt::format("read_sigma_profile: no data rows in '{}'", path));

  // The classes are concatenated, so the grid is however many rows precede
  // the first repeat of the opening sigma value.
  size_t n = sigma_values.size();
  for (size_t i = 1; i < sigma_values.size(); i++) {
    if (std::abs(sigma_values[i] - sigma_values[0]) < 1e-12) {
      n = i;
      break;
    }
  }
  if (n == 0 || sigma_values.size() % n != 0)
    throw std::runtime_error(fmt::format(
        "read_sigma_profile: {} rows is not a whole number of {}-bin profiles",
        sigma_values.size(), n));
  const int num_classes = static_cast<int>(sigma_values.size() / n);

  ProfileFile out;
  out.name = meta.value("name", std::string{});
  out.area = meta.value("area [A^2]", 0.0);
  out.component.volume = meta.value("volume [A^3]", 0.0);
  // Profiles written before the dispersion term, and the published databases,
  // carry neither field; the component then has no dispersion parameter.
  if (meta.contains("dispersion e/kB [K]") && meta.contains("dispersion class")) {
    out.component.dispersion.epsilon =
        meta.at("dispersion e/kB [K]").get<double>();
    out.component.dispersion.klass = dispersion_class_from_name(
        meta.at("dispersion class").get<std::string>());
    out.component.dispersion.known = true;
  }
  out.component.profile.grid =
      Grid{static_cast<int>(n), sigma_values.front(), sigma_values[n - 1]};
  out.component.profile.values = Mat(n, num_classes);
  for (int c = 0; c < num_classes; c++)
    for (size_t i = 0; i < n; i++)
      out.component.profile.values(i, c) = psigma_values[c * n + i];
  return out;
}

void write_sigma_profile(const std::string &path, const std::string &name,
                         const Profile &profile, const Parameters &params,
                         double area, double volume,
                         const Dispersion &dispersion) {
  nlohmann::json meta{
      {"name", name},
      {"area [A^2]", area},
      {"volume [A^3]", volume},
      {"r_av [A]", params.r_av},
      {"f_decay", params.f_decay},
      {"sigma_hb [e/A^2]", params.sigma_hb},
      {"averaging", params.resolves_hbond_classes() ? "Hsieh" : "Mullins"},
      {"generator", "occ"},
  };
  if (dispersion.known) {
    meta["dispersion e/kB [K]"] = dispersion.epsilon;
    meta["dispersion class"] =
        std::string(dispersion_class_name(dispersion.klass));
  }

  auto output =
      fmt::output_file(path, fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
  output.print("# meta: {}\n", meta.dump());
  output.print("# Rows are given as: sigma [e/A^2] followed by a space, then "
               "psigmaA [A^2]\n");
  output.print("# In the case of three sigma profiles, the order is NHB, OH, "
               "then OT\n");

  Vec centers = profile.grid.centers();
  for (int c = 0; c < profile.num_classes(); c++)
    for (int i = 0; i < profile.grid.n; i++)
      output.print("{:.3f} {:.14e}\n", centers(i), profile.values(i, c));
}

} // namespace occ::solvent::sigma
