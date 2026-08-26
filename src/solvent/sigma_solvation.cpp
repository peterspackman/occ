#include <algorithm>
#include <filesystem>
#include <fmt/core.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/sigma_io.h>
#include <occ/solvent/sigma_solvation.h>
#include <stdexcept>

namespace fs = std::filesystem;

namespace occ::solvent::sigma {

namespace {

constexpr double ANGS2_TO_BOHR2 = occ::units::ANGSTROM_TO_BOHR *
                                  occ::units::ANGSTROM_TO_BOHR;

std::string profile_filename(const std::string &name) {
  return occ::util::to_lower_copy(name) + ".sigma";
}

} // namespace

ProfileStore::ProfileStore(std::vector<std::string> search_paths)
    : m_search_paths(std::move(search_paths)) {}

ProfileStore ProfileStore::standard() {
  return ProfileStore({(fs::path(solvent_data_path()) / "sigma").string(),
                       fs::current_path().string()});
}

bool ProfileStore::contains(const std::string &name) const {
  const auto filename = profile_filename(name);
  return std::any_of(m_search_paths.begin(), m_search_paths.end(),
                     [&](const std::string &directory) {
                       return fs::exists(fs::path(directory) / filename);
                     });
}

Component ProfileStore::get(const std::string &name) const {
  const auto filename = profile_filename(name);
  for (const auto &directory : m_search_paths) {
    const auto path = fs::path(directory) / filename;
    if (fs::exists(path))
      return read_sigma_profile(path.string()).component;
  }
  throw std::runtime_error(fmt::format(
      "no sigma profile for solvent '{}' (looked for {} in: {}). Generate one "
      "with: occ sigma {}.xyz -o {}",
      name, filename, occ::util::join(m_search_paths, ", "),
      occ::util::to_lower_copy(name), filename));
}

std::vector<std::string> ProfileStore::available() const {
  std::vector<std::string> names;
  for (const auto &directory : m_search_paths) {
    if (!fs::is_directory(directory))
      continue;
    for (const auto &entry : fs::directory_iterator(directory)) {
      if (entry.path().extension() != ".sigma")
        continue;
      auto stem = entry.path().stem().string();
      if (std::find(names.begin(), names.end(), stem) == names.end())
        names.push_back(std::move(stem));
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

Component mix_components(const std::vector<Component> &components,
                         const Vec &mole_fractions) {
  if (components.empty())
    throw std::runtime_error("mix_components: no components");
  if (mole_fractions.size() != static_cast<Eigen::Index>(components.size()))
    throw std::runtime_error(
        fmt::format("mix_components: {} fractions for {} components",
                    mole_fractions.size(), components.size()));

  const double sum = mole_fractions.sum();
  if (sum <= 0.0)
    throw std::runtime_error("mix_components: mole fractions sum to zero");
  const Vec x = mole_fractions / sum;

  std::vector<Profile> profiles;
  profiles.reserve(components.size());
  for (const auto &component : components)
    profiles.push_back(component.profile);

  Component mixture;
  mixture.profile = mix_profiles(profiles, x);
  mixture.volume = 0.0;
  for (size_t k = 0; k < components.size(); k++)
    mixture.volume += x(k) * components[k].volume;
  return mixture;
}

SolventModel::SolventModel(Component solvent, Parameters params,
                           PotentialOptions options)
    : m_solvent(std::move(solvent)), m_params(std::move(params)),
      m_options(options) {
  const auto kernel = build_kernel(m_solvent.profile.grid, m_params,
                                   m_options.temperature);
  m_potential = solve_sigma_potential(m_solvent.profile, kernel, m_options);
}

Vec SolventModel::segment_energies(const Segments &solute) const {
  const Mat field = m_potential.mu / m_params.a_eff;
  Vec energies = contract_segments(solute, m_potential.grid, field,
                                   m_params.hbond_split());
  return energies / occ::units::AU_TO_KCAL_PER_MOL;
}

Vec SolventModel::segment_reorganisation(const Segments &solute) const {
  const double rt = m_params.gas_constant * m_options.temperature;
  const Mat field = m_potential.variance / (2.0 * rt * m_params.a_eff);
  Vec lambda = contract_segments(solute, m_potential.grid, field,
                                 m_params.hbond_split());
  return lambda / occ::units::AU_TO_KCAL_PER_MOL;
}

Vec SolventModel::segment_hbond_area(const Segments &solute) const {
  return contract_segments(solute, m_potential.grid,
                           m_potential.hbond_probability,
                           m_params.hbond_split());
}

occ::scrf::SolvationSurface
SolventModel::solvation_surface(const Segments &solute) const {
  occ::scrf::SolvationSurface surface;
  surface.positions = solute.positions;
  surface.areas = solute.areas * ANGS2_TO_BOHR2;
  surface.atom_index = solute.atom_index;
  surface.energies = segment_energies(solute);
  return surface;
}

} // namespace occ::solvent::sigma
