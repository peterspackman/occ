#include <algorithm>
#include <cctype>
#include <occ/io/grid_settings.h>
#include <stdexcept>

namespace occ::io {

GridSettings GridSettings::from_grid_quality(GridQuality quality) {
  GridSettings settings;
  switch (quality) {
  case GridQuality::Coarse:
    settings.radial_points = 23;
    settings.max_angular_points = 170;
    settings.min_angular_points = 110;
    break;
  case GridQuality::Standard:
    settings.radial_points = 50;
    settings.max_angular_points = 194;
    settings.min_angular_points = 110;
    break;
  case GridQuality::Fine:
    settings.radial_points = 75;
    settings.max_angular_points = 302;
    settings.min_angular_points = 194;
    break;
  case GridQuality::VeryFine:
    settings.radial_points = 99;
    settings.max_angular_points = 590;
    settings.min_angular_points = 302;
    break;
  }
  return settings;
}

GridSettings get_grid_settings(GridQuality quality) {
  return GridSettings::from_grid_quality(quality);
}

std::string grid_quality_to_string(GridQuality quality) {
  switch (quality) {
  case GridQuality::Coarse:
    return "Coarse";
  case GridQuality::Standard:
    return "Standard";
  case GridQuality::Fine:
    return "Fine";
  case GridQuality::VeryFine:
    return "VeryFine";
  default:
    return "Unknown";
  }
}

GridQuality grid_quality_from_string(const std::string &str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "coarse")
    return GridQuality::Coarse;
  if (lower_str == "standard")
    return GridQuality::Standard;
  if (lower_str == "fine")
    return GridQuality::Fine;
  if (lower_str == "veryfine")
    return GridQuality::VeryFine;
  if (lower_str == "very_fine")
    return GridQuality::VeryFine;
  if (lower_str == "very-fine")
    return GridQuality::VeryFine;

  throw std::invalid_argument("Unknown grid quality: " + str);
}

} // namespace occ::io