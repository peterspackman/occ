#include <algorithm>
#include <fmt/core.h>
#include <occ/isosurface/orbital_index.h>
#include <sstream>

namespace occ::isosurface {

// Format for output
std::string OrbitalIndex::format() const {
  switch (reference) {
  case Reference::Absolute:
    return std::to_string(offset); // Convert to 1-based for display
  case Reference::HOMO:
    return offset == 0 ? "HOMO" : fmt::format("HOMO{:+d}", offset);
  case Reference::LUMO:
    return offset == 0 ? "LUMO" : fmt::format("LUMO{:+d}", offset);
  }
  throw std::runtime_error("Invalid orbital reference type");
}

int OrbitalIndex::resolve(int num_alpha, int num_beta) const {
  int homo = num_alpha - 1;
  switch (reference) {
  case Reference::Absolute:
    return offset;
  case Reference::HOMO:
    return homo + offset;
  case Reference::LUMO:
    return (homo + 1) + offset;
  }
  throw std::runtime_error("Invalid orbital reference type");
}

std::vector<OrbitalIndex> parse_orbital_descriptions(const std::string &input) {
  std::vector<OrbitalIndex> indices;

  std::vector<std::string> orbital_specs;
  std::stringstream ss(input);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    if (!token.empty()) {
      orbital_specs.push_back(token);
    }
  }

  for (const auto &spec : orbital_specs) {
    std::string spec_lower = spec;
    std::transform(spec_lower.begin(), spec_lower.end(), spec_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Remove all spaces
    spec_lower.erase(
        std::remove_if(spec_lower.begin(), spec_lower.end(), ::isspace),
        spec_lower.end());

    OrbitalIndex idx;

    // Check if it's a pure number
    if (std::all_of(spec_lower.begin(), spec_lower.end(),
                    [](char c) { return std::isdigit(c) || c == '-'; })) {
      try {
        int num = std::stoi(spec_lower);
        if (num == 0) {
          throw std::runtime_error("Orbital indices must be non-zero");
        }
        idx.reference = OrbitalIndex::Reference::Absolute;
        idx.offset = num - 1; // Convert to 0-based
      } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Invalid orbital index: {}", spec));
      }
    } else {
      bool is_homo = spec_lower.find("homo") == 0;
      bool is_lumo = spec_lower.find("lumo") == 0;

      if (!is_homo && !is_lumo) {
        throw std::runtime_error(
            fmt::format("Invalid orbital specification: {}", spec));
      }

      idx.reference = is_homo ? OrbitalIndex::Reference::HOMO
                              : OrbitalIndex::Reference::LUMO;

      // Just parse everything after homo/lumo as the offset
      std::string offset_str = spec_lower.substr(4);
      try {
        idx.offset = offset_str.empty() ? 0 : std::stoi(offset_str);
      } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Invalid orbital offset in: {}", spec));
      }
    }
    indices.push_back(idx);
  }

  if (indices.empty()) {
    throw std::runtime_error("No valid orbital specifications provided");
  }
  return indices;
}

} // namespace occ::isosurface
