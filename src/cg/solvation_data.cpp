#include <occ/cg/solvation_data.h>

namespace occ::cg {

double CavitySurface::total_energy() const {
  double total = 0.0;
  for (const auto &field : energies)
    total += field.values.sum();
  return total;
}

double SolvationData::total_energy() const {
  double total = 0.0;
  for (const auto &cavity : cavities)
    total += cavity.total_energy();
  return total;
}

const CavitySurface *SolvationData::find(std::string_view name) const {
  for (const auto &cavity : cavities) {
    if (cavity.name == name)
      return &cavity;
  }
  return nullptr;
}

namespace {
CavitySurface *find_mutable(SolvationData &data, std::string_view name) {
  for (auto &cavity : data.cavities) {
    if (cavity.name == name)
      return &cavity;
  }
  return nullptr;
}
} // namespace

CavitySurface *coulomb_cavity(SolvationData &d) {
  return find_mutable(d, "coulomb");
}
const CavitySurface *coulomb_cavity(const SolvationData &d) {
  return d.find("coulomb");
}
CavitySurface *cds_cavity(SolvationData &d) { return find_mutable(d, "cds"); }
const CavitySurface *cds_cavity(const SolvationData &d) {
  return d.find("cds");
}

CavitySurface &add_cavity(SolvationData &data, const std::string &name,
                          const Mat3N &positions, const Vec &areas,
                          const Vec &energies) {
  CavitySurface cavity;
  cavity.name = name;
  cavity.positions = positions;
  cavity.areas = areas;
  cavity.energies.push_back({name, energies});
  data.cavities.push_back(std::move(cavity));
  return data.cavities.back();
}

SolvationData from_scrf_surfaces(const occ::scrf::SolvationSurfaces &surfaces) {
  SolvationData out;
  if (surfaces.coulomb) {
    const auto &c = *surfaces.coulomb;
    add_cavity(out, "coulomb", c.positions, c.areas, c.energies);
  }
  if (surfaces.cds) {
    const auto &d = *surfaces.cds;
    add_cavity(out, "cds", d.positions, d.areas, d.energies);
  }
  out.total_solvation_energy = surfaces.total_energy();
  return out;
}

} // namespace occ::cg
