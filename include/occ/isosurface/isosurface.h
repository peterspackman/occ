#pragma once
#include <occ/isosurface/deformation_density.h>
#include <occ/isosurface/eeq_esp.h>
#include <occ/isosurface/electric_potential.h>
#include <occ/isosurface/electron_density.h>
#include <occ/isosurface/orbital_index.h>
#include <occ/isosurface/promolecule_density.h>
#include <occ/isosurface/stockholder_weight.h>
#include <occ/isosurface/surface_types.h>
#include <occ/isosurface/void.h>

namespace occ::isosurface {

using Property = std::variant<FVec, IVec>;
using PropertyMap = ankerl::unordered_dense::map<std::string, Property>;

struct IsosurfaceProperties {
  template <typename T> void add(const std::string &name, const T &values) {
    properties[name] = Property(values);
  }

  bool has_property(const std::string &name) const {
    return properties.contains(name);
  }

  template <typename T> const T *get(const std::string &name) const {
    auto it = properties.find(name);
    if (it != properties.end()) {
      return std::get_if<T>(&it->second);
    }
    return nullptr;
  }

  void merge(const IsosurfaceProperties &other) {
    properties.insert(other.properties.begin(), other.properties.end());
  }

  inline size_t count() const { return properties.size(); }

  PropertyMap properties;
};

struct Isosurface {
  FMat3N vertices;
  IMat3N faces;
  FMat3N normals;
  FVec gaussian_curvature;
  FVec mean_curvature;
  IsosurfaceProperties properties;
};

struct IsosurfaceGenerationParameters {
  double isovalue{0.0};
  double separation{1.0}; // in Bohr by default
  double background_density{0.0};
  OrbitalIndex surface_orbital_index;
  std::vector<OrbitalIndex> property_orbital_indices;
  bool flip_normals{false};
  bool binary_output{true};
  SurfaceKind surface_kind{SurfaceKind::PromoleculeDensity};
  std::vector<PropertyKind> properties;
};

class IsosurfaceCalculator {
public:
  IsosurfaceCalculator() = default;

  void set_molecule(const occ::core::Molecule &mol);
  void set_environment(const occ::core::Molecule &env);
  void set_wavefunction(const occ::qm::Wavefunction &wfn);
  void set_crystal(const occ::crystal::Crystal &crystal);
  void set_parameters(const IsosurfaceGenerationParameters &params);

  bool validate();
  void compute();
  FVec compute_surface_property(PropertyKind) const;

  inline const auto &isosurface() const { return m_isosurface; }

  bool requires_crystal() const;
  bool requires_wavefunction() const;
  bool requires_environment() const;

  inline bool have_crystal() const { return m_crystal != std::nullopt; }
  inline bool have_wavefunction() const {
    return m_wavefunction.atoms.size() > 0;
  }
  inline bool have_environment() const { return m_molecule.size() > 0; }

  inline const std::string &error_message() const { return m_error_message; }

private:
  void compute_default_atom_surface_properties();
  void compute_isosurface();
  Isosurface m_isosurface;

  occ::core::Molecule m_molecule;
  occ::core::Molecule m_environment;
  occ::qm::Wavefunction m_wavefunction;
  std::optional<occ::crystal::Crystal> m_crystal;
  IsosurfaceGenerationParameters m_params;
  std::string m_error_message;
};

} // namespace occ::isosurface
