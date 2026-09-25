#pragma once
#include <fmt/format.h>
#include <occ/cg/solvation_data.h>
#include <occ/core/molecule.h>
#include <occ/dft/dft.h>
#include <occ/io/json_cache.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>

namespace occ::cg {

struct SMDSettings {
  std::string method{"b3lyp"};
  std::string basis{"6-31g**"};
  bool pure_spherical{false};
  double temperature{298.0}; // Kelvin
};

class SMDCalculator {
public:
  SMDCalculator(const std::string &basename,
                const std::vector<occ::core::Molecule> &molecules,
                const std::vector<occ::qm::Wavefunction> &wavefunctions,
                const std::string &solvent, occ::io::JsonCache &cache,
                const SMDSettings &settings = SMDSettings{});

  struct Result {
    std::vector<SolvationData> surfaces;
    std::vector<occ::qm::Wavefunction> wavefunctions;
  };

  Result calculate();

private:
  struct CacheKeys {
    std::string surface;
    std::string wavefunction;

    CacheKeys(const std::string &basename, size_t idx,
              const std::string &solvent)
        : surface(fmt::format("{}_{}_{}_surface.json", basename, idx, solvent)),
          wavefunction(
              fmt::format("{}_{}_{}.owf.json", basename, idx, solvent)) {}
  };

  bool try_load_cached(const CacheKeys &keys, SolvationData &,
                       occ::qm::Wavefunction &) const;

  std::pair<SolvationData, occ::qm::Wavefunction>
  perform_calculation(const occ::core::Molecule &mol,
                      const occ::qm::Wavefunction &gas_wfn, size_t index);

  void save_calculation(const CacheKeys &keys, const SolvationData &surface,
                        occ::qm::Wavefunction &wfn) const;

  void calculate_free_energy_components(SolvationData &surface,
                                        const occ::core::Molecule &mol,
                                        double original_energy,
                                        double solvated_energy,
                                        double surface_energy) const;

  std::string m_basename;
  std::string m_solvent;
  SMDSettings m_settings;
  const std::vector<occ::core::Molecule> &m_molecules;
  const std::vector<occ::qm::Wavefunction> &m_gas_wavefunctions;
  occ::io::JsonCache &m_cache;
};

} // namespace occ::cg
