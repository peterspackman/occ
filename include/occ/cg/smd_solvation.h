#pragma once
#include <filesystem>
#include <occ/cg/solvent_surface.h>
#include <occ/core/molecule.h>
#include <occ/dft/dft.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>

namespace occ::cg {

struct SMDSettings {
  std::string method{"b3lyp"};
  std::string basis{"6-31g**"};
  bool pure_spherical{false};
  double convergence_threshold{0.0};
  double temperature{298.0}; // Kelvin
};

class SMDCalculator {
public:
  SMDCalculator(const std::string &basename,
                const std::vector<occ::core::Molecule> &molecules,
                const std::vector<occ::qm::Wavefunction> &wavefunctions,
                const std::string &solvent,
                const SMDSettings &settings = SMDSettings{});

  struct Result {
    std::vector<SMDSolventSurfaces> surfaces;
    std::vector<occ::qm::Wavefunction> wavefunctions;
  };

  Result calculate();

private:
  struct CacheFiles {
    std::filesystem::path surface_path;
    std::filesystem::path wavefunction_path;

    CacheFiles(const std::string &basename, size_t idx,
               const std::string &solvent) {
      surface_path =
          fmt::format("{}_{}_{}_surface.json", basename, idx, solvent);
      wavefunction_path =
          fmt::format("{}_{}_{}.owf.json", basename, idx, solvent);
    }

    bool exists() const {
      return std::filesystem::exists(surface_path) &&
             std::filesystem::exists(wavefunction_path);
    }
  };

  bool try_load_cached(const CacheFiles &cache, SMDSolventSurfaces &,
                       occ::qm::Wavefunction &) const;

  std::pair<SMDSolventSurfaces, occ::qm::Wavefunction>
  perform_calculation(const occ::core::Molecule &mol,
                      const occ::qm::Wavefunction &gas_wfn, size_t index);

  void save_calculation(const CacheFiles &cache,
                        const SMDSolventSurfaces &surface,
                        occ::qm::Wavefunction &wfn) const;

  void calculate_free_energy_components(SMDSolventSurfaces &surface,
                                        const occ::core::Molecule &mol,
                                        double original_energy,
                                        double solvated_energy,
                                        double surface_energy) const;

  std::string m_basename;
  std::string m_solvent;
  SMDSettings m_settings;
  const std::vector<occ::core::Molecule> &m_molecules;
  const std::vector<occ::qm::Wavefunction> &m_gas_wavefunctions;
};

} // namespace occ::cg
