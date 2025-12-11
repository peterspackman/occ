#pragma once
#include <occ/core/dimer.h>
#include <occ/elastic_fit/pes.h>
#include <occ/interaction/interaction_json.h>


namespace occ::elastic_fit {

struct FittingSettings {
  PotentialType potential_type = PotentialType::LJ;
  bool include_positive = false;
  bool max_to_zero = false;
  double scale_factor = 2.0;
  double temperature = 0.0;
  double gulp_scale = 0.01;
  LinearSolverType solver_type = LinearSolverType::SVD;
  double svd_threshold = 1e-12;
  bool animate_phonons = false;
  bool save_debug_matrices = false;
  occ::IVec3 shrinking_factors{1, 1, 1};
  occ::Vec3 shift{0.0, 0.0, 0.0};
  std::string gulp_file; // Generate GULP output if not empty
};

struct FittingResults {
  double lattice_energy = 0.0;
  occ::Mat6 elastic_tensor;
  double energy_shift_applied = 0.0;
  size_t discarded_positive_pairs = 0;
  double discarded_total_energy = 0.0;
  size_t total_potentials_created = 0;
  std::vector<std::string> gulp_strings;
};

class ElasticFitter {
public:
  explicit ElasticFitter(const FittingSettings &settings);

  // Main fitting function
  FittingResults
  fit_elastic_tensor(const occ::interaction::ElatResults &elat_data);

  // Generate GULP input strings
  std::vector<std::string>
  generate_gulp_input(const occ::interaction::ElatResults &elat_data,
                      const FittingResults &results) const;

  // Utility methods for output
  static void print_elastic_tensor(
      const occ::Mat6 &tensor,
      const std::string &title = "Elastic constant matrix: (Units=GPa)");
  static void save_elastic_tensor(const occ::Mat6 &tensor,
                                  const std::string &filename);

private:
  FittingSettings m_settings;

  // Core fitting methods
  PES build_pes_from_elat_data(const occ::interaction::ElatResults &elat_data);
  std::unique_ptr<PotentialBase>
  create_potential_from_dimer(const occ::core::Dimer &dimer,
                              double adjusted_energy) const;

  // GULP generation methods (standalone)
  std::vector<std::string> generate_gulp_crystal_strings(
      const occ::interaction::ElatResults &elat_data) const;
  std::vector<std::string> generate_gulp_potential_strings() const;

  // Determine energy shift strategy
  double
  calculate_energy_shift(const occ::interaction::ElatResults &elat_data) const;
};

} // namespace occ::elastic_fit