#pragma once
#include <occ/elastic_fit/potentials.h>

namespace occ::elastic_fit {

enum class LinearSolverType { LU, SVD, QR, LDLT };

class PES {
private:
  crystal::Crystal m_crystal;
  std::vector<std::unique_ptr<PotentialBase>> m_potentials;
  double m_scale_factor;
  double m_shift_factor;
  size_t m_n_molecules = 0;
  double m_temperature = 0.0;

public:
  explicit PES(const crystal::Crystal &crystal) : m_crystal(crystal) {}

  inline void add_potential(std::unique_ptr<PotentialBase> pot) {
    m_potentials.push_back(std::move(pot));
  }

  inline double lattice_energy() const {

    double energy = 0.0;
    for (const auto &pot : m_potentials) {
      energy += pot->energy();
    }
    return energy / 2.0;
  }

  inline size_t number_of_potentials() const { return m_potentials.size(); }

  inline void set_shift(double shift) { m_shift_factor = shift; }

  inline void set_scale(double scale) { m_scale_factor = scale; }

  inline void set_temperature(double temperature) {
    m_temperature = temperature;
  }

  inline const double &get_temperature() const { return m_temperature; }

  inline double shift() const { return m_shift_factor; }

  inline const crystal::Crystal &crystal() const { return m_crystal; }

  inline size_t num_unique_molecules() {
    if (m_n_molecules > 0) {
      return m_n_molecules;
    }
    size_t n_molecules = 0;
    for (const auto &pot : m_potentials) {
      const auto [idx_0, idx_1] = pot->uc_pair_indices();
      if (idx_0 > n_molecules) {
        n_molecules = idx_0;
      }
      if (idx_1 > n_molecules) {
        n_molecules = idx_1;
      }
    }
    n_molecules++;
    m_n_molecules = n_molecules;
    return n_molecules;
  }

  occ::Mat inv_mass_matrix();

  occ::Mat6 compute_elastic_tensor(
      double volume, LinearSolverType solver_type = LinearSolverType::SVD,
      double svd_threshold = 1e-12, bool save_debug_matrices = false);

  occ::CMat compute_fm_at_kpoint(const occ::Vec3 &kp);

  std::pair<occ::Vec, occ::Mat>
  compute_phonons_at_kpoint(const occ::CMat &Dyn_ij);
  void phonons(const occ::IVec3 &shrinking_factors, const occ::Vec3 shift,
               bool animate);

  void animate_phonons(const occ::Vec &frequencies,
                       const occ::Mat &eigenvectors, const occ::Vec3 &kpoint);

  static inline std::pair<int, int> voigt_notation(int voigt) {
    switch (voigt) {
    case 0:
      return {0, 0}; // xx
    case 1:
      return {1, 1}; // yy
    case 2:
      return {2, 2}; // zz
    case 3:
      return {1, 2}; // yz
    case 4:
      return {0, 2}; // xz
    case 5:
      return {0, 1}; // xy
    default:
      throw std::runtime_error("Invalid voigt index.");
    }
  }

  static inline occ::Mat solve_linear_system(const occ::Mat &A,
                                             const occ::Mat &B,
                                             LinearSolverType solver_type,
                                             double svd_threshold = 1e-12) {
    switch (solver_type) {
    case LinearSolverType::LU:
      return A.lu().solve(B);
    case LinearSolverType::SVD: {
      Eigen::JacobiSVD<occ::Mat> svd(A,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
      return svd.solve(B);
    }
    case LinearSolverType::QR:
      return A.householderQr().solve(B);
    case LinearSolverType::LDLT:
      return A.ldlt().solve(B);
    default:
      throw std::runtime_error("Unknown linear solver type");
    }
  }
};

} // namespace occ::elastic_fit
