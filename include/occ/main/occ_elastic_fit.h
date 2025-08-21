#pragma once
#include "occ/crystal/crystal.h"
#include <CLI/App.hpp>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace occ::main {

struct EFSettings;

enum class LinearSolverType { LU, SVD, QR, LDLT };

struct Morse {
  double D0;
  double r0;
  double alpha;
  occ::Vec3 r_vector;
  occ::Vec3 r_hat;

  Morse(double D0, double r0, double alpha, const occ::Vec3 &r_vec)
      : D0(D0), r0(r0), alpha(alpha), r_vector(r_vec) {

    double r_norm = r_vector.norm();
    if (std::abs(r0 - r_norm) > 1e-10) {
      throw std::invalid_argument("Input distance vector misaligned.");
    }
    r_hat = r_vector / r0;
  }

  std::string to_string() const {
    return fmt::format("Morse(D0={:.3f}, r0={:.3f}, alpha={:.3f})", D0, r0,
                       alpha);
  }

  double energy(double r) const {
    return D0 *
           (std::exp(-2 * alpha * (r - r0)) - 2 * std::exp(-alpha * (r - r0)));
  }
  double energy() const { return -D0; }

  double first_derivative(double r) const {
    return 2 * D0 * alpha *
           (std::exp(-alpha * (r - r0)) - std::exp(-2 * alpha * (r - r0)));
  }
  double first_derivative() const { return 0.0; }

  double second_derivative(double r) const {
    return 2 * D0 * alpha * alpha *
           (2 * std::exp(-2 * alpha * (r - r0)) - std::exp(-alpha * (r - r0)));
  }
  double second_derivative() const { return 2 * D0 * alpha * alpha; }
};

struct LJ {
  double r0;
  double eps;
  occ::Vec3 r_vector;
  occ::Vec3 r_hat;
  double r012, r06;

  LJ(double eps, double r0, const occ::Vec3 &r_vec)
      : eps(eps), r0(r0), r_vector(r_vec) {

    double r_norm = r_vector.norm();
    if (std::abs(r0 - r_norm) > 1e-10) {
      throw std::invalid_argument("Input distance vector misaligned.");
    }
    r_hat = r_vector / r0;
    r012 = std::pow(r0, 12);
    r06 = std::pow(r0, 6);
  }

  std::string to_string() const {
    return fmt::format("LJ(eps={:.3f}, r0={:.3f})", eps, r0);
  }

  double energy(double r) const {
    return eps * (r012 / std::pow(r, 12) - 2 * r06 / std::pow(r, 6));
  }
  double energy() const { return -eps; }

  double first_derivative(double r) const {
    return eps * (-12 * r012 / std::pow(r, 13) + 12 * r06 / std::pow(r, 7));
  }
  double first_derivative() const { return 0.0; }

  double second_derivative(double r) const {
    return eps * (156 * r012 / std::pow(r, 14) - 84 * r06 / std::pow(r, 8));
  }
  double second_derivative() const { return 72 * eps / std::pow(r0, 2); }
};

struct LJ_A {
  double r0;
  double eps;
  occ::Vec3 r_vector;
  occ::Vec3 r_hat;
  double r012;

  LJ_A(double eps, double r0, const occ::Vec3 &r_vec)
      : eps(eps), r0(r0), r_vector(r_vec) {

    double r_norm = r_vector.norm();
    if (std::abs(r0 - r_norm) > 1e-10) {
      throw std::invalid_argument("Input distance vector misaligned.");
    }
    r_hat = r_vector / r0;
    r012 = std::pow(r0, 12);
  }

  std::string to_string() const {
    return fmt::format("LJ_A(eps={:.3f}, r0={:.3f})", eps, r0);
  }

  double energy(double r) const { return eps * (r012 / std::pow(r, 12)); }
  double energy() const { return -eps; }

  double first_derivative(double r) const {
    return eps * (-12 * r012 / std::pow(r, 13));
  }
  double first_derivative() const { return -12 * eps / r0; }

  double second_derivative(double r) const {
    return eps * (156 * r012 / std::pow(r, 14));
  }
  double second_derivative() const { return 156 * eps / std::pow(r0, 2); }
};

enum class PotentialType { MORSE, LJ, LJ_A };

class PotentialBase {
  using PairIndices = std::pair<int, int>;

public:
  virtual ~PotentialBase() = default;
  virtual double energy(double r) const = 0;
  virtual double energy() const = 0;
  virtual double first_derivative(double r) const = 0;
  virtual double first_derivative() const = 0;
  virtual double second_derivative(double r) const = 0;
  virtual double second_derivative() const = 0;
  virtual std::string to_string() const = 0;

  inline void set_pair_indices(const PairIndices &pair_indices) {
    m_pair_indices = pair_indices;
  }

  inline const PairIndices &pair_indices() const { return m_pair_indices; }

  inline void set_uc_pair_indices(const PairIndices &uc_pair_indices) {
    m_uc_pair_indices = uc_pair_indices;
  }

  inline const PairIndices &uc_pair_indices() const {
    return m_uc_pair_indices;
  }

  inline void set_pair_mass(std::pair<double, double> m) { m_pair_mass = m; }

  inline const std::pair<double, double> &pair_mass() { return m_pair_mass; }

  occ::Vec3 r_vector;
  occ::Vec3 r_hat;
  PairIndices m_pair_indices, m_uc_pair_indices;
  std::pair<double, double> m_pair_mass;
  double r0;
};

class MorseWrapper : public PotentialBase {
private:
  Morse morse;

public:
  MorseWrapper(double D0, double r0, double alpha, const occ::Vec3 &r_vec)
      : morse(D0, r0, alpha, r_vec) {
    this->r_vector = r_vec;
    this->r_hat = morse.r_hat;
    this->r0 = r0;
  }

  double energy(double r) const override { return morse.energy(r); }
  double energy() const override { return morse.energy(); }
  double first_derivative(double r) const override {
    return morse.first_derivative(r);
  }
  double first_derivative() const override { return morse.first_derivative(); }
  double second_derivative(double r) const override {
    return morse.second_derivative(r);
  }
  double second_derivative() const override {
    return morse.second_derivative();
  }
  std::string to_string() const override { return morse.to_string(); }
};

class LJWrapper : public PotentialBase {
private:
  LJ lj;

public:
  LJWrapper(double eps, double r0, const occ::Vec3 &r_vec)
      : lj(eps, r0, r_vec) {
    this->r_vector = r_vec;
    this->r_hat = lj.r_hat;
    this->r0 = r0;
  }

  double energy(double r) const override { return lj.energy(r); }
  double energy() const override { return lj.energy(); }
  double first_derivative(double r) const override {
    return lj.first_derivative(r);
  }
  double first_derivative() const override { return lj.first_derivative(); }
  double second_derivative(double r) const override {
    return lj.second_derivative(r);
  }
  double second_derivative() const override { return lj.second_derivative(); }
  std::string to_string() const override { return lj.to_string(); }
};

class LJ_AWrapper : public PotentialBase {
private:
  LJ_A lj_a;

public:
  LJ_AWrapper(double eps, double r0, const occ::Vec3 &r_vec)
      : lj_a(eps, r0, r_vec) {
    this->r_vector = r_vec;
    this->r_hat = lj_a.r_hat;
    this->r0 = r0;
  }

  double energy(double r) const override { return lj_a.energy(r); }
  double energy() const override { return lj_a.energy(); }
  double first_derivative(double r) const override {
    return lj_a.first_derivative(r);
  }
  double first_derivative() const override { return lj_a.first_derivative(); }
  double second_derivative(double r) const override {
    return lj_a.second_derivative(r);
  }
  double second_derivative() const override { return lj_a.second_derivative(); }
  std::string to_string() const override { return lj_a.to_string(); }
};

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

  occ::Mat6
  compute_elastic_tensor(double volume,
                         LinearSolverType solver_type = LinearSolverType::SVD,
                         double svd_threshold = 1e-12);

  occ::CMat compute_fm_at_kpoint(const occ::Vec3 &kp);

  std::pair<occ::Vec, occ::Mat>
  compute_phonons_at_kpoint(const occ::CMat &Dyn_ij);
  void phonon_density_of_states(const occ::IVec3 &shrinking_factors,
                                const occ::Vec3 shift, bool animate);

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

struct MonkhorstPack {
  size_t shrink_x, shrink_y, shrink_z;
  occ::Vec3 shift;
  occ::Mat3N grid;

  inline void generate_grid() {
    size_t n_kpoints = shrink_x * shrink_y * shrink_z;
    grid = occ::Mat3N(3, n_kpoints);
    size_t i_point = 0;
    double x_shift =
        shift[0] ? shift[0] : 1.0 / static_cast<double>(shrink_x) / 2;
    double y_shift =
        shift[1] ? shift[1] : 1.0 / static_cast<double>(shrink_y) / 2;
    double z_shift =
        shift[2] ? shift[2] : 1.0 / static_cast<double>(shrink_z) / 2;
    for (size_t x = 0; x < shrink_x; x++) {
      double x_point = static_cast<double>(x) / static_cast<double>(shrink_x);
      for (size_t y = 0; y < shrink_y; y++) {
        double y_point = static_cast<double>(y) / static_cast<double>(shrink_y);
        for (size_t z = 0; z < shrink_z; z++) {
          double z_point =
              static_cast<double>(z) / static_cast<double>(shrink_z);
          occ::Vec3 point = occ::Vec3(x_point + x_shift, y_point + y_shift,
                                      z_point + z_shift);
          grid.col(i_point) = point;
          i_point++;
        }
      }
    }
  }

  MonkhorstPack(const occ::IVec3 &shrinking_factors, const occ::Vec3 &shift)
      : shrink_x(shrinking_factors[0]), shrink_y(shrinking_factors[1]),
        shrink_z(shrinking_factors[2]), shift(shift) {
    this->generate_grid();
  }

  inline const size_t size() const { return grid.cols(); }

  inline auto begin() const { return grid.colwise().begin(); }
  inline auto end() const { return grid.colwise().end(); }
};

struct EFSettings {
  std::string json_filename;
  std::string output_file = "elastic_tensor.txt";
  std::string potential_type_str = "lj";
  PotentialType potential_type = PotentialType::LJ;
  bool include_positive = false;
  bool max_to_zero = false;
  double scale_factor = 2.0;
  double temperature = 0.0;
  double gulp_scale = 0.01;
  std::string gulp_file{""};
  LinearSolverType solver_type = LinearSolverType::SVD;
  std::string solver_type_str = "svd";
  double svd_threshold = 1e-12;
  bool animate_phonons = false;
  std::vector<size_t> shrinking_factors_raw{1};
  occ::IVec3 shrinking_factors{1, 1, 1};
  std::vector<double> shift_raw{0.0};
  occ::Vec3 shift{0.0, 0.0, 0.0};
};

CLI::App *add_elastic_fit_subcommand(CLI::App &app);
void run_elastic_fit_subcommand(const EFSettings &settings);

} // namespace occ::main
