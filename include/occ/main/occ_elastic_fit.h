#pragma once
#include <CLI/App.hpp>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace occ::main {

using occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

inline void print_cvector(const occ::CVec &vec, int per_line) {
  std::string line;

  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    std::complex<double> val = vec(i);

    if (std::abs(val.real()) < 1e-10 && val.imag() != 0.0) {
      line += fmt::format("{:7.2f}", -std::abs(val.imag()));
    } else {
      line += fmt::format("{:7.2f}", val.real());
    }

    // Check if we need to print the line
    if ((i + 1) % per_line == 0 || i == vec.size() - 1) {
      spdlog::info("{}", line);
      line.clear();
    }
  }
}

inline void print_matrix_full(const occ::Mat6 &matrix, int precision = 6,
                              int width = 12) {
  for (int i = 0; i < 6; ++i) {
    std::string row;
    for (int j = 0; j < 6; ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix_upper_triangle(const occ::Mat6 &matrix,
                                        int precision = 3, int width = 9) {

  for (int i = 0; i < 6; ++i) {
    std::string row;
    for (int k = 0; k < i; ++k) {
      row += fmt::format("{:{}}", "", width);
    }
    for (int j = i; j < 6; ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix(const occ::Mat6 &matrix,
                         bool upper_triangle_only = false, int precision = 3,
                         int width = 9) {
  if (upper_triangle_only) {
    print_matrix_upper_triangle(matrix, precision, width);
  } else {
    print_matrix_full(matrix, precision, width);
  }
}

inline void save_matrix(const occ::Mat &matrix, const std::string &filename,
                        std::vector<std::string> comments = {},
                        bool upper_triangle_only = false, int width = 6) {
  std::ofstream file(filename);
  occ::log::info("Writing matrix to file {}", filename);
  for (const auto &comment : comments) {
    file << "# " << comment << std::endl;
  }
  file << std::fixed << std::setprecision(4);

  if (upper_triangle_only) {
    int count = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = i; j < matrix.cols(); ++j) {
        file << std::setw(12) << matrix(i, j);
        count++;
        if (count % width == 0) {
          file << std::endl;
        } else {
          file << " ";
        }
      }
    }
    if (count % width != 0) {
      file << std::endl;
    }
    file.close();
    return;
  }

  int count = 0;
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file << std::setw(12) << matrix(i, j);
      count++;
      if (count % width == 0) {
        file << std::endl;
      } else {
        file << " ";
      }
    }
  }
  if (count % width != 0) {
    file << std::endl;
  }

  file.close();
}

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
  std::vector<std::unique_ptr<PotentialBase>> m_potentials;
  double m_scale_factor;
  double m_shift_factor;

public:
  explicit PES(double scale_factor_val = 1.0)
      : m_scale_factor(scale_factor_val) {}

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

  inline double shift() const { return m_shift_factor; }

  inline occ::Mat6 voigt_elastic_tensor_from_hessian(double volume) {
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
    int dim = 3 * n_molecules;

    occ::Mat6 D_ee = occ::Mat6::Zero();       // Strain-Strain
    occ::Mat D_ei = occ::Mat::Zero(6, dim);   // Strain-Cart
    occ::Mat D_ij = occ::Mat::Zero(dim, dim); // Cart-Cart

    occ::Mat Mass_inv_ij = occ::Mat::Zero(dim, dim); // Dynamical Cart-Cart

    for (const auto &pot : m_potentials) {
      const occ::Vec3 &r_hat = pot->r_hat;
      const occ::Vec3 &r_vector = pot->r_vector;
      double r = pot->r0;
      double r2 = r * r;
      double dU_dr = pot->first_derivative();
      double d2U_dr2 = pot->second_derivative();
      const auto [idx_0, idx_1] = pot->uc_pair_indices();
      const auto [m0, m1] = pot->pair_mass();

      for (int p = 0; p < 6; p++) {
        auto [alpha, beta] = this->voigt_notation(p);
        for (int q = 0; q < 6; q++) {
          auto [gamma, delta] = this->voigt_notation(q);

          double term1 = (r_vector[alpha] * r_vector[beta] * r_vector[gamma] *
                          r_vector[delta]) *
                         d2U_dr2 / r2;

          // NOTE: term2 will be zero for us by definition.
          // double term2 = dU_dr / r *
          //                ((alpha == gamma ? r_hat[beta] * r_hat[delta] * r2:
          //                0) +
          //                 (alpha == delta ? r_hat[beta] * r_hat[gamma] * r2:
          //                 0) + (beta == gamma ? r_hat[alpha] * r_hat[delta] *
          //                 r2: 0) + (beta == delta ? r_hat[alpha] *
          //                 r_hat[gamma] * r2: 0));
          // D_ee(p, q) += term1 + term2;

          D_ee(p, q) += term1;
        }
        for (int coord = 0; coord < 3; coord++) {
          double mixed_term =
              d2U_dr2 * r_vector[alpha] * r_vector[beta] * r_vector[coord] / r2;

          // NOTE: zero by definition.
          // mixed_term += dU_dr / r *
          //               ((alpha == coord ? r_vector[beta]: 0) +
          //                (beta == coord ? r_vector[alpha]: 0));

          D_ei(p, idx_0 * 3 + coord) -= mixed_term;
          D_ei(p, idx_1 * 3 + coord) += mixed_term;
        }
      }
      for (int alpha = 0; alpha < 3; alpha++) {
        for (int beta = 0; beta < 3; beta++) {
          double e = d2U_dr2 * r_hat[alpha] * r_hat[beta];

          // NOTE: zero by definition
          // if (alpha == beta) {
          //   e += dU_dr / r;
          // }

          D_ij(idx_0 * 3 + alpha, idx_0 * 3 + beta) += e;
          D_ij(idx_1 * 3 + alpha, idx_1 * 3 + beta) += e;
          D_ij(idx_0 * 3 + alpha, idx_1 * 3 + beta) -= e;
          D_ij(idx_1 * 3 + alpha, idx_0 * 3 + beta) -= e;

          Mass_inv_ij(idx_0 * 3 + alpha, idx_0 * 3 + beta) =
              1 / std::sqrt(m0 * m0);
          Mass_inv_ij(idx_1 * 3 + alpha, idx_1 * 3 + beta) =
              1 / std::sqrt(m1 * m1);
          Mass_inv_ij(idx_0 * 3 + alpha, idx_1 * 3 + beta) =
              1 / std::sqrt(m0 * m1);
          Mass_inv_ij(idx_1 * 3 + alpha, idx_0 * 3 + beta) =
              1 / std::sqrt(m1 * m0);
        }
      }
      for (int alpha = 0; alpha < 3; alpha++) {
        for (int beta = 0; beta < 3; beta++) {
        }
      }
    }
    D_ee /= 2;
    D_ei /= 2;
    D_ij /= 2;
    occ::Mat Dyn_ij = Mass_inv_ij * D_ij;
    save_matrix(D_ij, "D_ij.txt",
                {"Cartesian-cartesian Hessian "
                 "(kJ/mol/Ang**2)"});
    save_matrix(D_ij / 96.485, "D_ij_gulp.txt",
                {"Cartesian-cartesian Hessian"
                 "(eV/Ang**2)"},
                false, 3);
    save_matrix(Mass_inv_ij, "Mass_inv_ij.txt",
                {"Inverted mass matrix"
                 "(1/kg)"},
                false);
    save_matrix(
        Dyn_ij, "Dyn_ij.txt",
        {"Dynamical cartesian-cartesian Hessian as upper right triangle "
         "(kJ/Ang**2/kg)"},
        false);
    save_matrix(D_ei.transpose(), "D_ei.txt",
                {"Strain-cartesian second derivative matrix as upper right "
                 "triangle (kJ/mol/Ang)"},
                false);
    save_matrix(D_ei.transpose() / 96.485, "D_ei_gulp.txt",
                {"Strain-cartesian second derivative "
                 "(eV/Ang)"},
                false, 3);
    save_matrix(D_ee, "D_ee.txt",
                {"Strain-strain second derivative matrix "
                 "(kJ/mol)"},
                false);
    save_matrix(D_ee / 96.485, "D_ee_gulp.txt",
                {"Strain-strain second derivative matrix (eV)"}, false, 3);
    Eigen::EigenSolver<occ::Mat> es(Dyn_ij);
    occ::CVec eigenvalues = es.eigenvalues();
    occ::log::info("Phonon frequencies (cm-1):");
    print_cvector(eigenvalues, 6);
    auto D_ij_inv = D_ij.inverse();
    occ::Mat6 correction = D_ei * D_ij_inv * D_ei.transpose();
    occ::Mat6 C =
        (D_ee - correction) / volume * KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    return C;
  }

  inline occ::Mat6
  compute_voigt_elastic_tensor_analytical(double volume) const {
    occ::Mat3 C[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        C[i][j] = occ::Mat3::Zero();
      }
    }

    for (const auto &pot : m_potentials) {
      const occ::Vec3 &r_hat = pot->r_hat;
      double r = pot->r0;
      double d2V = pot->second_derivative();
      double prefactor = d2V * r * r;

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
              C[i][j](k, l) +=
                  prefactor * r_hat[i] * r_hat[j] * r_hat[k] * r_hat[l];
            }
          }
        }
      }
    }

    return this->to_voigt(C) / volume / 2.0 * KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
  }

  static std::pair<int, int> voigt_notation(int voigt) {
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

  static occ::Mat6 to_voigt(const occ::Mat3 C[3][3]) {
    std::unordered_map<int, int> voigt_map = {
        {0 * 3 + 0, 0}, {1 * 3 + 1, 1}, {2 * 3 + 2, 2},
        {1 * 3 + 2, 3}, {2 * 3 + 1, 3}, {0 * 3 + 2, 4},
        {2 * 3 + 0, 4}, {0 * 3 + 1, 5}, {1 * 3 + 0, 5}};

    occ::Mat6 C_voigt = occ::Mat6::Zero();

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            auto alpha_it = voigt_map.find(i * 3 + j);
            auto beta_it = voigt_map.find(k * 3 + l);

            if (alpha_it == voigt_map.end() || beta_it == voigt_map.end()) {
              throw std::runtime_error("Voigt mapping error");
            }

            int alpha = alpha_it->second;
            int beta = beta_it->second;

            double factor = 1.0;
            if ((alpha < 3 && beta >= 3) || (alpha >= 3 && beta < 3)) {
              factor = 0.5;
            }
            if (alpha >= 3 && beta >= 3) {
              factor = 0.25;
            }

            C_voigt(alpha, beta) += factor * C[i][j](k, l);
          }
        }
      }
    }
    return C_voigt;
  }
};

struct EFSettings {
  std::string json_filename;
  std::string output_file = "elastic_tensor.txt";
  std::string potential_type = "lj";
  bool include_positive = false;
  bool max_to_zero = false;
  double scale_factor = 2.0;
  double gulp_scale = 0.01;
  std::string gulp_file{""};
};

CLI::App *add_elastic_fit_subcommand(CLI::App &app);
void run_elastic_fit_subcommand(const EFSettings &settings);

} // namespace occ::main
