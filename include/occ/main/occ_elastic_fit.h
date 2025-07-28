#pragma once
#include "occ/core/units.h"
#include <CLI/App.hpp>
#include <cmath>
#include <occ/core/linear_algebra.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace occ::main {

using occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

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

enum class PotentialType { MORSE, LJ };

class PotentialBase {
public:
  virtual ~PotentialBase() = default;
  virtual double energy(double r) const = 0;
  virtual double energy() const = 0;
  virtual double first_derivative(double r) const = 0;
  virtual double first_derivative() const = 0;
  virtual double second_derivative(double r) const = 0;
  virtual double second_derivative() const = 0;
  virtual std::string to_string() const = 0;

  occ::Vec3 r_vector;
  occ::Vec3 r_hat;
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

class PES {
private:
  std::vector<std::unique_ptr<PotentialBase>> m_potentials;
  double m_scale_factor;

public:
  explicit PES(double scale_factor_val = 1.0)
      : m_scale_factor(scale_factor_val) {}

  void add_potential(std::unique_ptr<PotentialBase> pot) {
    m_potentials.push_back(std::move(pot));
  }

  double lattice_energy() const {
    double energy = 0.0;
    for (const auto &pot : m_potentials) {
      energy += pot->energy();
    }
    return energy / 2.0;
  }

  size_t number_of_potentials() const { return m_potentials.size(); }

  occ::Mat6 compute_voigt_elastic_tensor_analytical(double volume) const {
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

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              C[i][j](k, l) +=
                  prefactor * r_hat[i] * r_hat[j] * r_hat[k] * r_hat[l];
            }
          }
        }
      }
    }

    return to_voigt(C) / volume / 2.0 * KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
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
  double scale_factor = 2.0;
};

CLI::App *add_elastic_fit_subcommand(CLI::App &app);
void run_elastic_fit_subcommand(const EFSettings &settings);

} // namespace occ::main
