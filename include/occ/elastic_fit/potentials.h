#pragma once
#include <cmath>
#include <occ/core/log.h>
#include <occ/crystal/crystal.h>
#include <stdexcept>


namespace occ::elastic_fit {

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

} // namespace occ::elastic_fit
