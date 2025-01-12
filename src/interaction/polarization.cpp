#include <array>
#include <occ/interaction/polarization.h>

namespace occ::interaction {

static const std::array<double, 110> Thakkar_atomic_polarizability{
    4.50,   1.38,   164.04, 37.74,  20.43,  11.67,  7.26,   5.24,   3.70,
    2.66,   162.88, 71.22,  57.79,  37.17,  24.93,  19.37,  14.57,  11.09,
    291.10, 157.90, 142.30, 114.30, 97.30,  94.70,  75.50,  63.90,  57.70,
    51.10,  45.50,  38.35,  52.91,  40.80,  29.80,  26.24,  21.13,  16.80,
    316.20, 199.00, 153.00, 121.00, 106.00, 86.00,  77.00,  65.00,  58.00,
    32.00,  52.46,  47.55,  68.67,  57.30,  42.20,  38.10,  32.98,  27.06,
    396.00, 273.50, 210.00, 200.00, 190.00, 212.00, 203.00, 194.00, 187.00,
    159.00, 172.00, 165.00, 159.00, 153.00, 147.00, 145.30, 148.00, 109.00,
    88.00,  75.00,  65.00,  57.00,  51.00,  44.00,  36.06,  34.73,  71.72,
    60.05,  48.60,  43.62,  40.73,  33.18,  315.20, 246.20, 217.00, 217.00,
    171.00, 153.00, 167.00, 165.00, 157.00, 155.00, 153.00, 138.00, 133.00,
    161.00, 123.00, 118.00, 0.00,   0.00,   0.00,   0.00,   0.00,   0.00,
    0.00,   0.00};

// Atomic polarizibilities for charged species
// if not assigned, these should be the same as the uncharged
// +/- are in the same table, double charges are implicit e.g. for Ca
// + will be SIGNIFICANTLY smaller than neutral, - should be a bit larger than
// neutral val for iodine was interpolated

static const std::array<double, 110> Charged_atomic_polarizibility{
    4.50,   1.38,   0.19,   0.052,  20.43,  11.67,  7.26,   5.24,   7.25,
    2.66,   0.986,  0.482,  57.79,  37.17,  24.93,  19.37,  21.20,  11.09,
    5.40,   3.20,   142.30, 114.30, 97.30,  94.70,  75.50,  63.90,  57.70,
    51.10,  45.50,  38.35,  52.91,  40.80,  29.80,  26.24,  27.90,  16.80,
    9.10,   5.80,   153.00, 121.00, 106.00, 86.00,  77.00,  65.00,  58.00,
    32.00,  52.46,  47.55,  68.67,  57.30,  42.20,  38.10,  39.60,  27.06,
    15.70,  10.60,  210.00, 200.00, 190.00, 212.00, 203.00, 194.00, 187.00,
    159.00, 172.00, 165.00, 159.00, 153.00, 147.00, 145.30, 148.00, 109.00,
    88.00,  75.00,  65.00,  57.00,  51.00,  44.00,  36.06,  34.73,  71.72,
    60.05,  48.60,  43.62,  40.73,  33.18,  20.40,  13.40,  217.00, 217.00,
    171.00, 153.00, 167.00, 165.00, 157.00, 155.00, 153.00, 138.00, 133.00,
    161.00, 123.00, 118.00, 0.00,   0.00,   0.00,   0.00,   0.00,   0.00,
    0.00,   0.00};

double ce_model_polarization_energy(const occ::IVec &atomic_numbers,
                                    const occ::Mat3N &field, bool charged) {
  auto fsq = field.colwise().squaredNorm();
  const auto &polarizabilities =
      charged ? Charged_atomic_polarizibility : Thakkar_atomic_polarizability;
  double epol = 0.0;
  for (auto i = 0; i < atomic_numbers.rows(); i++) {
    int n = atomic_numbers(i);
    double pol_fac = polarizabilities[n - 1];
    epol += pol_fac * fsq(i);
  }
  return epol * -0.5;
}

double polarization_energy(const occ::Vec &polarizabilities,
                           const occ::Mat3N &field) {
  auto fsq = field.colwise().squaredNorm();
  double epol = 0.0;
  for (auto i = 0; i < polarizabilities.rows(); i++) {
    epol += polarizabilities(i) * fsq(i);
  }
  return epol * -0.5;
}

} // namespace occ::interaction
