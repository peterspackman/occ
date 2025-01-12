#pragma once
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>

namespace occ::crystal {

/**
 * \brief A struct representing a lattice vector triplet (h, k, l) in a crystal
 * lattice.
 *
 * The `HKL` struct is used to represent a triplet of integers (h, k, l) that
 * defines a lattice vector in a crystal lattice. The triplet is typically used
 * to index the reciprocal lattice of the crystal lattice, with h, k, and l
 * being the Miller indices of the lattice vector in the reciprocal lattice.
 *
 * The `HKL` struct provides methods to calculate the magnitude of the lattice
 * vector using the lattice parameters of the crystal lattice, and to compare
 * `HKL` objects using the magnitude of the lattice vectors.
 */
struct HKL {
  int h{0}, k{0}, l{0};

  /**
   * \brief Calculates the magnitude of the lattice vector represented by this
   * `HKL` object.
   *
   * The magnitude of the lattice vector is calculated using the lattice
   * parameters of the crystal lattice, as follows:
   *
   * \f[
   *
   *  d = \sqrt{h^2 a^2 + k^2 b^2 + l^2 c^2 + 2 hk ab \cos \gamma + 2 hl ac
   * \cos \beta + 2 kl bc \cos \alpha}
   *
   * \f]
   *
   * where (h, k, l) are the Miller indices of the lattice vector, and (a, b,
   * c) and
   * (\f$\alpha\f$, \f$\beta\f$, \f$\gamma\f$) are the lattice parameters of
   * the crystal lattice.
   *
   * \param lattice The lattice parameters of the crystal lattice, represented
   * as a 3x3 matrix. The matrix must have the lattice vectors as columns, in
   * the order (a, b, c).
   *
   * \return The magnitude of the lattice vector represented by this `HKL`
   * object, as a floating-point value.
   */
  double d(const Mat3 &lattice) const;

  /// The maximum representable HKL structure
  static HKL maximum() {
    return {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max()};
  }

  /// The minimum representable HKL structure
  static HKL minimum() {
    return {std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
            std::numeric_limits<int>::min()};
  }

  static HKL floor(const Vec3 &vec, double tolerance = 1e-8);
  static HKL ceil(const Vec3 &vec);

  Vec3 vector() const;
  static HKL from_vector(const Vec3 &vec);

  inline bool operator==(const HKL &rhs) const {
    return (h == rhs.h) && (k == rhs.k) && (l == rhs.l);
  }

  inline HKL operator-(const HKL &rhs) const {
    return HKL{h - rhs.h, k - rhs.k, l - rhs.l};
  }

  inline bool operator<(const HKL &rhs) const {
    return std::tie(h, k, l) < std::tie(rhs.h, rhs.k, rhs.l);
  }

  inline bool operator>(const HKL &rhs) const {
    return std::tie(h, k, l) > std::tie(rhs.h, rhs.k, rhs.l);
  }

  inline HKL operator+(const HKL &rhs) const {
    return HKL{h + rhs.h, k + rhs.k, l + rhs.l};
  }
  inline HKL &operator+=(const HKL &rhs) {
    h += rhs.h;
    k += rhs.k;
    l += rhs.l;
    return *this;
  }

  inline HKL &operator-=(const HKL &rhs) {
    h -= rhs.h;
    k -= rhs.k;
    l -= rhs.l;
    return *this;
  }
};
} // namespace occ::crystal

template <> struct fmt::formatter<occ::crystal::HKL> : nested_formatter<int> {
  auto format(const occ::crystal::HKL &, format_context &ctx) const
      -> format_context::iterator;
};
