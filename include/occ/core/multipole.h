#pragma once
#include <array>
#include <fmt/format.h>
#include <occ/core/linear_algebra.h>

namespace occ::core {

/**
 * The number of unique multipole components for a given angular momentum.
 *
 * \param L the angular momentum.
 *
 * \returns an unsigned int with the count of the number of components.
 */
inline constexpr unsigned int num_unique_multipole_components(int L) {
  return (L + 1) * (L + 2) / 2;
}

/**
 * The number of tensor multipole components for a given angular momentum.
 *
 * \param L the angular momentum.
 *
 * \returns an unsigned int with the count of the number of components.
 */
inline constexpr unsigned int num_multipole_components_tensor(int L) {
  unsigned int result = 1;
  for (int i = 1; i <= L; i++) {
    result *= 3;
  }
  return result;
}

/**
 * The total number of unique multipole components up to and including a given
 * angular momentum.
 *
 * \param L the angular momentum.
 *
 * \returns an unsigned int with the count of the number of components.
 */
inline constexpr unsigned int total_num_multipole_components(int L) {
  unsigned int n{0};
  for (unsigned int i = 0; i <= L; i++) {
    n += num_unique_multipole_components(i);
  }
  return n;
}

/**
 * The names of multipole components in order
 */
inline constexpr std::array<const char *, 35> multipole_component_names{
    "q",     "Dx",    "Dy",    "Dz",    "Qxx",   "Qxy",   "Qxz",
    "Qyy",   "Qyz",   "Qzz",   "Oxxx",  "Oxxy",  "Oxxz",  "Oxyy",
    "Oxyz",  "Oxzz",  "Oyyy",  "Oyyz",  "Oyzz",  "Ozzz",  "Hxxxx",
    "Hxxxy", "Hxxxz", "Hxxyy", "Hxxyz", "Hxxzz", "Hxyyy", "Hxyyz",
    "Hxyzz", "Hxzzz", "Hyyyy", "Hyyyz", "Hyyzz", "Hyzzz", "Hzzzz"};

/**
 * Templated storage class for Multipole expansions
 */
template <unsigned int L> struct Multipole {
  /// storage alias for 3 components of a dipole
  using Dipole = std::array<double, 3>;
  /// storage alias for the 6 components of a quadrupole
  using Quadrupole = std::array<double, 6>;
  /// storage alias for the 10 components of an octupole
  using Octupole = std::array<double, 10>;
  /// storage alias for the 15 components of a hexadecapole
  using Hexadecapole = std::array<double, 15>;

  static constexpr unsigned int num_components{
      total_num_multipole_components(L)};
  std::array<double, num_components> components;

  /**
   * The monopole (charge) part of this multipole expansion `q`
   *
   * \returns monopole.
   */
  double charge() const { return components[0]; }

  /**
   * The dipole part of this multipole expansion (`x y z`).
   *
   * \returns the (3 unique) components of the dipole
   */
  Dipole dipole() const {
    static_assert(L > 0, "No dipole for a multipole with angular momentum < 1");
    return {components[1], components[2], components[3]};
  }

  /**
   * The quadrupole part of this multipole expansion (`xx xy xz yy yz
   * zz`).
   *
   * \returns the (6 unique) components of the quadrupole
   */
  Quadrupole quadrupole() const {
    static_assert(L > 1,
                  "No quadrupole for a multipole with angular momentum < 2");
    return {components[4], components[5], components[6],
            components[7], components[8], components[9]};
  }

  /**
   * The octupole part of this multipole expansion (`xxx xxy xxz xyy xyz
   * xzz yyy yyz yzz zzz`).
   *
   * \returns the (10 unique) components of the octupole
   */
  Octupole octupole() const {
    static_assert(L > 2,
                  "No octupole for a multipole with angular momentum < 3");
    return {components[10], components[11], components[12], components[13],
            components[14], components[15], components[16], components[17],
            components[18], components[19]};
  }
  /**
   * The hexadecapole part of this multipole expansion (`xxxx xxxz xxyy xxyz
   * xxzz xyyy xyyz xyzz xzzz yyyy yyyz yyzz yzzz zzzz`).
   *
   * \returns the (15 unique) components of the hexadecapole
   */
  Hexadecapole hexadecapole() const {
    static_assert(L > 3,
                  "No hexadecapole for a multipole with angular momentum < 4");
    return {components[20], components[21], components[22], components[23],
            components[24], components[25], components[26], components[27],
            components[28], components[29], components[30], components[31],
            components[32], components[33], components[34]};
  }

  /**
   * Operator overload to add two multipoles together.
   * \returns a Multipole of order max(L1, L2).
   */
  template <unsigned int L2>
  Multipole<std::max(L, L2)> operator+(const Multipole<L2> &rhs) const {
    constexpr unsigned int LM = std::max(L, L2);
    Multipole<LM> result;
    std::fill(result.components.begin(), result.components.end(), 0.0);
    for (unsigned int i = 0; i < num_components; i++) {
      result.components[i] += components[i];
    }

    for (unsigned int i = 0; i < rhs.num_components; i++) {
      result.components[i] += rhs.components[i];
    }
    return result;
  }

  inline std::string to_string() const {
    constexpr auto names = multipole_component_names;
    if constexpr (L == 0) {
      return fmt::format("{:5s} {:12.6f}\n", names[0], components[0]);
    } else if constexpr (L == 1) {
      return fmt::format("{:5s} {:12.6f}\n"
                         "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} "
                         "{:12.6f}\n",
                         names[0], components[0], names[1], components[1],
                         names[2], components[2], names[3], components[3]);
    } else if constexpr (L == 2) {
      return fmt::format("{:5s} {:12.6f}\n"
                         "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
                         "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
                         "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n",
                         names[0], components[0], names[1], components[1],
                         names[2], components[2], names[3], components[3],
                         names[4], components[4], names[5], components[5],
                         names[6], components[6], names[7], components[7],
                         names[8], components[8], names[9], components[9]);
    } else if constexpr (L == 3) {
      return fmt::format(
          "{:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}\n",
          names[0], components[0], names[1], components[1], names[2],
          components[2], names[3], components[3], names[4], components[4],
          names[5], components[5], names[6], components[6], names[7],
          components[7], names[8], components[8], names[9], components[9],
          names[10], components[10], names[11], components[11], names[12],
          components[12], names[13], components[13], names[14], components[14],
          names[15], components[15], names[16], components[16], names[17],
          components[17], names[18], components[18], names[19], components[19]);
    } else if constexpr (L == 4) {
      return fmt::format(
          "{:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n"
          "{:5s} {:12.6f}    {:5s} {:12.6f}    {:5s} {:12.6f}\n",
          names[0], components[0], names[1], components[1], names[2],
          components[2], names[3], components[3], names[4], components[4],
          names[5], components[5], names[6], components[6], names[7],
          components[7], names[8], components[8], names[9], components[9],
          names[10], components[10], names[11], components[11], names[12],
          components[12], names[13], components[13], names[14], components[14],
          names[15], components[15], names[16], components[16], names[17],
          components[17], names[18], components[18], names[19], components[19],
          names[20], components[20], names[21], components[21], names[22],
          components[22], names[23], components[23], names[24], components[24],
          names[25], components[25], names[26], components[26], names[27],
          components[27], names[28], components[28], names[29], components[29],
          names[30], components[30], names[31], components[31], names[32],
          components[32], names[33], components[33], names[34], components[34]);
    }
  }
};

} // namespace occ::core
