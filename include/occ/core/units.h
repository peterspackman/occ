#pragma once
#include <cmath>

namespace occ::units {
constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
constexpr double ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM;
constexpr double AU_TO_KJ_PER_MOL = 2625.499639;
constexpr double AU_TO_PER_CM = 219474.63;
constexpr double AU_TO_KCAL_PER_MOL = 627.5096080305927;
constexpr double EV_TO_KJ_PER_MOL = 96.48530749925973;
constexpr double AU_TO_EV = 27.211399;
constexpr double AU_TO_KELVIN = 315777.09;
constexpr double KJ_TO_KCAL = AU_TO_KCAL_PER_MOL / AU_TO_KJ_PER_MOL;
constexpr double PI = 3.14159265358979323846;

template <typename T> constexpr auto radians(T x) { return x * PI / 180; }

template <typename T> constexpr auto degrees(T x) { return x * 180 / PI; }

template <typename T> constexpr auto angstroms(T x) {
  return BOHR_TO_ANGSTROM * x;
}

} // namespace occ::units
