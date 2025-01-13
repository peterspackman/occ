#pragma once

namespace occ::constants {

template <class T>
constexpr T pi =
    T(3.141592653589793238462643383279502884197169399375105820974944);
template <class T>
constexpr T two_pi = T(6.2831853071795864769252867665590057683943388015061);
template <class T>
constexpr T sqrt_pi = T(1.7724538509055160272981674833411451827975);
template <class T>
constexpr T sqrt_half_pi = T(1.253314137315500251207882642405522626503);
template <class T>
constexpr T sqrt_two_pi = T(2.506628274631000502415765284811045253007);
template <class T>
constexpr T e =
    T(2.7182818284590452353602874713526624977572470936999595749669676);

template <class T>
constexpr T angstroms_per_bohr = T(0.52917721092000002923994491662820121095);
template <class T>
constexpr T bohr_per_angstrom = T(1.889726124565061881906545595284207);

template <class T>
constexpr T kj_per_kcal = T(0.2390057361376673040152963671128107);
template <class T>
constexpr T kcal_per_kj = T(4.18400000000000000000000000000000000);

template <class T> constexpr T rad_per_deg = T(pi<T> / 180.0);
template <class T> constexpr T deg_per_rad = T(180.0 / pi<T>);

template <class T> constexpr T planck = T(6.62607015000000000000000e-34);

template <class T> constexpr T avogadro = T(6.02214076000000000000000e23);

template <class T> constexpr T speed_of_light = T(2.99792458000000000000000e8);

template <class T> constexpr T molar_gas_constant = T(8.314462618);

template <class T> constexpr T molar_mass_constant = T(0.99999999965e-3);

template <class T> constexpr T celsius = T(273.15);

template <class T> constexpr T boltzmann = T(1.38066244e-23);

} // namespace occ::constants
