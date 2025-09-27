#pragma once
#include <cmath>
#include <occ/core/constants.h>

namespace occ::units {
// Length conversions (CODATA 2018)
constexpr double BOHR_TO_ANGSTROM = 0.52917721067;
constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;
constexpr double BOHR_TO_METER = BOHR_TO_ANGSTROM * 1e-10;
constexpr double METER_TO_BOHR = 1.0 / BOHR_TO_METER;

// Energy conversions (CODATA 2018) 
constexpr double AU_TO_JOULE = 4.3597447222071e-18;  // Hartree to Joule
constexpr double JOULE_TO_AU = 1.0 / AU_TO_JOULE;
constexpr double AU_TO_KJ_PER_MOL = AU_TO_JOULE * constants::avogadro<double> / 1000.0;
constexpr double AU_TO_KCAL_PER_MOL = AU_TO_KJ_PER_MOL / 4.184;
constexpr double AU_TO_EV = 27.211386245988;
constexpr double AU_TO_KELVIN = AU_TO_JOULE / constants::boltzmann<double>;
constexpr double EV_TO_JOULE = 1.602176634e-19;
constexpr double EV_TO_KJ_PER_MOL = EV_TO_JOULE * constants::avogadro<double> / 1000.0;
constexpr double KJ_TO_KCAL = 1.0 / 4.184;

// Frequency/wavenumber conversions
// 1 Hartree = (E_h / hc) cm^-1 where E_h is Hartree energy
constexpr double AU_TO_PER_CM = AU_TO_JOULE / (constants::planck<double> * constants::speed_of_light<double> * 100.0);
constexpr double PER_CM_TO_AU = 1.0 / AU_TO_PER_CM;

// Mass conversions
// AMU = molar mass constant / Avogadro = 1 g/mol / N_A
constexpr double AMU_TO_KG = constants::molar_mass_constant<double> / constants::avogadro<double>;
constexpr double AMU_TO_AU = 1822.888486209;  // atomic mass unit to electron mass
constexpr double AU_TO_AMU = 1.0 / AMU_TO_AU;

// Speed of light in different units
constexpr double SPEED_OF_LIGHT_CM_PER_S = constants::speed_of_light<double> * 100.0;  // m/s to cm/s
constexpr double SPEED_OF_LIGHT_AU = 137.035999084;  // in atomic units (fine structure constant^-1)

// Mathematical constants (for backward compatibility)
constexpr double PI = constants::pi<double>;

// Other conversions
constexpr double KJ_PER_MOL_PER_ANGSTROM3_TO_GPA = 1.6605388;
constexpr double LIGHTSPEED = 299792458.0;
constexpr double PLANCK = 6.62607015e-34;
constexpr double AVOGADRO = 6.02214076e23;
constexpr double BOLTZMANN = 1.380649e-23;

// Pressure conversions
constexpr double GPA_TO_PA = 1e9;
constexpr double PA_TO_GPA = 1e-9;

// Density conversions
constexpr double G_CM3_TO_KG_M3 = 1000.0;
constexpr double KG_M3_TO_G_CM3 = 1e-3;

template <typename T> constexpr auto radians(T x) { return x * constants::pi<double> / 180; }

template <typename T> constexpr auto degrees(T x) { return x * 180 / constants::pi<double>; }

template <typename T> constexpr auto angstroms(T x) {
  return BOHR_TO_ANGSTROM * x;
}

} // namespace occ::units
