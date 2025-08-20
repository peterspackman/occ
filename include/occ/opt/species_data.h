#pragma once
#ifdef USE_GLOBAL_SPECIES_DATA
#include <occ/core/element.h>
#endif

namespace occ::opt {

// Species data - covalent and van der Waals radii for elements
// Values from pyberny/species-data.csv
struct SpeciesData {
  int atomic_number;
  float covalent_radius; // Angstroms
  float vdw_radius;      // Angstroms
};

// Static array with species data (atomic numbers 1-103)
// 1.6 used as a default where no data, 2.0 for vdw
static constexpr SpeciesData SPECIES_DATA[] = {
    {0, 0.0f, 0.0f},            // Dummy entry for index 0
    {1, 0.38f, 1.6404493538f},  // H
    {2, 0.32f, 1.4023196089f},  // He
    {3, 1.34f, 2.2013771973f},  // Li
    {4, 0.9f, 2.2066689695f},   // Be
    {5, 0.82f, 2.0584993504f},  // B
    {6, 0.77f, 1.8997461871f},  // C
    {7, 0.75f, 1.7674518844f},  // N
    {8, 0.73f, 1.6880753028f},  // O
    {9, 0.71f, 1.6086987211f},  // F
    {10, 0.69f, 1.5399056837f}, // Ne
    {11, 1.54f, 1.9738309967f}, // Na
    {12, 1.3f, 2.2595866905f},  // Mg
    {13, 1.18f, 2.2913373232f}, // Al
    {14, 1.11f, 2.2225442858f}, // Si
    {15, 1.06f, 2.1220006157f}, // P
    {16, 1.02f, 2.0426240341f}, // S
    {17, 0.99f, 1.9632474524f}, // Cl
    {18, 0.97f, 1.8785790987f}, // Ar
    {19, 1.96f, 1.9632474524f}, // K
    {20, 1.74f, 2.4606740307f}, // Ca
    {21, 1.44f, 2.428923398f},  // Sc
    {22, 1.36f, 2.3865892212f}, // Ti
    {23, 1.25f, 2.3495468164f}, // V
    {24, 1.27f, 2.1114170715f}, // Cr
    {25, 1.39f, 2.1008335273f}, // Mn
    {26, 1.25f, 2.2384196021f}, // Fe
    {27, 1.26f, 2.2119607416f}, // Co
    {28, 1.21f, 2.0214569456f}, // Ni
    {29, 1.38f, 1.989706313f},  // Cu
    {30, 1.31f, 2.1272923878f}, // Zn
    {31, 1.26f, 2.2172525137f}, // Ga
    {32, 1.22f, 2.2225442858f}, // Ge
    {33, 1.19f, 2.1749183368f}, // As
    {34, 1.16f, 2.137875932f},  // Se
    {35, 1.14f, 2.0796664388f}, // Br
    {36, 1.1f, 2.0214569456f},  // Kr
    {37, 2.11f, 1.9685392245f}, // Rb
    {38, 1.92f, 2.4024645375f}, // Sr
    {39, 1.62f, 2.5480411882f}, // Y
    {40, 1.48f, 2.3971727654f}, // Zr
    {41, 1.37f, 2.241859254f},  // Nb
    {42, 1.45f, 2.1690973875f}, // Mo
    {43, 1.56f, 2.1569263116f}, // Tc
    {44, 1.26f, 2.1142217107f}, // Ru
    {45, 1.35f, 2.0902499831f}, // Rh
    {46, 1.31f, 1.9367885919f}, // Pd
    {47, 1.53f, 2.0214569456f}, // Ag
    {48, 1.48f, 2.1114170715f}, // Cd
    {49, 1.44f, 2.239467373f},  // In
    {50, 1.41f, 2.2770495385f}, // Sn
    {51, 1.38f, 2.2627617538f}, // Sb
    {52, 1.35f, 2.23312783f},   // Te
    {53, 1.33f, 2.2066689695f}, // I
    {54, 1.3f, 2.1590430205f},  // Xe
    {55, 2.25f, 2.0002898572f}, // Cs
    {56, 1.98f, 2.524175296f},  // Ba
    {57, 1.69f, 2.0f},          // La (no vdw radius)
    {58, 1.6f, 2.0f},           // Ce (no cov/vdw radii)
    {59, 1.6f, 2.0f},           // Pr
    {60, 1.6f, 2.0f},           // Nd
    {61, 1.6f, 2.0f},           // Pm
    {62, 1.6f, 2.0f},           // Sm
    {63, 1.6f, 2.0f},           // Eu
    {64, 1.6f, 2.0f},           // Gd
    {65, 1.6f, 2.0f},           // Tb
    {66, 1.6f, 2.0f},           // Dy
    {67, 1.6f, 2.0f},           // Ho
    {68, 1.6f, 2.0f},           // Er
    {69, 1.6f, 2.0f},           // Tm
    {70, 1.6f, 2.0f},           // Yb
    {71, 1.6f, 2.0f},           // Lu
    {72, 1.5f, 2.2278360579f},  // Hf
    {73, 1.38f, 2.1960854252f}, // Ta
    {74, 1.46f, 2.1590430205f}, // W
    {75, 1.59f, 2.1272923878f}, // Re
    {76, 1.28f, 2.0320404899f}, // Os
    {77, 1.37f, 2.1167088436f}, // Ir
    {78, 1.28f, 2.0743746667f}, // Pt
    {79, 1.44f, 2.0426240341f}, // Au
    {80, 1.49f, 2.1061252994f}, // Hg
    {81, 1.48f, 2.0690828946f}, // Tl
    {82, 1.47f, 2.280753779f},  // Pb
    {83, 1.46f, 2.2860455511f}, // Bi
    {84, 1.6f, 2.1680390331f},  // Po (no cov radius)
    {85, 1.6f, 2.1537512484f},  // At (no cov radius)
    {86, 1.45f, 2.2384196021f}, // Rn
    {87, 1.6f, 2.0f},           // Fr (no cov/vdw radii)
    {88, 1.6f, 2.0f},           // Ra
    {89, 1.6f, 2.0f},           // Ac
    {90, 1.6f, 2.0f},           // Th
    {91, 1.6f, 2.0f},           // Pa
    {92, 1.6f, 2.0f},           // U
};

constexpr size_t SPECIES_COUNT = sizeof(SPECIES_DATA) / sizeof(SpeciesData);

inline float get_covalent_radius(int atomic_number) {
#ifndef USE_GLOBAL_SPECIES_DATA
  if (atomic_number <= 0 || atomic_number >= static_cast<int>(SPECIES_COUNT)) {
    return 0.0f;
  }
  return SPECIES_DATA[atomic_number].covalent_radius;
#else
  return occ::core::Element(atomic_number).covalent_radius();
#endif
}

inline float get_vdw_radius(int atomic_number) {
#ifndef USE_GLOBAL_SPECIES_DATA
  if (atomic_number <= 0 || atomic_number >= static_cast<int>(SPECIES_COUNT)) {
    return 0.0f;
  }
  return SPECIES_DATA[atomic_number].vdw_radius;
#else
  return occ::core::Element(atomic_number).covalent_radius();
#endif
}

} // namespace occ::opt
