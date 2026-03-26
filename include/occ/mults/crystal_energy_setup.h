#pragma once
#include <occ/mults/rigid_molecule.h>
#include <occ/mults/short_range.h>
#include <occ/mults/force_field_params.h>
#include <occ/crystal/unitcell.h>
#include <map>
#include <string>
#include <vector>

namespace occ::io {
struct StructureInput;
}

namespace occ::crystal {
class Crystal;
}

namespace occ::mults {

class MultipoleSource;

/// Complete specification of a crystal energy calculation.
///
/// This is the single entry point for setting up CrystalEnergy.
/// All molecule data is explicit — no Crystal molecule assembly needed.
struct CrystalEnergySetup {
    crystal::UnitCell unit_cell;
    std::vector<RigidMolecule> molecules; ///< all UC molecules

    // Short-range parameters (kJ/mol, Angstrom)
    // Element-based: key = (Z1, Z2) with Z1 <= Z2
    std::map<std::pair<int, int>, BuckinghamParams> buckingham_params;
    // Type-based: key = (type1, type2) — used when sites have short_range_type > 0
    std::map<std::pair<int, int>, BuckinghamParams> typed_buckingham;
    // Type code labels (for logging)
    std::map<int, std::string> type_labels;

    // Anisotropic repulsion (optional)
    std::map<std::pair<int, int>, AnisotropicRepulsionParams> aniso_params;

    // Force field type
    ForceFieldType force_field = ForceFieldType::BuckinghamDE;

    // Settings
    double cutoff_radius = 15.0;
    bool use_ewald = true;
    double ewald_accuracy = 1e-8;
    double ewald_eta = 0.0;  ///< 0 = auto
    int ewald_kmax = 0;      ///< 0 = auto
    int max_interaction_order = 4;

    // Optional tapering
    double taper_on = 0.0;
    double taper_off = 0.0;
    int taper_order = 3;
};

/// Build CrystalEnergySetup from a StructureInput (explicit rigid bodies).
/// No Crystal object needed — orientations are taken directly from the JSON.
CrystalEnergySetup
from_structure_input(const occ::io::StructureInput &si);

/// Build CrystalEnergySetup from a Crystal + MultipleSources (CIF path).
/// Uses Crystal's unit_cell_molecules() to extract orientations.
/// This is safe because Crystal built the molecules itself from the CIF.
CrystalEnergySetup
from_crystal_and_multipoles(const occ::crystal::Crystal &crystal,
                            const std::vector<MultipoleSource> &multipoles);

/// Convert a CrystalEnergySetup to a StructureInput for serialization.
/// Uses the current molecule placements (COM, angle_axis, parity).
occ::io::StructureInput
to_structure_input(const CrystalEnergySetup &setup,
                   const std::string &title = "");

} // namespace occ::mults
