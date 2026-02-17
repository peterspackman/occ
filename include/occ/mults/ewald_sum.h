#pragma once
#include <occ/mults/crystal_energy.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::mults {

/// Parameters for the Ewald summation.
struct EwaldParams {
    double alpha = 0.35;        ///< Screening parameter (1/Angstrom)
    int kmax = 8;               ///< Maximum reciprocal lattice index
    bool include_dipole = true; ///< Include qmu + mumu corrections (DMACRYS does all three)
};

/// A single charge+dipole site for Ewald summation.
struct EwaldSite {
    Vec3 position;   ///< Position in Angstrom
    double charge;   ///< Charge in a.u. (Q00)
    Vec3 dipole;     ///< Dipole moment in a.u. (e*Bohr)
    int mol_index;   ///< Index of the molecule this site belongs to
};

/// Result from Ewald correction computation.
struct EwaldResult {
    double energy = 0.0;           ///< Energy correction (kJ/mol)
    std::vector<Vec3> site_forces; ///< Force correction per site (kJ/mol/Ang)
};

/// Compute Ewald correction: (Ewald total) - (truncated real-space sum).
///
/// The correction is: reciprocal + self - erf(inter) - erf(intra),
/// where the erf terms cancel the real-space truncation error.
///
/// @param sites           All charge+dipole sites (positions in Angstrom)
/// @param unit_cell       Crystal unit cell
/// @param neighbors       Neighbor pair list (for real-space erf correction)
/// @param mol_site_indices  Mapping: mol_index -> list of site indices in `sites`
/// @param cutoff_radius   COM cutoff for electrostatic pair gate (Angstrom)
/// @param use_com_gate    Apply COM gate to erf correction (match main loop)
/// @param elec_site_cutoff Per-site cutoff in Angstrom (0 = none)
/// @param params          Ewald parameters (alpha, kmax, dipole flag)
EwaldResult compute_ewald_correction(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params);

/// Gather EwaldSites from CartesianMolecules (pure data extraction).
std::vector<EwaldSite> gather_ewald_sites(
    const std::vector<CartesianMolecule>& cart_mols,
    bool include_dipole);

/// Build mol_site_indices from CartesianMolecules.
std::vector<std::vector<size_t>> build_mol_site_indices(
    const std::vector<CartesianMolecule>& cart_mols);

} // namespace occ::mults
