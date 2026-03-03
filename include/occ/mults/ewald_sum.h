#pragma once
#include <occ/mults/crystal_energy.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cutoff_spline.h>
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

/// Ewald correction with analytical site-position Hessian.
///
/// site_hessian layout is Cartesian-by-site:
/// [x1,y1,z1, x2,y2,z2, ...] in kJ/mol/Angstrom^2.
struct EwaldResultWithHessian : EwaldResult {
    /// d^2E / dx_i dx_j in Cartesian site layout [x1,y1,z1,x2,...]
    /// Units: kJ/mol/Angstrom^2.
    Mat site_hessian;
};

/// Pre-computed reciprocal lattice vectors and coefficients for Ewald.
/// The lattice is fixed during strain FD loops, so G-vectors and
/// exp(-G²/4α²)/G² can be cached and reused across evaluations.
struct EwaldLatticeCache {
    struct GVector {
        Vec3 G;        ///< Reciprocal lattice vector (Bohr)
        double coeff;  ///< exp(-G²/4α²) / G²
    };
    std::vector<GVector> g_vectors;
    double four_pi_over_vol = 0.0;
    double alpha_bohr = 0.0;
    double two_alpha_over_sqrt_pi = 0.0;
};

/// Build an EwaldLatticeCache from unit cell and parameters.
EwaldLatticeCache build_ewald_lattice_cache(
    const crystal::UnitCell& unit_cell,
    const EwaldParams& params);

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
/// @param taper           Optional DMACRYS-style real-space radial taper
EwaldResult compute_ewald_correction(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params,
    const CutoffSpline* taper = nullptr,
    const EwaldLatticeCache* lattice_cache = nullptr);

/// Compute Ewald correction and analytical site-position Hessian.
EwaldResultWithHessian compute_ewald_correction_with_hessian(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params,
    const CutoffSpline* taper = nullptr,
    const EwaldLatticeCache* lattice_cache = nullptr);

/// Gather EwaldSites from CartesianMolecules (pure data extraction).
std::vector<EwaldSite> gather_ewald_sites(
    const std::vector<CartesianMolecule>& cart_mols,
    bool include_dipole);

/// Build mol_site_indices from CartesianMolecules.
std::vector<std::vector<size_t>> build_mol_site_indices(
    const std::vector<CartesianMolecule>& cart_mols);

} // namespace occ::mults
