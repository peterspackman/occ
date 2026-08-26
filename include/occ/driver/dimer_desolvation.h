#pragma once
#include <occ/crystal/crystal.h>
#include <occ/driver/cg_solvation_model.h>
#include <occ/driver/sigma_solvation.h>
#include <occ/qm/wavefunction.h>
#include <vector>

namespace occ::driver {

/// Desolvation cost of forming each unique dimer,
/// `G_solv(A) + G_solv(B) − G_solv(AB)`, in Hartree and positive when
/// forming the contact costs solvation.
///
/// This is the per-contact quantity a pairwise decomposition actually wants:
/// the buried region is absent from the dimer cavity by construction, so no
/// surface has to be attributed to a neighbour by proximity.
///
/// No dimer SCF is involved — the potential at the dimer cavity is the
/// superposition of the two monomer wavefunctions, the same frozen-density
/// approximation the CE model already uses for its electrostatic term. The
/// conductor cavity is solvent independent, so `E_diel` is computed once;
/// only the residual contraction depends on the solvent.
///
/// Indexed by unique dimer index, matching `dimers.unique_dimers`.
/// Only dimers whose nearest atom-atom distance is within `max_distance`
/// are computed; the rest carry no attributed surface and are left at zero.
std::vector<double> dimer_desolvation(
    const crystal::Crystal &crystal,
    const std::vector<qm::Wavefunction> &conductor_wavefunctions,
    const crystal::CrystalDimers &dimers, const SolventSpec &solvent,
    const SigmaSolvationSettings &settings = {},
    double max_distance = 4.0);

} // namespace occ::driver
