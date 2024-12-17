#pragma once
#include <string>
#include <vector>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>
#include <occ/cg/solvent_surface.h>
#include <occ/cg/solvation_contribution.h>

namespace occ::main {

using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;

enum class SolvationPartitionScheme {
    NearestAtom,
    NearestAtomDnorm,
    ElectronDensity,
};

// Partition solvent surface contributions
std::vector<occ::cg::SolvationContribution> partition_solvent_surface(
    SolvationPartitionScheme scheme, 
    const Crystal &crystal,
    const std::string &mol_name, 
    const std::vector<occ::qm::Wavefunction> &wfns,
    const occ::cg::SMDSolventSurfaces &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent);

} // namespace occ::main
