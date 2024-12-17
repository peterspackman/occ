#include <filesystem>
#include <fmt/os.h>
#include <occ/cg/solvation_types.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/point_group.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/gto/density.h>
#include <occ/io/eigen_json.h>
#include <occ/io/wavefunction_json.h>
#include <occ/main/solvation_partition.h>
#include <occ/qm/scf.h>
#include <occ/solvent/solvation_correction.h>

namespace fs = std::filesystem;

namespace occ::main {

using occ::cg::NeighborAtoms;
using occ::cg::SMDSolventSurfaces;
using occ::cg::SolvationContribution;
using occ::core::Dimer;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::solvent::SolvationCorrectedProcedure;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;

auto calculate_wfn_transform(const Wavefunction &wfn, const Molecule &m,
                             const Crystal &c) {
  using occ::crystal::SymmetryOperation;
  int sint = m.asymmetric_unit_symop()(0);
  SymmetryOperation symop(sint);
  occ::Mat3N positions = wfn.positions() * BOHR_TO_ANGSTROM;

  occ::Mat3 rotation =
      c.unit_cell().direct() * symop.rotation() * c.unit_cell().inverse();
  occ::Vec3 translation =
      (m.centroid() - (rotation * positions).rowwise().mean()) /
      BOHR_TO_ANGSTROM;
  return std::make_pair(rotation, translation);
}

std::vector<SolvationContribution> partition_by_electron_density(
    const Crystal &crystal, const std::string &mol_name,
    const std::vector<Wavefunction> &wfns, const SMDSolventSurfaces &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent) {

  std::vector<SolvationContribution> energy_contribution(neighbors.size());

  // Calculate densities for both coulomb and CDS surfaces
  Mat density_coulomb(surface.coulomb.size(), nearest_neighbors.size());
  Mat density_cds(surface.cds.size(), nearest_neighbors.size());

  for (int i = 0; i < nearest_neighbors.size(); i++) {
    const auto &dimer = nearest_neighbors[i].dimer;
    Molecule mol_B = dimer.b();
    const auto &wfnb = wfns[mol_B.asymmetric_molecule_idx()];
    Wavefunction B = wfns[mol_B.asymmetric_molecule_idx()];
    auto transform_b = calculate_wfn_transform(wfnb, mol_B, crystal);
    B.apply_transformation(transform_b.first, transform_b.second);

    density_coulomb.col(i) = occ::density::evaluate_density_on_grid<0>(
        B.basis, B.mo.D, surface.coulomb.positions);
    density_cds.col(i) = occ::density::evaluate_density_on_grid<0>(
        B.basis, B.mo.D, surface.cds.positions);
  }

  // Process coulomb contributions
  for (int i = 0; i < surface.coulomb.size(); i++) {
    double total_density = density_coulomb.row(i).array().sum();
    for (int m_idx = 0; m_idx < density_coulomb.cols(); m_idx++) {
      double proportion = density_coulomb(i, m_idx) / total_density;
      double energy =
          surface.coulomb.energies(i) + surface.electronic_energies(i);
      energy_contribution[m_idx].add_coulomb(energy * proportion);
      energy_contribution[m_idx].add_coulomb_area(surface.coulomb.areas(i) *
                                                  proportion);
    }
  }

  // Process CDS contributions
  for (int i = 0; i < surface.cds.size(); i++) {
    double total_density = density_cds.row(i).array().sum();
    for (int m_idx = 0; m_idx < density_cds.cols(); m_idx++) {
      double proportion = density_cds(i, m_idx) / total_density;
      energy_contribution[m_idx].add_cds(surface.cds.energies(i) * proportion);
      energy_contribution[m_idx].add_cds_area(surface.cds.areas(i) *
                                              proportion);
    }
  }

  // Exchange contributions between pairs
  for (int i = 0; i < neighbors.size(); i++) {
    for (int j = i + 1; j < neighbors.size(); j++) {
      const auto &d1 = neighbors[i].dimer;
      const auto &d2 = neighbors[j].dimer;
      if (d1.equivalent_in_opposite_frame(d2)) {
        energy_contribution[i].exchange_with(energy_contribution[j]);
        break;
      }
    }
  }

  return energy_contribution;
}

std::vector<SolvationContribution>
compute_solvation_energy_breakdown_nearest_atom(
    const Crystal &crystal, const std::string &mol_name,
    const SMDSolventSurfaces &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent, bool dnorm) {

  using occ::units::angstroms;
  std::vector<SolvationContribution> energy_contribution(neighbors.size());

  auto natoms = NeighborAtoms(nearest_neighbors);

  auto closest_idx = [&](const Vec3 &x, const Mat3N &pos, const Vec &vdw) {
    Eigen::Index idx = 0;
    Vec norms = (natoms.positions.colwise() - x).colwise().norm();
    if (dnorm) {
      norms.array() -= vdw.array();
      norms.array() /= vdw.array();
    }
    norms.minCoeff(&idx);
    return idx;
  };

  // Process coulomb contributions
  auto cfile =
      fmt::output_file(fmt::format("{}_coulomb.txt", mol_name),
                       fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
  cfile.print("{}\nx y z e neighbor\n", surface.coulomb.size());

  for (size_t i = 0; i < surface.coulomb.size(); i++) {
    occ::Vec3 x = surface.coulomb.positions.col(i);
    Eigen::Index idx = closest_idx(x, natoms.positions, natoms.vdw_radii);
    auto m_idx = natoms.molecule_index(idx);

    energy_contribution[m_idx].add_coulomb(surface.coulomb.energies(i) +
                                           surface.electronic_energies(i));
    energy_contribution[m_idx].add_coulomb_area(surface.coulomb.areas(i));

    cfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n", angstroms(x(0)),
                angstroms(x(1)), angstroms(x(2)), surface.coulomb.energies(i),
                m_idx);
  }

  // Process CDS contributions
  auto cdsfile =
      fmt::output_file(fmt::format("{}_cds.txt", mol_name),
                       fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
  cdsfile.print("{}\nx y z e neighbor\n", surface.cds.size());

  for (size_t i = 0; i < surface.cds.size(); i++) {
    occ::Vec3 x = surface.cds.positions.col(i);
    Eigen::Index idx = closest_idx(x, natoms.positions, natoms.vdw_radii);
    auto m_idx = natoms.molecule_index(idx);

    energy_contribution[m_idx].add_cds(surface.cds.energies(i));
    energy_contribution[m_idx].add_cds_area(surface.cds.areas(i));

    cdsfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                  angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                  surface.cds.energies(i), m_idx);
  }

  // Exchange contributions between pairs
  for (int i = 0; i < neighbors.size(); i++) {
    auto &ci = energy_contribution[i];
    if(ci.has_been_exchanged()) continue;
    for (int j = i; j < neighbors.size(); j++) {
      auto &cj = energy_contribution[j];
      if(ci.has_been_exchanged()) continue;
      const auto &d1 = neighbors[i].dimer;
      const auto &d2 = neighbors[j].dimer;
      if (d1.equivalent_in_opposite_frame(d2)) {
        energy_contribution[i].exchange_with(energy_contribution[j]);
        break;
      }
    }
  }

  return energy_contribution;
}

std::vector<SolvationContribution> partition_solvent_surface(
    SolvationPartitionScheme scheme, const Crystal &crystal,
    const std::string &mol_name, const std::vector<occ::qm::Wavefunction> &wfns,
    const SMDSolventSurfaces &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent) {

  switch (scheme) {
  case SolvationPartitionScheme::NearestAtom:
    return compute_solvation_energy_breakdown_nearest_atom(
        crystal, mol_name, surface, neighbors, nearest_neighbors, solvent,
        false);
  case SolvationPartitionScheme::NearestAtomDnorm:
    return compute_solvation_energy_breakdown_nearest_atom(
        crystal, mol_name, surface, neighbors, nearest_neighbors, solvent,
        true);
  case SolvationPartitionScheme::ElectronDensity:
    return partition_by_electron_density(crystal, mol_name, wfns, surface,
                                         neighbors, nearest_neighbors, solvent);
  default:
    throw std::runtime_error("Not implemented");
  }
}

} // namespace occ::main
