#include <filesystem>
#include <fmt/os.h>
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

struct NeighbourAtoms {
    Mat3N positions;
    IVec mol_idx;
    IVec atomic_numbers;
    Vec vdw_radii;
};

NeighbourAtoms
atom_environment(const CrystalDimers::MoleculeNeighbors &neighbors) {
    size_t num_atoms = 0;
    for (const auto &[n, unique_index] : neighbors) {
        num_atoms += n.b().size();
    }

    NeighbourAtoms result{Mat3N(3, num_atoms), IVec(num_atoms), IVec(num_atoms),
                          Vec(num_atoms)};
    size_t current_idx = 0;
    size_t i = 0;
    for (const auto &[n, unique_index] : neighbors) {
        const auto &mol = n.b();
        size_t N = mol.size();
        result.mol_idx.block(current_idx, 0, N, 1).array() = i;
        result.atomic_numbers.block(current_idx, 0, N, 1).array() =
            mol.atomic_numbers().array();
        result.vdw_radii.block(current_idx, 0, N, 1).array() =
            mol.vdw_radii().array() / BOHR_TO_ANGSTROM;
        result.positions.block(0, current_idx, 3, N) =
            mol.positions() / BOHR_TO_ANGSTROM;
        current_idx += N;
        i++;
    }
    return result;
}

void pair_solvent_energy_contributions(
    const CrystalDimers::MoleculeNeighbors &neighbors,
    std::vector<SolventNeighborContribution> &energy_contribution) {
    // found A -> B contribution, now find B -> A
    for (int i = 0; i < neighbors.size(); i++) {
        auto &ci = energy_contribution[i];
        if (ci.neighbor_set)
            continue;
        const auto &d1 = neighbors[i].dimer;

        for (int j = i; j < neighbors.size(); j++) {
            if (ci.neighbor_set)
                break;
            auto &cj = energy_contribution[j];
            if (cj.neighbor_set)
                continue;
            const auto &d2 = neighbors[j].dimer;
            if (d1.equivalent_in_opposite_frame(d2)) {
                ci.neighbor_set = true;
                cj.neighbor_set = true;
                occ::log::debug("Interaction paired {}<->{}", i, j);
                ci.assign(cj);
                continue;
            }
        }
    }
}

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

std::vector<SolventNeighborContribution> partition_by_electron_density(
    const Crystal &crystal, const std::string &mol_name,
    const std::vector<Wavefunction> &wfns,
    const SolvatedSurfaceProperties &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent) {

    std::vector<SolventNeighborContribution> energy_contribution(
        neighbors.size());

    Mat density_coulomb(surface.coulomb_pos.cols(), nearest_neighbors.size());
    Mat density_cds(surface.cds_pos.cols(), nearest_neighbors.size());
    for (int i = 0; i < nearest_neighbors.size(); i++) {
        const auto &dimer = nearest_neighbors[i].dimer;
        Molecule mol_B = dimer.b();
        const auto &wfnb = wfns[mol_B.asymmetric_molecule_idx()];
        Wavefunction B = wfns[mol_B.asymmetric_molecule_idx()];
        auto transform_b = calculate_wfn_transform(wfnb, mol_B, crystal);
        B.apply_transformation(transform_b.first, transform_b.second);
        const auto &pos_B = mol_B.positions();
        const auto pos_B_t = B.positions() * BOHR_TO_ANGSTROM;
        assert(occ::util::all_close(pos_B, pos_B_t, 1e-5, 1e-5));
        density_coulomb.col(i) = occ::density::evaluate_density_on_grid<0>(
            B.basis, B.mo.D, surface.coulomb_pos);
        density_cds.col(i) = occ::density::evaluate_density_on_grid<0>(
            B.basis, B.mo.D, surface.cds_pos);
    }

    for (int i = 0; i < surface.coulomb_pos.cols(); i++) {
        double total_density = density_coulomb.row(i).array().sum();
        for (int m_idx = 0; m_idx < density_coulomb.cols(); m_idx++) {
            auto &contrib = energy_contribution[m_idx];
            double proportion = density_coulomb(i, m_idx) / total_density;
            contrib.coulomb.ab +=
                (surface.e_coulomb(i) + surface.e_ele(i)) * proportion;
            contrib.coulomb_area.ab += surface.a_coulomb(i) * proportion;
        }
    }

    for (int i = 0; i < surface.cds_pos.cols(); i++) {
        double total_density = density_cds.row(i).array().sum();
        for (int m_idx = 0; m_idx < density_cds.cols(); m_idx++) {
            auto &contrib = energy_contribution[m_idx];
            double proportion = density_cds(i, m_idx) / total_density;
            contrib.cds.ab += surface.e_cds(i) * proportion;
            contrib.cds_area.ab += surface.a_cds(i) * proportion;
        }
    }
    pair_solvent_energy_contributions(neighbors, energy_contribution);
    return energy_contribution;
}

std::vector<SolventNeighborContribution>
compute_solvation_energy_breakdown_nearest_atom(
    const Crystal &crystal, const std::string &mol_name,
    const SolvatedSurfaceProperties &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent, bool dnorm) {
    using occ::units::angstroms;
    std::vector<SolventNeighborContribution> energy_contribution(
        neighbors.size());

    occ::IVec neighbor_idx_coul(surface.coulomb_pos.cols());
    occ::IVec neighbor_idx_cds(surface.cds_pos.cols());
    auto natoms = atom_environment(nearest_neighbors);

    auto closest_idx = [&](const Vec3 &x, const Mat3N &pos, const Vec &vdw) {
        Eigen::Index idx = 0;
        Vec norms = (natoms.positions.colwise() - x).colwise().norm();
        if (dnorm) {
            norms.array() -= vdw.array();
            norms.array() /= vdw.array();
            double r = norms.minCoeff(&idx);
        } else {
            double r = norms.minCoeff(&idx);
        }
        return idx;
    };

    auto cfile =
        fmt::output_file(fmt::format("{}_coulomb.txt", mol_name),
                         fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    cfile.print("{}\nx y z e neighbor\n", neighbor_idx_coul.rows());
    // coulomb breakdown
    for (size_t i = 0; i < neighbor_idx_coul.rows(); i++) {
        occ::Vec3 x = surface.coulomb_pos.col(i);
        Eigen::Index idx = closest_idx(x, natoms.positions, natoms.vdw_radii);
        auto m_idx = natoms.mol_idx(idx);
        auto &contrib = energy_contribution[m_idx];
        contrib.coulomb.ab += surface.e_coulomb(i) + surface.e_ele(i);
        neighbor_idx_coul(i) = m_idx;
        contrib.coulomb_area.ab += surface.a_coulomb(i);
        cfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                    angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                    surface.e_coulomb(i), m_idx);
    }

    auto cdsfile =
        fmt::output_file(fmt::format("{}_cds.txt", mol_name),
                         fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    cdsfile.print("{}\nx y z e neighbor\n", neighbor_idx_cds.rows());
    // cds breakdown
    for (size_t i = 0; i < neighbor_idx_cds.rows(); i++) {
        occ::Vec3 x = surface.cds_pos.col(i);
        Eigen::Index idx = closest_idx(x, natoms.positions, natoms.vdw_radii);
        auto m_idx = natoms.mol_idx(idx);
        auto &contrib = energy_contribution[m_idx];
        contrib.cds.ab += surface.e_cds(i);
        contrib.cds_area.ab += surface.a_cds(i);
        neighbor_idx_cds(i) = m_idx;
        cdsfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                      angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                      surface.e_cds(i), m_idx);
    }

    pair_solvent_energy_contributions(neighbors, energy_contribution);
    return energy_contribution;
}

std::vector<SolventNeighborContribution> partition_solvent_surface(
    SolvationPartitionScheme scheme, const Crystal &crystal,
    const std::string &mol_name, const std::vector<occ::qm::Wavefunction> &wfns,
    const SolvatedSurfaceProperties &surface,
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
                                             neighbors, nearest_neighbors,
                                             solvent);
    default:
        throw std::runtime_error("Not implemented");
    }
}

std::pair<std::vector<SolvatedSurfaceProperties>, std::vector<Wavefunction>>
calculate_solvated_surfaces(const std::string &basename,
                            const std::vector<Molecule> &mols,
                            const std::vector<Wavefunction> &wfns,
                            const std::string &solvent_name,
                            const std::string &method,
                            const std::string &basis_name) {
    std::vector<SolvatedSurfaceProperties> result;
    using occ::dft::DFT;
    using occ::solvent::SolvationCorrectedProcedure;
    std::vector<Wavefunction> solvated_wfns;
    size_t index = 0;
    for (const auto &wfn : wfns) {
        fs::path props_path(fmt::format("{}_{}_{}_surface.json", basename,
                                        index, solvent_name));
        fs::path json_wfn_path(
            fmt::format("{}_{}_{}.owf.json", basename, index, solvent_name));
        if (fs::exists(props_path)) {
            occ::log::info("Loading surface properties from {}",
                           props_path.string());
            {
                std::ifstream ifs(props_path.string());
                auto jf = nlohmann::json::parse(ifs);
                result.emplace_back(jf.get<SolvatedSurfaceProperties>());
            }

            if (fs::exists(json_wfn_path)) {
                occ::log::info("Loading solvated wavefunction from {}",
                               json_wfn_path.string());
                solvated_wfns.emplace_back(Wavefunction::load(json_wfn_path.string()));
            } else {
                occ::qm::AOBasis basis =
                    occ::qm::AOBasis::load(wfn.atoms, basis_name);
                double original_energy = wfn.energy.total;
                occ::log::debug("Total energy (gas) {:.3f}", original_energy);
                basis.set_pure(false);
                occ::log::debug(
                    "Loaded basis set, {} shells, {} basis functions",
                    basis.size(), basis.nbf());
                occ::dft::DFT ks(method, basis);
                SolvationCorrectedProcedure<DFT> proc_solv(ks, solvent_name);
                SCF<SolvationCorrectedProcedure<DFT>> scf(proc_solv,
                                                          wfn.mo.kind);
                scf.convergence_settings.incremental_fock_threshold = 0.0;
                scf.set_charge_multiplicity(wfn.charge(), wfn.multiplicity());
                double e = scf.compute_scf_energy();
		occ::log::info("Writing solvated wavefunction file to {}", json_wfn_path.string());
                Wavefunction wfn = scf.wavefunction();
		wfn.save(json_wfn_path.string());
                solvated_wfns.push_back(wfn);
            }
        } else {
            occ::qm::AOBasis basis =
                occ::qm::AOBasis::load(wfn.atoms, basis_name);
            double original_energy = wfn.energy.total;
            occ::log::debug("Total energy (gas) {:.3f}", original_energy);
            basis.set_pure(false);
            occ::log::debug("Loaded basis set, {} shells, {} basis functions",
                            basis.size(), basis.nbf());
            occ::dft::DFT ks(method, basis);
            SolvationCorrectedProcedure<DFT> proc_solv(ks, solvent_name);
            SCF<SolvationCorrectedProcedure<DFT>> scf(proc_solv, wfn.mo.kind);
            scf.convergence_settings.incremental_fock_threshold = 0.0;
            scf.set_charge_multiplicity(wfn.charge(), wfn.multiplicity());
            double e = scf.compute_scf_energy();
            Wavefunction wfn = scf.wavefunction();

	    occ::log::info("Writing solvated wavefunction file to {}", json_wfn_path.string());
	    wfn.save(json_wfn_path.string());
            solvated_wfns.push_back(wfn);

            SolvatedSurfaceProperties props;
            props.coulomb_pos = proc_solv.surface_positions_coulomb();
            props.cds_pos = proc_solv.surface_positions_cds();
            auto coul_areas = proc_solv.surface_areas_coulomb();
            auto cds_areas = proc_solv.surface_areas_cds();
            props.e_cds = proc_solv.surface_cds_energy_elements();
            props.a_coulomb = coul_areas;
            props.a_cds = cds_areas;
            auto nuc = proc_solv.surface_nuclear_energy_elements();
            auto elec = proc_solv.surface_electronic_energy_elements(scf.mo);
            auto pol = proc_solv.surface_polarization_energy_elements();
            props.e_coulomb = nuc + elec + pol;
            occ::log::debug("sum e_nuc {:12.6f}", nuc.array().sum());
            occ::log::debug("sum e_ele {:12.6f}", elec.array().sum());
            occ::log::debug("sum e_pol {:12.6f}", pol.array().sum());
            occ::log::debug("sum e_cds {:12.6f}", props.e_cds.array().sum());
            double esolv = nuc.array().sum() + elec.array().sum() +
                           pol.array().sum() + props.e_cds.array().sum();

            // dG_gas
            const auto &mol = mols[index];
            double Gr = mol.rotational_free_energy(298);
            occ::core::MolecularPointGroup pg(mol);
            double Gt = mol.translational_free_energy(298);

            constexpr double R = 8.31446261815324;
            constexpr double RT = 298 * R / 1000;
            Gr += RT * std::log(pg.symmetry_number());
            props.dg_correction = (
                                      // dG concentration in kJ/mol
                                      1.89 / occ::units::KJ_TO_KCAL
                                      // 2 RT contribution from enthalpy
                                      - 2 * RT) /
                                  occ::units::AU_TO_KJ_PER_MOL;
            props.dg_gas =
                // Gr + Gt contribution from gas
                (Gt + Gr) / occ::units::AU_TO_KJ_PER_MOL;

            props.dg_ele = e - original_energy - esolv;
            occ::log::debug("total e_solv {:12.6f} ({:.3f} kJ/mol)", esolv,
                            esolv * occ::units::AU_TO_KJ_PER_MOL);

            occ::log::info("SCF difference         (au)       {: 9.3f}",
                           e - original_energy);
            occ::log::debug("SCF difference         (kJ/mol)   {: 9.3f}",
                            occ::units::AU_TO_KJ_PER_MOL *
                                (e - original_energy));
            occ::log::debug("total E solv (surface) (kj/mol)   {: 9.3f}",
                            esolv * occ::units::AU_TO_KJ_PER_MOL);
            occ::log::debug("orbitals E_solv        (kj/mol)   {: 9.3f}",
                            props.dg_ele * occ::units::AU_TO_KJ_PER_MOL);
            occ::log::debug("CDS E_solv   (surface) (kj/mol)   {: 9.3f}",
                            props.e_cds.array().sum() *
                                occ::units::AU_TO_KJ_PER_MOL);
            occ::log::debug("nuc E_solv   (surface) (kj/mol)   {: 9.3f}",
                            nuc.array().sum() * occ::units::AU_TO_KJ_PER_MOL);
            occ::log::debug("pol E_solv   (surface) (kj/mol)   {: 9.3f}",
                            pol.array().sum() * occ::units::AU_TO_KJ_PER_MOL);
            occ::log::debug("ele E_solv   (surface) (kj/mol)   {: 9.3f}",
                            elec.array().sum() * occ::units::AU_TO_KJ_PER_MOL);

            props.esolv = e - original_energy;
            props.e_ele =
                (props.dg_ele / coul_areas.array().sum()) * coul_areas.array();

            {
		occ::log::info("Writing solvated surface properties to {}", props_path.string());
                std::ofstream ofs(props_path.string());
                nlohmann::json j = props;
                ofs << j;
            }
            result.push_back(props);
        }
        index++;
    }
    return {result, solvated_wfns};
}

} // namespace occ::main
