#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <occ/core/kabsch.h>
#include <occ/core/logger.h>
#include <occ/core/point_group.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/cifparser.h>
#include <occ/io/crystalgrower.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/occ_input.h>
#include <occ/main/pair_energy.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/main/single_point.h>
#include <scn/scn.h>

namespace fs = std::filesystem;
using occ::core::Dimer;
using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::crystal::SymmetryOperation;
using occ::hf::HartreeFock;
using occ::interaction::CEModelInteraction;
using occ::qm::BasisSet;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::solvent::COSMO;
using occ::units::AU_TO_KCAL_PER_MOL;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;
using occ::util::all_close;

struct InteractionE {
    double actual{0.0};
    double fictitious{0.0};
};

struct SolvatedSurfaceProperties {
    double esolv{0.0};
    double dg_ele{0.0};
    double dg_gas{0.0};
    double dg_correction{0.0};
    occ::Mat3N coulomb_pos;
    occ::Mat3N cds_pos;
    occ::Vec e_coulomb;
    occ::Vec e_cds;
    occ::Vec e_ele;
    occ::Vec a_coulomb;
    occ::Vec a_cds;

    void save(const std::string &filename) {
        auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
        size_t N = coulomb_pos.cols();
        output.print("{}\ncoulomb esolv = {} dg_ele = {} dg_gas = {}\n", N, esolv, dg_ele, dg_gas);
        for(size_t i = 0; i < N; i++)
        {
            output.print("{: 12.6f} {: 12.6f} {: 12.6f} "
                         "{: 20.12f} {: 20.12f} {: 20.12f}\n",
                         coulomb_pos(0, i), 
                         coulomb_pos(1, i), 
                         coulomb_pos(2, i), 
                         a_coulomb(i),
                         e_coulomb(i),
                         e_ele(i));
        }
        N = cds_pos.cols();
        output.print("{}\ncds\n", N);
        for(size_t i = 0; i < N; i++)
        {
            output.print("{: 12.6f} {: 12.6f} {: 12.6f} "
                         "{: 20.12f} {: 20.12f}\n",
                         cds_pos(0, i), 
                         cds_pos(1, i), 
                         cds_pos(2, i), 
                         a_cds(i),
                         e_cds(i));
        }
    }

    static SolvatedSurfaceProperties load(const std::string &filename) {
        std::ifstream file(filename);
        std::string line;
        size_t N;
        SolvatedSurfaceProperties props;
        double coul_x, coul_y, coul_z, cds_x, cds_y, cds_z;
        double e_coul, e_cds, e_ele;
        double area;
        std::getline(file, line);
        scn::scan_default(line, N);
        std::getline(file, line);
        scn::scan(line, "coulomb esolv = {} dg_ele = {} dg_gas = {}", props.esolv, props.dg_ele, props.dg_gas);
        props.coulomb_pos = occ::Mat3N::Zero(3, N);
        props.e_coulomb = occ::Vec::Zero(N);
        props.e_ele = occ::Vec::Zero(N);
        props.a_coulomb = occ::Vec::Zero(N);
        for(size_t i = 0; i < N; i++) {
            std::getline(file, line);
            scn::scan_default(line, coul_x, coul_y, coul_z, area, e_coul, e_ele);
            props.coulomb_pos(0, i) = coul_x;
            props.coulomb_pos(1, i) = coul_y;
            props.coulomb_pos(2, i) = coul_z;
            props.a_coulomb(i) = area;
            props.e_coulomb(i) = e_coul;
            props.e_ele(i) = e_ele;
        }
        std::getline(file, line);
        scn::scan_default(line, N);
        props.cds_pos = occ::Mat3N::Zero(3, N);
        props.e_cds = occ::Vec::Zero(N);
        props.a_cds = occ::Vec::Zero(N);
        std::getline(file, line);
        for(size_t i = 0; i < N; i++) {
            std::getline(file, line);
            scn::scan_default(line, cds_x, cds_y, cds_z, area, e_cds);
            props.cds_pos(0, i) = cds_x;
            props.cds_pos(1, i) = cds_y;
            props.cds_pos(2, i) = cds_z;
            props.a_cds(i) = area;
            props.e_cds(i) = e_cds;
        }
        return props;
    }
};

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

CEModelInteraction::EnergyComponents read_energy_components(const std::string &line) {
    CEModelInteraction::EnergyComponents components;
    scn::scan(line, "{{ e_coul: {}, e_rep: {}, e_pol: {}, e_disp: {}, e_tot: {} }}",
              components.coulomb, components.exchange_repulsion,
              components.polarization, components.dispersion, components.total);
    return components;
}

Wavefunction calculate_wavefunction(const Molecule &mol,
                                    const std::string &name) {
    fs::path fchk_path(fmt::format("{}.fchk", name));
    if (fs::exists(fchk_path)) {
        fmt::print("Loading gas phase wavefunction from {}\n", fchk_path);
        using occ::io::FchkReader;
        FchkReader fchk(fchk_path.string());
        return Wavefunction(fchk);
    }

    const std::string method = "b3lyp";
    const std::string basis_name = "6-31G**";
    occ::io::OccInput input;
    input.method.name = method;
    input.basis.name = basis_name;
    input.geometry.set_molecule(mol);
    auto wfn = occ::main::single_point_calculation(input);

    occ::io::FchkWriter fchk(fchk_path.string());
    fchk.set_title(fmt::format("{} {}/{} generated by occ-ng",
                               fchk_path.stem(), method, basis_name));
    fchk.set_method(method);
    fchk.set_basis_name(basis_name);
    wfn.save(fchk);
    fchk.write();
    return wfn;
}

std::vector<Wavefunction>
calculate_wavefunctions(const std::string &basename,
                        const std::vector<Molecule> &molecules) {
    std::vector<Wavefunction> wfns;
    size_t index = 0;
    for (const auto &m : molecules) {
        fmt::print("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", index,
                   "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                       Element(atom.atomic_number).symbol(), atom.x, atom.y,
                       atom.z);
        }
        std::string name = fmt::format("{}_{}", basename, index);
        wfns.emplace_back(calculate_wavefunction(m, name));
        index++;
    }
    return wfns;
}

std::pair<std::vector<SolvatedSurfaceProperties>, std::vector<Wavefunction>>
calculate_solvated_surfaces(const std::string &basename,
                            const std::vector<Molecule> &mols,
                            const std::vector<Wavefunction> &wfns,
                            const std::string &solvent_name) {
    std::vector<SolvatedSurfaceProperties> result;
    using occ::dft::DFT;
    using occ::solvent::SolvationCorrectedProcedure;
    const std::string method = "b3lyp";
    const std::string basis_name = "6-31G**";
    std::vector<Wavefunction> solvated_wfns;
    size_t index = 0;
    for (const auto &wfn : wfns) {
        fs::path props_path(fmt::format("{}_{}_{}_surface.txt", basename, index, solvent_name));
        fs::path fchk_path(fmt::format("{}_{}_{}.fchk", basename, index, solvent_name));
        if (fs::exists(props_path)) {
            fmt::print("Loading surface properties from {}\n", props_path);
            result.emplace_back(SolvatedSurfaceProperties::load(props_path.string()));

            if(fs::exists(fchk_path)) {
                fmt::print("Loading solvated wavefunction from {}\n", fchk_path);
                using occ::io::FchkReader;
                FchkReader fchk(fchk_path.string());
                solvated_wfns.emplace_back(Wavefunction(fchk));
            }
            else {
                BasisSet basis(basis_name, wfn.atoms);
                double original_energy = wfn.energy.total;
                fmt::print("Total energy (gas) {:.3f}\n", original_energy);
                basis.set_pure(false);
                fmt::print("Loaded basis set, {} shells, {} basis functions\n",
                           basis.size(), libint2::nbf(basis));
                occ::dft::DFT ks(method, basis, wfn.atoms, SpinorbitalKind::Restricted);
                SolvationCorrectedProcedure<DFT> proc_solv(ks, solvent_name);
                SCF<SolvationCorrectedProcedure<DFT>, SpinorbitalKind::Restricted> scf(
                    proc_solv);
                scf.start_incremental_F_threshold = 0.0;
                scf.set_charge_multiplicity(0, 1);
                double e = scf.compute_scf_energy();
                Wavefunction wfn = scf.wavefunction();
                occ::io::FchkWriter fchk(fchk_path.string());
                fchk.set_title(fmt::format("{} {}/{} solvent={} generated by occ-ng",
                                           fchk_path.stem(), method, basis_name, solvent_name));
                fchk.set_method(method);
                fchk.set_basis_name(basis_name);
                wfn.save(fchk);
                fchk.write();
                solvated_wfns.push_back(wfn);
            }
        } else {
            BasisSet basis(basis_name, wfn.atoms);
            double original_energy = wfn.energy.total;
            fmt::print("Total energy (gas) {:.3f}\n", original_energy);
            basis.set_pure(false);
            fmt::print("Loaded basis set, {} shells, {} basis functions\n",
                       basis.size(), libint2::nbf(basis));
            occ::dft::DFT ks(method, basis, wfn.atoms, SpinorbitalKind::Restricted);
            SolvationCorrectedProcedure<DFT> proc_solv(ks, solvent_name);
            SCF<SolvationCorrectedProcedure<DFT>, SpinorbitalKind::Restricted> scf(
                proc_solv);
            scf.start_incremental_F_threshold = 0.0;
            scf.set_charge_multiplicity(0, 1);
            double e = scf.compute_scf_energy();
            Wavefunction wfn = scf.wavefunction();
            occ::io::FchkWriter fchk(fchk_path.string());
            fchk.set_title(fmt::format("{} {}/{} solvent={} generated by occ-ng",
                                       fchk_path.stem(), method, basis_name, solvent_name));
            fchk.set_method(method);
            fchk.set_basis_name(basis_name);
            wfn.save(fchk);
            fchk.write();
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
            auto elec = proc_solv.surface_electronic_energy_elements(
                SpinorbitalKind::Restricted, scf.mo);
            auto pol = proc_solv.surface_polarization_energy_elements();
            props.e_coulomb = nuc + elec + pol;
            occ::log::debug("sum e_nuc {:12.6f}\n", nuc.array().sum());
            occ::log::debug("sum e_ele {:12.6f}\n", elec.array().sum());
            occ::log::debug("sum e_pol {:12.6f}\n", pol.array().sum());
            occ::log::debug("sum e_cds {:12.6f}\n", props.e_cds.array().sum());
            double esolv = nuc.array().sum() + elec.array().sum() +
                           pol.array().sum() + props.e_cds.array().sum();

            // dG_gas
            const auto& mol = mols[index];
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
                - 2 * RT ) / occ::units::AU_TO_KJ_PER_MOL;
            props.dg_gas = 
                // Gr + Gt contribution from gas
                (Gt + Gr) / occ::units::AU_TO_KJ_PER_MOL;

            props.dg_ele = e - original_energy - esolv;
            occ::log::debug("total e_solv {:12.6f} ({:.3f} kJ/mol)\n", esolv,
                            esolv * occ::units::AU_TO_KJ_PER_MOL);

            fmt::print("SCF difference         (au)       {: 9.3f}\n", e - original_energy);
            fmt::print("SCF difference         (kJ/mol)   {: 9.3f}\n", occ::units::AU_TO_KJ_PER_MOL * (e - original_energy));
            fmt::print("total E solv (surface) (kj/mol)   {: 9.3f}\n", esolv * occ::units::AU_TO_KJ_PER_MOL);
            fmt::print("orbitals E_solv        (kj/mol)   {: 9.3f}\n", props.dg_ele * occ::units::AU_TO_KJ_PER_MOL);
            fmt::print("CDS E_solv   (surface) (kj/mol)   {: 9.3f}\n", props.e_cds.array().sum() * occ::units::AU_TO_KJ_PER_MOL);
            fmt::print("nuc E_solv   (surface) (kj/mol)   {: 9.3f}\n", nuc.array().sum() * occ::units::AU_TO_KJ_PER_MOL);
            fmt::print("pol E_solv   (surface) (kj/mol)   {: 9.3f}\n", pol.array().sum() * occ::units::AU_TO_KJ_PER_MOL);
            fmt::print("ele E_solv   (surface) (kj/mol)   {: 9.3f}\n", elec.array().sum() * occ::units::AU_TO_KJ_PER_MOL);

            props.esolv = e - original_energy;
            props.e_ele =
                (props.dg_ele / coul_areas.array().sum()) * coul_areas.array();
            props.save(props_path.string());
            result.push_back(props);
        }
        index++;
    }
    return {result, solvated_wfns};
}

void compute_monomer_energies(std::vector<Wavefunction> &wfns) {
    size_t complete = 0;
    for (auto &wfn : wfns) {
        fmt::print("Calculating unique monomer energies for molecule {}\n",
                   complete, wfns.size());
        std::cout << std::flush;
        HartreeFock hf(wfn.atoms, wfn.basis);
	occ::Mat schwarz = hf.compute_schwarz_ints();
        occ::interaction::compute_ce_model_energies(wfn, hf, 1e-8, schwarz);
        complete++;
    }
    fmt::print("Finished calculating {} unique monomer energies\n", complete);
}

void write_xyz_neighbors(const std::string &filename, const std::vector<Dimer> &neighbors) {
    auto neigh = fmt::output_file(filename,
        fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);

    size_t natom = std::accumulate(neighbors.begin(), neighbors.end(), 0,
                                   [](size_t a, const auto &dimer) {
                                       return a + dimer.b().size();
                                   });

    neigh.print("{}\nel x y z idx\n", natom);

    size_t j = 0;
    for (const auto &dimer : neighbors) {
        auto pos = dimer.b().positions();
        auto els = dimer.b().elements();
        for (size_t a = 0; a < dimer.b().size(); a++) {
            neigh.print("{:.3s} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                        els[a].symbol(), pos(0, a), pos(1, a),
                        pos(2, a), j);
        }
        j++;
    }
}

std::pair<occ::IVec, occ::Mat3N>
environment(const std::vector<Dimer> &neighbors) {
    size_t num_atoms = 0;
    for (const auto &n : neighbors) {
        num_atoms += n.b().size();
    }

    occ::IVec mol_idx(num_atoms);
    occ::Mat3N positions(3, num_atoms);
    size_t current_idx = 0;
    size_t i = 0;
    for (const auto &n : neighbors) {
        size_t N = n.b().size();
        mol_idx.block(current_idx, 0, N, 1).array() = i;
        positions.block(0, current_idx, 3, N) =
            n.b().positions() / BOHR_TO_ANGSTROM;
        current_idx += N;
        i++;
    }
    return {mol_idx, positions};
}

struct SolventNeighborContribution {
    struct AsymPair {
        double ab{0.0}, ba{0.0};
        inline double total() const { return ab + ba; }
        void assign(AsymPair &other) {
            other.ba = ab;
            ba = other.ab;
        }
    };

    inline double total() const { 
        return coulomb.total()
             + cds.total();
    }

    AsymPair coulomb;
    AsymPair cds;
    AsymPair cds_area;
    AsymPair coulomb_area;
    bool neighbor_set{false};

    void assign(SolventNeighborContribution &other) {
        coulomb.assign(other.coulomb);
        cds.assign(other.cds);
        coulomb_area.assign(other.coulomb_area);
        cds_area.assign(other.cds_area);
    }
};

std::vector<SolventNeighborContribution> compute_solvation_energy_breakdown(
    const Crystal &crystal, const std::string &mol_name,
    const SolvatedSurfaceProperties &surface,
    const std::vector<Dimer> &neighbors, const std::string &solvent) {
    using occ::units::angstroms;
    std::vector<SolventNeighborContribution> energy_contribution(
        neighbors.size());

    occ::Mat3N neigh_pos;
    occ::IVec mol_idx;
    occ::IVec neighbor_idx_coul(surface.coulomb_pos.cols());
    occ::IVec neighbor_idx_cds(surface.cds_pos.cols());
    std::tie(mol_idx, neigh_pos) = environment(neighbors);

    auto cfile =
        fmt::output_file(fmt::format("{}_coulomb.txt", mol_name),
                         fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    cfile.print("{}\nx y z e neighbor\n", neighbor_idx_coul.rows());
    // coulomb breakdown
    for (size_t i = 0; i < neighbor_idx_coul.rows(); i++) {
        occ::Vec3 x = surface.coulomb_pos.col(i);
        Eigen::Index idx = 0;
        double r =
            (neigh_pos.colwise() - x).colwise().squaredNorm().minCoeff(&idx);
        auto m_idx = mol_idx(idx);
        auto &contrib = energy_contribution[m_idx];
        contrib.coulomb.ab +=
            surface.e_coulomb(i) + surface.e_ele(i);
        neighbor_idx_coul(i) = m_idx;
        contrib.coulomb_area.ab += surface.a_coulomb(i);
        cfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                    angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                    surface.e_coulomb(i), mol_idx(idx));
    }

    auto cdsfile =
        fmt::output_file(fmt::format("{}_cds.txt", mol_name),
                         fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    cdsfile.print("{}\nx y z e neighbor\n", neighbor_idx_cds.rows());
    // cds breakdown
    for (size_t i = 0; i < neighbor_idx_cds.rows(); i++) {
        occ::Vec3 x = surface.cds_pos.col(i);
        Eigen::Index idx = 0;
        double r =
            (neigh_pos.colwise() - x).colwise().squaredNorm().minCoeff(&idx);
        auto m_idx = mol_idx(idx);
        auto &contrib = energy_contribution[m_idx];
        contrib.cds.ab += surface.e_cds(i);
        contrib.cds_area.ab += surface.a_cds(i);
        neighbor_idx_cds(i) = m_idx;
        cdsfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                      angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                      surface.e_cds(i), mol_idx(idx));
    }

    // found A -> B contribution, now find B -> A
    for (int i = 0; i < neighbors.size(); i++) {
        auto& ci = energy_contribution[i];
        if(ci.neighbor_set) continue;
        const auto &d1 = neighbors[i];

        for (int j = i; j < neighbors.size(); j++) {
            auto& cj = energy_contribution[j];
            if(cj.neighbor_set) continue;
            const auto &d2 = neighbors[j];
            if (d1.equivalent_in_opposite_frame(d2)) {
                ci.neighbor_set = true;
                cj.neighbor_set = true;
                fmt::print("Interaction paired {}<->{}\n", i, j);
                ci.assign(cj);
                continue;
            }
        }
    }
    return energy_contribution;
}
struct AssignedEnergy {
    bool is_nn{true};
    double energy{0.0};
};

std::vector<AssignedEnergy> assign_interaction_terms_to_nearest_neighbours(
    int i, const std::vector<SolventNeighborContribution> &solv,
    const occ::crystal::CrystalDimers &crystal_dimers,
    const std::vector<CEModelInteraction::EnergyComponents> dimer_energies) {
    std::vector<AssignedEnergy> crystal_contributions(solv.size());
    double total_taken{0.0};
    const auto &n = crystal_dimers.molecule_neighbors[i];
    for (size_t k1 = 0; k1 < solv.size(); k1++) {
        const auto& s1 = solv[k1];
        if ((abs(s1.coulomb.total()) != 0.0 || abs(s1.cds.total()) != 0.0))
            continue;
        crystal_contributions[k1].is_nn = false;
        auto v = n[k1].v_ab().normalized();
        auto unique_dimer_idx = crystal_dimers.unique_dimer_idx[i][k1];
        total_taken += dimer_energies[unique_dimer_idx].total_kjmol();
        double total_dp = 0.0;
        for (size_t k2 = 0; k2 < solv.size(); k2++) {
            const auto& s2 = solv[k2];
            if (abs(s2.coulomb.total()) == 0.0 || abs(s2.cds.total()) == 0.0)
                continue;
            if (k1 == k2)
                continue;
            auto v2 = n[k2].v_ab().normalized();
            double dp = v.dot(v2);
            if (dp <= 0.0)
                continue;
            total_dp += dp;
        }
        for (size_t k2 = 0; k2 < solv.size(); k2++) {
            const auto& s2 = solv[k2];
            if (abs(s2.coulomb.total()) == 0.0 || abs(s2.cds.total()) == 0.0)
                continue;
            if (k1 == k2)
                continue;
            auto v2 = n[k2].v_ab().normalized();
            double dp = v.dot(v2);
            if (dp <= 0.0)
                continue;
            crystal_contributions[k2].is_nn = true;
            crystal_contributions[k2].energy +=
                (dp / total_dp) *
                dimer_energies[unique_dimer_idx].total_kjmol();
        }
    }
    double total_reassigned{0.0};
    for (size_t k1 = 0; k1 < solv.size(); k1++) {
        if (!crystal_contributions[k1].is_nn)
            continue;
        fmt::print("{}: {:.3f}\n", k1, crystal_contributions[k1].energy);
        total_reassigned += crystal_contributions[k1].energy;
    }
    occ::log::debug("Total taken from non-nearest neighbors: {:.3f} kJ/mol\n",
               total_taken);
    occ::log::debug("Total assigned to nearest neighbors: {:.3f} kJ/mol\n",
               total_reassigned);
    return crystal_contributions;
}

int main(int argc, char **argv) {
    cxxopts::Options options(
        "occ-interactions",
        "Interactions of molecules with neighbours in a crystal");
    double radius = 0.0, cg_radius = 0.0;
    using occ::parallel::nthreads;
    std::string cif_filename{""};
    std::string solvent{"water"};
    std::string wfn_choice{"gas"};
    // clang-format off
    options.add_options()
        ("h,help", "Print help")
        ("i,input", "Input CIF", cxxopts::value<std::string>(cif_filename))
        ("t,threads", "Number of threads", 
         cxxopts::value<int>(nthreads)->default_value("1"))
        ("r,radius", "maximum radius (angstroms) for neighbours",
         cxxopts::value<double>(radius)->default_value("3.8"))
        ("c,cg-radius", "maximum radius (angstroms) for nearest neighbours in cg file (must be <= radius)",
         cxxopts::value<double>(cg_radius)->default_value("3.8"))
        ("s,solvent", "Solvent name", cxxopts::value<std::string>(solvent))
        ("w,wavefunction-choice", "Choice of wavefunctions",
         cxxopts::value<std::string>(wfn_choice));
    // clang-format on
    options.parse_positional({"input"});

    occ::log::set_level(occ::log::level::info);
    spdlog::set_level(spdlog::level::info);
    libint2::Shell::do_enforce_unit_normalization(true);
    libint2::initialize();

    occ::timing::StopWatch global_timer;
    global_timer.start();

    try {
        options.parse(argc, argv);
    } catch (const std::runtime_error &err) {
        occ::log::error("error when parsing command line arguments: {}",
                        err.what());
        fmt::print("{}", options.help());
        exit(1);
    }

    fmt::print("Parallelized over {} threads & {} Eigen threads\n", nthreads,
               Eigen::nbThreads());

    const std::string error_format =
        "Exception:\n    {}\nTerminating program.\n";
    try {
        occ::timing::StopWatch sw;
        sw.start();
        std::string basename = fs::path(cif_filename).stem();
        Crystal c_symm = read_crystal(cif_filename);
        sw.stop();
        double tprev = sw.read();
        fmt::print("Loaded crystal from {} in {:.6f} seconds\n", cif_filename,
                   tprev);

        sw.start();
        auto molecules = c_symm.symmetry_unique_molecules();
        sw.stop();

        fmt::print(
            "Found {} symmetry unique molecules in {} in {:.6f} seconds\n",
            molecules.size(), cif_filename, sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        auto wfns = calculate_wavefunctions(basename, molecules);
        sw.stop();

        fmt::print("Gas phase wavefunctions took {:.6f} seconds\n",
                   sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        std::vector<Wavefunction> solvated_wavefunctions;
        std::vector<SolvatedSurfaceProperties> surfaces;
        std::tie(surfaces, solvated_wavefunctions) =
            calculate_solvated_surfaces(basename, molecules, wfns, solvent);
        sw.stop();
        fmt::print("Solution phase wavefunctions took {:.6f} seconds\n",
                   sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        compute_monomer_energies(wfns);
        compute_monomer_energies(solvated_wavefunctions);
        sw.stop();
        fmt::print("Computing monomer energies took {:.6f} seconds\n",
                   sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        auto crystal_dimers = c_symm.symmetry_unique_dimers(radius);
        sw.stop();

        const auto &dimers = crystal_dimers.unique_dimers;
        fmt::print("Found {} symmetry unique dimers in {:.6f} seconds\n",
                   dimers.size(), sw.read() - tprev);

        if (dimers.size() < 1) {
            fmt::print("No dimers found using neighbour radius {:.3f}\n",
                       radius);
            exit(0);
        }

        const std::string row_fmt_string =
            "{:>7.2f} {:>7.2f} {:>20s} {: 7.2f} "
            "{: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f}\n";


        fmt::print(
            "Calculating unique pair interactions using {} wavefunctions\n",
            wfn_choice);
        const auto &wfns_a = [&]() {
            if (wfn_choice == "gas")
                return wfns;
            else
                return solvated_wavefunctions;
        }();
        const auto &wfns_b = [&]() {
            if (wfn_choice == "solvent")
                return solvated_wavefunctions;
            else
                return wfns;
        }();


        using occ::main::ce_model_energies;
        auto dimer_energies = ce_model_energies(c_symm, dimers, wfns_a, wfns_b, basename);

        const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
        std::vector<std::vector<SolventNeighborContribution>>
            solvation_breakdowns;
        std::vector<std::vector<InteractionE>> interaction_energies_vec(mol_neighbors.size());
        double dG_solubility{0.0};
        for (size_t i = 0; i < mol_neighbors.size(); i++) {
            const auto &n = mol_neighbors[i];
            std::string molname = fmt::format("{}_{}_{}", basename, i, solvent);
            auto solv = compute_solvation_energy_breakdown(
                c_symm, molname, surfaces[i], n, solvent);
            solvation_breakdowns.push_back(solv);
            auto crystal_contributions =
                assign_interaction_terms_to_nearest_neighbours(
                    i, solv, crystal_dimers, dimer_energies);
            auto& interactions = interaction_energies_vec[i];
            interactions.reserve(n.size());
            double Gr = molecules[i].rotational_free_energy(298);
            occ::core::MolecularPointGroup pg(molecules[i]);
            fmt::print("Molecule {} point group = {}, symmetry number = {}\n",
                       i, pg.point_group_string(), pg.symmetry_number());
            double Gt = molecules[i].translational_free_energy(298);
            double molar_mass = molecules[i].molar_mass();

            fmt::print("Neighbors for asymmetric molecule {}\n", i);

            fmt::print("{:>7s} {:>7s} {:>20s} "
                       "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}\n",
                       "Rn", "Rc", "Symop", "E_crys", "A_coul", "E_scoul", "A_cds", "E_scds", "E_gas", "E_nn", "E_int");
            fmt::print("============================="
                       "============================="
                       "=============================\n");

            size_t j = 0;
            CEModelInteraction::EnergyComponents total;

            // write neighbors file for molecule i
            {
                std::string neighbors_filename = fmt::format("{}_{}_neighbors.xyz", basename, i);
                write_xyz_neighbors(neighbors_filename, n);
            }

            size_t num_neighbors = std::accumulate(crystal_contributions.begin(), crystal_contributions.end(), 0,
                    [](size_t a, const AssignedEnergy &x) { return x.is_nn ? a + 1 : a; });

            // TODO adjust for co-crystals!
            double gas_term_per_interaction = 2 * surfaces[i].dg_gas / num_neighbors;
            double correction_term_per_interaction = 2 * surfaces[i].dg_correction / num_neighbors;

            double total_interaction_energy{0.0};

            for (const auto &dimer : n) {
                auto s_ab = c_symm.dimer_symmetry_string(dimer);
                size_t idx = crystal_dimers.unique_dimer_idx[i][j];
                double rn = dimer.nearest_distance();
                double rc = dimer.centroid_distance();
                const auto &e =
                    dimer_energies[crystal_dimers.unique_dimer_idx[i][j]];
                double ecoul = e.coulomb_kjmol(), erep = e.exchange_kjmol(),
                       epol = e.polarization_kjmol(),
                       edisp = e.dispersion_kjmol(), etot = e.total_kjmol();
                total.coulomb += ecoul;
                total.exchange_repulsion += erep;
                total.polarization += epol;
                total.dispersion += edisp;
                total.total += etot;

                double e_int = etot;
                double solv_cont = solv[j].total() * AU_TO_KJ_PER_MOL;
                double gas_term = crystal_contributions[j].is_nn ? gas_term_per_interaction * AU_TO_KJ_PER_MOL : 0.0;
                double correction_term = crystal_contributions[j].is_nn ? correction_term_per_interaction * AU_TO_KJ_PER_MOL : 0.0;

                e_int = correction_term + gas_term + solv_cont - etot - crystal_contributions[j].energy;
                interactions.push_back({e_int, e_int - gas_term});
                auto &sj = solv[j];

                fmt::print(
                    row_fmt_string, rn, rc, s_ab,
                    etot,
                    sj.coulomb_area.total(),
                    sj.coulomb.total() * AU_TO_KJ_PER_MOL,
                    sj.cds_area.total(),
                    sj.cds.total() * AU_TO_KJ_PER_MOL,
                    gas_term,
                    crystal_contributions[j].energy,
                    e_int);
                if(crystal_contributions[j].is_nn) total_interaction_energy += e_int;
                j++;
            }
            constexpr double R = 8.31446261815324;
            constexpr double RT = 298 * R / 1000;
            fmt::print("\nFree energy estimates at T = 298 K, P = 1 atm., "
                       "units: kJ/mol\n");
            fmt::print(
                "-------------------------------------------------------\n");
            fmt::print(
                "lattice energy (crystal)             {: 9.3f}  (E_lat)\n",
                0.5 * total.total);
            Gr += RT * std::log(pg.symmetry_number());
            fmt::print(
                "rotational free energy (molecule)    {: 9.3f}  (E_rot)\n", Gr);
            fmt::print(
                "translational free energy (molecule) {: 9.3f}  (E_trans)\n",
                Gt);
            // includes concentration shift
            double dG_solv = 
                surfaces[i].esolv * occ::units::AU_TO_KJ_PER_MOL + 
                1.89 / occ::units::KJ_TO_KCAL;
            fmt::print(
                "solvation free energy (molecule)     {: 9.3f}  (E_solv)\n",
                dG_solv);
            double dH_sub = -0.5 * total.total - 2 * RT;
            fmt::print("\u0394H sublimation                       {: 9.3f}\n",
                       dH_sub); 
            double dS_sub = Gr + Gt;
            fmt::print("\u0394S sublimation                       {: 9.3f}\n",
                       dS_sub);
            double dG_sub = dH_sub + dS_sub;
            fmt::print("\u0394G sublimation                       {: 9.3f}\n",
                       dG_sub);
            dG_solubility = dG_solv + dG_sub;
            fmt::print("\u0394G solution                          {: 9.3f}\n",
                       dG_solubility);
            double equilibrium_constant = std::exp(-dG_solubility / RT);
            fmt::print("equilibrium_constant                 {: 9.2e}\n",
                       equilibrium_constant);
            fmt::print("log S                                {: 9.3f}\n",
                    std::log10(equilibrium_constant));
            fmt::print("solubility (g/L)                     {: 9.2e}\n",
                       equilibrium_constant * molar_mass * 1000);
            fmt::print("Total E_int (~ 2 x \u0394G solution)      {: 9.3f}\n", total_interaction_energy);
        }

        auto uc_dimers = c_symm.unit_cell_dimers(cg_radius);
        auto &uc_neighbors = uc_dimers.molecule_neighbors;

        // write CG structure file
        {
            std::string cg_structure_filename = fmt::format("{}_cg.txt", basename);
            fmt::print("Writing crystalgrower structure file to '{}'\n", cg_structure_filename);
            occ::io::crystalgrower::StructureWriter cg_structure_writer(cg_structure_filename);
            cg_structure_writer.write(c_symm, uc_dimers);
        }

        // map interactions surrounding UC molecules to symmetry unique interactions
        for (size_t i = 0; i < uc_neighbors.size(); i++) {
            const auto &m = c_symm.unit_cell_molecules()[i];
            size_t asym_idx = m.asymmetric_molecule_idx();
            const auto &m_asym = c_symm.symmetry_unique_molecules()[asym_idx];
            auto &n = uc_neighbors[i];
            fmt::print("Molecule {} has {} neighbours within {:.3f}\n", i, n.size(), cg_radius);
            occ::log::debug("uc = {} asym = {}\n", i, asym_idx);
            int s_int = m.asymmetric_unit_symop()(0);

            SymmetryOperation symop(s_int);

            const auto &rotation = symop.rotation();
            occ::log::debug("Asymmetric unit symop: {} (has handedness change: {})\n",
                            symop.to_string(),
                            rotation.determinant() < 0);

            occ::log::debug("Neighbors for unit cell molecule {} ({})\n", i,
                            n.size());

            occ::log::debug("{:<7s} {:>7s} {:>10s} {:>7s} {:>7s}\n", "N", "b", "Tb",
                            "E_int", "R");

            size_t j = 0;
            const auto &n_asym = mol_neighbors[asym_idx];
            const auto &interaction_energies = interaction_energies_vec[asym_idx];

            for (auto &dimer : n) {

                auto shift_b = dimer.b().cell_shift();
                auto idx_b = dimer.b().unit_cell_molecule_idx();

                size_t idx{0};
                bool match_type{false};
                for (idx = 0; idx < n_asym.size(); idx++) {
                    if (dimer.equivalent(n_asym[idx])) {
                        break;
                    }
                    if (dimer.equivalent_under_rotation(n_asym[idx], rotation)) {
                        match_type = true;
                        break;
                    }
                }
                if (idx >= n_asym.size()) {
                    throw std::runtime_error(
                        fmt::format("No matching interaction found for uc_mol "
                                    "= {}, dimer = {}\n",
                                    i, j));
                }
                double rn = dimer.nearest_distance();
                double rc = dimer.centroid_distance();

                double e_int = interaction_energies[idx].fictitious * occ::units::KJ_TO_KCAL;

                dimer.set_interaction_energy(e_int);
                occ::log::debug(
                    "{:<7d} {:>7d} {:>10s} {:7.2f} {:7.3f} {}\n", j, idx_b,
                    fmt::format("{},{},{}", shift_b[0], shift_b[1], shift_b[2]),
                    e_int, rc, match_type);
                j++;
            }
        }

        // write CG net file
        {
            std::string cg_net_filename = fmt::format("{}_net.txt", basename);
            fmt::print("Writing crystalgrower net file to '{}'\n", cg_net_filename);
            occ::io::crystalgrower::NetWriter cg_net_writer(cg_net_filename);
            cg_net_writer.write(c_symm, uc_dimers);
        }

    } catch (const char *ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::string &ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::exception &ex) {
        fmt::print(error_format, ex.what());
        return 1;
    } catch (...) {
        fmt::print("Exception:\n- Unknown...\n");
        return 1;
    }

    global_timer.stop();
    fmt::print("Program exiting successfully after {:.6f} seconds\n",
               global_timer.read());

    return 0;
}
