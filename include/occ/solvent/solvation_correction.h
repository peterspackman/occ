#pragma once
#include <occ/qm/spinorbital.h>
#include <occ/core/energy_components.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/solvent/parameters.h>
#include <libint2/atom.h>

#include <occ/solvent/cosmo.h>


namespace occ::solvent {

using occ::qm::SpinorbitalKind;

class ContinuumSolvationModel
{
public:
    ContinuumSolvationModel(const std::vector<libint2::Atom>&, const std::string& solvent = "water");

    void set_solvent(const std::string&);
    const std::string& solvent() const { return m_solvent_name; }

    const Mat3N &nuclear_positions() const { return m_nuclear_positions; } 
    const Mat3N &surface_positions() const { return m_surface_positions_coulomb; } 
    const Vec &surface_areas() const { return m_surface_areas_coulomb; } 
    const Vec &nuclear_charges() const { return m_nuclear_charges; } 
    size_t num_surface_points() const { return m_surface_areas_coulomb.rows(); }
    void set_surface_potential(const Vec&);
    const Vec& apparent_surface_charge();

    double surface_polarization_energy();
    double smd_cds_energy() const;
    Vec smd_cds_energy_elements() const;
    Vec surface_polarization_energy_elements() const;
    void write_surface_file(const std::string& filename);


private:
    std::string m_filename{"solvent.xyz"};
    std::string m_solvent_name;
    Mat3N m_nuclear_positions;
    Vec m_nuclear_charges;
    Mat3N m_surface_positions_coulomb, m_surface_positions_cds;
    Vec m_surface_areas_coulomb, m_surface_areas_cds;
    Vec m_surface_potential;
    Vec m_asc;
    IVec m_surface_atoms_coulomb, m_surface_atoms_cds;
    bool m_asc_needs_update{true};
    SMDSolventParameters m_params;

    COSMO m_cosmo;
};


template<typename Proc>
class SolvationCorrectedProcedure
{
public:
    SolvationCorrectedProcedure(Proc &proc, const std::string &solvent = "water") : m_atoms(proc.atoms()), m_proc(proc),
        m_solvation_model(proc.atoms(), solvent)
    {
        occ::Mat3N pos(3, m_atoms.size());
        occ::IVec nums(m_atoms.size());
        for(int i = 0; i < m_atoms.size(); i++)
        {
            pos(0, i) = m_atoms[i].x;
            pos(1, i) = m_atoms[i].y;
            pos(2, i) = m_atoms[i].z;
            nums(i) = m_atoms[i].atomic_number;
        }
        m_qn = m_proc.nuclear_electric_potential_contribution(m_solvation_model.surface_positions());
        m_point_charges.reserve(m_qn.rows());

        for(int i = 0; i < m_solvation_model.num_surface_points(); i++)
        {
            const auto& pt = m_solvation_model.surface_positions().col(i);
            m_point_charges.push_back({0.0, {pt(0), pt(1), pt(2)}});
        }

        m_cds_solvation_energy = m_solvation_model.smd_cds_energy();
    }

    const auto &shellpair_list() const { return m_proc.shellpair_list(); }
    const auto &shellpair_data() const { return m_proc.shellpair_data(); }
    const auto &atoms() const { return m_proc.atoms(); }
    const auto &basis() const { return m_proc.basis(); }

    void set_system_charge(int charge) { m_proc.set_system_charge(charge); }
    int system_charge() const { return m_proc.system_charge(); }
    int num_e() const { return m_proc.num_e(); }

    double two_electron_energy() const { return m_proc.two_electron_energy(); }
    double two_electron_energy_alpha() const { return m_proc.two_electron_energy_alpha(); }
    double two_electron_energy_beta() const { return m_proc.two_electron_energy_beta(); }
    bool usual_scf_energy() const { return m_proc.usual_scf_energy(); }

    double nuclear_repulsion_energy() const { return m_proc.nuclear_repulsion_energy(); }

    void update_scf_energy(occ::core::EnergyComponents &energy, bool incremental) const
    { 

        m_proc.update_scf_energy(energy, incremental);
        if(incremental)
        {
            energy["solvation.electronic"] += m_electronic_solvation_energy;
            energy["solvation.surface"] += m_surface_solvation_energy;
            energy["solvation.nuclear"] += m_nuclear_solvation_energy;
        }
        else
        {
            energy["solvation.electronic"] = m_electronic_solvation_energy;
            energy["solvation.nuclear"] = m_nuclear_solvation_energy;
            energy["solvation.surface"] = m_surface_solvation_energy;
            energy["solvation.CDS"] = m_cds_solvation_energy;
        }
        energy["total"] = energy["electronic"] + energy["nuclear.repulsion"]
            + energy["solvation.nuclear"] + energy["solvation.surface"]
            + energy["solvation.CDS"];
    }

    auto compute_kinetic_matrix() { return m_proc.compute_kinetic_matrix(); }

    auto compute_overlap_matrix() { return m_proc.compute_overlap_matrix(); }

    auto compute_nuclear_attraction_matrix() { return m_proc.compute_nuclear_attraction_matrix(); }
    
    auto compute_schwarz_ints() { return m_proc.compute_schwarz_ints(); }

    void update_core_hamiltonian(SpinorbitalKind kind, const occ::Mat &D, occ::Mat &H)
    {
        occ::timing::start(occ::timing::category::solvent);
        occ::Vec v = (m_qn + m_proc.electronic_electric_potential_contribution(kind, D, m_solvation_model.surface_positions()));
        m_solvation_model.set_surface_potential(v);
        auto asc = m_solvation_model.apparent_surface_charge();
        for(int i = 0; i < m_point_charges.size(); i++)
        {
            m_point_charges[i].first = asc(i);
        }
        double surface_energy = m_solvation_model.surface_polarization_energy();
        // fmt::print("PCM non-electrostatic energy: {:.12f}\n", cds_energy);
        m_nuclear_solvation_energy = m_qn.dot(asc);
        m_surface_solvation_energy = - surface_energy;
        m_electronic_solvation_energy = 0.0;
        // fmt::print("Surface electrostatic energy: {:.12f}\n", surface_energy);
        m_X = m_proc.compute_point_charge_interaction_matrix(m_point_charges);

        switch(kind)
        {
            case SpinorbitalKind::Restricted:
            {
                m_electronic_solvation_energy = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(D, H);
                H += m_X;
                m_electronic_solvation_energy = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(D, H) - m_electronic_solvation_energy;
                break;
            }
            case SpinorbitalKind::Unrestricted:
            {
                m_electronic_solvation_energy = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(D, H);
                H.alpha() += m_X;
                H.beta() += m_X;
                m_electronic_solvation_energy = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(D, H) - m_electronic_solvation_energy;
                break;
            }
            case SpinorbitalKind::General:
            {
                m_electronic_solvation_energy = 2 * occ::qm::expectation<SpinorbitalKind::General>(D, H);
                H.alpha_alpha() += m_X;
                H.beta_beta() += m_X;
                m_electronic_solvation_energy = 2 * occ::qm::expectation<SpinorbitalKind::General>(D, H) - m_electronic_solvation_energy;
                break;
            }
        }
        occ::timing::stop(occ::timing::category::solvent);
    }


    Mat compute_fock(SpinorbitalKind kind, const Mat &D,
                    double precision = std::numeric_limits<double>::epsilon(),
                    const Mat &Schwarz = Mat()) const
    {
        return m_proc.compute_fock(kind, D, precision, Schwarz);
    }

    void set_solvent(const std::string &solvent)
    {
        m_solvation_model.set_solvent(solvent);
    }

    void write_surface_file(const std::string &filename)
    {
        m_solvation_model.write_surface_file(filename);
    }

private:
    const std::vector<libint2::Atom> &m_atoms;
    Proc &m_proc;
    ContinuumSolvationModel m_solvation_model;
    std::vector<std::pair<double, std::array<double, 3>>> m_point_charges;
    double m_electronic_solvation_energy{0.0}, m_nuclear_solvation_energy{0.0},
           m_surface_solvation_energy, m_cds_solvation_energy;
    Mat m_X;
    Vec m_qn;
};

}
