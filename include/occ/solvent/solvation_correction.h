#pragma once
#include <occ/qm/spinorbital.h>
#include <occ/qm/energy_components.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <libint2/atom.h>

#ifdef USING_PCMSolver
#include "PCMSolver/pcmsolver.h"
#else
#include <occ/solvent/cosmo.h>
#endif


namespace occ::solvent {

using occ::qm::SpinorbitalKind;

class ContinuumSolvationModel
{
public:
    ContinuumSolvationModel(const std::vector<libint2::Atom>&, const std::string& solvent = "water");
    ~ContinuumSolvationModel();

    void set_solvent(const std::string&);
    const std::string& solvent() const { return m_solvent_name; }

    const Mat3N &nuclear_positions() const { return m_nuclear_positions; } 
    const Mat3N &surface_positions() const { return m_surface_positions; } 
    const Vec &surface_areas() const { return m_surface_areas; } 
    const Vec &nuclear_charges() const { return m_nuclear_charges; } 
    size_t num_surface_points() const { return m_surface_areas.rows(); }
    void set_surface_potential(const Vec&);
    const Vec& apparent_surface_charge();

    double surface_polarization_energy();
    double smd_cds_energy() const;

private:
    std::string m_solvent_name{"water"};
    Mat3N m_nuclear_positions;
    Vec m_nuclear_charges;
    Mat3N m_surface_positions;
    Vec m_surface_areas;
    Vec m_surface_potential;
    Vec m_asc;
    IVec m_surface_atoms;
    bool m_asc_needs_update{true};
#ifdef USING_PCMSolver
    const char * m_surface_potential_label = "OCC_TOTAL_MEP";
    const char * m_asc_label = "OCC_TOTAL_ASC";
    pcmsolver_context_t *m_pcm_context;
#else
    COSMO m_cosmo;
#endif
};


template<typename Proc>
class SolvationCorrectedProcedure
{
public:
    SolvationCorrectedProcedure(Proc &proc) : m_atoms(proc.atoms()), m_proc(proc),
        m_solvation_model(proc.atoms())
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

    void update_scf_energy(occ::qm::EnergyComponents &energy) const
    { 
        m_proc.update_scf_energy(energy);
        energy.solvation = m_solvation_energy;
    }

    auto compute_kinetic_matrix() { return m_proc.compute_kinetic_matrix(); }

    auto compute_overlap_matrix() { return m_proc.compute_overlap_matrix(); }

    auto compute_nuclear_attraction_matrix() { return m_proc.compute_nuclear_attraction_matrix(); }
    
    auto compute_schwarz_ints() { return m_proc.compute_schwarz_ints(); }

    void update_core_hamiltonian(SpinorbitalKind kind, const occ::MatRM &D, occ::MatRM &H)
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
        double cds_energy = m_solvation_model.smd_cds_energy();
        m_nuclear_solvation_energy = m_qn.dot(asc);
        m_solvation_energy = m_nuclear_solvation_energy - surface_energy + cds_energy;
        m_X = m_proc.compute_point_charge_interaction_matrix(m_point_charges);
        double e_X = 0.0;

        switch(kind)
        {
            case SpinorbitalKind::Restricted:
            {
                e_X = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(D, H);
                H += m_X;
                e_X = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(D, H) - e_X;
                break;
            }
            case SpinorbitalKind::Unrestricted:
            {
                e_X = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(D, H);
                H.alpha() += m_X;
                H.beta() += m_X;
                e_X = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(D, H) - e_X;
                break;
            }
            case SpinorbitalKind::General:
            {
                e_X = 2 * occ::qm::expectation<SpinorbitalKind::General>(D, H);
                H.alpha_alpha() += m_X;
                H.beta_beta() += m_X;
                e_X = 2 * occ::qm::expectation<SpinorbitalKind::General>(D, H) - e_X;
                break;
            }
        }
        occ::timing::stop(occ::timing::category::solvent);
    }


    MatRM compute_fock(SpinorbitalKind kind, const MatRM &D,
                    double precision = std::numeric_limits<double>::epsilon(),
                    const MatRM &Schwarz = MatRM()) const
    {
        return m_proc.compute_fock(kind, D, precision, Schwarz);
    }

    void set_solvent(const std::string &solvent)
    {
        m_solvation_model.set_solvent(solvent);
    }


private:
    const std::vector<libint2::Atom> &m_atoms;
    Proc &m_proc;
    ContinuumSolvationModel m_solvation_model;
    std::vector<std::pair<double, std::array<double, 3>>> m_point_charges;
    double m_solvation_energy{0.0}, m_nuclear_solvation_energy{0.0};
    occ::MatRM m_X;
    occ::Vec m_qn;
};

}