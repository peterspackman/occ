#pragma once
#include <occ/solvent/surface.h>
#include <occ/solvent/cosmo.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/energy_components.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>


namespace occ::solvent {

using occ::qm::SpinorbitalKind;

template<typename Proc>
class SolvationCorrectedProcedure
{
public:
    SolvationCorrectedProcedure(Proc &proc) : m_atoms(proc.atoms()), m_proc(proc), m_solv(78.39)
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
        occ::Vec radii = occ::solvent::cosmo::solvation_radii(nums);
        m_surface = occ::solvent::surface::solvent_surface(radii, nums, pos);

        m_point_charges.reserve(m_surface.vertices.cols());
        m_qn = m_proc.nuclear_electric_potential_contribution(m_surface.vertices);

        for(int i = 0; i < m_surface.vertices.cols(); i++)
        {
            const auto& pt = m_surface.vertices.col(i);
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
        occ::Vec v = - (m_qn + m_proc.electronic_electric_potential_contribution(kind, D, m_surface.vertices));
        auto result = m_solv(m_surface.vertices, m_surface.areas, v);

        std::ofstream pts("points.xyz");
        fmt::print(pts, "{}\n", v.rows());

        m_nuclear_solvation_energy = m_qn.dot(result.converged);
        m_solvation_energy = m_nuclear_solvation_energy - result.energy;
        m_X = m_proc.compute_point_charge_interaction_matrix(m_point_charges);
        double e_X = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(D, m_X);

        fmt::print(pts, "esurf={:12.5f} enuc={:12.5f} eele={:12.5f}\n", result.energy, m_nuclear_solvation_energy, e_X);

        for(int i = 0; i < m_point_charges.size(); i++)
        {
            m_point_charges[i].first = result.converged(i);
            fmt::print(pts, "{:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}\n",
                       m_surface.vertices(0, i),
                       m_surface.vertices(1, i),
                       m_surface.vertices(2, i),
                       v(i),
                       result.converged(i)
                       );
        }

        occ::timing::stop(occ::timing::category::solvent);
        switch(kind)
        {
            case SpinorbitalKind::Restricted:
            {
                H += m_X;
                break;
            }
            case SpinorbitalKind::Unrestricted:
            {
                H.alpha() += m_X;
                H.beta() += m_X;
            }
            case SpinorbitalKind::General:
            {
                H.alpha_alpha() += m_X;
                H.alpha_beta() += m_X;
                H.beta_alpha() += m_X;
                H.beta_beta() += m_X;

            }
        }
    }


    MatRM compute_fock(SpinorbitalKind kind, const MatRM &D,
                    double precision = std::numeric_limits<double>::epsilon(),
                    const MatRM &Schwarz = MatRM()) const
    {
        return m_proc.compute_fock(kind, D, precision, Schwarz);
    }


private:
    const std::vector<libint2::Atom> &m_atoms;
    Proc &m_proc;
    COSMO m_solv;
    occ::solvent::surface::Surface m_surface;
    std::vector<std::pair<double, std::array<double, 3>>> m_point_charges;
    double m_solvation_energy{0.0}, m_nuclear_solvation_energy{0.0};
    occ::MatRM m_X;
    occ::Vec m_qn;
};

}
