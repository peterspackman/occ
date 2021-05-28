#pragma once
#include <occ/solvent/surface.h>
#include <occ/solvent/cosmo.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/energy_components.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>


namespace occ::solvent {


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

        nwcosmo = occ::Mat(32, 4);
        nwcosmo << 
             0.47489957658797588,      0.47489957658797588,       1.6214085751519076,       3.0254455781422177E-002,
              1.6214085751519074,      0.47489957658797577,      0.47489957658797577,       3.0254455781422614E-002,
             0.47489957658797577,       1.6214085751519074,      0.47489957658797577,       3.0254455781422284E-002,
              1.0132497196747725,       1.0132497196747725,       1.0132497196747725,       3.1860206846178983E-002,
            -0.47489957658797588,      0.47489957658797588,       1.6214085751519076,       3.0254455781422069E-002,
            -0.47489957658797577,       1.6214085751519074,      0.47489957658797577,       3.0254455781422069E-002,
             -1.6214085751519074,      0.47489957658797577,      0.47489957658797577,       3.0254455781422177E-002,
             -1.0132497196747725,       1.0132497196747725,       1.0132497196747725,       3.1860206846179205E-002,
            -0.47489957658797588,     -0.47489957658797588,       1.6214085751519076,       3.0254455781421739E-002,
             -1.6214085751519074,     -0.47489957658797577,      0.47489957658797577,       3.0254455781422069E-002,
            -0.47489957658797577,      -1.6214085751519074,      0.47489957658797577,       3.0254455781422284E-002,
             -1.0132497196747725,      -1.0132497196747725,       1.0132497196747725,       3.1860206846179531E-002,
             0.47489957658797588,     -0.47489957658797588,       1.6214085751519076,       3.0254455781422013E-002,
             0.47489957658797577,      -1.6214085751519074,      0.47489957658797577,       3.0254455781422069E-002,
              1.6214085751519074,     -0.47489957658797577,      0.47489957658797577,       3.0254455781422232E-002,
              1.0132497196747725,      -1.0132497196747725,       1.0132497196747725,       3.1860206846179094E-002,
             0.47489957658797588,      0.47489957658797588,      -1.6214085751519076,       3.0254455781421795E-002,
              1.6214085751519074,      0.47489957658797577,     -0.47489957658797577,       3.0254455781421906E-002,
             0.47489957658797577,       1.6214085751519074,     -0.47489957658797577,       3.0254455781422177E-002,
              1.0132497196747725,       1.0132497196747725,      -1.0132497196747725,       3.1860206846179365E-002,
            -0.47489957658797588,      0.47489957658797588,      -1.6214085751519076,       3.0254455781422069E-002,
            -0.47489957658797577,       1.6214085751519074,     -0.47489957658797577,       3.0254455781422503E-002,
             -1.6214085751519074,      0.47489957658797577,     -0.47489957658797577,       3.0254455781422069E-002,
             -1.0132497196747725,       1.0132497196747725,      -1.0132497196747725,       3.1860206846179316E-002,
            -0.47489957658797588,     -0.47489957658797588,      -1.6214085751519076,       3.0254455781422069E-002,
             -1.6214085751519074,     -0.47489957658797577,     -0.47489957658797577,       3.0254455781422284E-002,
            -0.47489957658797577,      -1.6214085751519074,     -0.47489957658797577,       3.0254455781422232E-002,
             -1.0132497196747725,      -1.0132497196747725,      -1.0132497196747725,       3.1860206846178823E-002,
             0.47489957658797588,     -0.47489957658797588,      -1.6214085751519076,       3.0254455781422069E-002,
             0.47489957658797577,      -1.6214085751519074,     -0.47489957658797577,       3.0254455781422177E-002,
              1.6214085751519074,     -0.47489957658797577,     -0.47489957658797577,       3.0254455781422340E-002,
              1.0132497196747725,      -1.0132497196747725,      -1.0132497196747725,       3.1860206846179094E-002;

        m_surface.vertices = nwcosmo.block(0, 0, 32, 3).transpose() * occ::units::ANGSTROM_TO_BOHR;
        m_surface.areas = occ::Vec(32);
        m_surface.areas.setConstant(1.0);
        m_point_charges.reserve(m_surface.vertices.cols());
        occ::Vec qn = m_proc.nuclear_electric_potential_contribution(m_surface.vertices);
        auto result = m_solv(m_surface.vertices, m_surface.areas, qn);
        m_nuclear_solvation_energy = result.energy;
        fmt::print("Nuclear solvation energy term: {}\n", m_nuclear_solvation_energy);
        for(int i = 0; i < m_surface.vertices.cols(); i++)
        {
            const auto& pt = m_surface.vertices.col(i);
            m_point_charges.push_back({result.converged(i), {pt(0), pt(1), pt(2)}});
        }
        m_h = m_proc.compute_point_charge_interaction_matrix(m_point_charges);
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

    void update_core_hamiltonian(occ::qm::SpinorbitalKind kind, const occ::MatRM &D, occ::MatRM &H)
    {
        occ::timing::start(occ::timing::category::solvent);
        occ::Vec v = m_proc.electronic_electric_potential_contribution(D, m_surface.vertices);
        auto result = m_solv(m_surface.vertices, m_surface.areas, v);

        std::ofstream pts("points.xyz");
        fmt::print(pts, "{}\n\n", v.rows());

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

        m_solvation_energy = result.energy + m_nuclear_solvation_energy;
        fmt::print("Energy {}\n", result.energy);

        m_X = m_proc.compute_point_charge_interaction_matrix(m_point_charges);
        occ::timing::stop(occ::timing::category::solvent);
        H += m_h + m_X;
    }


    MatRM compute_fock(occ::qm::SpinorbitalKind kind, const MatRM &D,
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
    occ::MatRM m_h, m_X;
    occ::Mat nwcosmo;
};

}
