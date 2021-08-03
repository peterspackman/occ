#pragma once
#include <occ/qm/spinorbital.h>
#include <occ/core/energy_components.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/solvent/parameters.h>
#include <occ/core/atom.h>
#include <occ/core/logger.h>
#include <occ/solvent/cosmo.h>


namespace occ::solvent {

using occ::qm::SpinorbitalKind;

class ContinuumSolvationModel
{
public:
    ContinuumSolvationModel(const std::vector<occ::core::Atom>&, const std::string& solvent = "water");

    void set_solvent(const std::string&);
    const std::string& solvent() const { return m_solvent_name; }

    const Mat3N &nuclear_positions() const { return m_nuclear_positions; } 
    const Mat3N &surface_positions_coulomb() const { return m_surface_positions_coulomb; } 
    const Mat3N &surface_positions_cds() const { return m_surface_positions_cds; } 
    const Vec &surface_areas_coulomb() const { return m_surface_areas_coulomb; } 
    const Vec &surface_areas_cds() const { return m_surface_areas_cds; } 
    const Vec &nuclear_charges() const { return m_nuclear_charges; } 
    size_t num_surface_points() const { return m_surface_areas_coulomb.rows(); }
    void set_surface_potential(const Vec&);
    const Vec& apparent_surface_charge();

    double surface_polarization_energy();
    double surface_charge() const { return m_asc.array().sum(); }
    double smd_cds_energy() const;

    Vec surface_cds_energy_elements() const;
    Vec surface_polarization_energy_elements() const;

    template<typename Proc>
    Vec surface_nuclear_energy_elements(const Proc& proc) const
    {
        Vec qn = proc.nuclear_electric_potential_contribution(m_surface_positions_coulomb);
        qn.array() *= m_asc.array();
        return qn;
    }

    template <typename Proc>
    Vec surface_electronic_energy_elements(const SpinorbitalKind kind, const Mat& D, const Proc& p) const
    {
        Vec result(m_surface_areas_coulomb.rows());
        Mat X;
        std::vector<std::pair<double, std::array<double, 3>>> point_charges;
        point_charges.emplace_back(0, std::array<double, 3>{0.0, 0.0, 0.0});
        for(int i = 0; i < m_surface_areas_coulomb.rows(); i++)
        {
            point_charges[0].first = m_asc(i);
            point_charges[0].second[0] = m_surface_positions_coulomb(0, i);
            point_charges[0].second[1] = m_surface_positions_coulomb(1, i);
            point_charges[0].second[2] = m_surface_positions_coulomb(2, i);
            X = p.compute_point_charge_interaction_matrix(point_charges);
            switch(kind)
            {
                case SpinorbitalKind::Restricted:
                {
                    result(i) = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(D, X);
                    break;
                }
                case SpinorbitalKind::Unrestricted:
                {
                    result(i) = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(D, X);
                    break;
                }
                case SpinorbitalKind::General:
                {
                    result(i) = 2 * occ::qm::expectation<SpinorbitalKind::General>(D, X);
                    break;
                }
            }

        }
        return result;
    }

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
        m_qn = m_proc.nuclear_electric_potential_contribution(m_solvation_model.surface_positions_coulomb());
        m_point_charges.reserve(m_qn.rows());

        for(int i = 0; i < m_solvation_model.num_surface_points(); i++)
        {
            const auto& pt = m_solvation_model.surface_positions_coulomb().col(i);
            m_point_charges.push_back({0.0, {pt(0), pt(1), pt(2)}});
        }

        m_cds_solvation_energy = m_solvation_model.smd_cds_energy();
    }

    bool supports_incremental_fock_build() const { return m_proc.supports_incremental_fock_build(); }
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
        occ::Vec v = (m_qn + m_proc.electronic_electric_potential_contribution(kind, D, m_solvation_model.surface_positions_coulomb()));
        m_solvation_model.set_surface_potential(v);
        auto asc = m_solvation_model.apparent_surface_charge();
        for(int i = 0; i < m_point_charges.size(); i++)
        {
            m_point_charges[i].first = asc(i);
        }
        double surface_energy = m_solvation_model.surface_polarization_energy();
        m_nuclear_solvation_energy = m_qn.dot(asc);
        m_surface_solvation_energy = surface_energy;
        m_electronic_solvation_energy = 0.0;
        occ::log::debug("PCM surface polarization energy: {:.12f}", surface_energy);
        occ::log::debug("PCM surface charge: {:.12f}", m_solvation_model.surface_charge());
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

    auto surface_positions_coulomb() const { return m_solvation_model.surface_positions_coulomb(); }
    auto surface_positions_cds() const { return m_solvation_model.surface_positions_cds(); }
    auto surface_areas_coulomb() const { return m_solvation_model.surface_areas_coulomb(); }
    auto surface_areas_cds() const { return m_solvation_model.surface_areas_cds(); }

    auto surface_cds_energy_elements() const { return m_solvation_model.surface_cds_energy_elements(); }
    auto surface_polarization_energy_elements() const { return m_solvation_model.surface_polarization_energy_elements(); }

    auto surface_nuclear_energy_elements() const { return m_solvation_model.surface_nuclear_energy_elements(m_proc); }
    auto surface_electronic_energy_elements(const SpinorbitalKind kind, const Mat& D) const
    {
        return m_solvation_model.surface_electronic_energy_elements(kind, D, m_proc);
    }

    template<unsigned int order = 1>
    inline auto compute_electronic_multipole_matrices(const Vec3 &o = {0.0, 0.0, 0.0}) const
    {
        return m_proc.template compute_electronic_multipole_matrices<order>(o);
    }

    template<unsigned int order = 1>
    inline auto compute_electronic_multipoles(SpinorbitalKind k, const Mat& D, const Vec3 &o = {0.0, 0.0, 0.0}) const
    {
        return m_proc.template compute_electronic_multipoles<order>(k, D, o);
    }

    template<unsigned int order = 1>
    inline auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0, 0.0}) const
    {
        return m_proc.template compute_nuclear_multipoles<order>(o);
    }


private:
    const std::vector<occ::core::Atom> &m_atoms;
    Proc &m_proc;
    ContinuumSolvationModel m_solvation_model;
    std::vector<std::pair<double, std::array<double, 3>>> m_point_charges;
    double m_electronic_solvation_energy{0.0}, m_nuclear_solvation_energy{0.0},
           m_surface_solvation_energy, m_cds_solvation_energy;
    Mat m_X;
    Vec m_qn;
};

}
