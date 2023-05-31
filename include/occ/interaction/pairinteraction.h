#pragma once
#include <fmt/core.h>
#include <memory>
#include <occ/core/units.h>
#include <occ/interaction/polarization.h>
#include <occ/qm/hf_fwd.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

namespace occ::interaction {

using occ::qm::HartreeFock;
using occ::qm::Wavefunction;

struct CEParameterizedModel {
    const double coulomb{1.0};
    const double exchange_repulsion{1.0};
    const double polarization{1.0};
    const double dispersion{1.0};
    const std::string name{"Unscaled"};
    bool xdm{false};
    double xdm_a1{0.21}, xdm_a2{3.05};
    double scaled_total(double coul, double ex, double pol, double disp) const {
        return coulomb * coul + exchange_repulsion * ex + polarization * pol +
               dispersion * disp;
    }
};

inline CEParameterizedModel CE_HF_321G{1.019, 0.811,   0.651,
                                       0.901, "CE-HF", false};
inline CEParameterizedModel CE_B3LYP_631Gdp{1.0573, 0.6177,     0.7399,
                                            0.8708, "CE-B3LYP", false};
inline CEParameterizedModel CE_XDM_FIT{1.0, 1.0, 1.0, 1.0, "CE-XDM-FIT", true};
inline CEParameterizedModel CE2_XDM{1.0, 0.485, 0.803, 1.0, "CE2-XDM", true};

inline CEParameterizedModel CE_HF_SVP{1.0,         0.485, 0.803, 1.0,
                                      "CE-HF-SVP", true,  0.65,  1.7};
inline CEParameterizedModel CE_LDA_SVP{1.0,          0.485, 0.803, 1.0,
                                       "CE-LDA-SVP", true,  0.65,  1.7};
inline CEParameterizedModel CE_BLYP_SVP{1.0,           0.485, 0.803, 1.0,
                                        "CE-BLYP-SVP", true,  0.65,  1.7};
inline CEParameterizedModel CE_B3LYP_SVP{
    1.0, 0.485, 0.803, 1.0, "CE-B3LYP-SVP", true, 0.65, 1.7};
inline CEParameterizedModel CE_WB97M_SVP{
    1.0, 0.485, 0.803, 1.0, "CE-WB97M-SVP", true, 0.65, 1.7};
inline CEParameterizedModel CE_WB97X_SVP{
    1.0, 0.485, 0.803, 1.0, "CE-WB97X-SVP", true, 0.65, 1.7};

struct CEMonomerCalculationParameters {
    double precision{1e-12};
    Mat Schwarz;
    bool xdm{false};
    bool neglect_exchange{false};
};

void compute_ce_model_energies(
    Wavefunction &wfn, HartreeFock &hf,
    const CEMonomerCalculationParameters &params = {});

template <typename Procedure>
double compute_polarization_energy(const Wavefunction &wfn_a,
                                   const Procedure &proc_a,
                                   const Wavefunction &wfn_b,
                                   const Procedure &proc_b) {
    // fields (incl. sign) have been checked and agree with both finite
    // difference method and occ
    auto pos_a = wfn_a.positions();
    auto pos_b = wfn_b.positions();

    occ::Mat3N field_a =
        proc_b.electronic_electric_field_contribution(wfn_b.mo, pos_a);
    field_a += proc_b.nuclear_electric_field_contribution(pos_a);
    occ::log::debug("Field at A due to B\n{}\n",
                    field_a.colwise().squaredNorm());
    occ::Mat3N field_b =
        proc_a.electronic_electric_field_contribution(wfn_a.mo, pos_b);
    field_b += proc_a.nuclear_electric_field_contribution(pos_b);
    occ::log::debug("Field at B due to A\n{}\n",
                    field_b.colwise().squaredNorm());

    using occ::interaction::ce_model_polarization_energy;
    double e_pol = 0.0;
    if (wfn_a.have_xdm_parameters && wfn_b.have_xdm_parameters) {
        occ::log::trace("Using XDM polarizability");
        using occ::interaction::polarization_energy;
        e_pol = polarization_energy(wfn_a.xdm_polarizabilities, field_a) +
                polarization_energy(wfn_b.xdm_polarizabilities, field_b);
    } else {
        // if charged atoms, use charged atomic polarizabilities
        bool charged_a = (wfn_a.atoms.size() == 1) && (wfn_a.charge() != 0);
        bool charged_b = (wfn_b.atoms.size() == 1) && (wfn_b.charge() != 0);
        e_pol = ce_model_polarization_energy(wfn_a.atomic_numbers(), field_a,
                                             charged_a) +
                ce_model_polarization_energy(wfn_b.atomic_numbers(), field_b,
                                             charged_b);
    }
    return e_pol;
}

inline CEParameterizedModel ce_model_from_string(const std::string &s) {
    if (s == "ce-b3lyp")
        return CE_B3LYP_631Gdp;
    if (s == "ce-hf")
        return CE_HF_321G;
    if (s == "ce-xdm-fit")
        return CE_XDM_FIT;
    const std::string ce2{"ce2-xdm"};
    if (s.compare(0, ce2.size(), ce2) == 0)
        return CE2_XDM;
    if (s == "ce-b3lyp")
        return CE_B3LYP_631Gdp;
    if (s == "ce-hf")
        return CE_HF_321G;
    if (s == "ce-xdm-fit")
        return CE_XDM_FIT;
    if (s == "ce-hf-svp")
        return CE_HF_SVP;
    if (s == "ce-lda-svp")
        return CE_LDA_SVP;
    if (s == "ce-blyp-svp")
        return CE_BLYP_SVP;
    if (s == "ce-b3lyp-svp")
        return CE_B3LYP_SVP;
    if (s == "ce-wb97x-svp")
        return CE_WB97X_SVP;
    if (s == "ce-wb97m-svp")
        return CE_WB97M_SVP;
    if (s == "ce-xdm-wb97m-v")
        return CE_WB97M_SVP;
    return CEParameterizedModel{};
}

struct CEEnergyComponents {
    double coulomb{0.0};
    double exchange_repulsion{0.0};
    double polarization{0.0};
    double dispersion{0.0};
    double total{0.0};
    double exchange_component{0.0};
    double repulsion_component{0.0};
    double nonorthogonal_term{0.0};
    double orthogonal_term{0.0};
    bool is_computed{false};

    double coulomb_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * coulomb;
    }
    double exchange_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * exchange_repulsion;
    }
    double polarization_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * polarization;
    }
    double dispersion_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * dispersion;
    }

    double repulsion_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * repulsion_component;
    }

    double exchange_component_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * exchange_component;
    }

    double total_kjmol() const { return occ::units::AU_TO_KJ_PER_MOL * total; }

    inline auto operator-(const CEEnergyComponents &rhs) {
        return CEEnergyComponents{coulomb - rhs.coulomb,
                                  exchange_repulsion - rhs.exchange_repulsion,
                                  polarization - rhs.polarization,
                                  dispersion - rhs.dispersion,
                                  total - rhs.total,
                                  true};
    }

    inline auto operator+(const CEEnergyComponents &rhs) {
        return CEEnergyComponents{coulomb + rhs.coulomb,
                                  exchange_repulsion + rhs.exchange_repulsion,
                                  polarization + rhs.polarization,
                                  dispersion + rhs.dispersion,
                                  total + rhs.total,
                                  true};
    }
};

struct CEModelInteraction {
    CEModelInteraction(const CEParameterizedModel &);
    CEEnergyComponents operator()(Wavefunction &, Wavefunction &) const;
    CEEnergyComponents dft_pair(const std::string &, Wavefunction &,
                                Wavefunction &) const;
    void set_use_density_fitting(bool value = true);
    void compute_monomer_energies(Wavefunction &) const;
    void set_use_xdm_dimer_parameters(bool value = true);

    const auto &scale_factors() const { return m_scale_factors; }

  private:
    CEParameterizedModel m_scale_factors;
    bool m_use_density_fitting{false};
    bool m_use_xdm_dimer_parameters{false};
};

} // namespace occ::interaction
