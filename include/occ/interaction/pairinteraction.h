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
    double xdm_a1{1.0}, xdm_a2{1.0};
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

void compute_ce_model_energies(Wavefunction &wfn, HartreeFock &hf,
                               double precision, const Mat &Schwarz,
                               bool xdm = false);

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
    return CEParameterizedModel{};
}

struct CEEnergyComponents {
    double coulomb{0.0};
    double exchange_repulsion{0.0};
    double polarization{0.0};
    double dispersion{0.0};
    double total{0.0};
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

  private:
    CEParameterizedModel scale_factors;
    bool m_use_density_fitting{false};
};

} // namespace occ::interaction
