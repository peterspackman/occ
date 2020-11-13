#pragma once
#include "ints.h"
#include "wavefunction.h"
#include "basisset.h"
#include "spinorbital.h"
#include "polarization.h"
#include <memory>

namespace tonto::interaction {
using tonto::MatRM;
using tonto::Vec;
using tonto::qm::BasisSet;
using tonto::qm::SpinorbitalKind;
using tonto::qm::Wavefunction;

struct CEParameterizedModel {
    const double coulomb{1.0};
    const double exchange_repulsion{1.0};
    const double polarization{1.0};
    const double dispersion{1.0};
    const std::string name{"Unscaled"};
    double scaled_total(double coul, double ex, double pol, double disp) const
    {
        return coulomb * coul + exchange_repulsion * ex + polarization * pol + dispersion * disp;
    }
};

inline CEParameterizedModel CE_HF_321G{1.019, 0.811, 0.651, 0.901, "CE-HF"};
inline CEParameterizedModel CE_B3LYP_631Gdp{1.0573, 0.6177, 0.7399, 0.8708, "CE-B3LYP"};

std::pair<MatRM, Vec> merge_molecular_orbitals(const MatRM&, const MatRM&, const Vec&, const Vec&);
BasisSet merge_basis_sets(const BasisSet&, const BasisSet&);
std::vector<libint2::Atom> merge_atoms(const std::vector<libint2::Atom>&, const std::vector<libint2::Atom>&);

template<typename Procedure>
double compute_polarization_energy(const Wavefunction &wfn_a, const Procedure &proc_a,
                                   const Wavefunction &wfn_b, const Procedure &proc_b)
{
    // fields (incl. sign) have been checked and agree with both finite difference
    // method and tonto
    auto pos_a = wfn_a.positions();
    auto pos_b = wfn_b.positions();

    tonto::Mat3N field_a = proc_b.electronic_electric_field_contribution(wfn_b.D, pos_a);
    field_a += proc_b.nuclear_electric_field_contribution(pos_a);
    tonto::Mat3N field_b = proc_a.electronic_electric_field_contribution(wfn_a.D, pos_b);
    field_b += proc_a.nuclear_electric_field_contribution(pos_b);

    using tonto::pol::ce_model_polarization_energy;
    double e_pol = ce_model_polarization_energy(wfn_a.atomic_numbers(), field_a) +
                   ce_model_polarization_energy(wfn_b.atomic_numbers(), field_b);
    return e_pol;
}


inline CEParameterizedModel ce_model_from_string(const std::string& s)
{
    if(s == "ce-b3lyp") return CE_B3LYP_631Gdp;
    if(s == "ce-hf") return CE_HF_321G;
    return CEParameterizedModel{};
}

struct CEModelInteraction
{
    struct EnergyComponents {
        double coulomb{0.0};
        double exchange_repulsion{0.0};
        double polarization{0.0};
        double dispersion{0.0};
        double total{0.0};
    };
    CEModelInteraction(const CEParameterizedModel&);
    EnergyComponents operator()(Wavefunction&, Wavefunction&) const;
private:
    CEParameterizedModel scale_factors;
};

}
