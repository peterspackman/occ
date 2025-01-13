#pragma once
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/interaction/polarization.h>
#include <occ/qm/hf_fwd.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>
#include <string>

namespace occ::interaction {

using qm::HartreeFock;
using qm::Wavefunction;

struct CEParameterizedModel {
  double coulomb{1.0};
  double exchange{1.0};
  double repulsion{1.0};
  double polarization{1.0};
  double dispersion{1.0};
  std::string name{"Unscaled"};
  std::string method{"b3lyp"};
  std::string basis{"6-31g**"};
  bool xdm{false};
  double xdm_a1{0.65}, xdm_a2{1.70};
  double scaled_total(double coul, double ex, double rep, double pol,
                      double disp) const {
    return coulomb * coul + exchange * ex + repulsion * rep +
           polarization * pol + dispersion * disp;
  }
};

inline CEParameterizedModel CE_HF_321G{1.019,   0.811, 0.811,   0.651, 0.901,
                                       "CE-HF", "hf",  "3-21g", false};
inline CEParameterizedModel CE_B3LYP_631Gdp{1.0573,  0.6177,    0.6177,
                                            0.7399,  0.8708,    "CE-B3LYP",
                                            "b3lyp", "6-31g**", false};
inline CEParameterizedModel CE_XDM_FIT{
    1.0,       1.0,        1.0,  1.0,  1.0, "CE-XDM-FIT",
    "wb97m-v", "def2-svp", true, 0.65, 1.70};

inline constexpr double CE1p_XDM_KREP{0.77850434};

inline CEParameterizedModel CE1_XDM_B3LYP{
    1.0,     1.0,        CE1p_XDM_KREP, CE1p_XDM_KREP, 1.0, "CE-1p-B3LYP",
    "b3lyp", "def2-svp", true,          0.65,          1.70};
inline CEParameterizedModel CE2_XDM_WB97MV{
    1.0,     0.485,      0.485, 0.803, 1.0, "CE2p-XDM-wB97M-V",
    "b3lyp", "def2-svp", true,  0.65,  1.70};

inline CEParameterizedModel CE5_XDM_WB97MV{
    1.0051,    0.6705,     0.6,  0.7929, 1.0509, "CE5p-XDM-wB97M-V",
    "wb97m-v", "def2-svp", true, 0.65,   1.70};

inline CEParameterizedModel CE1_XDM = CE1_XDM_B3LYP;
inline CEParameterizedModel CE2_XDM = CE2_XDM_WB97MV;
inline CEParameterizedModel CE5_XDM = CE5_XDM_WB97MV;

struct CEMonomerCalculationParameters {
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

  Mat3N field_a =
      proc_b.electronic_electric_field_contribution(wfn_b.mo, pos_a);
  Vec3 tmp = field_a.colwise().squaredNorm();
  log::debug("Field (ele) at A due to B [{:.5f}, {:.5f}, {:.5f}]", tmp(0),
             tmp(1), tmp(2));
  Mat3N field_a_n = proc_b.nuclear_electric_field_contribution(pos_a);
  tmp = field_a_n.colwise().squaredNorm();
  log::debug("Field (nuc) at A due to B [{:.5f}, {:.5f}, {:.5f}]", tmp(0),
             tmp(1), tmp(2));

  field_a += field_a_n;
  tmp = field_a.colwise().squaredNorm();
  log::debug("Field (net) at A due to B [{:.5f}, {:.5f}, {:.5f}]", tmp(0),
             tmp(1), tmp(2));

  Mat3N field_b =
      proc_a.electronic_electric_field_contribution(wfn_a.mo, pos_b);
  Mat3N field_b_n = proc_a.nuclear_electric_field_contribution(pos_b);
  tmp = field_b.colwise().squaredNorm();
  log::debug("Field (ele) at B due to A [{:.5f}, {:.5f}, {:.5f}]", tmp(0),
             tmp(1), tmp(2));
  tmp = field_b_n.colwise().squaredNorm();
  log::debug("Field (nuc) at B due to A [{:.5f}, {:.5f}, {:.5f}]", tmp(0),
             tmp(1), tmp(2));
  field_b += field_b_n;
  tmp = field_b.colwise().squaredNorm();
  log::debug("Field (net) at B due to A [{:.5f}, {:.5f}, {:.5f}]", tmp(0),
             tmp(1), tmp(2));

  using interaction::ce_model_polarization_energy;
  double e_pol = 0.0;
  if (wfn_a.have_xdm_parameters && wfn_b.have_xdm_parameters) {
    log::trace("Using XDM polarizability");
    using interaction::polarization_energy;
    e_pol = polarization_energy(wfn_a.xdm_polarizabilities, field_a) +
            polarization_energy(wfn_b.xdm_polarizabilities, field_b);
  } else {
    // if charged atoms, use charged atomic polarizabilities
    bool charged_a = (wfn_a.atoms.size() == 1) && (wfn_a.charge() != 0);
    bool charged_b = (wfn_b.atoms.size() == 1) && (wfn_b.charge() != 0);
    log::debug("using charged atom polarizabilities: A={} B={}", charged_a,
               charged_b);
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
  if (s == "ce-1p" || s == "ce1p" || s == "ce-1p-xdm")
    return CE1_XDM;
  if (s == "ce-2p" || s == "ce2p" || s == "ce-2p-xdm")
    return CE2_XDM;
  if (s == "ce-5p" || s == "ce5p" || s == "ce-5p-xdm")
    return CE5_XDM;
  if (s == "ce-1p-wb97m-v" || s == "ce1p-wb97m-v" || s == "ce-1p-wb97m-v")
    return CE1_XDM;
  if (s == "ce-2p-wb97m-v" || s == "ce2p-wb97m-v" || s == "ce-2p-wb97m-v")
    return CE2_XDM;
  if (s == "ce-5p-wb97m-v" || s == "ce5p-wb97m-v" || s == "ce-5p-wb97m-v")
    return CE5_XDM;
  log::warn("Unknown model, defaulting to CE-1p");
  return CE1_XDM;
}

struct CEEnergyComponents {
  double coulomb{0.0};
  double exchange{0.0};
  double repulsion{0.0};
  double polarization{0.0};
  double dispersion{0.0};
  double total{0.0};
  double exchange_repulsion{0.0};
  double nonorthogonal_term{0.0};
  double orthogonal_term{0.0};
  bool is_computed{false};

  double coulomb_kjmol() const { return units::AU_TO_KJ_PER_MOL * coulomb; }
  double exchange_repulsion_kjmol() const {
    return units::AU_TO_KJ_PER_MOL * exchange_repulsion;
  }
  double polarization_kjmol() const {
    return units::AU_TO_KJ_PER_MOL * polarization;
  }
  double dispersion_kjmol() const {
    return units::AU_TO_KJ_PER_MOL * dispersion;
  }

  double repulsion_kjmol() const { return units::AU_TO_KJ_PER_MOL * repulsion; }

  double exchange_kjmol() const { return units::AU_TO_KJ_PER_MOL * exchange; }

  double total_kjmol() const { return units::AU_TO_KJ_PER_MOL * total; }

  inline auto operator-(const CEEnergyComponents &rhs) {
    return CEEnergyComponents{coulomb - rhs.coulomb,
                              exchange - rhs.exchange,
                              repulsion - rhs.repulsion,
                              polarization - rhs.polarization,
                              dispersion - rhs.dispersion,
                              total - rhs.total,
                              exchange_repulsion - rhs.exchange_repulsion,
                              nonorthogonal_term - rhs.nonorthogonal_term,
                              orthogonal_term - rhs.orthogonal_term,
                              true};
  }

  inline auto operator+(const CEEnergyComponents &rhs) {
    return CEEnergyComponents{coulomb + rhs.coulomb,
                              exchange + rhs.exchange,
                              repulsion + rhs.repulsion,
                              polarization + rhs.polarization,
                              dispersion + rhs.dispersion,
                              total + rhs.total,
                              exchange_repulsion + rhs.exchange_repulsion,
                              nonorthogonal_term + rhs.nonorthogonal_term,
                              orthogonal_term + rhs.orthogonal_term,
                              true};
  }

  inline auto operator+=(const CEEnergyComponents &rhs) { *this = *this + rhs; }

  inline auto operator-=(const CEEnergyComponents &rhs) { *this = *this - rhs; }
};

struct CEModelInteraction {
  CEModelInteraction(const CEParameterizedModel &);
  CEEnergyComponents operator()(Wavefunction &, Wavefunction &) const;
  void set_use_density_fitting(bool value = true);
  void compute_monomer_energies(Wavefunction &) const;
  void set_use_xdm_dimer_parameters(bool value = true);
  inline bool using_xdm_dimer_parameters() const {
    return m_use_xdm_dimer_parameters;
  }

  const auto &scale_factors() const { return m_scale_factors; }

private:
  CEParameterizedModel m_scale_factors;
  bool m_use_density_fitting{false};
  bool m_use_xdm_dimer_parameters{false};
};

} // namespace occ::interaction
