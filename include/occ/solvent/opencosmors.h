#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/solvent/sigma_profile.h>
#include <vector>

/// The openCOSMO-RS solvation free energy expression.
///
/// openCOSMO-RS assembles a solvation free energy as
///
///     dG_solv = E_diel + RT ln(gamma_inf) - sum_a tau_a A_a
///               - omega_ring n_ring - RT ln(v_gas/v_liquid) - eta
///
/// (Grigorash et al., Chem. Eng. Sci., 2025, eq. 16). The first two terms are
/// what `sigma_solvation` already produces: the gas-to-conductor dielectric
/// energy and the conductor-to-solvent residual. The remaining terms are the
/// cavity-formation work and the reference-state bookkeeping, and without
/// them the total is not a solvation free energy on any absolute scale.
///
/// The `tau_a A_a` term is a sum over atoms of a surface tension times that
/// atom's cavity area, so it is additive over surface segments and partitions
/// over contacts exactly as the residual does. The other three are molecular
/// constants.
namespace occ::solvent::sigma {

/// Parameters of the solvation free energy expression.
///
/// Defaults are the openCOSMO-RS 24a set, whose published units are
/// kcal/mol. These were regressed alongside the openCOSMO-RS interaction
/// kernel, which is not the COSMO-SAC kernel `Parameters` selects; combining
/// them is an approximation rather than the published model.
struct SolvationParameters {
  /// Per-element surface tension, kcal/mol/Å², by atomic number. Elements
  /// absent from the set contribute nothing and are reported by
  /// `unparameterised_elements`.
  ankerl::unordered_dense::map<int, double> tau;
  double eta{0.0};        ///< constant offset, kcal/mol
  double omega_ring{0.0}; ///< per ring, kcal/mol

  static SolvationParameters opencosmors_24a();
};

/// Per-segment cavity contribution `−τ_Z(i) a_i`, in Hartree.
///
/// Negative, so it stabilises. Additive over segments by construction, which
/// is what lets `occ cg` attribute it to contacts without further modelling.
/// Segments on an element outside the parameter set contribute zero.
Vec segment_cavity_energies(const Segments &segments,
                            const SolvationParameters &params);

/// Atomic numbers present in `segments` that the parameter set does not
/// cover, sorted and deduplicated. Empty when every element is covered.
std::vector<int> unparameterised_elements(const Segments &segments,
                                          const SolvationParameters &params);

/// Interaction parameters of the openCOSMO-RS model.
///
/// Defaults are the 24a set. `mf_alpha` and `hb_c` are in J/mol/Å²/e², so the
/// pairwise energies come out in J/mol.
struct RSParameters {
  double a_eff{5.9248470};           ///< Å²
  double r_av{0.5};                  ///< Å, σ averaging radius
  double r_corr{1.0};                ///< Å, σ⊥ averaging radius
  double sigma_orth_factor{0.816};
  double mf_alpha{7.2847361e06};     ///< misfit prefactor
  double mf_f_corr{2.4};             ///< σ⊥ weight in the misfit
  double hb_c{4.3311555e07};         ///< H-bond prefactor at 298.15 K
  double hb_c_T{1.5};                ///< temperature dependence
  double hb_sigma_thresh{9.6112460e-03}; ///< e/Å²
  double comb_z{10.0};               ///< coordination number
  double comb_a_std{4.1623570e01};   ///< Å², area normalisation

  static RSParameters opencosmors_24a();
};

/// One component reduced to what the kernel needs: its segment descriptors,
/// areas, and the cavity volume the combinatorial term uses.
struct RSComponent {
  Vec sigma;      ///< e/Å², averaged on `r_av`
  Vec sigma_orth; ///< e/Å²
  Vec area;       ///< Å² per segment
  double volume{0.0}; ///< Å³
  /// Cavity area for the combinatorial term, Å². The discretised segment
  /// areas need not sum to it exactly, so it is carried separately; zero
  /// falls back to the sum.
  double cavity_area{0.0};

  double total_area() const {
    return (cavity_area > 0.0) ? cavity_area : area.sum();
  }
  Eigen::Index size() const { return area.size(); }

  /// Pull the descriptors out of `segments`, which must have had
  /// `average_sigma` and `average_sigma_orth` applied. A non-positive
  /// `cavity_area` defaults to the segment sum.
  static RSComponent from_segments(const Segments &segments, double volume,
                                   double cavity_area = 0.0);
};

/// Mole-fraction-weighted mixture of solvent components, as the pooled
/// segment ensemble the kernel sees.
///
/// Segment areas are scaled by mole fraction so the pooled ensemble carries
/// the mixture's segment fractions; volume and cavity area become the
/// mole-fraction averages the combinatorial term expects. Fractions are
/// normalised.
RSComponent mix_rs_components(const std::vector<RSComponent> &components,
                              const Vec &mole_fractions);

/// Pairwise interaction free energy between every segment of `a` and every
/// segment of `b`, in J/mol. Misfit plus hydrogen bonding; the sign
/// convention is that hydrogen bonding is negative.
Mat rs_interaction_energies(const RSComponent &a, const RSComponent &b,
                            const RSParameters &params, double temperature);

/// Controls for the segment activity fixed point.
struct RSOptions {
  double temperature{298.15};
  double mixing{0.7};       ///< successive-substitution damping
  double tolerance{1e-12};  ///< on max relative change in Γ
  int max_iterations{5000};
  bool throw_on_failure{true};
};

/// Residual `ln γ` per component, with the ideal conductor as reference.
///
/// Solves the pooled-segment fixed point `Γ_i = 1/Σ_j X_j Γ_j exp(−A_ij/RT)`
/// in log space, then sums `(a_i/a_eff) ln Γ_i` over each component's own
/// segments. Subtract the pure-component values for a pure-component
/// reference state.
Vec rs_residual_ln_gamma(const std::vector<RSComponent> &components,
                         const Vec &mole_fractions,
                         const RSParameters &params,
                         const RSOptions &options = {});

/// Staverman–Guggenheim combinatorial `ln γ`, in the volume/area form
/// openCOSMO-RS uses. Zero for a pure component, so it needs no reference
/// state subtraction.
Vec rs_combinatorial_ln_gamma(const std::vector<RSComponent> &components,
                              const Vec &mole_fractions,
                              const RSParameters &params);

/// A solvent held as its converged segment activities, ready to accept any
/// solute at infinite dilution.
///
/// The fixed point is solved once for the pure solvent at construction; each
/// solute is then a single test-particle evaluation against it, which is what
/// makes a solvent screen cheap and what lets `occ cg` reuse one solvent
/// across every surface patch.
class RSSolventModel {
public:
  explicit RSSolventModel(RSComponent solvent, RSParameters params = {},
                          RSOptions options = {});

  /// Per-segment residual energy `(a_i/a_eff) RT ln Γ_i` for a solute at
  /// infinite dilution, in Hartree. Additive over segments, so it partitions
  /// over contacts without further modelling.
  Vec segment_energies(const RSComponent &solute) const;

  /// `RT ln γ_res` for the solute at infinite dilution, in Hartree. The sum
  /// of `segment_energies`.
  double residual_energy(const RSComponent &solute) const;

  /// `RT ln γ_comb` for the solute at infinite dilution, in Hartree.
  double combinatorial_energy(const RSComponent &solute) const;

  const RSComponent &solvent() const { return m_solvent; }
  const RSParameters &parameters() const { return m_params; }
  const RSOptions &options() const { return m_options; }

private:
  RSComponent m_solvent;
  RSParameters m_params;
  RSOptions m_options;
  Vec m_ln_gamma; ///< pure-solvent segment activities
  Vec m_fraction; ///< pure-solvent segment mole fractions
};

/// The terms of the openCOSMO-RS solvation free energy, each in Hartree.
struct RSSolvationEnergy {
  double dielectric{0.0};       ///< E_diel, gas to ideal conductor
  double residual{0.0};         ///< RT ln γ_res at infinite dilution
  double combinatorial{0.0};    ///< RT ln γ_comb at infinite dilution
  double cavity{0.0};           ///< −Σ τ_α A_α
  double ring{0.0};             ///< −ω_ring n_ring
  double reference_state{0.0};  ///< −RT ln(v_gas/v_liquid)
  double constant{0.0};         ///< −η

  double total() const {
    return dielectric + residual + combinatorial + cavity + ring +
           reference_state + constant;
  }
};

/// Assemble the solvation free energy of a solute at infinite dilution.
///
/// `dielectric` is the gas-to-conductor energy in Hartree, which the caller
/// already has from the conductor SCF. `segments` supplies the per-element
/// areas for the cavity term, so it must be the solute's own cavity.
/// `volume_liquid` is the solute's liquid molar volume in Å³ per molecule;
/// pass a non-positive value to drop the reference-state term.
RSSolvationEnergy rs_solvation_free_energy(
    const RSSolventModel &solvent, const RSComponent &solute,
    const Segments &segments, double dielectric, int num_rings,
    double volume_liquid, const SolvationParameters &solvation_params);

/// Cavity area per element, Å², keyed by atomic number.
ankerl::unordered_dense::map<int, double>
area_per_element(const Segments &segments);

/// The molecular terms `−ω_ring n_ring − RT ln(v_gas/v_liquid) − η`, in
/// Hartree.
///
/// `volume_liquid` is the solute's liquid molar volume in Å³ per molecule;
/// the ideal-gas volume is evaluated at `temperature` and 1 bar. Pass a
/// non-positive volume to drop the reference-state term, leaving the ring
/// correction and the constant.
double molecular_solvation_terms(int num_rings, double volume_liquid,
                                 double temperature,
                                 const SolvationParameters &params);

} // namespace occ::solvent::sigma
