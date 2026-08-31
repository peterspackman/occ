#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <occ/solvent/cosmors_segments.h>
#include <vector>

/// The openCOSMO-RS solvation free energy expression.
///
/// openCOSMO-RS assembles a solvation free energy as
///
///     dG_solv = E_diel + RT ln(gamma_inf) - sum_a tau_a A_a
///               - omega_ring n_ring - RT ln(v_gas/v_liquid) - eta
///
/// (Grigorash et al., Chem. Eng. Sci., 2025, eq. 16), term by term:
///
///  - `E_diel`, the gas-to-ideal-conductor energy. The caller supplies it,
///    since it comes from the conductor SCF rather than from this model.
///  - `RT ln(gamma_inf)`, the activity coefficient at infinite dilution,
///    split here into its residual (segment misfit and hydrogen bonding) and
///    combinatorial (Staverman-Guggenheim size and shape) parts.
///  - `-sum_a tau_a A_a`, the van der Waals term: a per-element surface
///    tension times that element's cavity area. It is a sum over atoms, so
///    it is additive over surface segments and partitions over contacts
///    exactly as the residual does.
///  - `-omega_ring n_ring`, an empirical correction for ring strain.
///  - `-RT ln(v_gas/v_liquid)`, moving from a 1 bar ideal gas reference to
///    the pure liquid.
///  - `-eta`, the fitted intercept of the regression.
///
/// There is deliberately no cavity-formation term: the work of opening the
/// cavity was not modelled separately in the regression, and whatever part
/// of it is not already in the van der Waals term is absorbed into `eta`.
/// Without the last four terms the total is not a solvation free energy on
/// any absolute scale.
namespace occ::solvent::cosmors {

/// Number of rings, for the ring correction: the cycle rank of the bond
/// graph, `bonds - atoms + 1` for a connected molecule. Zero without bonds,
/// which is also the right answer for an acyclic solute.
int ring_count(const core::Molecule &molecule);

/// Every parameter of the openCOSMO-RS model.
///
/// The kernel, the combinatorial term and the free-energy assembly were
/// regressed together, so they are one set: mixing constants from different
/// parameterisations is not a variant of the model, it is a different model.
/// The defaults are the 24a set, and `v24a()` states it explicitly.
struct Parameters {
  /// @name Segment descriptors
  /// @{
  double a_eff{5.9248470};   ///< Å², effective contact area
  double r_av{0.5};          ///< Å, σ averaging radius
  double r_corr{1.0};        ///< Å, σ⊥ averaging radius
  double sigma_orth_factor{0.816};
  /// @}

  /// @name Interaction kernel
  /// `mf_alpha` and `hb_c` are in J/mol/Å²/e², so pairwise energies come out
  /// in J/mol.
  /// @{
  double mf_alpha{7.2847361e06};         ///< misfit prefactor
  double mf_f_corr{2.4};                 ///< σ⊥ weight in the misfit
  double hb_c{4.3311555e07};             ///< H-bond prefactor, at `hb_t_ref`
  double hb_c_T{1.5};                    ///< temperature dependence of `hb_c`
  /// Temperature the H-bond prefactor was fitted at, K. Not a setting: the
  /// run temperature is `ActivityOptions::temperature`.
  double hb_t_ref{298.15};
  double hb_sigma_thresh{9.6112460e-03}; ///< e/Å², donor/acceptor threshold
  /// @}

  /// @name Combinatorial term
  /// @{
  double comb_z{10.0};             ///< coordination number
  double comb_a_std{4.1623570e01}; ///< Å², area normalisation
  /// @}

  /// @name Free-energy assembly
  /// Published in kcal/mol, converted at use.
  /// @{
  /// Per-element van der Waals surface tension, kcal/mol/Å², by atomic
  /// number. Elements absent from the map contribute nothing and are
  /// reported by `unparameterised_elements`.
  ankerl::unordered_dense::map<int, double> tau{
      {1, 2.933803e-02},  {6, 2.287904e-02},  {7, 7.007681e-04},
      {8, 3.545052e-03},  {9, 5.608829e-03},  {14, 4.215503e-03},
      {15, 3.607977e-03}, {16, 3.498700e-02}, {17, 3.414282e-02},
      {35, 4.085111e-02},
  };
  double eta{-4.448499};            ///< fitted intercept, kcal/mol
  double omega_ring{2.6302510e-01}; ///< ring correction, kcal/mol per ring
  /// @}

  /// Names the defaults as the published openCOSMO-RS 24a set. The values
  /// live in the member initialisers above so there is one copy of them;
  /// `[cosmors][parameters]` pins those against the reference distribution.
  static Parameters v24a() { return Parameters{}; }
};

/// One component reduced to what the kernel needs: its segment descriptors,
/// areas, and the cavity volume the combinatorial term uses.
struct Component {
  Vec sigma;           ///< e/Å², averaged on `r_av`
  Vec sigma_orth;      ///< e/Å²
  Vec area;            ///< Å² per segment
  IVec atomic_number;  ///< parent atom of each segment, for the vdW term
  double volume{0.0};  ///< Å³
  /// Cavity area for the combinatorial term, Å². The discretised segment
  /// areas need not sum to it exactly, so it is carried separately; zero
  /// falls back to the sum.
  double cavity_area{0.0};

  double total_area() const {
    return (cavity_area > 0.0) ? cavity_area : area.sum();
  }
  Eigen::Index size() const { return area.size(); }

  /// Pull the descriptors out of `segments`, which must have had
  /// `average_sigma` and `average_sigma_orth` applied on the same
  /// `Parameters`. A non-positive `cavity_area` defaults to the segment sum.
  static Component from_segments(const Segments &segments, double volume,
                                 double cavity_area = 0.0);
};

/// Per-segment van der Waals contribution `−τ_Z(i) a_i`, in Hartree.
///
/// Negative, so it stabilises. Additive over segments by construction, which
/// is what lets `occ cg` attribute it to contacts without further modelling.
/// Segments on an element outside the parameter set contribute zero.
Vec segment_vdw_energies(const Component &component,
                         const Parameters &params);

/// Atomic numbers in `component` that `params.tau` does not cover, sorted and
/// deduplicated. Empty when every element is covered.
std::vector<int> unparameterised_elements(const Component &component,
                                          const Parameters &params);

/// Mole-fraction-weighted mixture of solvent components, as the pooled
/// segment ensemble the kernel sees.
///
/// Segment areas are scaled by mole fraction so the pooled ensemble carries
/// the mixture's segment fractions; volume and cavity area become the
/// mole-fraction averages the combinatorial term expects. Fractions are
/// normalised.
Component mix_components(const std::vector<Component> &components,
                              const Vec &mole_fractions);

/// Pairwise interaction free energy between every segment of `a` and every
/// segment of `b`, in J/mol. Misfit plus hydrogen bonding; the sign
/// convention is that hydrogen bonding is negative.
Mat interaction_energies(const Component &a, const Component &b,
                            const Parameters &params, double temperature);

/// Controls for the segment activity fixed point.
struct ActivityOptions {
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
Vec residual_ln_gamma(const std::vector<Component> &components,
                         const Vec &mole_fractions,
                         const Parameters &params,
                         const ActivityOptions &options = {});

/// Staverman–Guggenheim combinatorial `ln γ`, in the volume/area form
/// openCOSMO-RS uses. Zero for a pure component, so it needs no reference
/// state subtraction.
Vec combinatorial_ln_gamma(const std::vector<Component> &components,
                              const Vec &mole_fractions,
                              const Parameters &params);

/// A solvent held as its converged segment activities, ready to accept any
/// solute at infinite dilution.
///
/// The fixed point is solved once for the pure solvent at construction; each
/// solute is then a single test-particle evaluation against it, which is what
/// makes a solvent screen cheap and what lets `occ cg` reuse one solvent
/// across every surface patch.
class SolventModel {
public:
  explicit SolventModel(Component solvent, Parameters params = {},
                          ActivityOptions options = {});

  /// Per-segment residual energy `(a_i/a_eff) RT ln Γ_i` for a solute at
  /// infinite dilution, in Hartree. Additive over segments, so it partitions
  /// over contacts without further modelling.
  Vec segment_energies(const Component &solute) const;

  /// `RT ln γ_res` for the solute at infinite dilution, in Hartree. The sum
  /// of `segment_energies`.
  double residual_energy(const Component &solute) const;

  /// `RT ln γ_comb` for the solute at infinite dilution, in Hartree.
  double combinatorial_energy(const Component &solute) const;

  const Component &solvent() const { return m_solvent; }
  const Parameters &parameters() const { return m_params; }
  const ActivityOptions &options() const { return m_options; }

private:
  Component m_solvent;
  Parameters m_params;
  ActivityOptions m_options;
  Vec m_ln_gamma; ///< pure-solvent segment activities
  Vec m_fraction; ///< pure-solvent segment mole fractions
};

/// The terms of the openCOSMO-RS solvation free energy, each in Hartree.
struct SolvationEnergy {
  /// E_diel, gas to ideal conductor. Supplied by the caller, and the whole
  /// SCF difference: the variational ½σφ part plus the electronic relaxation
  /// cost. `occ cg` reports those two separately, as its `dielectric` and
  /// `electronic` channels, because only the first is per-segment.
  double dielectric{0.0};
  double residual{0.0};        ///< RT ln γ_res at infinite dilution
  double combinatorial{0.0};   ///< RT ln γ_comb at infinite dilution
  double vdw{0.0};             ///< −Σ_a τ_a A_a
  double ring{0.0};            ///< −ω_ring n_ring
  double reference_state{0.0}; ///< −RT ln(v_gas/v_liquid)
  double eta{0.0};             ///< −η, the fitted intercept

  double total() const {
    return dielectric + residual + combinatorial + vdw + ring +
           reference_state + eta;
  }
};

/// Assemble the solvation free energy of a solute at infinite dilution.
///
/// `dielectric` is the gas-to-conductor energy in Hartree, which the caller
/// already has from the conductor SCF. `volume_liquid` is the volume of one
/// solute molecule in the liquid, Å³; pass a non-positive value to drop the
/// reference-state term. The parameters are the solvent model's own, so the
/// kernel and the assembly cannot disagree.
SolvationEnergy solvation_free_energy(const SolventModel &solvent,
                                      const Component &solute,
                                      double dielectric, int num_rings,
                                      double volume_liquid);

/// Cavity area per element, Å², keyed by atomic number.
ankerl::unordered_dense::map<int, double>
area_per_element(const Component &component);

} // namespace occ::solvent::cosmors
