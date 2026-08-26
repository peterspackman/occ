#pragma once
#include <occ/scrf/surfaces.h>
#include <occ/solvent/sigma_activity.h>
#include <string>
#include <vector>

namespace occ::solvent::sigma {

/// Resolves a solvent name to its σ-profile, read from `<name>.sigma` under
/// the search paths. Profiles are produced by `occ sigma`; nothing is
/// computed here.
class ProfileStore {
public:
  explicit ProfileStore(std::vector<std::string> search_paths);

  /// `$OCC_DATA_PATH/solvent/sigma`, then the working directory.
  static ProfileStore standard();

  bool contains(const std::string &name) const;

  /// Throws naming the paths searched and how to generate the profile.
  Component get(const std::string &name) const;

  /// Names of every profile found, sorted.
  std::vector<std::string> available() const;

  const std::vector<std::string> &search_paths() const {
    return m_search_paths;
  }

private:
  std::vector<std::string> m_search_paths;
};

/// Mole-fraction-weighted mixture of solvent components.
///
/// The profile is `Σ_k x_k A_k p_k(σ)` (what `mix_profiles` produces) and the
/// cavity volume is weighted the same way, so the combinatorial term sees a
/// consistent pseudo-component. Fractions are normalised.
Component mix_components(const std::vector<Component> &components,
                         const Vec &mole_fractions);

/// A solvent held as its converged σ-potential, ready to contract against
/// any solute's segments.
///
/// The potential is solved once at construction. Every additional solute —
/// or every patch of a crystal surface — is then a contraction, which is
/// what makes a solvent screen cheap compared with re-running an SCF per
/// solvent.
class SolventModel {
public:
  explicit SolventModel(Component solvent,
                        Parameters params = Parameters::cosmo_sac_2010(),
                        PotentialOptions options = {});

  const Potential &potential() const { return m_potential; }
  const Parameters &parameters() const { return m_params; }
  const Component &solvent() const { return m_solvent; }

  /// Per-segment residual solvation energy `a_i μ_S(σ̄_i)/a_eff`, in Hartree.
  /// Additive over segments by construction, so it partitions over contacts
  /// without any further modelling.
  Vec segment_energies(const Segments &solute) const;

  /// Per-segment reorganisation descriptor `a_i Var_S(σ̄_i)/(2·RT·a_eff)`, in
  /// Hartree.
  ///
  /// \warning Mean-field. This is the variance of one segment's pairing
  /// energy over its partner ensemble in a model that treats pairs as
  /// independent — a per-site descriptor, not a Marcus reorganisation
  /// energy.
  Vec segment_reorganisation(const Segments &solute) const;

  /// Per-segment hydrogen-bonded area `a_i p_HB(σ̄_i)`, in Å². Additive, so
  /// it partitions over contacts the same way the energies do.
  Vec segment_hbond_area(const Segments &solute) const;

  /// Solute segments and their energies in the shape `occ::cg` consumes.
  occ::scrf::SolvationSurface solvation_surface(const Segments &solute) const;

private:
  Component m_solvent;
  Parameters m_params;
  PotentialOptions m_options;
  Potential m_potential;
};

} // namespace occ::solvent::sigma
