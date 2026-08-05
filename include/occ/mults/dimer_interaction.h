#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/force_field_params.h>
#include <occ/mults/multipole_interactions.h>
#include <utility>
#include <vector>

namespace occ::mults {

/// One molecule's oriented (lab-frame) DMA sites, ready for a dimer interaction.
///
/// \c positions are in Bohr (atomic units, the DMA-native frame). Each site
/// carries a multipole and an atomic number; the atomic number labels the site
/// for the element-based exp-6 (Buckingham/Williams) lookup. DMA sites usually
/// coincide with atoms, so one site list serves both the electrostatic and the
/// short-range sums.
struct MoleculeMultipoles {
  std::vector<occ::dma::Mult> multipoles; ///< one per site
  Mat3N positions;                        ///< 3 x N, Bohr
  std::vector<int> atomic_numbers;        ///< N, for the exp-6 element lookup
  /// Optional NEIGHCRYS short-range type codes (one per site) for the typed
  /// exp-6 sets (FIT / W99); empty -> element-based exp-6 lookup.
  std::vector<int> type_codes{};

  /// Number of sites (== columns of \c positions).
  Eigen::Index size() const { return positions.cols(); }

  /// The parallel arrays must have one entry per site; type_codes is optional
  /// but, when present, must also match.
  bool is_valid() const {
    const Eigen::Index n = positions.cols();
    return n == static_cast<Eigen::Index>(multipoles.size()) &&
           n == static_cast<Eigen::Index>(atomic_numbers.size()) &&
           (type_codes.empty() ||
            n == static_cast<Eigen::Index>(type_codes.size()));
  }
};

/// Components of a classical dimer interaction energy (DMA-multipole
/// electrostatics + exp-6 repulsion/dispersion), all in kJ/mol. There is no
/// induction term: this model has no polarization by construction.
struct DimerInteractionEnergy {
  double electrostatic{0.0}; ///< multipole-multipole (kJ/mol)
  double repulsion{0.0};     ///< exp-6 A*exp(-B r) sum (kJ/mol)
  double dispersion{0.0};    ///< exp-6 -C/r^6 sum (kJ/mol)
  double total() const { return electrostatic + repulsion + dispersion; }
};

/// Classical interaction energy between two molecules from their distributed
/// multipoles plus an element-based exp-6 (Buckingham/Williams) short-range
/// term. Electrostatics are evaluated in Bohr/Hartree and the exp-6 in the
/// Angstrom/(kJ/mol) Williams convention; the result is reported in kJ/mol.
///
/// Element pairs without exp-6 parameters in \p ff are skipped (the caller is
/// responsible for ensuring coverage); only the electrostatic term contributes
/// for those pairs.
DimerInteractionEnergy
dimer_interaction_energy(const MoleculeMultipoles &a,
                         const MoleculeMultipoles &b,
                         const ForceFieldParams &ff,
                         const MultipoleInteractions::Config &elec_config = {});

/// Build a ForceFieldParams loaded with the built-in element-based Williams DE
/// Buckingham parameters (H, C, N, O).
ForceFieldParams williams_de_force_field();

/// Build a ForceFieldParams for the FIT (Williams/Cox) typed exp-6 set: typed
/// pair parameters (with the H_F1/H_F2 split) plus the element-based Williams DE
/// table as a fallback for any atom whose type is unparameterised. Atom typing
/// is enabled (callers must populate MoleculeMultipoles::type_codes).
ForceFieldParams fit_force_field();

/// Build a ForceFieldParams for the W99 (Williams 1999) typed exp-6 set, with
/// the element-based Williams DE table as a fallback. Atom typing is enabled.
ForceFieldParams williams_typed_force_field();

/**
 * @brief A selectable short-range (repulsion-dispersion) parameter set.
 *
 * The single place that maps a user-facing name onto one of the builders
 * above, shared by `occ cg -m` and `occ dma --csp-force-field` so the two
 * agree on what "fit" or "w99" means. Adding a set is one entry in
 * short_range_model_registry() and one branch in make_force_field().
 */
struct ShortRangeModel {
  std::string name;        ///< canonical name, recorded in written output
  std::string description; ///< one line, shown in --help
  /// Atom typing the parameters are indexed by: "neighcrys" (W99 labels),
  /// "neighcrys-fit" (FIT labels), or "none" for element-based.
  std::string atom_typing;
  std::vector<std::string> aliases;

  bool typed() const { return atom_typing != "none"; }
};

/// Every selectable short-range parameter set.
const std::vector<ShortRangeModel> &short_range_model_registry();

/// Canonical names and aliases, for CLI validation.
std::vector<std::string> short_range_model_names();

/// Resolve an exact name or alias.
/// @throws std::runtime_error if the name is not known.
const ShortRangeModel &short_range_model_from_string(const std::string &name);

/// Resolve by substring, for compound model names like "dma-fit" or
/// "williams-99". Falls back to the element-based set when nothing matches.
const ShortRangeModel &short_range_model_for_model_name(const std::string &model_name);

/// Build the ForceFieldParams for a registry entry.
ForceFieldParams make_force_field(const ShortRangeModel &model);

/// Among the given element atomic numbers, the unique pairs (Zmin, Zmax) that
/// \p ff has no exp-6 parameters for. dimer_interaction_energy() contributes
/// electrostatics only for such pairs, so callers can warn about incomplete
/// repulsion/dispersion coverage. \p elements may contain duplicates; the
/// result is sorted and de-duplicated.
std::vector<std::pair<int, int>>
missing_exp6_pairs(const std::vector<int> &elements, const ForceFieldParams &ff);

} // namespace occ::mults
