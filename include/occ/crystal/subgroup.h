#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <optional>
#include <string>
#include <vector>

namespace occ::crystal {

/**
 * \brief Whether a group-subgroup relation preserves translations or class.
 */
enum class SubgroupType {
  /// translationengleiche: the lattice is kept, point group symmetry is lost
  Translationengleiche,
  /// klassengleiche: the point group is kept, translations are lost
  Klassengleiche,
};

/**
 * \brief A maximal subgroup of a space group.
 *
 * The relation is described by the subgroup's space group number, the index
 * [G:H], and the transformation to the subgroup's standard setting: a basis
 * change \f$P\f$ (columns are the new basis vectors in terms of the old) and an
 * origin shift \f$p\f$, following the International Tables convention
 *
 * \f[ (a', b', c') = (a, b, c) P, \qquad x' = P^{-1}(x - p) \f]
 *
 * \note \f$\det P \ne 1\f$ in general, and it is **not** equal to the index
 * either. Both the parent and the subgroup are given in their conventional
 * (possibly centred) cells, so
 *
 * \f[ \det P = \text{index} \times \frac{\text{centring of } H}{\text{centring of } G} \f]
 *
 * For example P2 (2 operations) has a klassengleiche subgroup C2 (4 operations)
 * at index 2, with \f$\det P = 4\f$; and a centred parent may transform to a
 * primitive cell, giving \f$\det P < 1\f$. The invariant that always holds,
 * counting symmetry operations per unit volume, is
 *
 * \f[ n_G \cdot \det P = n_H \cdot \text{index} \f]
 *
 * so the naive \f$|H| \cdot \text{index} = |G|\f$ is *not* a valid check.
 */
struct MaximalSubgroup {
  /// Space group number of the parent
  int parent{0};
  /// Space group number of the subgroup
  int subgroup{0};
  /// The index [G:H]
  int index{1};
  SubgroupType type{SubgroupType::Translationengleiche};
  /// Basis change; columns are the new basis vectors in terms of the old
  Mat3 basis_transform{Mat3::Identity()};
  /// Origin shift, in fractional coordinates of the *parent* cell
  Vec3 origin_shift{Vec3::Zero()};

  inline bool is_translationengleiche() const {
    return type == SubgroupType::Translationengleiche;
  }
  inline bool is_klassengleiche() const {
    return type == SubgroupType::Klassengleiche;
  }

  /// The 3x4 matrix [P | p] as an ITA-style string, e.g. "a-b,a+b,c"
  std::string to_string() const;
};

/**
 * \brief The maximal subgroups of a space group.
 *
 * \param space_group_number the ITA number of the parent, 1-230
 *
 * \throws std::out_of_range if the number is not in 1-230
 *
 * \note Klassengleiche relations are tabulated only up to index 9. Maximal
 * *isomorphic* subgroups are infinite in number, so no finite table can be
 * complete.
 */
const std::vector<MaximalSubgroup> &maximal_subgroups(int space_group_number);

/// The maximal subgroups of a given type.
std::vector<MaximalSubgroup> maximal_subgroups(int space_group_number,
                                               SubgroupType type);

/**
 * \brief A subgroup reached by descending one or more maximal-subgroup steps.
 *
 * Non-maximal subgroups are reached by composing maximal relations along a path
 * through the graph. The transformation and index of the composite are the
 * composition of the steps taken.
 */
struct SubgroupPath {
  /// Space group number of the subgroup reached
  int subgroup{0};
  /// The total index, i.e. the product of the indices of each step
  int index{1};
  /// The composed basis change
  Mat3 basis_transform{Mat3::Identity()};
  /// The composed origin shift, in fractional coordinates of the parent cell
  Vec3 origin_shift{Vec3::Zero()};
  /// The maximal-subgroup steps taken, from the parent down
  std::vector<MaximalSubgroup> steps;

  /// True if every step was translationengleiche (i.e. the lattice is kept)
  bool is_translationengleiche() const;
  /// The number of steps taken from the parent
  inline size_t depth() const { return steps.size(); }
};

struct SubgroupSearchParameters {
  /// Only report subgroups whose total index is at most this
  int max_index{4};
  /// Only descend this many maximal-subgroup steps
  int max_depth{4};
  /// Follow translationengleiche relations
  bool translationengleiche{true};
  /// Follow klassengleiche relations
  bool klassengleiche{true};
};

/**
 * \brief All subgroups reachable from a space group, by traversing the graph of
 * maximal-subgroup relations.
 *
 * Composes the basis change and origin shift along each path. Paths reaching the
 * same subgroup by the same transformation are deduplicated.
 *
 * \note These are **paths, not distinct subgroups**. Two different descents can
 * land on the same subgroup carrying different transformations, and both are
 * reported. For example P2_1/c reaches P1 at index 4 three times -- via P-1, via
 * P2_1 and via Pc -- and since P1 is unconstrained in its origin, all three are
 * in fact the same subgroup. The descent route is itself meaningful (it is what
 * Bilbao's SUBGROUPGRAPH shows), so they are kept; deduplicate on the actual
 * symmetry operations if you want distinct subgroups.
 *
 * \note Subgroups are reported individually, not grouped into conjugacy classes
 * under the affine normalizer (which is how Bilbao presents them).
 */
std::vector<SubgroupPath>
subgroup_paths(int space_group_number,
               const SubgroupSearchParameters &params = {});

/**
 * \brief A change of setting from a space group to one of its subgroups.
 *
 * Both MaximalSubgroup and SubgroupPath convert to one of these; it is what
 * `to_subgroup` actually needs.
 */
struct SubgroupTransform {
  /// Space group number of the subgroup.
  ///
  /// Zero, i.e. invalid, by default: a default-constructed transform is not a
  /// no-op, and silently defaulting to 1 would mean "descend to P1" -- which
  /// `to_subgroup` would dutifully do, destroying the symmetry. It fails loudly
  /// instead.
  int subgroup{0};
  /// Basis change; columns are the new basis vectors in terms of the old
  Mat3 basis_transform{Mat3::Identity()};
  /// Origin shift, in fractional coordinates of the parent cell
  Vec3 origin_shift{Vec3::Zero()};

  SubgroupTransform() = default;
  SubgroupTransform(const MaximalSubgroup &);
  SubgroupTransform(const SubgroupPath &);
};

/**
 * \brief Re-describe a crystal in one of its subgroups.
 *
 * The structure itself is unchanged -- the same atoms in the same places. What
 * changes is the description: the cell is transformed by \f$P\f$ (for a
 * klassengleiche relation this is a supercell, so \f$\det P > 1\f$ and the atoms
 * are tiled), the origin is shifted by \f$p\f$, and the asymmetric unit is
 * re-derived under the subgroup's symmetry, splitting orbits that the parent
 * held together.
 *
 * The subgroup's symmetry operations are taken directly from its space group
 * number: because the transformation is tabulated, no symmetry identification is
 * needed.
 *
 * The asymmetric unit is chosen molecule by molecule, so a molecule comes
 * through whole wherever the subgroup's symmetry allows it -- see
 * `with_molecular_asymmetric_unit`. Pass `molecular_asymmetric_unit = false` to
 * take the cheaper atom-by-atom choice, which is equally valid but can scatter a
 * molecule across symmetry images.
 *
 * \param crystal the parent structure
 * \param transform the subgroup relation to apply
 * \param tolerance fractional-coordinate tolerance for matching atoms
 * \param molecular_asymmetric_unit keep molecules intact in the asymmetric unit
 *
 * \returns the same structure, described in the subgroup
 *
 * \throws std::runtime_error if the subgroup's symmetry does not in fact hold
 *         for the transformed structure (which would mean the transformation and
 *         the subgroup's standard setting disagree)
 */
Crystal to_subgroup(const Crystal &crystal, const SubgroupTransform &transform,
                    double tolerance = 1e-4,
                    bool molecular_asymmetric_unit = true);

/**
 * \brief Compose two subgroup transformations, applied left to right.
 *
 * \f$x'' = P_2^{-1}(P_1^{-1}(x - p_1) - p_2) = (P_1 P_2)^{-1}(x - (p_1 + P_1 p_2))\f$
 */
SubgroupTransform compose(const SubgroupTransform &first,
                          const SubgroupTransform &second);

/**
 * \brief Re-describe a crystal in the standard (ITA reference) setting of its
 * space group.
 *
 * P2_1/c, P2_1/a and P2_1/n are all space group 14, and tabulated data --
 * including the subgroup relations here -- is given for the standard setting
 * only. Real structures frequently are not in it: of 47 experimental structures
 * checked, several were P2_1/a or P2_1/n. Converting first makes the tables
 * apply.
 *
 * The structure is unchanged; only its description is.
 */
Crystal to_standard_setting(const Crystal &crystal, double tolerance = 1e-4);

/**
 * \brief Re-choose a crystal's asymmetric unit so that groups of atoms stay
 * together.
 *
 * The asymmetric unit is a *choice*: any set of atoms that generates the unit
 * cell exactly once under the space group will do. Picking one representative
 * per orbit in whatever order the atoms happen to come in is valid but
 * chemically useless -- it can scatter a molecule across several symmetry
 * images, so the asymmetric unit is a handful of atoms from here and a handful
 * from there.
 *
 * Walking the atoms group by group instead makes each group land in the
 * asymmetric unit whole, wherever the symmetry permits it. Where it does not --
 * a group sitting on a special position, mapped onto itself by part of the
 * group -- only its own asymmetric part can be taken, which is correct and
 * unavoidable.
 *
 * \param crystal the structure to re-describe (unchanged; only the choice of
 *        asymmetric unit differs)
 * \param groups one entry per unit cell atom, in the order of
 *        `crystal.unit_cell_atoms()`, giving the group that atom belongs to.
 *        Atoms sharing a value are kept together where symmetry allows.
 * \param tolerance fractional-coordinate tolerance for matching atoms
 *
 * \throws std::invalid_argument if `groups` is not one entry per unit cell atom
 */
Crystal with_grouped_asymmetric_unit(const Crystal &crystal, const IVec &groups,
                                     double tolerance = 1e-4);

/**
 * \brief Re-choose the asymmetric unit so that whole molecules stay together.
 *
 * `with_grouped_asymmetric_unit` with the groups taken from the crystal's own
 * molecules. This is what makes "Z' = 1 with the whole molecule as the
 * asymmetric unit" a construction rather than a coincidence.
 */
Crystal with_molecular_asymmetric_unit(const Crystal &crystal,
                                       double tolerance = 1e-4);

/**
 * \brief The number of molecules in the asymmetric unit, Z'.
 *
 * \f[ Z' = \frac{Z}{\text{multiplicity of the general position}} \f]
 *
 * i.e. the number of molecules in the unit cell divided by the number of
 * symmetry operations. \f$Z' < 1\f$ means the molecule sits on a special
 * position: it is mapped onto itself by some symmetry operation of the crystal,
 * so only a fraction of it (a half for an inversion centre, a third for a
 * 3-fold, ...) is crystallographically independent. Benzene in Pbca has
 * \f$Z' = 1/2\f$.
 */
double z_prime(const Crystal &crystal);

/**
 * \brief True if the asymmetric unit consists of whole molecules.
 *
 * When a molecule sits on a special position the asymmetric unit holds only part
 * of it, and the rest is generated by symmetry. This is what makes \f$Z' < 1\f$
 * structures awkward to work with: the asymmetric unit is not a chemically
 * meaningful object.
 */
bool has_whole_molecule_asymmetric_unit(const Crystal &crystal);

struct ZPrimeSearchParameters {
  /// The Z' wanted in the subgroup, usually 1. If unset, any Z' is accepted and
  /// the search just takes the lowest-index subgroup meeting the other
  /// requirements -- which is what you want when Z' = 1 is unreachable with
  /// whole molecules. A molecule sitting on a mirror plane, for instance, may
  /// already have Z' = 1 while the asymmetric unit is still only half a
  /// molecule; dropping the mirror makes it whole but takes Z' to 2.
  std::optional<double> target{1.0};
  /// Only consider subgroups up to this index
  int max_index{4};
  /// Only descend this many maximal-subgroup steps
  int max_depth{2};
  /// Follow translationengleiche relations (these keep the cell)
  bool translationengleiche{true};
  /// Follow klassengleiche relations (these enlarge the cell)
  bool klassengleiche{false};
  /// Require the resulting asymmetric unit to hold whole molecules
  bool require_whole_molecules{true};
  /// Fractional-coordinate tolerance for matching atoms
  double tolerance{1e-4};
};

/**
 * \brief Find a subgroup in which the crystal has the target Z'.
 *
 * The common case: a molecule sitting on a special position gives \f$Z' < 1\f$,
 * so the asymmetric unit is only a fragment. Descending into a subgroup that
 * lacks the offending symmetry element makes the whole molecule independent,
 * without changing the structure at all.
 *
 * Candidates are tried in order of increasing index, so the least drastic
 * descent that works is the one returned. Only translationengleiche relations
 * are followed by default, since those keep the unit cell.
 *
 * \returns the transformation to apply (feed it to `to_subgroup`), or nullopt if
 *          no subgroup within the search bounds gives the target Z'
 */
std::optional<SubgroupTransform>
find_subgroup_for_z_prime(const Crystal &crystal,
                          const ZPrimeSearchParameters &params = {});

} // namespace occ::crystal

namespace occ::crystal::impl {
// defined in the generated subgroup_data.cpp
extern const int num_subgroup_edges;
extern const int subgroup_offsets[231];
extern const unsigned char subgroup_edge_data[];
} // namespace occ::crystal::impl
