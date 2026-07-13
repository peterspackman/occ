#pragma once
#include <occ/core/linear_algebra.h>
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

} // namespace occ::crystal

namespace occ::crystal::impl {
// defined in the generated subgroup_data.cpp
extern const int num_subgroup_edges;
extern const int subgroup_offsets[231];
extern const unsigned char subgroup_edge_data[];
} // namespace occ::crystal::impl
