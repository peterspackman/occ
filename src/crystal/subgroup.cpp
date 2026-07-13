#include <Eigen/LU>
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <occ/crystal/subgroup.h>
#include <stdexcept>

namespace occ::crystal {

namespace {

constexpr int edge_stride = 27; // 3 header bytes + 12 (numerator, denominator)

MaximalSubgroup decode_edge(int parent, int edge) {
  const unsigned char *p = impl::subgroup_edge_data + edge * edge_stride;
  MaximalSubgroup result;
  result.parent = parent;
  result.subgroup = static_cast<int>(p[0]);
  result.index = static_cast<int>(p[1]);
  result.type = p[2] == 0 ? SubgroupType::Translationengleiche
                          : SubgroupType::Klassengleiche;

  // 3x4 [P | p], row-major, each entry a (numerator, denominator) pair
  const unsigned char *entry = p + 3;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      const int numerator = static_cast<signed char>(*entry++);
      const int denominator = static_cast<int>(*entry++);
      const double value = static_cast<double>(numerator) / denominator;
      if (j < 3)
        result.basis_transform(i, j) = value;
      else
        result.origin_shift(i) = value;
    }
  }
  return result;
}

void check_space_group_number(int n) {
  if (n < 1 || n > 230)
    throw std::out_of_range(
        fmt::format("space group number must be 1-230, got {}", n));
}

// Format one column of the basis change as e.g. "a-b" or "2c"
std::string format_axis(const Vec3 &column) {
  constexpr const char *names[3] = {"a", "b", "c"};
  std::string result;
  for (int i = 0; i < 3; i++) {
    const double v = column(i);
    if (std::abs(v) < 1e-9)
      continue;
    if (v > 0 && !result.empty())
      result += "+";
    if (v < 0)
      result += "-";
    const double magnitude = std::abs(v);
    if (std::abs(magnitude - 1.0) > 1e-9) {
      // render the small rationals that actually occur
      if (std::abs(magnitude - std::round(magnitude)) < 1e-9)
        result += fmt::format("{}", static_cast<int>(std::round(magnitude)));
      else
        result += fmt::format("{:g}", magnitude);
    }
    result += names[i];
  }
  return result.empty() ? "0" : result;
}

} // namespace

std::string MaximalSubgroup::to_string() const {
  std::string result =
      fmt::format("{},{},{}", format_axis(basis_transform.col(0)),
                  format_axis(basis_transform.col(1)),
                  format_axis(basis_transform.col(2)));
  if (!origin_shift.isZero(1e-9))
    result += fmt::format(";{:g},{:g},{:g}", origin_shift(0), origin_shift(1),
                          origin_shift(2));
  return result;
}

const std::vector<MaximalSubgroup> &maximal_subgroups(int space_group_number) {
  check_space_group_number(space_group_number);

  // decoded once, on first use
  static const std::vector<std::vector<MaximalSubgroup>> table = []() {
    std::vector<std::vector<MaximalSubgroup>> result(231);
    for (int sg = 1; sg <= 230; sg++) {
      const int begin = impl::subgroup_offsets[sg - 1];
      const int end = impl::subgroup_offsets[sg];
      result[sg].reserve(end - begin);
      for (int edge = begin; edge < end; edge++)
        result[sg].push_back(decode_edge(sg, edge));
    }
    return result;
  }();

  return table[space_group_number];
}

std::vector<MaximalSubgroup> maximal_subgroups(int space_group_number,
                                               SubgroupType type) {
  std::vector<MaximalSubgroup> result;
  for (const auto &subgroup : maximal_subgroups(space_group_number)) {
    if (subgroup.type == type)
      result.push_back(subgroup);
  }
  return result;
}

bool SubgroupPath::is_translationengleiche() const {
  return std::all_of(steps.begin(), steps.end(),
                     [](const MaximalSubgroup &step) {
                       return step.is_translationengleiche();
                     });
}

std::vector<SubgroupPath>
subgroup_paths(int space_group_number,
               const SubgroupSearchParameters &params) {
  check_space_group_number(space_group_number);

  std::vector<SubgroupPath> result;
  if (params.max_index < 2 || params.max_depth < 1)
    return result;

  // Different paths can arrive at the same subgroup with the same
  // transformation; keep only one. Quantize the transformation so that
  // floating point noise doesn't defeat the deduplication -- every entry is a
  // rational with denominator at most 12, so twelfths are exact.
  auto key_of = [](const SubgroupPath &path) {
    std::string key = fmt::format("{}|", path.subgroup);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++)
        key += fmt::format("{},", std::llround(path.basis_transform(i, j) * 12));
      key += fmt::format("{};", std::llround(path.origin_shift(i) * 12));
    }
    return key;
  };
  ankerl::unordered_dense::set<std::string> seen;

  auto follows = [&params](const MaximalSubgroup &step) {
    return step.is_translationengleiche() ? params.translationengleiche
                                          : params.klassengleiche;
  };

  // breadth-first over the graph of maximal-subgroup relations
  std::vector<SubgroupPath> frontier;
  SubgroupPath root;
  root.subgroup = space_group_number;
  frontier.push_back(root);

  for (int depth = 0; depth < params.max_depth && !frontier.empty(); depth++) {
    std::vector<SubgroupPath> next;
    for (const SubgroupPath &path : frontier) {
      for (const MaximalSubgroup &step : maximal_subgroups(path.subgroup)) {
        if (!follows(step))
          continue;
        const int index = path.index * step.index;
        if (index > params.max_index)
          continue;

        // compose:  x' = P2^-1 (P1^-1 (x - p1) - p2)
        //              = (P1 P2)^-1 (x - (p1 + P1 p2))
        SubgroupPath child;
        child.subgroup = step.subgroup;
        child.index = index;
        child.basis_transform = path.basis_transform * step.basis_transform;
        child.origin_shift =
            path.origin_shift + path.basis_transform * step.origin_shift;
        child.steps = path.steps;
        child.steps.push_back(step);

        if (!seen.insert(key_of(child)).second)
          continue;
        result.push_back(child);
        next.push_back(child);
      }
    }
    frontier = std::move(next);
  }

  std::sort(result.begin(), result.end(),
            [](const SubgroupPath &a, const SubgroupPath &b) {
              return std::tie(a.index, a.subgroup) <
                     std::tie(b.index, b.subgroup);
            });
  return result;
}

SubgroupTransform::SubgroupTransform(const MaximalSubgroup &s)
    : subgroup(s.subgroup), basis_transform(s.basis_transform),
      origin_shift(s.origin_shift) {}

SubgroupTransform::SubgroupTransform(const SubgroupPath &p)
    : subgroup(p.subgroup), basis_transform(p.basis_transform),
      origin_shift(p.origin_shift) {}

namespace {

// wrap a fractional coordinate into [0, 1)
inline double wrap(double x) {
  double r = x - std::floor(x);
  // guard against -0.0 and values that round up to exactly 1
  if (r >= 1.0 - 1e-12 || r < 0.0)
    r = 0.0;
  return r;
}

inline Vec3 wrap(const Vec3 &v) {
  return Vec3(wrap(v(0)), wrap(v(1)), wrap(v(2)));
}

// smallest component-wise difference between two fractional coordinates,
// accounting for periodicity
inline double periodic_distance(const Vec3 &a, const Vec3 &b) {
  Vec3 d = a - b;
  for (int i = 0; i < 3; i++)
    d(i) -= std::round(d(i));
  return d.norm();
}

} // namespace

Crystal to_subgroup(const Crystal &crystal, const SubgroupTransform &transform,
                    double tolerance) {
  const Mat3 &p_matrix = transform.basis_transform;
  const Vec3 &p_shift = transform.origin_shift;
  const double det = p_matrix.determinant();
  if (std::abs(det) < 1e-9)
    throw std::runtime_error("subgroup basis transform is singular");

  // The tabulated transformations target the ITA standard setting. For the 24
  // groups with two origin choices (Pnnn, Fddd, ...) SpaceGroup(number) gives
  // origin choice 1, but the standard is choice 2 -- so ask for the standard
  // setting explicitly rather than trusting the by-number lookup.
  const SpaceGroup subgroup_sg = SpaceGroup(transform.subgroup).standard_setting();
  const Mat3 p_inverse = p_matrix.inverse();

  // The new cell: (a', b', c') = (a, b, c) P, i.e. columns transform by P.
  //
  // from_lattice_vectors, not the UnitCell(Mat3) constructor: the latter
  // canonicalizes the basis to the conventional orientation (a along x), which
  // silently *rotates* the structure whenever P permutes axes. Preserving the
  // basis keeps the child's cartesian coordinates identical to the parent's, so
  // the two structures can be compared directly.
  const UnitCell new_cell =
      UnitCell::from_lattice_vectors(crystal.unit_cell().direct() * p_matrix);

  // The new cell may span several parent cells (det P > 1 for a klassengleiche
  // relation), so tile the parent unit cell over enough lattice translations to
  // cover it. The corners of the new cell, expressed in parent fractional
  // coordinates, bound the range: x = P x' + p.
  Vec3 lower = Vec3::Constant(std::numeric_limits<double>::max());
  Vec3 upper = Vec3::Constant(std::numeric_limits<double>::lowest());
  for (int i = 0; i < 8; i++) {
    Vec3 corner((i & 1) ? 1.0 : 0.0, (i & 2) ? 1.0 : 0.0, (i & 4) ? 1.0 : 0.0);
    Vec3 x = p_matrix * corner + p_shift;
    lower = lower.cwiseMin(x);
    upper = upper.cwiseMax(x);
  }

  const CrystalAtomRegion &parent_atoms = crystal.unit_cell_atoms();
  const AsymmetricUnit &parent_asym = crystal.asymmetric_unit();
  const Eigen::Index n_parent = parent_atoms.size();

  struct Site {
    Vec3 frac;
    int element;
    int parent_asym_index;
  };
  std::vector<Site> sites;

  for (int h = static_cast<int>(std::floor(lower(0))) - 1;
       h <= static_cast<int>(std::ceil(upper(0))) + 1; h++) {
    for (int k = static_cast<int>(std::floor(lower(1))) - 1;
         k <= static_cast<int>(std::ceil(upper(1))) + 1; k++) {
      for (int l = static_cast<int>(std::floor(lower(2))) - 1;
           l <= static_cast<int>(std::ceil(upper(2))) + 1; l++) {
        const Vec3 translation(h, k, l);
        for (Eigen::Index i = 0; i < n_parent; i++) {
          // into the new basis: x' = P^-1 (x - p)
          const Vec3 x = parent_atoms.frac_pos.col(i) + translation;
          const Vec3 x_new = wrap(p_inverse * (x - p_shift));

          const bool duplicate =
              std::any_of(sites.begin(), sites.end(), [&](const Site &s) {
                return periodic_distance(s.frac, x_new) < tolerance;
              });
          if (duplicate)
            continue;
          sites.push_back({x_new, parent_atoms.atomic_numbers(i),
                           static_cast<int>(parent_atoms.asym_idx(i))});
        }
      }
    }
  }

  // The transformed cell holds det(P) times as many atoms as the parent's.
  const double expected = n_parent * std::abs(det);
  if (std::abs(static_cast<double>(sites.size()) - expected) > 0.5) {
    throw std::runtime_error(fmt::format(
        "subgroup transform produced {} atoms in the new cell, expected {} "
        "({} parent atoms x det(P) = {:.3f})",
        sites.size(), static_cast<int>(std::llround(expected)), n_parent, det));
  }

  // Reduce to an asymmetric unit under the subgroup: one representative per
  // orbit. The parent held some of these orbits together; under H they split,
  // which is the whole point.
  const auto &subgroup_ops = subgroup_sg.symmetry_operations();
  std::vector<bool> assigned(sites.size(), false);
  std::vector<size_t> representatives;

  for (size_t i = 0; i < sites.size(); i++) {
    if (assigned[i])
      continue;
    representatives.push_back(i);
    assigned[i] = true;

    for (const auto &op : subgroup_ops) {
      const Vec3 image = wrap(op.rotation() * sites[i].frac + op.translation());
      bool matched = false;
      for (size_t j = 0; j < sites.size(); j++) {
        if (periodic_distance(sites[j].frac, image) < tolerance) {
          if (sites[j].element != sites[i].element)
            continue;
          assigned[j] = true;
          matched = true;
          break;
        }
      }
      if (!matched) {
        // The claimed subgroup symmetry does not hold for this structure. That
        // means the tabulated transformation and the subgroup's standard
        // setting disagree -- a bug, not a property of the input.
        throw std::runtime_error(fmt::format(
            "symmetry operation '{}' of space group {} does not map the "
            "transformed structure onto itself",
            op.to_string(), transform.subgroup));
      }
    }
  }

  const Eigen::Index n_asym = representatives.size();
  Mat3N positions(3, n_asym);
  IVec numbers(n_asym);
  Vec occupations(n_asym);
  std::vector<std::string> labels;
  labels.reserve(n_asym);

  // When an orbit splits, the new sites all descend from one parent site; give
  // them distinct labels.
  ankerl::unordered_dense::map<int, int> seen_parent_site;
  for (Eigen::Index i = 0; i < n_asym; i++) {
    const Site &site = sites[representatives[i]];
    positions.col(i) = site.frac;
    numbers(i) = site.element;
    occupations(i) = parent_asym.occupations(site.parent_asym_index);

    const std::string base = parent_asym.labels.empty()
                                 ? fmt::format("{}", site.element)
                                 : parent_asym.labels[site.parent_asym_index];
    const int n = seen_parent_site[site.parent_asym_index]++;
    labels.push_back(n == 0 ? base
                            : fmt::format("{}{}", base,
                                          static_cast<char>('a' + n - 1)));
  }

  AsymmetricUnit asym(positions, numbers, labels);
  asym.occupations = occupations;
  return Crystal(asym, subgroup_sg, new_cell);
}

SubgroupTransform compose(const SubgroupTransform &first,
                          const SubgroupTransform &second) {
  SubgroupTransform result;
  result.subgroup = second.subgroup;
  result.basis_transform = first.basis_transform * second.basis_transform;
  result.origin_shift =
      first.origin_shift + first.basis_transform * second.origin_shift;
  return result;
}

Crystal to_standard_setting(const Crystal &crystal, double tolerance) {
  const SpaceGroup &sg = crystal.space_group();
  if (sg.is_standard_setting())
    return crystal;

  auto [p_matrix, p_shift] = sg.standard_setting_transform();
  SubgroupTransform transform;
  transform.subgroup = sg.number();
  transform.basis_transform = p_matrix;
  transform.origin_shift = p_shift;
  return to_subgroup(crystal, transform, tolerance);
}

double z_prime(const Crystal &crystal) {
  const double n_molecules = crystal.unit_cell_molecules().size();
  const double n_symops = crystal.space_group().symmetry_operations().size();
  if (n_symops == 0)
    return 0.0;
  return n_molecules / n_symops;
}

bool has_whole_molecule_asymmetric_unit(const Crystal &crystal) {
  // If a molecule sits on a special position, symmetry has to complete it, so
  // the asymmetric unit holds fewer atoms than the symmetry-unique molecules do.
  size_t atoms_in_unique_molecules = 0;
  for (const auto &molecule : crystal.symmetry_unique_molecules())
    atoms_in_unique_molecules += molecule.size();
  return crystal.asymmetric_unit().size() == atoms_in_unique_molecules;
}

std::optional<SubgroupTransform>
find_subgroup_for_z_prime(const Crystal &crystal,
                          const ZPrimeSearchParameters &params) {
  auto satisfies = [&params](const Crystal &candidate) {
    if (params.target &&
        std::abs(z_prime(candidate) - *params.target) > 1e-9)
      return false;
    if (params.require_whole_molecules &&
        !has_whole_molecule_asymmetric_unit(candidate))
      return false;
    return true;
  };

  // already there?
  if (satisfies(crystal))
    return SubgroupTransform{};

  SubgroupSearchParameters search;
  search.max_index = params.max_index;
  search.max_depth = params.max_depth;
  search.translationengleiche = params.translationengleiche;
  search.klassengleiche = params.klassengleiche;

  // The tabulated transformations are written for the standard setting, but real
  // structures often aren't in it (P2_1/a and P2_1/n are both space group 14).
  // Convert first, and fold the setting change into the transformation we hand
  // back, so it applies to the crystal the caller actually gave us.
  const SpaceGroup &sg = crystal.space_group();
  SubgroupTransform to_standard;
  to_standard.subgroup = sg.number();
  std::tie(to_standard.basis_transform, to_standard.origin_shift) =
      sg.standard_setting_transform();

  // subgroup_paths returns candidates sorted by increasing index, so the first
  // hit is the least drastic descent that works.
  for (const auto &path : subgroup_paths(sg.number(), search)) {
    const SubgroupTransform transform =
        compose(to_standard, SubgroupTransform(path));
    try {
      Crystal candidate = to_subgroup(crystal, transform, params.tolerance);
      if (satisfies(candidate))
        return transform;
    } catch (const std::exception &) {
      // this descent doesn't apply to this structure; try the next
      continue;
    }
  }
  return std::nullopt;
}

} // namespace occ::crystal
