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

} // namespace occ::crystal
