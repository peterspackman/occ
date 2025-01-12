#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/crystal/hkl.h>

namespace occ::crystal {

struct SiteIndex {
  int offset{0};
  HKL hkl;

  inline bool operator==(const SiteIndex &rhs) const {
    return (offset == rhs.offset) && (hkl == rhs.hkl);
  }
};

struct SiteIndexHash {
  using is_avalanching = void;
  [[nodiscard]] auto operator()(SiteIndex const &idx) const noexcept
      -> uint64_t {
    static_assert(std::has_unique_object_representations_v<SiteIndex>);
    return ankerl::unordered_dense::detail::wyhash::hash(&idx, sizeof(idx));
  }
};

} // namespace occ::crystal
