#pragma once
#include <cmath>

namespace occ::gto {

enum ShellOrder { Default, Gaussian, Molden };

template <bool cartesian, ShellOrder order = Default, typename F>
inline void iterate_over_shell(F &f, int l) {
  if constexpr (order == ShellOrder::Default) {
    if constexpr (cartesian) {
      int i, j, k;
      for (i = l; i >= 0; i--) {
        for (j = l - i; j >= 0; j--) {
          k = l - i - j;
          f(i, j, k, l);
        }
      }
    } else {
      for (int m = -l; m <= l; m++) {
        f(l, m);
      }
    }
  } else if constexpr (order == ShellOrder::Gaussian) {
    if constexpr (cartesian) {
      switch (l) {
      case 0:
        f(0, 0, 0, l);
        break;
      case 1:
        f(1, 0, 0, l);
        f(0, 1, 0, l);
        f(0, 0, 1, l);
        break;
      case 2:
        f(2, 0, 0, l); // xx
        f(0, 2, 0, l); // yy
        f(0, 0, 2, l); // zz
        f(1, 1, 0, l); // xy
        f(1, 0, 1, l); // xz
        f(0, 1, 1, l); // yz
        break;
      case 3:
        f(3, 0, 0, l); // xxx
        f(0, 3, 0, l); // yyy
        f(0, 0, 3, l); // zzz
        f(1, 2, 0, l); // xyy
        f(2, 1, 0, l); // xxy
        f(2, 0, 1, l); // xxz
        f(1, 0, 2, l); // xzz
        f(0, 1, 2, l); // yzz
        f(0, 2, 1, l); // yyz
        f(1, 1, 1, l); // xyz
        break;
      /* Apparently GDMA expects this order
      case 4:
          f(4, 0, 0, l); // xxxx
          f(0, 4, 0, l); // yyyy
          f(0, 0, 4, l); // zzzz
          f(3, 1, 0, l); // xxxy
          f(3, 0, 1, l); // xxxz
          f(1, 3, 0, l); // xyyy
          f(0, 3, 1, l); // yyyz
          f(1, 0, 3, l); // xzzz
          f(0, 1, 3, l); // yzzz
          f(2, 2, 0, l); // xxyy
          f(2, 0, 2, l); // xxzz
          f(0, 2, 2, l); // yyzz
          f(2, 1, 1, l); // xxyz
          f(1, 2, 1, l); // xyyz
          f(1, 1, 2, l); // xyzz
          break;
      But this is the actual order G09 puts out...
      */
      default:
        int i, j, k;
        for (i = l; i >= 0; i--) {
          for (j = l - i; j >= 0; j--) {
            k = l - i - j;
            f(i, j, k, l);
          }
        }
        break;
      }
    } else {
      for (int m = 0; m != l + 1; m = (m > 0 ? -m : 1 - m)) {
        f(l, m);
      }
    }
  } else if constexpr (order == ShellOrder::Molden) {
    // only spherical case here
    for (int m = 0; m != l + 1; m = (m > 0 ? -m : 1 - m)) {
      f(l, m);
    }
  }
}

template <ShellOrder order = Default>
inline int shell_index_cartesian(int i, int j, int k, int l) = delete;

template <>
inline int shell_index_cartesian<ShellOrder::Default>(int i, int j, int k,
                                                      int l) {
  return ((((l - i + 1) * (l - i)) >> 1) + l - i - j);
}

template <>
inline int shell_index_cartesian<ShellOrder::Gaussian>(int i, int j, int k,
                                                       int l) {
  int idx_found = -1;
  int idx = 0;
  auto f = [&](int pi, int pj, int pk, int l) {
    if (pi == i && pj == j && pk == k) {
      idx_found = idx;
    }
    idx++;
  };
  iterate_over_shell<true, ShellOrder::Gaussian>(f, l);
  return idx_found;
}

template <ShellOrder order = Default>
inline int shell_index_spherical(int l, int m) = delete;

template <>
inline int shell_index_spherical<ShellOrder::Default>(int l, int m) {
  return m + l;
}

template <>
inline int shell_index_spherical<ShellOrder::Gaussian>(int l, int m) {
  return 2 * std::abs(m) + (m > 0 ? -1 : 0);
}

template <> inline int shell_index_spherical<ShellOrder::Molden>(int l, int m) {
  return 2 * std::abs(m) + (m > 0 ? -1 : 0);
}

} // namespace occ::gto
