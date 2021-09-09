#pragma once
#include <libint2/cgshell_ordering.h>
#include <libint2/shgshell_ordering.h>

namespace occ::gto {

template <bool cartesian, typename F>
inline void iterate_over_shell(F &f, int l) {
    if constexpr (cartesian) {
        int i, j, k;
        FOR_CART(i, j, k, l)
        f(i, j, k, l);
        END_FOR_CART
    } else {
        int m;
        FOR_SOLIDHARM(l, m)
        f(l, m);
        END_FOR_SOLIDHARM
    }
}

} // namespace occ::gto
