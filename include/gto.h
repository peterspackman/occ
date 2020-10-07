#include <libint2/shell.h>
#include <libint2/basis.h>
#include "linear_algebra.h"
#include <string>
#include <vector>
#include <array>
#include <libint2/cgshell_ordering.h>
#include <libint2/shgshell_ordering.h>

namespace tonto::gto {

struct Momenta {
    int l{0};
    int m{0};
    int n{0};

    std::string to_string() const {
        int am = l + m + n;
        static char lsymb[] = "SPDFGHIKMNOQRTUVWXYZ";
        if (am == 0) return std::string(1, lsymb[0]);

        std::string suffix = "";
        for(int i = 0; i < l; i++) suffix += "x";
        for(int i = 0; i < m; i++) suffix += "y";
        for(int i = 0; i < n; i++) suffix += "z";

        return std::string(1, lsymb[am]) + suffix;
    }
};

std::vector<Momenta> cartesian_ordering(int l) {
    if(l == 0) return {{0, 0, 0}};
    int i = 0, j = 0, k = 0;
    std::vector<Momenta> powers;
    FOR_CART(i,j,k,l)
        powers.push_back({i, j, k});
    END_FOR_CART
    return powers;
}
}
