#include <tonto/io/conversion.h>
#include <tonto/core/util.h>
#include <tonto/core/logger.h>
#include <tonto/qm/gto.h>

namespace tonto::io::conversion {

namespace orb {
tonto::MatRM from_gaussian(const tonto::qm::BasisSet &basis, const tonto::MatRM& mo)
{
    // no reordering should occur unless there are d, f, g, h etc. functions
    using tonto::util::index_of;
    if(tonto::qm::max_l(basis) < 2) return mo;
    struct xyz {
        uint_fast8_t x{0};
        uint_fast8_t y{0};
        uint_fast8_t z{0};
        bool operator ==(const xyz& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    };

    tonto::log::debug("Reordering MO coefficients from Gaussian ordering to internal convention");
    auto shell2bf = basis.shell2bf();
    tonto::MatRM result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for(size_t i = 0; i < basis.size(); i++)
    {
        const auto& shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if(l < 2) {
            result.block(bf_first, 0, shell_size, ncols) = mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> gaussian_order;
        switch(l) {
        case 2:
            gaussian_order = {
                // xx, yy, zz, xy, xz, yz
                {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                {1, 1, 0}, {1, 0, 1}, {0, 1, 1}
            };
            break;
        case 3:
            gaussian_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3},
                {1, 2, 0}, {2, 1, 0}, {2, 0, 1},
                {1, 0, 2}, {0, 1, 2}, {0, 2, 1},
                {1, 1, 1}
            };
            break;
        }
        if (gaussian_order.size() == 0) {
            tonto::log::warn("Unknown Gaussian ordering for shell with angular momentum {}, not reordering", l);
            continue;
        }

        int xp, yp, zp;
        size_t our_idx{0};
        FOR_CART(xp, yp, zp, l)
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp), static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            tonto::log::debug("Setting row {} <- row {}", our_idx, gaussian_idx);
            result.row(bf_first + our_idx) = mo.row(bf_first + gaussian_idx);
            double normalization_factor = tonto::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + our_idx) *= normalization_factor;
            our_idx++;
        END_FOR_CART
    }
    return result;
}


tonto::MatRM to_gaussian(const tonto::qm::BasisSet &basis, const tonto::MatRM &mo)
{
    // no reordering should occur unless there are d, f, g, h etc. functions
    using tonto::util::index_of;
    if(tonto::qm::max_l(basis) < 2) return mo;
    struct xyz {
        uint_fast8_t x{0};
        uint_fast8_t y{0};
        uint_fast8_t z{0};
        bool operator ==(const xyz& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    };

    tonto::log::debug("Reordering MO coefficients from Gaussian ordering to internal convention");
    auto shell2bf = basis.shell2bf();
    tonto::MatRM result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for(size_t i = 0; i < basis.size(); i++)
    {
        const auto& shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if(l < 2) {
            result.block(bf_first, 0, shell_size, ncols) = mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> gaussian_order;
        switch(l) {
        case 2:
            gaussian_order = {
                // xx, yy, zz, xy, xz, yz
                {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                {1, 1, 0}, {1, 0, 1}, {0, 1, 1}
            };
            break;
        case 3:
            gaussian_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3},
                {1, 2, 0}, {2, 1, 0}, {2, 0, 1},
                {1, 0, 2}, {0, 1, 2}, {0, 2, 1},
                {1, 1, 1}
            };
            break;
        }
        if (gaussian_order.size() == 0) {
            tonto::log::warn("Unknown Gaussian ordering for shell with angular momentum {}, not reordering", l);
            continue;
        }

        int xp, yp, zp;
        size_t our_idx{0};
        FOR_CART(xp, yp, zp, l)
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp), static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            tonto::log::debug("Setting row {} <- row {}", our_idx, gaussian_idx);
            result.row(bf_first + our_idx) = mo.row(bf_first + gaussian_idx);
            double normalization_factor = tonto::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + our_idx) *= normalization_factor;
            our_idx++;
        END_FOR_CART
    }
    return result;
}

}

}
