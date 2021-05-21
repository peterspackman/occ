#include <occ/io/conversion.h>
#include <occ/core/util.h>
#include <occ/core/logger.h>
#include <occ/gto/gto.h>

namespace occ::io::conversion {

namespace orb {
occ::MatRM from_gaussian(const occ::qm::BasisSet &basis, const occ::MatRM& mo)
{
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    if(occ::qm::max_l(basis) < 2) return mo;
    struct xyz {
        uint_fast8_t x{0};
        uint_fast8_t y{0};
        uint_fast8_t z{0};
        bool operator ==(const xyz& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    };

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to internal convention");
    auto shell2bf = basis.shell2bf();
    occ::MatRM result(mo.rows(), mo.cols());
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
            occ::log::warn("Unknown Gaussian ordering for shell with angular momentum {}, not reordering", l);
            continue;
        }

        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l)
        {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp), static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            occ::log::debug("Setting row {} <- {} ({}{}{})", our_idx, gaussian_idx, xp, yp, zp);
            result.row(bf_first + our_idx) = mo.row(bf_first + gaussian_idx);
            double normalization_factor = occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + our_idx) *= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}


occ::MatRM to_gaussian(const occ::qm::BasisSet &basis, const occ::MatRM &mo)
{
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    if(occ::qm::max_l(basis) < 2) return mo;
    struct xyz {
        uint_fast8_t x{0};
        uint_fast8_t y{0};
        uint_fast8_t z{0};
        bool operator ==(const xyz& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    };

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to internal convention");
    auto shell2bf = basis.shell2bf();
    occ::MatRM result(mo.rows(), mo.cols());
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
            occ::log::warn("Unknown Gaussian ordering for shell with angular momentum {}, not reordering", l);
            continue;
        }

        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l)
        {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp), static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            occ::log::debug("Setting row {} <- {} ({}{}{})", gaussian_idx, our_idx, xp, yp, zp);
            result.row(bf_first + gaussian_idx) = mo.row(bf_first + our_idx);
            double normalization_factor = occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + gaussian_idx) /= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

}

}
