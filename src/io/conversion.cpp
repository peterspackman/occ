#include <occ/core/logger.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/conversion.h>

namespace occ::io::conversion {

struct xyz {
    uint_fast8_t x{0};
    uint_fast8_t y{0};
    uint_fast8_t z{0};
    bool operator==(const xyz &rhs) const {
	return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct lm {
    int_fast8_t l{0};
    int_fast8_t m{0};
    int_fast8_t sign{1};
    bool operator==(const lm &rhs) const {
	return l == rhs.l && m == rhs.m;
    }
};

namespace orb {
Mat from_gaussian_order_cartesian(const occ::qm::BasisSet &basis,
                                  const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    if (occ::qm::max_l(basis) < 2)
        return mo;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if (l < 2) {
            result.block(bf_first, 0, shell_size, ncols) =
                mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> gaussian_order;
        switch (l) {
        case 2:
            gaussian_order = {// xx, yy, zz, xy, xz, yz
                              {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                              {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
            break;
        case 3:
            gaussian_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {1, 2, 0}, {2, 1, 0},
                {2, 0, 1}, {1, 0, 2}, {0, 1, 2}, {0, 2, 1}, {1, 1, 1}};
            break;
        /* Apparently GDMA expects this order
        case 4:
            gaussian_order = {
                // xxxx, yyyy, zzzz,
                // xxxy, xxxz, xyyy,
                // yyyz, xzzz, yzzz,
                // xxyy, xxzz, yyzz,
                // xxyz, xyyz, xyzz
                {4, 0, 0}, {0, 4, 0}, {0, 0, 4},
                {3, 1, 0}, {3, 0, 1}, {1, 3, 0},
                {0, 3, 1}, {1, 0, 3}, {0, 1, 3},
                {2, 2, 0}, {2, 0, 2}, {0, 2, 2},
                {2, 1, 1}, {1, 2, 1}, {1, 1, 2}
            };
            break;
        */
        // But this is the actual order G09 puts out...
        default:
            auto cc_order = occ::gto::cartesian_subshell_ordering(l);
            gaussian_order.reserve(cc_order.size());
            for (auto it = cc_order.rbegin(); it != cc_order.rend(); it++) {
                const auto &x = *it;
                gaussian_order.emplace_back(
                    xyz{static_cast<uint_fast8_t>(x.l),
                        static_cast<uint_fast8_t>(x.m),
                        static_cast<uint_fast8_t>(x.n)});
            }
            break;
        }
        if (gaussian_order.size() == 0) {
            occ::log::warn("Unknown Gaussian ordering for shell with angular "
                           "momentum {}, not reordering",
                           l);
            continue;
        }

        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l) {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp),
                  static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            occ::log::debug("Setting row {} <- {} ({}{}{})", our_idx,
                            gaussian_idx, xp, yp, zp);
            result.row(bf_first + our_idx) = mo.row(bf_first + gaussian_idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + our_idx) *= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

Mat to_gaussian_order_cartesian(const occ::qm::BasisSet &basis, const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    if (occ::qm::max_l(basis) < 2)
        return mo;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if (l < 2) {
            result.block(bf_first, 0, shell_size, ncols) =
                mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> gaussian_order;
        switch (l) {
        case 2:
            gaussian_order = {// xx, yy, zz, xy, xz, yz
                              {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                              {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
            break;
        case 3:
            gaussian_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {1, 2, 0}, {2, 1, 0},
                {2, 0, 1}, {1, 0, 2}, {0, 1, 2}, {0, 2, 1}, {1, 1, 1}};
            break;
        /* Apparently GDMA expects this order
        case 4:
            gaussian_order = {
                // xxxx, yyyy, zzzz,
                // xxxy, xxxz, xyyy,
                // yyyz, xzzz, yzzz,
                // xxyy, xxzz, yyzz,
                // xxyz, xyyz, xyzz
                {4, 0, 0}, {0, 4, 0}, {0, 0, 4},
                {3, 1, 0}, {3, 0, 1}, {1, 3, 0},
                {0, 3, 1}, {1, 0, 3}, {0, 1, 3},
                {2, 2, 0}, {2, 0, 2}, {0, 2, 2},
                {2, 1, 1}, {1, 2, 1}, {1, 1, 2}
            };
            break;
        */
        // But this is the actual order G09 puts out...
        default:
            auto cc_order = occ::gto::cartesian_subshell_ordering(l);
            gaussian_order.reserve(cc_order.size());
            for (auto it = cc_order.rbegin(); it != cc_order.rend(); it++) {
                const auto &x = *it;
                gaussian_order.emplace_back(
                    xyz{static_cast<uint_fast8_t>(x.l),
                        static_cast<uint_fast8_t>(x.m),
                        static_cast<uint_fast8_t>(x.n)});
            }
            break;
        }
        if (gaussian_order.size() == 0) {
            occ::log::warn("Unknown Gaussian ordering for shell with angular "
                           "momentum {}, not reordering",
                           l);
            continue;
        }

        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l) {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp),
                  static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            occ::log::debug("Setting row {} <- {} ({}{}{})", gaussian_idx,
                            our_idx, xp, yp, zp);
            result.row(bf_first + gaussian_idx) = mo.row(bf_first + our_idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + gaussian_idx) /= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

Mat to_gaussian_order_spherical(const occ::qm::BasisSet &basis, const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    if (occ::qm::max_l(basis) < 1)
        return mo;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if (l < 1) {
            result.block(bf_first, 0, shell_size, ncols) =
                mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> gaussian_order;
        switch (l) {
	case 1:
            result.row(bf_first) = mo.row(bf_first + 2); // x
            result.row(bf_first + 1) = mo.row(bf_first); // y
            result.row(bf_first + 2) = mo.row(bf_first + 1); // z
	    continue;
        case 2:
            gaussian_order = {// xx, yy, zz, xy, xz, yz
                              {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                              {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
            break;
        case 3:
            gaussian_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {1, 2, 0}, {2, 1, 0},
                {2, 0, 1}, {1, 0, 2}, {0, 1, 2}, {0, 2, 1}, {1, 1, 1}};
            break;
        /* Apparently GDMA expects this order
        case 4:
            gaussian_order = {
                // xxxx, yyyy, zzzz,
                // xxxy, xxxz, xyyy,
                // yyyz, xzzz, yzzz,
                // xxyy, xxzz, yyzz,
                // xxyz, xyyz, xyzz
                {4, 0, 0}, {0, 4, 0}, {0, 0, 4},
                {3, 1, 0}, {3, 0, 1}, {1, 3, 0},
                {0, 3, 1}, {1, 0, 3}, {0, 1, 3},
                {2, 2, 0}, {2, 0, 2}, {0, 2, 2},
                {2, 1, 1}, {1, 2, 1}, {1, 1, 2}
            };
            break;
        */
        // But this is the actual order G09 puts out...
        default:
            auto cc_order = occ::gto::cartesian_subshell_ordering(l);
            gaussian_order.reserve(cc_order.size());
            for (auto it = cc_order.rbegin(); it != cc_order.rend(); it++) {
                const auto &x = *it;
                gaussian_order.emplace_back(
                    xyz{static_cast<uint_fast8_t>(x.l),
                        static_cast<uint_fast8_t>(x.m),
                        static_cast<uint_fast8_t>(x.n)});
            }
            break;
        }
        if (gaussian_order.size() == 0) {
            occ::log::warn("Unknown Gaussian ordering for shell with angular "
                           "momentum {}, not reordering",
                           l);
            continue;
        }

        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l) {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp),
                  static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            occ::log::debug("Setting row {} <- {} ({}{}{})", gaussian_idx,
                            our_idx, xp, yp, zp);
            result.row(bf_first + gaussian_idx) = mo.row(bf_first + our_idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + gaussian_idx) /= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

Mat from_gaussian_order_spherical(const occ::qm::BasisSet &basis,
                                  const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    if (occ::qm::max_l(basis) < 1)
        return mo;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis[i];
        size_t bf_first = shell2bf[i];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        if (l < 1) {
            result.block(bf_first, 0, shell_size, ncols) =
                mo.block(bf_first, 0, shell_size, ncols);
            continue;
        }
        std::vector<xyz> gaussian_order;
        switch (l) {
	case 1:
            result.row(bf_first + 2) = mo.row(bf_first); // x
            result.row(bf_first) = mo.row(bf_first + 1); // y
            result.row(bf_first + 1) = mo.row(bf_first + 2); // z
	    continue;
        case 2:
            gaussian_order = {// xx, yy, zz, xy, xz, yz
                              {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                              {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
            break;
        case 3:
            gaussian_order = {
                // xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
                {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {1, 2, 0}, {2, 1, 0},
                {2, 0, 1}, {1, 0, 2}, {0, 1, 2}, {0, 2, 1}, {1, 1, 1}};
            break;
        /* Apparently GDMA expects this order
        case 4:
            gaussian_order = {
                // xxxx, yyyy, zzzz,
                // xxxy, xxxz, xyyy,
                // yyyz, xzzz, yzzz,
                // xxyy, xxzz, yyzz,
                // xxyz, xyyz, xyzz
                {4, 0, 0}, {0, 4, 0}, {0, 0, 4},
                {3, 1, 0}, {3, 0, 1}, {1, 3, 0},
                {0, 3, 1}, {1, 0, 3}, {0, 1, 3},
                {2, 2, 0}, {2, 0, 2}, {0, 2, 2},
                {2, 1, 1}, {1, 2, 1}, {1, 1, 2}
            };
            break;
        */
        // But this is the actual order G09 puts out...
        default:
            auto cc_order = occ::gto::cartesian_subshell_ordering(l);
            gaussian_order.reserve(cc_order.size());
            for (auto it = cc_order.rbegin(); it != cc_order.rend(); it++) {
                const auto &x = *it;
                gaussian_order.emplace_back(
                    xyz{static_cast<uint_fast8_t>(x.l),
                        static_cast<uint_fast8_t>(x.m),
                        static_cast<uint_fast8_t>(x.n)});
            }
            break;
        }
        if (gaussian_order.size() == 0) {
            occ::log::warn("Unknown Gaussian ordering for shell with angular "
                           "momentum {}, not reordering",
                           l);
            continue;
        }

        size_t our_idx{0};
        auto func = [&](int xp, int yp, int zp, int l) {
            xyz v{static_cast<uint_fast8_t>(xp), static_cast<uint_fast8_t>(yp),
                  static_cast<uint_fast8_t>(zp)};
            size_t gaussian_idx = index_of(v, gaussian_order);
            occ::log::debug("Setting row {} <- {} ({}{}{})", our_idx,
                            gaussian_idx, xp, yp, zp);
            result.row(bf_first + our_idx) = mo.row(bf_first + gaussian_idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(xp, yp, zp);
            result.row(bf_first + our_idx) *= normalization_factor;
            our_idx++;
        };
        occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

} // namespace orb

} // namespace occ::io::conversion
