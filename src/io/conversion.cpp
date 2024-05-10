#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/conversion.h>

namespace occ::io::conversion::orb {

template <typename Lambda>
qm::MolecularOrbitals convert_order(const qm::AOBasis &basis,
                                    const qm::MolecularOrbitals &mo,
                                    Lambda conv_func) {

    using SK = occ::qm::SpinorbitalKind;
    namespace block = occ::qm::block;
    qm::MolecularOrbitals result = mo;

    switch (mo.kind) {
    case SK::Restricted: {
        result.C = conv_func(basis, mo.C);
        break;
    }
    case SK::Unrestricted: {
        block::a(result.C) = conv_func(basis, block::a(mo.C));
        block::b(result.C) = conv_func(basis, block::b(mo.C));
        break;
    }
    case SK::General: {
        block::aa(result.C) = conv_func(basis, block::aa(mo.C));
        block::ab(result.C) = conv_func(basis, block::ab(mo.C));
        block::ba(result.C) = conv_func(basis, block::ba(mo.C));
        block::bb(result.C) = conv_func(basis, block::bb(mo.C));
        break;
    }
    }
    return result;
}

qm::MolecularOrbitals to_gaussian_order(const qm::AOBasis &basis,
                                        const qm::MolecularOrbitals &mo) {
    if (basis.is_pure()) {
        return convert_order(basis, mo, [](const qm::AOBasis &b, const Mat &m) {
            return to_gaussian_order_spherical(b, m);
        });
    } else {
        return convert_order(basis, mo, [](const qm::AOBasis &b, const Mat &m) {
            return to_gaussian_order_cartesian(b, m);
        });
    }
}

qm::MolecularOrbitals from_gaussian_order(const qm::AOBasis &basis,
                                          const qm::MolecularOrbitals &mo) {

    if (basis.is_pure()) {
        return convert_order(basis, mo, [](const qm::AOBasis &b, const Mat &m) {
            return from_gaussian_order_spherical(b, m);
        });
    } else {
        return convert_order(basis, mo, [](const qm::AOBasis &b, const Mat &m) {
            return from_gaussian_order_cartesian(b, m);
        });
    }
}

/* TODO
 *
 * Refactor into one method, generic across orders/normalization schemes
 *
 * Shouldn't be too hard, but should also write tests to ensure correctness
 *
 */
Mat from_gaussian_order_cartesian(const occ::qm::AOBasis &basis,
                                  const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    constexpr auto their_order = occ::gto::ShellOrder::Gaussian;
    if (basis.l_max() < 2)
        return mo;

    occ::log::debug(
        "Reordering cartesian MO coefficients from Gaussian ordering to "
        "internal convention");
    auto shell2bf = basis.first_bf();
    Mat result(mo.rows(), mo.cols());
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis.shells()[i];
        size_t bf_first = shell2bf[i];
        int l = shell.l;
        size_t idx = 0;
        auto func = [&](int pi, int pj, int pk, int pl) {
            int their_idx =
                occ::gto::shell_index_cartesian<their_order>(pi, pj, pk, l);
            result.row(bf_first + idx) = mo.row(bf_first + their_idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(pi, pj, pk);
            result.row(bf_first + idx) *= normalization_factor;
            occ::log::trace("Swapping (l={}, {}): {} (ours) <-> {} (theirs)", l,
                            occ::gto::component_label(pi, pj, pk, l), idx,
                            their_idx);
            idx++;
        };
        occ::gto::iterate_over_shell<true, occ::gto::ShellOrder::Default>(func,
                                                                          l);
    }
    return result;
}

Mat to_gaussian_order_cartesian(const occ::qm::AOBasis &basis, const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    constexpr auto their_order = occ::gto::ShellOrder::Gaussian;
    if (basis.l_max() < 2)
        return mo;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.first_bf();
    Mat result(mo.rows(), mo.cols());
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis.shells()[i];
        size_t bf_first = shell2bf[i];
        int l = shell.l;
        size_t idx = 0;
        auto func = [&](int pi, int pj, int pk, int pl) {
            int their_idx =
                occ::gto::shell_index_cartesian<their_order>(pi, pj, pk, l);
            result.row(bf_first + their_idx) = mo.row(bf_first + idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(pi, pj, pk);
            result.row(bf_first + their_idx) /= normalization_factor;
            occ::log::trace("Swapping (l={}, {}): {} (ours) <-> {} (theirs)", l,
                            occ::gto::component_label(pi, pj, pk, l), idx,
                            their_idx);
            idx++;
        };
        occ::gto::iterate_over_shell<true, occ::gto::ShellOrder::Default>(func,
                                                                          l);
    }
    return result;
}

Mat to_gaussian_order_spherical(const occ::qm::AOBasis &basis, const Mat &mo) {
    using occ::util::index_of;
    constexpr auto order = occ::gto::ShellOrder::Gaussian;
    if (basis.l_max() < 1)
        return mo;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.first_bf();
    Mat result(mo.rows(), mo.cols());
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis.shells()[i];
        size_t bf_first = shell2bf[i];
        int l = shell.l;

        if (l == 1) {
            // yzx -> xyz
            result.row(bf_first) = mo.row(bf_first + 2);
            result.row(bf_first + 1) = mo.row(bf_first);
            result.row(bf_first + 2) = mo.row(bf_first + 1);
            occ::log::trace("Swapping (l={}): (0, 1, 2) <-> (2, 0, 1)", l);
            continue;
        } else {
            size_t idx = 0;
            auto func = [&](int am, int m) {
                int their_idx = occ::gto::shell_index_spherical<order>(am, m);
                result.row(bf_first + their_idx) = mo.row(bf_first + idx);
                occ::log::trace("Swapping (l={}): {} <-> {}", l, their_idx,
                                idx);
                idx++;
            };
            occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(
                func, l);
        }
    }
    return result;
}

Mat from_gaussian_order_spherical(const occ::qm::AOBasis &basis,
                                  const Mat &mo) {
    using occ::util::index_of;
    if (basis.l_max() < 1)
        return mo;
    constexpr auto order = occ::gto::ShellOrder::Gaussian;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.first_bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis.shells()[i];
        size_t bf_first = shell2bf[i];
        int l = shell.l;
        if (l == 1) {
            // xyz -> yzx
            occ::log::trace("Swapping (l={}): (2, 0, 1) <-> (0, 1, 2)", l);
            result.block(bf_first, 0, 1, ncols) =
                mo.block(bf_first + 1, 0, 1, ncols);
            result.block(bf_first + 1, 0, 1, ncols) =
                mo.block(bf_first + 2, 0, 1, ncols);
            result.block(bf_first + 2, 0, 1, ncols) =
                mo.block(bf_first, 0, 1, ncols);
        } else {
            size_t idx = 0;
            auto func = [&](int am, int m) {
                int their_idx = occ::gto::shell_index_spherical<order>(am, m);
                result.row(bf_first + idx) = mo.row(bf_first + their_idx);
                occ::log::trace("Swapping (l={}): {} <-> {}", l, idx,
                                their_idx);
                idx++;
            };
            occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(
                func, l);
        }
    }
    return result;
}

} // namespace occ::io::conversion::orb
