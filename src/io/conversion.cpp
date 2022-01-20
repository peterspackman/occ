#include <occ/core/logger.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/io/conversion.h>

namespace occ::io::conversion::orb {

Mat from_gaussian_order_cartesian(const occ::qm::BasisSet &basis,
                                  const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    constexpr auto their_order = occ::gto::ShellOrder::Gaussian;
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
        int l = shell.contr[0].l;
	size_t idx = 0;
	auto func = [&](int pi, int pj, int pk, int pl) {
	    int their_idx = occ::gto::shell_index_cartesian<their_order>(pi, pj, pk, l);
	    result.row(bf_first + idx) = mo.row(bf_first + their_idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(pi, pj, pk);
            result.row(bf_first + idx) *= normalization_factor;
	    occ::log::debug("Swapping (l={}, {}): {} (ours) <-> {} (theirs)", l, 
		    occ::gto::component_label(pi, pj, pk, l), idx, their_idx);
	    idx++;
	};
	occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

Mat to_gaussian_order_cartesian(const occ::qm::BasisSet &basis, const Mat &mo) {
    // no reordering should occur unless there are d, f, g, h etc. functions
    using occ::util::index_of;
    constexpr auto their_order = occ::gto::ShellOrder::Gaussian;
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
        int l = shell.contr[0].l;
	size_t idx = 0;
	auto func = [&](int pi, int pj, int pk, int pl) {
	    int their_idx = occ::gto::shell_index_cartesian<their_order>(pi, pj, pk, l);
	    result.row(bf_first + their_idx) = mo.row(bf_first + idx);
            double normalization_factor =
                occ::gto::cartesian_normalization_factor(pi, pj, pk);
            result.row(bf_first + their_idx) /= normalization_factor;
	    occ::log::debug("Swapping (l={}, {}): {} (ours) <-> {} (theirs)", l, 
		    occ::gto::component_label(pi, pj, pk, l), idx, their_idx);
	    idx++;
	};
	occ::gto::iterate_over_shell<true>(func, l);
    }
    return result;
}

Mat to_gaussian_order_spherical(const occ::qm::BasisSet &basis, const Mat &mo) {
    using occ::util::index_of;
    constexpr auto order = occ::gto::ShellOrder::Gaussian;
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
        int l = shell.contr[0].l;

	if(l == 1) {
	    // yzx -> xyz
	    result.row(bf_first) = mo.row(bf_first + 2);
	    result.row(bf_first + 1) = mo.row(bf_first);
	    result.row(bf_first + 2) = mo.row(bf_first + 1);
	    occ::log::debug("Swapping (l={}): (0, 1, 2) <-> (2, 0, 1)", l);
	    continue;
	}
	else {
	    size_t idx = 0;
	    auto func = [&](int am, int m) {
		int their_idx = occ::gto::shell_index_spherical<order>(am, m);
		result.row(bf_first + their_idx) = mo.row(bf_first + idx);
		occ::log::debug("Swapping (l={}): {} <-> {}", l, their_idx, idx);
		idx++;
	    };
	    occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(func, l);
	}
    }
    return result;
}

Mat from_gaussian_order_spherical(const occ::qm::BasisSet &basis,
                                  const Mat &mo) {
    using occ::util::index_of;
    if (occ::qm::max_l(basis) < 1)
        return mo;
    constexpr auto order = occ::gto::ShellOrder::Gaussian;

    occ::log::debug("Reordering MO coefficients from Gaussian ordering to "
                    "internal convention");
    auto shell2bf = basis.shell2bf();
    Mat result(mo.rows(), mo.cols());
    size_t ncols = mo.cols();
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &shell = basis[i];
        size_t bf_first = shell2bf[i];
        int l = shell.contr[0].l;
	if(l == 1) {
	    // xyz -> yzx
	    occ::log::debug("Swapping (l={}): (2, 0, 1) <-> (0, 1, 2)", l); 
	    result.block(bf_first, 0, 1, ncols) = mo.block(bf_first + 1, 0, 1, ncols);
	    result.block(bf_first + 1, 0, 1, ncols) = mo.block(bf_first + 2, 0, 1, ncols);
	    result.block(bf_first + 2, 0, 1, ncols) = mo.block(bf_first, 0, 1, ncols);
	}
	else {
	    size_t idx = 0;
	    auto func = [&](int am, int m) {
		int their_idx = occ::gto::shell_index_spherical<order>(am, m);
		result.row(bf_first + idx)= mo.row(bf_first + their_idx);
		occ::log::debug("Swapping (l={}): {} <-> {}", l, idx, their_idx);
		idx++;
	    };
	    occ::gto::iterate_over_shell<false, occ::gto::ShellOrder::Default>(func, l);
	}
    }
    return result;
}

} // namespace occ::io::conversion::orb
