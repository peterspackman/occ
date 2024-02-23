#include <occ/qm/partitioning.h>

namespace occ::qm {

namespace impl {
template <SpinorbitalKind kind>
Vec mulliken(const AOBasis &basis, const Mat &D, Eigen::Ref<const Mat> op) {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;

    auto nbf = basis.nbf();
    Vec N = Vec::Zero(basis.atoms().size());
    auto bf2atom = basis.bf_to_atom();
    if constexpr (kind == R) {
        Vec pop = (D * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    } else if constexpr (kind == U) {
        Vec pop = ((block::b(D) + block::a(D)) * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    } else if constexpr (kind == G) {
        Vec pop =
            ((block::aa(D) + block::ab(D) + block::ba(D) + block::bb(D)) * op)
                .diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    }
    return N;
}
}

Vec mulliken_partition(const AOBasis &basis, const MolecularOrbitals &mo, Eigen::Ref<const Mat> op) {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;

    switch(mo.kind) {
	case U: return impl::mulliken<U>(basis, mo.D, op);
	case G: return impl::mulliken<G>(basis, mo.D, op);
	default: return impl::mulliken<R>(basis, mo.D, op);
    }
}

} // namespace occ::qm
