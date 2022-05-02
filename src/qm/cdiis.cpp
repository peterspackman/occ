#include <occ/qm/cdiis.h>
#include <occ/qm/opmatrix.h>

namespace occ::qm {

double rms_error_diis(const Mat &commutator) {
    return commutator.norm() / commutator.size();
}

double maximum_error_diis(const Mat &commutator) {
    return commutator.array().abs().maxCoeff();
}

double minimum_error_diis(const Mat &commutator) {
    return commutator.array().abs().maxCoeff();
}

auto commutator(const Eigen::Ref<const Mat> S, const Eigen::Ref<const Mat> D,
                const Eigen::Ref<const Mat> F) {
    return S * D * F - F * D * S;
}

Mat CDIIS::update(const Mat &overlap, const Mat &D, const Mat &F) {
    // we have an unrestricted problem
    Mat comm;
    if (D.rows() != D.cols()) {
        comm = Mat(D.rows(), D.cols());
        block::a(comm) =
            commutator(block::a(overlap), block::a(D), block::a(F));
        block::b(comm) =
            commutator(block::b(overlap), block::b(D), block::b(F));
    } else {
        comm = commutator(overlap, D, F);
    }
    m_max_error = maximum_error_diis(comm);
    m_min_error = minimum_error_diis(comm);
    Mat result = F;
    occ::core::diis::DIIS::extrapolate(result, comm);
    return result;
}

} // namespace occ::qm
