#include <fmt/ostream.h>
#include <occ/core/parallel.h>
#include <occ/qm/density_fitting.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace occ::df {

DFFockEngine::DFFockEngine(const BasisSet &_obs, const BasisSet &_dfbs)
    : obs(_obs), dfbs(_dfbs), nbf(_obs.nbf()), ndf(_dfbs.nbf()),
      ints(_dfbs.nbf()) {
    std::tie(m_shellpair_list, m_shellpair_data) =
        occ::ints::compute_shellpairs(obs);
    Mat V = occ::ints::compute_2body_2index_ints(dfbs); // V = (P|Q) in df basis
    Vinv = V.inverse(); // V^-1
    Vinv_sqrt = Vinv.sqrt();
    m_engines.reserve(occ::parallel::nthreads);
    m_engines.emplace_back(libint2::Operator::coulomb,
                           std::max(obs.max_nprim(), dfbs.max_nprim()),
                           std::max(obs.max_l(), dfbs.max_l()), 0);
    m_engines[0].set(libint2::BraKet::xs_xx);
    for (size_t i = 1; i < occ::parallel::nthreads; ++i) {
        m_engines.push_back(m_engines[0]);
    }

    V_LLt = Eigen::LLT<Mat>(V);
    Mat I = Mat::Identity(ndf, ndf);
    L = Eigen::LLT<Mat>(Vinv).matrixL(); // (P|Q)^-1/2 
    Linv_t = V_LLt.matrixL().solve(I).transpose();
}

void DFFockEngine::populate_integrals() {
  if (!m_have_integrals) {
    m_ints = Mat::Zero(nbf * nbf, ndf);
    auto lambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
            size_t n2, size_t bf3, size_t n3, const double *buf) {
        size_t offset = 0;
        for (size_t i = bf1; i < bf1 + n1; i++) {
            auto x = Eigen::Map<Mat>(m_ints.col(i).data(), nbf, nbf);
            Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
            x.block(bf2, bf3, n2, n3) = buf_mat;
            if (bf2 != bf3)
                x.block(bf3, bf2, n3, n2) = buf_mat.transpose();
            offset += n2 * n3;
        }
    };

    three_center_integral_helper(lambda);
    m_have_integrals = true;
  }
}

Mat DFFockEngine::compute_J(const MolecularOrbitals &mo) {

    populate_integrals();

    Vec g(ndf);
    for (int r = 0; r < ndf; r++) {
        const auto tr = Eigen::Map<const Mat>(m_ints.col(r).data(), nbf, nbf);
        g(r) = (mo.D.array() * tr.array()).sum();
    }
    // fmt::print("g\n{}\n", g);
    Vec d = V_LLt.solve(g);
    Mat J = Mat::Zero(nbf, nbf);
    for (int r = 0; r < ndf; r++) {
        const auto tr = Eigen::Map<const Mat>(m_ints.col(r).data(), nbf, nbf);
        J += d(r) * tr;
    }
    return 2 * J;
}

Mat DFFockEngine::compute_K(const MolecularOrbitals &mo) {
    Mat K = Mat::Zero(nbf, nbf);
    // temporaries
    Mat iuP = Mat::Zero(nbf, ndf);
    Mat B(nbf, ndf);
    for(size_t i = 0; i < mo.Cocc.cols(); i++) {
        auto c = mo.Cocc.col(i);
        for(size_t r = 0; r < ndf; r++) {
            const auto vu = Eigen::Map<const Mat>(m_ints.col(r).data(), nbf, nbf);
            iuP.col(r) = (vu * c);
        }
        B = iuP * Vinv_sqrt;
        K.noalias() += B * B.transpose();
    }
    return K;
}

std::pair<Mat, Mat> DFFockEngine::compute_JK(const MolecularOrbitals &mo) {
    return {compute_J(mo), compute_K(mo)};
}

inline int upper_triangle_index(const int N, const int i, const int j) {
    return (2 * N * i - i * i - i + 2 * j) / 2;
}

inline int lower_triangle_index(const int N, const int i, const int j) {
    return upper_triangle_index(N, j, i);
}

} // namespace occ::df
