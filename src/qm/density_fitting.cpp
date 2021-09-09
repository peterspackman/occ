#include <fmt/ostream.h>
#include <occ/core/parallel.h>
#include <occ/qm/density_fitting.h>

namespace occ::df {

DFFockEngine::DFFockEngine(const BasisSet &_obs, const BasisSet &_dfbs)
    : obs(_obs), dfbs(_dfbs), nbf(_obs.nbf()), ndf(_dfbs.nbf()),
      ints(_dfbs.nbf()) {
    Mat V = occ::ints::compute_2body_2index_ints(dfbs);
    Vinv = V.inverse();
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
    auto L = V_LLt.matrixL();
    Linv_t = L.solve(I).transpose();
}

Mat DFFockEngine::compute_2body_fock_dfC(const Mat &C_occ) {

    // using first time? compute 3-center ints and transform to inv sqrt
    // representation
    if (!ints_populated) {
        auto lambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                          size_t n2, size_t bf3, size_t n3, const double *buf) {
            size_t offset = 0;
            for (size_t i = bf1; i < bf1 + n1; i++) {
                auto &x = ints[i];
                x.resize(nbf, nbf);
                for (size_t j = bf2; j < bf2 + n2; j++) {
                    for (size_t k = bf3; k < bf3 + n3; k++) {
                        x(j, k) = buf[offset];
                        offset++;
                    }
                }
            }
        };

        three_center_integral_helper(lambda);
        ints_populated = true;
    }

    // compute exchange
    /*
       for(size_t i = 0; i < ints.size(); i++)
       {
       fmt::print("{}\n{}\n", i, ints[i]);
       }
       */

    Mat J = Mat::Zero(nbf, nbf);
    Mat D = C_occ * C_occ.transpose();

    for (int r = 0; r < ndf; r++) {
        const auto &tr = ints[r];
        for (int s = 0; s < ndf; s++) {
            const auto &ts = ints[s];
            double Vrs = Vinv(r, s);
            if (abs(Vrs) < 1e-12)
                continue;
            for (int i = 0; i < nbf; i++)
                for (int j = 0; j < nbf; j++)
                    for (int k = 0; k < nbf; k++)
                        for (int l = 0; l < nbf; l++) {
                            double v = Vrs * tr(i, j) * ts(k, l);
                            J(i, j) += D(k, l) * v;
                            J(k, l) += D(i, j) * v;
                            J(i, k) -= 0.25 * D(j, l) * v;
                            J(j, l) -= 0.25 * D(i, k) * v;
                            J(i, l) -= 0.25 * D(j, k) * v;
                            J(j, k) -= 0.25 * D(i, l) * v;
                        }
        }
    }
    return J;
}

Mat DFFockEngine::compute_J(const Mat &D) {

    // using first time? compute 3-center ints and transform to inv sqrt
    // representation
    if (!ints_populated) {
        auto lambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                          size_t n2, size_t bf3, size_t n3, const double *buf) {
            size_t offset = 0;
            for (size_t i = bf1; i < bf1 + n1; i++) {
                auto &x = ints[i];
                x.resize(nbf, nbf);
                x.block(bf2, bf3, n2, n3) =
                    Eigen::Map<const MatRM>(&buf[offset], n2, n3);
                offset += n2 * n3;
            }
        };

        three_center_integral_helper(lambda);
        ints_populated = true;
        for (auto &x : ints) {
            if (x.isZero())
                x.resize(0, 0);
        }
    }

    // compute exchange
    /*
       for(size_t i = 0; i < ints.size(); i++)
       {
       fmt::print("{}\n{}\n", i, ints[i]);
       }
       */

    Vec g(ndf);
    for (int r = 0; r < ndf; r++) {
        const auto &tr = ints[r];
        if (tr.rows() == 0)
            continue;
        g(r) = (D.array() * tr.array()).sum();
    }
    // fmt::print("g\n{}\n", g);
    Vec d = V_LLt.solve(g);
    Mat J = Mat::Zero(nbf, nbf);
    for (int r = 0; r < ndf; r++) {
        const auto &tr = ints[r];
        if (tr.rows() == 0)
            continue;
        J += d(r) * tr;
    }
    return 2 * J;
}

Mat DFFockEngine::compute_J_direct(const Mat &D) const {

    using occ::parallel::nthreads;
    std::vector<Vec> gg(nthreads);
    std::vector<Mat> JJ(nthreads);
    for (int i = 0; i < nthreads; i++) {
        gg[i] = Vec::Zero(ndf);
        JJ[i] = Mat::Zero(nbf, nbf);
    }

    auto glambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                       size_t n2, size_t bf3, size_t n3, const double *buf) {
        auto &g = gg[thread_id];
        size_t offset = 0;
        for (size_t i = bf1; i < bf1 + n1; i++) {
            g(i) += (D.block(bf2, bf3, n2, n3).array() *
                     Eigen::Map<const MatRM>(&buf[offset], n2, n3).array())
                        .sum();
            offset += n2 * n3;
        }
    };

    three_center_integral_helper(glambda);

    for (int i = 1; i < nthreads; i++)
        gg[0] += gg[i];

    Vec d = V_LLt.solve(gg[0]);

    auto Jlambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                       size_t n2, size_t bf3, size_t n3, const double *buf) {
        auto &J = JJ[thread_id];
        size_t offset = 0;
        for (size_t i = bf1; i < bf1 + n1; i++) {
            J.block(bf2, bf3, n2, n3) +=
                d(i) * Eigen::Map<const MatRM>(&buf[offset], n2, n3);
            offset += n2 * n3;
        }
    };

    three_center_integral_helper(Jlambda);

    for (int i = 1; i < nthreads; i++)
        JJ[0] += JJ[i];
    return 2 * JJ[0];
}

inline int upper_triangle_index(const int N, const int i, const int j) {
    return (2 * N * i - i * i - i + 2 * j) / 2;
}

inline int lower_triangle_index(const int N, const int i, const int j) {
    return upper_triangle_index(N, j, i);
}

} // namespace occ::df
