#include <fmt/ostream.h>
#include <occ/core/parallel.h>
#include <occ/qm/density_fitting.h>

namespace occ::df {

DFFockEngine::DFFockEngine(const BasisSet &_obs, const BasisSet &_dfbs)
    : obs(_obs), dfbs(_dfbs), nbf(_obs.nbf()), ndf(_dfbs.nbf()),
      ints(_dfbs.nbf()) {
    std::tie(m_shellpair_list, m_shellpair_data) =
        occ::ints::compute_shellpairs(obs);

    occ::timing::start(occ::timing::category::df);
    Mat V = occ::ints::compute_2body_2index_ints(dfbs); // V = (P|Q) in df basis
    occ::timing::stop(occ::timing::category::df);
    m_engines.reserve(occ::parallel::nthreads);

    occ::timing::start(occ::timing::category::engine_construct);
    m_engines.emplace_back(libint2::Operator::coulomb,
                           std::max(obs.max_nprim(), dfbs.max_nprim()),
                           std::max(obs.max_l(), dfbs.max_l()), 0);
    m_engines[0].set(libint2::BraKet::xs_xx);
    for (size_t i = 1; i < occ::parallel::nthreads; ++i) {
        m_engines.push_back(m_engines[0]);
    }
    occ::timing::stop(occ::timing::category::engine_construct);

    occ::timing::start(occ::timing::category::la);
    V_LLt = Eigen::LLT<Mat>(V);
    Mat Vsqrt = Eigen::SelfAdjointEigenSolver<Mat>(V).operatorSqrt();
    Vsqrt_LLt = Eigen::LLT<Mat>(Vsqrt);
    occ::timing::stop(occ::timing::category::la);
}

DFFockEngine::Policy policy_choice(const DFFockEngine &df) {
    if (df.memory_limit >= df.integral_storage_max_size()) {
        return DFFockEngine::Policy::Stored;
    }
    return DFFockEngine::Policy::Direct;
}

Mat DFFockEngine::compute_J(const MolecularOrbitals &mo, double precision,
                            const Mat &Schwarz, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if (policy == Policy::Direct) {
        return compute_J_direct(mo, precision, Schwarz);
    } else {
        return compute_J_stored(mo);
    }
}

Mat DFFockEngine::compute_K(const MolecularOrbitals &mo, double precision,
                            const Mat &Schwarz, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if (policy == Policy::Direct) {
        return compute_K_direct(mo, precision, Schwarz);
    } else {
        return compute_K_stored(mo);
    }
}

Mat DFFockEngine::compute_fock(const MolecularOrbitals &mo, double precision,
                               const Mat &Schwarz, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if (policy == Policy::Direct) {
        return compute_fock_direct(mo, precision, Schwarz);
    } else {
        return compute_fock_stored(mo);
    }
}

std::pair<Mat, Mat> DFFockEngine::compute_JK(const MolecularOrbitals &mo,
                                             double precision,
                                             const Mat &Schwarz,
                                             Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if (policy == Policy::Direct) {
        return compute_JK_direct(mo, precision, Schwarz);
    } else {
        return compute_JK_stored(mo);
    }
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

        Mat D(nbf, nbf);
        D.setConstant(1.0);
        three_center_integral_helper(lambda, D);
        m_have_integrals = true;
    }
}

inline auto g_lambda_direct_r(std::vector<Vec> &gg,
                              const MolecularOrbitals &mo) {
    return [&](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
               size_t bf3, size_t n3, const double *buf) {
        auto &g = gg[thread_id];
        size_t offset = 0;
        if (bf2 != bf3) {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                g(i) += (mo.D.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                            .sum();
                g(i) += (mo.D.block(bf3, bf2, n3, n2).array() *
                         buf_mat.transpose().array())
                            .sum();
                offset += n2 * n3;
            }
        } else {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                g(i) += (mo.D.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                            .sum();
                offset += n2 * n3;
            }
        }
    };
}

inline auto g_lambda_direct_u(std::vector<Vec> &gg_alpha,
                              std::vector<Vec> &gg_beta,
                              const MolecularOrbitals &mo) {
    return [&](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
               size_t bf3, size_t n3, const double *buf) {
        auto &ga = gg_alpha[thread_id];
        auto &gb = gg_beta[thread_id];
        const auto Da = qm::block::a(mo.D);
        const auto Db = qm::block::b(mo.D);
        size_t offset = 0;
        if (bf2 != bf3) {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                ga(i) += (Da.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                gb(i) += (Db.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                ga(i) += (Da.block(bf3, bf2, n3, n2).array() *
                          buf_mat.transpose().array())
                             .sum();
                gb(i) += (Db.block(bf3, bf2, n3, n2).array() *
                          buf_mat.transpose().array())
                             .sum();
                offset += n2 * n3;
            }
        } else {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                ga(i) += (Da.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                gb(i) += (Db.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                offset += n2 * n3;
            }
        }
    };
}

inline auto j_lambda_direct_r(std::vector<Mat> &JJ, const Vec &d) {
    return [&](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
               size_t bf3, size_t n3, const double *buf) {
        auto &J = JJ[thread_id];
        size_t offset = 0;
        if (bf2 != bf3) {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                J.block(bf2, bf3, n2, n3) += d(i) * buf_mat;
                J.block(bf3, bf2, n3, n2) += d(i) * buf_mat.transpose();
                offset += n2 * n3;
            }
        } else {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                J.block(bf2, bf3, n2, n3) += d(i) * buf_mat;
                offset += n2 * n3;
            }
        }
    };
}

inline auto j_lambda_direct_u(std::vector<Mat> &JJ, const Vec &da,
                              const Vec &db) {
    return [&](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
               size_t bf3, size_t n3, const double *buf) {
        auto Ja = qm::block::a(JJ[thread_id]);
        auto Jb = qm::block::a(JJ[thread_id]);
        size_t offset = 0;
        if (bf2 != bf3) {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                Ja.block(bf2, bf3, n2, n3) += da(i) * buf_mat;
                Ja.block(bf3, bf2, n3, n2) += da(i) * buf_mat.transpose();
                Jb.block(bf2, bf3, n2, n3) += db(i) * buf_mat;
                Jb.block(bf3, bf2, n3, n2) += db(i) * buf_mat.transpose();
                offset += n2 * n3;
            }
        } else {
            for (size_t i = bf1; i < bf1 + n1; i++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                Ja.block(bf2, bf3, n2, n3) += da(i) * buf_mat;
                Jb.block(bf2, bf3, n2, n3) += db(i) * buf_mat;
                offset += n2 * n3;
            }
        }
    };
}

inline auto k_lambda_direct_r(std::vector<Mat> &iuP,
                              const MolecularOrbitals &mo) {
    size_t nmo = mo.Cocc.cols();
    return [&, nmo](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
                    size_t bf3, size_t n3, const double *buf) {
        for (size_t i = 0; i < mo.Cocc.cols(); i++) {
            auto &iuPx = iuP[nmo * thread_id + i];
            auto c2 = mo.Cocc.block(bf2, i, n2, 1);
            auto c3 = mo.Cocc.block(bf3, i, n3, 1);

            size_t offset = 0;
            if (bf2 != bf3) {
                for (size_t r = bf1; r < bf1 + n1; r++) {
                    Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                    iuPx.block(bf2, r, n2, 1) += buf_mat * c3;
                    iuPx.block(bf3, r, n3, 1) += (buf_mat.transpose() * c2);
                    offset += n2 * n3;
                }
            } else {
                for (size_t r = bf1; r < bf1 + n1; r++) {
                    Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                    iuPx.block(bf2, r, n2, 1) += buf_mat * c3;
                    offset += n2 * n3;
                }
            }
        }
    };
}

inline auto k_lambda_direct_u(std::vector<Mat> &iuPa, std::vector<Mat> &iuPb,
                              const MolecularOrbitals &mo) {
    size_t nmo = mo.Cocc.cols();
    return [&, nmo](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
                    size_t bf3, size_t n3, const double *buf) {
        for (size_t i = 0; i < mo.Cocc.cols(); i++) {
            auto &iuPxa = iuPa[nmo * thread_id + i];
            auto &iuPxb = iuPb[nmo * thread_id + i];
            auto c2a = qm::block::a(mo.Cocc).block(bf2, i, n2, 1);
            auto c3a = qm::block::a(mo.Cocc).block(bf3, i, n3, 1);
            auto c2b = qm::block::b(mo.Cocc).block(bf2, i, n2, 1);
            auto c3b = qm::block::b(mo.Cocc).block(bf3, i, n3, 1);

            size_t offset = 0;
            if (bf2 != bf3) {
                for (size_t r = bf1; r < bf1 + n1; r++) {
                    Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                    iuPxa.block(bf2, r, n2, 1) += buf_mat * c3a;
                    iuPxa.block(bf3, r, n3, 1) += (buf_mat.transpose() * c2a);

                    iuPxb.block(bf2, r, n2, 1) += buf_mat * c3b;
                    iuPxb.block(bf3, r, n3, 1) += (buf_mat.transpose() * c2b);
                    offset += n2 * n3;
                }
            } else {
                for (size_t r = bf1; r < bf1 + n1; r++) {
                    Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                    iuPxa.block(bf2, r, n2, 1) += buf_mat * c3a;
                    iuPxb.block(bf2, r, n2, 1) += buf_mat * c3b;
                    offset += n2 * n3;
                }
            }
        }
    };
}

inline auto jk_lambda_direct_r(std::vector<Vec> &gg, std::vector<Mat> &iuP,
                               const MolecularOrbitals &mo) {
    return [&](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
               size_t bf3, size_t n3, const double *buf) {
        auto &g = gg[thread_id];
        size_t offset = 0;
        size_t nocc = mo.Cocc.cols();
        auto c3 = mo.Cocc.block(bf3, 0, n3, nocc);
        Mat c2_term(n2, nocc);

        if (bf2 != bf3) {
            auto c2 = mo.Cocc.block(bf2, 0, n2, nocc);
            Mat c3_term(n3, nocc);
            for (size_t r = bf1; r < bf1 + n1; r++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                g(r) += (mo.D.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                            .sum();
                c2_term = buf_mat * c3;
                g(r) += (mo.D.block(bf3, bf2, n3, n2).array() *
                         buf_mat.transpose().array())
                            .sum();
                c3_term = buf_mat.transpose() * c2;

                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPx = iuP[i];
                    iuPx.block(bf2, r, n2, 1) += c2_term.block(0, i, n2, 1);
                    iuPx.block(bf3, r, n3, 1) += c3_term.block(0, i, n3, 1);
                }
                offset += n2 * n3;
            }
        } else {
            for (size_t r = bf1; r < bf1 + n1; r++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                g(r) += (mo.D.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                            .sum();
                c2_term = buf_mat * c3;

                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPx = iuP[i];
                    iuPx.block(bf2, r, n2, 1) += c2_term.block(0, i, n2, 1);
                }
                offset += n2 * n3;
            }
        }
    };
}

inline auto jk_lambda_direct_u(std::vector<Vec> &gg_alpha,
                               std::vector<Vec> &gg_beta,
                               std::vector<Mat> &iuPa, std::vector<Mat> &iuPb,
                               const MolecularOrbitals &mo) {
    return [&](int thread_id, size_t bf1, size_t n1, size_t bf2, size_t n2,
               size_t bf3, size_t n3, const double *buf) {
        auto &ga = gg_alpha[thread_id];
        auto &gb = gg_beta[thread_id];
        size_t offset = 0;
        size_t nocc = mo.Cocc.cols();
        auto c3a = qm::block::a(mo.Cocc).block(bf3, 0, n3, nocc);
        auto c3b = qm::block::b(mo.Cocc).block(bf3, 0, n3, nocc);
        auto Da = qm::block::a(mo.D);
        auto Db = qm::block::b(mo.D);
        Mat c2_term_a(n2, nocc);
        Mat c2_term_b(n2, nocc);

        if (bf2 != bf3) {
            auto c2a = qm::block::a(mo.Cocc).block(bf2, 0, n2, nocc);
            auto c2b = qm::block::b(mo.Cocc).block(bf2, 0, n2, nocc);
            Mat c3_term_a(n3, nocc);
            Mat c3_term_b(n3, nocc);
            for (size_t r = bf1; r < bf1 + n1; r++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);

                ga(r) += (Da.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                c2_term_a = buf_mat * c3a;

                gb(r) += (Db.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                c2_term_b = buf_mat * c3b;

                ga(r) += (Da.block(bf3, bf2, n3, n2).array() *
                          buf_mat.transpose().array())
                             .sum();
                c3_term_a = buf_mat.transpose() * c2a;

                gb(r) += (Db.block(bf3, bf2, n3, n2).array() *
                          buf_mat.transpose().array())
                             .sum();
                c3_term_b = buf_mat.transpose() * c2b;

                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPxa = iuPa[i];
                    iuPxa.block(bf2, r, n2, 1) += c2_term_a.block(0, i, n2, 1);
                    iuPxa.block(bf3, r, n3, 1) += c3_term_a.block(0, i, n3, 1);
                    auto &iuPxb = iuPb[i];
                    iuPxb.block(bf2, r, n2, 1) += c2_term_b.block(0, i, n2, 1);
                    iuPxb.block(bf3, r, n3, 1) += c3_term_b.block(0, i, n3, 1);
                }
                offset += n2 * n3;
            }
        } else {
            for (size_t r = bf1; r < bf1 + n1; r++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                ga(r) += (Da.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                c2_term_a = buf_mat * c3a;
                gb(r) += (Db.block(bf2, bf3, n2, n3).array() * buf_mat.array())
                             .sum();
                c2_term_b = buf_mat * c3b;
                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPxa = iuPa[i];
                    iuPxa.block(bf2, r, n2, 1) += c2_term_a.block(0, i, n2, 1);
                    auto &iuPxb = iuPb[i];
                    iuPxb.block(bf2, r, n2, 1) += c2_term_b.block(0, i, n2, 1);
                }
                offset += n2 * n3;
            }
        }
    };
}

Mat DFFockEngine::compute_J_direct(const MolecularOrbitals &mo,
                                   double precision, const Mat &Schwarz) {

    using occ::parallel::nthreads;
    bool unrestricted = (mo.kind == qm::SpinorbitalKind::Unrestricted);
    std::vector<Vec> gg(nthreads);
    std::vector<Mat> JJ(nthreads);
    for (int i = 0; i < nthreads; i++) {
        gg[i] = Vec::Zero(ndf);
        JJ[i] = Mat::Zero(nbf, nbf);
    }

    auto glambda = g_lambda_direct_r(gg, mo);
    three_center_integral_helper(glambda, mo.D, precision, Schwarz);

    for (int i = 1; i < nthreads; i++) {
        gg[0] += gg[i];
    }

    Vec d = V_LLt.solve(gg[0]);

    auto jlambda = j_lambda_direct_r(JJ, d);
    three_center_integral_helper(jlambda, mo.D, precision, Schwarz);

    for (int i = 1; i < nthreads; i++) {
        JJ[0] += JJ[i];
    }
    return (JJ[0] + JJ[0].transpose());
}

Mat DFFockEngine::compute_K_direct(const MolecularOrbitals &mo,
                                   double precision, const Mat &Schwarz) {
    using occ::parallel::nthreads;
    size_t nmo = mo.Cocc.cols();
    Mat K = Mat::Zero(nbf, nbf);
    std::vector<Mat> iuP(nmo * nthreads);
    Mat B(nbf, ndf);
    for (auto &x : iuP)
        x = Mat::Zero(nbf, ndf);

    auto klambda = k_lambda_direct_r(iuP, mo);
    three_center_integral_helper(klambda, mo.D, precision, Schwarz);

    for (size_t i = nmo; i < nmo * nthreads; i++) {
        iuP[i % nmo] += iuP[i];
    }

    for (size_t i = 0; i < nmo; i++) {
        B = Vsqrt_LLt.solve(iuP[i].transpose());
        K.noalias() += B.transpose() * B;
    }

    return 0.5 * (K + K.transpose());
}

std::pair<Mat, Mat> DFFockEngine::compute_JK_direct(const MolecularOrbitals &mo,
                                                    double precision,
                                                    const Mat &Schwarz) {

    using occ::parallel::nthreads;
    size_t nmo = mo.Cocc.cols();

    std::vector<Vec> gg(nthreads);
    std::vector<Mat> JJ(nthreads);
    std::vector<Mat> KK(nthreads);

    std::vector<Mat> iuP(nmo);

    for (auto &x : iuP)
        x = Mat::Zero(nbf, ndf);

    for (int i = 0; i < nthreads; i++) {
        gg[i] = Vec::Zero(ndf);
        JJ[i] = Mat::Zero(nbf, nbf);
        KK[i] = Mat::Zero(nbf, nbf);
    }

    auto jk_lambda_1 = jk_lambda_direct_r(gg, iuP, mo);
    three_center_integral_helper(jk_lambda_1, mo.D, precision, Schwarz);

    for (int i = 1; i < nthreads; i++)
        gg[0] += gg[i];

    auto klambda = [&](int thread_id) {
        Mat B(nbf, ndf);
        for (size_t i = 0; i < nmo; i++) {
            if (i % nthreads != thread_id)
                continue;
            B = Vsqrt_LLt.solve(iuP[i].transpose());
            KK[thread_id].noalias() += B.transpose() * B;
        }
    };

    occ::parallel::parallel_do(klambda);

    occ::timing::start(occ::timing::category::la);
    Vec d = V_LLt.solve(gg[0]);
    occ::timing::stop(occ::timing::category::la);

    auto jlambda = j_lambda_direct_r(JJ, d);
    three_center_integral_helper(jlambda, mo.D, precision, Schwarz);

    for (int i = 1; i < nthreads; i++) {
        JJ[0] += JJ[i];
        KK[0] += KK[i];
    }

    return {JJ[0] + JJ[0].transpose(), 0.5 * (KK[0] + KK[0].transpose())};
}

Mat DFFockEngine::compute_fock_direct(const MolecularOrbitals &mo,
                                      double precision, const Mat &Schwarz) {
    Mat J, K;
    std::tie(J, K) = compute_JK_direct(mo, precision, Schwarz);
    J.noalias() -= K;
    return J;
}

Mat DFFockEngine::compute_J_stored(const MolecularOrbitals &mo) {

    populate_integrals();

    Vec g(ndf);
    for (int r = 0; r < ndf; r++) {
        const auto tr = Eigen::Map<const Mat>(m_ints.col(r).data(), nbf, nbf);
        g(r) = (mo.D.array() * tr.array()).sum();
    }
    Vec d = V_LLt.solve(g);
    Mat J = Mat::Zero(nbf, nbf);
    for (int r = 0; r < ndf; r++) {
        const auto tr = Eigen::Map<const Mat>(m_ints.col(r).data(), nbf, nbf);
        J += d(r) * tr;
    }
    return 2 * J;
}

Mat DFFockEngine::compute_K_stored(const MolecularOrbitals &mo) {
    Mat K = Mat::Zero(nbf, nbf);
    // temporaries
    Mat iuP = Mat::Zero(nbf, ndf);
    Mat B(nbf, ndf);
    for (size_t i = 0; i < mo.Cocc.cols(); i++) {
        auto c = mo.Cocc.col(i);
        for (size_t r = 0; r < ndf; r++) {
            const auto vu =
                Eigen::Map<const Mat>(m_ints.col(r).data(), nbf, nbf);
            iuP.col(r) = (vu * c);
        }
        B = Vsqrt_LLt.solve(iuP.transpose());
        K.noalias() += B.transpose() * B;
    }
    return K;
}

std::pair<Mat, Mat>
DFFockEngine::compute_JK_stored(const MolecularOrbitals &mo) {
    return {compute_J_stored(mo), compute_K_stored(mo)};
}

Mat DFFockEngine::compute_fock_stored(const MolecularOrbitals &mo) {
    Mat J, K;
    std::tie(J, K) = compute_JK(mo);
    J.noalias() -= K;
    return J;
}

inline int upper_triangle_index(const int N, const int i, const int j) {
    return (2 * N * i - i * i - i + 2 * j) / 2;
}

inline int lower_triangle_index(const int N, const int i, const int j) {
    return upper_triangle_index(N, j, i);
}

} // namespace occ::df
