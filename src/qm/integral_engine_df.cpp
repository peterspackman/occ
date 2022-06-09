#include <occ/core/timings.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm {

using ShellPairList = std::vector<std::vector<size_t>>;
using ShellList = std::vector<OccShell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellKind = OccShell::Kind;
using Op = cint::Operator;
using IntegralResult = IntegralEngine::IntegralResult<3>;

IntegralEngineDF::IntegralEngineDF(const AtomList &atoms, const ShellList &ao,
                                   const ShellList &df)
    : m_ao_engine(atoms, ao), m_aux_engine(atoms, df), m_integral_store(0, 0) {
    m_ao_engine.set_auxiliary_basis(df, false);
    occ::timing::start(occ::timing::category::df);
    Mat V = m_aux_engine.one_electron_operator(
        Op::coulomb); // V = (P|Q) in df basis
    occ::timing::stop(occ::timing::category::df);

    occ::timing::start(occ::timing::category::la);
    V_LLt = Eigen::LLT<Mat>(V);
    Mat Vsqrt = Eigen::SelfAdjointEigenSolver<Mat>(V).operatorSqrt();
    Vsqrt_LLt = Eigen::LLT<Mat>(Vsqrt);
    occ::timing::stop(occ::timing::category::la);
}

template <typename Func>
void inner_loop(Func &f, const IntegralEngine &ao, const IntegralEngine &aux,
                const IntegralEngine::IntegralResult<3> &args) {
    const auto incr = args.dims[0] * args.dims[1];
    const auto first_bf_aux = aux.first_bf()[args.shell[2]];
    const auto bf1 = ao.first_bf()[args.shell[0]];
    const auto bf2 = ao.first_bf()[args.shell[1]];

    for (size_t p = 0, offset = 0; p < args.dims[2]; p++, offset += incr) {
        const auto bf_aux = first_bf_aux + p;
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        f(bf_aux, bf1, bf2, buf_mat);
    }
}

inline auto g_lambda_direct_r(std::vector<Vec> &gg,
                              const MolecularOrbitals &mo) {
    return [&](const IntegralResult &args) {
        auto &g = gg[args.thread];
        size_t offset = 0;
        if (args.bf[0] != args.bf[1]) {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                g(i) += (mo.D.block(args.bf[0], args.bf[1], args.dims[0],
                                    args.dims[1])
                             .array() *
                         buf_mat.array())
                            .sum();
                g(i) += (mo.D.block(args.bf[1], args.bf[0], args.dims[1],
                                    args.dims[0])
                             .array() *
                         buf_mat.transpose().array())
                            .sum();
                offset += args.dims[0] * args.dims[1];
            }
        } else {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                g(i) += (mo.D.block(args.bf[0], args.bf[1], args.dims[0],
                                    args.dims[1])
                             .array() *
                         buf_mat.array())
                            .sum();
                offset += args.dims[0] * args.dims[1];
            }
        }
    };
}

inline auto g_lambda_direct_u(std::vector<Vec> &gg_alpha,
                              std::vector<Vec> &gg_beta,
                              const MolecularOrbitals &mo) {
    return [&](const IntegralResult &args) {
        auto &ga = gg_alpha[args.thread];
        auto &gb = gg_beta[args.thread];
        const auto Da = qm::block::a(mo.D);
        const auto Db = qm::block::b(mo.D);
        size_t offset = 0;
        if (args.bf[0] != args.bf[1]) {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                ga(i) += (Da.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                gb(i) += (Db.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                ga(i) += (Da.block(args.bf[1], args.bf[0], args.dims[1],
                                   args.dims[0])
                              .array() *
                          buf_mat.transpose().array())
                             .sum();
                gb(i) += (Db.block(args.bf[1], args.bf[0], args.dims[1],
                                   args.dims[0])
                              .array() *
                          buf_mat.transpose().array())
                             .sum();
                offset += args.dims[0] * args.dims[1];
            }
        } else {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                ga(i) += (Da.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                gb(i) += (Db.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                offset += args.dims[0] * args.dims[1];
            }
        }
    };
}

inline auto j_lambda_direct_r(std::vector<Mat> &JJ, const Vec &d) {
    return [&](const IntegralResult &args) {
        auto &J = JJ[args.thread];
        size_t offset = 0;
        if (args.bf[0] != args.bf[1]) {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                    d(i) * buf_mat;
                J.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
                    d(i) * buf_mat.transpose();
                offset += args.dims[0] * args.dims[1];
            }
        } else {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                    d(i) * buf_mat;
                offset += args.dims[0] * args.dims[1];
            }
        }
    };
}

inline auto j_lambda_direct_u(std::vector<Mat> &JJ, const Vec &da,
                              const Vec &db) {
    return [&](const IntegralResult &args) {
        auto Ja = qm::block::a(JJ[args.thread]);
        auto Jb = qm::block::a(JJ[args.thread]);
        size_t offset = 0;
        if (args.bf[0] != args.bf[1]) {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                    da(i) * buf_mat;
                Ja.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
                    da(i) * buf_mat.transpose();
                Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                    db(i) * buf_mat;
                Jb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
                    db(i) * buf_mat.transpose();
                offset += args.dims[0] * args.dims[1];
            }
        } else {
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                    da(i) * buf_mat;
                Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                    db(i) * buf_mat;
                offset += args.dims[0] * args.dims[1];
            }
        }
    };
}

inline auto k_lambda_direct_r(std::vector<Mat> &iuP,
                              const MolecularOrbitals &mo) {
    size_t nmo = mo.Cocc.cols();
    return [&, nmo](const IntegralResult &args) {
        for (size_t i = 0; i < mo.Cocc.cols(); i++) {
            auto &iuPx = iuP[nmo * args.thread + i];
            auto c2 = mo.Cocc.block(args.bf[0], i, args.dims[0], 1);
            auto c3 = mo.Cocc.block(args.bf[1], i, args.dims[1], 1);

            size_t offset = 0;
            if (args.bf[0] != args.bf[1]) {
                for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2];
                     r++) {
                    Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                                  args.dims[0], args.dims[1]);
                    iuPx.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3;
                    iuPx.block(args.bf[1], r, args.dims[1], 1) +=
                        (buf_mat.transpose() * c2);
                    offset += args.dims[0] * args.dims[1];
                }
            } else {
                for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2];
                     r++) {
                    Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                                  args.dims[0], args.dims[1]);
                    iuPx.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3;
                    offset += args.dims[0] * args.dims[1];
                }
            }
        }
    };
}

inline auto k_lambda_direct_u(std::vector<Mat> &iuPa, std::vector<Mat> &iuPb,
                              const MolecularOrbitals &mo) {
    size_t nmo = mo.Cocc.cols();
    return [&, nmo](const IntegralResult &args) {
        for (size_t i = 0; i < mo.Cocc.cols(); i++) {
            auto &iuPxa = iuPa[nmo * args.thread + i];
            auto &iuPxb = iuPb[nmo * args.thread + i];
            auto c2a =
                qm::block::a(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
            auto c3a =
                qm::block::a(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);
            auto c2b =
                qm::block::b(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
            auto c3b =
                qm::block::b(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);

            size_t offset = 0;
            if (args.bf[0] != args.bf[1]) {
                for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2];
                     r++) {
                    Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                                  args.dims[0], args.dims[1]);
                    iuPxa.block(args.bf[0], r, args.dims[0], 1) +=
                        buf_mat * c3a;
                    iuPxa.block(args.bf[1], r, args.dims[1], 1) +=
                        (buf_mat.transpose() * c2a);

                    iuPxb.block(args.bf[0], r, args.dims[0], 1) +=
                        buf_mat * c3b;
                    iuPxb.block(args.bf[1], r, args.dims[1], 1) +=
                        (buf_mat.transpose() * c2b);
                    offset += args.dims[0] * args.dims[1];
                }
            } else {
                for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2];
                     r++) {
                    Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                                  args.dims[0], args.dims[1]);
                    iuPxa.block(args.bf[0], r, args.dims[0], 1) +=
                        buf_mat * c3a;
                    iuPxb.block(args.bf[0], r, args.dims[0], 1) +=
                        buf_mat * c3b;
                    offset += args.dims[0] * args.dims[1];
                }
            }
        }
    };
}

inline auto jk_lambda_direct_r(std::vector<Vec> &gg, std::vector<Mat> &iuP,
                               const MolecularOrbitals &mo) {
    return [&](const IntegralResult &args) {
        auto &g = gg[args.thread];
        size_t offset = 0;
        size_t nocc = mo.Cocc.cols();
        auto c3 = mo.Cocc.block(args.bf[1], 0, args.dims[1], nocc);
        Mat c2_term(args.dims[0], nocc);

        if (args.bf[0] != args.bf[1]) {
            auto c2 = mo.Cocc.block(args.bf[0], 0, args.dims[0], nocc);
            Mat c3_term(args.dims[1], nocc);
            for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                g(r) += (mo.D.block(args.bf[0], args.bf[1], args.dims[0],
                                    args.dims[1])
                             .array() *
                         buf_mat.array())
                            .sum();
                c2_term = buf_mat * c3;
                g(r) += (mo.D.block(args.bf[1], args.bf[0], args.dims[1],
                                    args.dims[0])
                             .array() *
                         buf_mat.transpose().array())
                            .sum();
                c3_term = buf_mat.transpose() * c2;

                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPx = iuP[i];
                    iuPx.block(args.bf[0], r, args.dims[0], 1) +=
                        c2_term.block(0, i, args.dims[0], 1);
                    iuPx.block(args.bf[1], r, args.dims[1], 1) +=
                        c3_term.block(0, i, args.dims[1], 1);
                }
                offset += args.dims[0] * args.dims[1];
            }
        } else {
            for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                g(r) += (mo.D.block(args.bf[0], args.bf[1], args.dims[0],
                                    args.dims[1])
                             .array() *
                         buf_mat.array())
                            .sum();
                c2_term = buf_mat * c3;

                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPx = iuP[i];
                    iuPx.block(args.bf[0], r, args.dims[0], 1) +=
                        c2_term.block(0, i, args.dims[0], 1);
                }
                offset += args.dims[0] * args.dims[1];
            }
        }
    };
}

inline auto jk_lambda_direct_u(std::vector<Vec> &gg_alpha,
                               std::vector<Vec> &gg_beta,
                               std::vector<Mat> &iuPa, std::vector<Mat> &iuPb,
                               const MolecularOrbitals &mo) {
    return [&](const IntegralResult &args) {
        auto &ga = gg_alpha[args.thread];
        auto &gb = gg_beta[args.thread];
        size_t offset = 0;
        size_t nocc = mo.Cocc.cols();
        auto c3a =
            qm::block::a(mo.Cocc).block(args.bf[1], 0, args.dims[1], nocc);
        auto c3b =
            qm::block::b(mo.Cocc).block(args.bf[1], 0, args.dims[1], nocc);
        auto Da = qm::block::a(mo.D);
        auto Db = qm::block::b(mo.D);
        Mat c2_term_a(args.dims[0], nocc);
        Mat c2_term_b(args.dims[0], nocc);

        if (args.bf[0] != args.bf[1]) {
            auto c2a =
                qm::block::a(mo.Cocc).block(args.bf[0], 0, args.dims[0], nocc);
            auto c2b =
                qm::block::b(mo.Cocc).block(args.bf[0], 0, args.dims[0], nocc);
            Mat c3_term_a(args.dims[1], nocc);
            Mat c3_term_b(args.dims[1], nocc);
            for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);

                ga(r) += (Da.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                c2_term_a = buf_mat * c3a;

                gb(r) += (Db.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                c2_term_b = buf_mat * c3b;

                ga(r) += (Da.block(args.bf[1], args.bf[0], args.dims[1],
                                   args.dims[0])
                              .array() *
                          buf_mat.transpose().array())
                             .sum();
                c3_term_a = buf_mat.transpose() * c2a;

                gb(r) += (Db.block(args.bf[1], args.bf[0], args.dims[1],
                                   args.dims[0])
                              .array() *
                          buf_mat.transpose().array())
                             .sum();
                c3_term_b = buf_mat.transpose() * c2b;

                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPxa = iuPa[i];
                    iuPxa.block(args.bf[0], r, args.dims[0], 1) +=
                        c2_term_a.block(0, i, args.dims[0], 1);
                    iuPxa.block(args.bf[1], r, args.dims[1], 1) +=
                        c3_term_a.block(0, i, args.dims[1], 1);
                    auto &iuPxb = iuPb[i];
                    iuPxb.block(args.bf[0], r, args.dims[0], 1) +=
                        c2_term_b.block(0, i, args.dims[0], 1);
                    iuPxb.block(args.bf[1], r, args.dims[1], 1) +=
                        c3_term_b.block(0, i, args.dims[1], 1);
                }
                offset += args.dims[0] * args.dims[1];
            }
        } else {
            for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                ga(r) += (Da.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                c2_term_a = buf_mat * c3a;
                gb(r) += (Db.block(args.bf[0], args.bf[1], args.dims[0],
                                   args.dims[1])
                              .array() *
                          buf_mat.array())
                             .sum();
                c2_term_b = buf_mat * c3b;
                for (int i = 0; i < mo.Cocc.cols(); i++) {
                    auto &iuPxa = iuPa[i];
                    iuPxa.block(args.bf[0], r, args.dims[0], 1) +=
                        c2_term_a.block(0, i, args.dims[0], 1);
                    auto &iuPxb = iuPb[i];
                    iuPxb.block(args.bf[0], r, args.dims[0], 1) +=
                        c2_term_b.block(0, i, args.dims[0], 1);
                }
                offset += args.dims[0] * args.dims[1];
            }
        }
    };
}

Mat stored_coulomb_kernel_r(const Mat &ints, const AOBasis &aobasis,
                            const AOBasis &auxbasis,
                            const MolecularOrbitals &mo,
                            const Eigen::LLT<Mat> V_LLt) {
    const auto nbf = aobasis.nbf();
    const auto ndf = auxbasis.nbf();
    Vec g(ndf);
    for (int r = 0; r < ndf; r++) {
        const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
        g(r) = (mo.D.array() * tr.array()).sum();
    }
    Vec d = V_LLt.solve(g);
    Mat J = Mat::Zero(nbf, nbf);
    for (int r = 0; r < ndf; r++) {
        const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
        J += d(r) * tr;
    }
    return 2 * J;
}

Mat stored_exchange_kernel_r(const Mat &ints, const AOBasis &aobasis,
                             const AOBasis &auxbasis,
                             const MolecularOrbitals &mo,
                             const Eigen::LLT<Mat> Vsqrt_LLt) {
    const auto nbf = aobasis.nbf();
    const auto ndf = auxbasis.nbf();
    Mat K = Mat::Zero(nbf, nbf);
    // temporaries
    Mat iuP = Mat::Zero(nbf, ndf);
    Mat B(nbf, ndf);
    for (size_t i = 0; i < mo.Cocc.cols(); i++) {
        auto c = mo.Cocc.col(i);
        for (size_t r = 0; r < ndf; r++) {
            const auto vu = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
            iuP.col(r) = (vu * c);
        }
        B = Vsqrt_LLt.solve(iuP.transpose());
        K.noalias() += B.transpose() * B;
    }
    return K;
}

template <ShellKind kind, typename Lambda>
void three_center_aux_kernel(Lambda &f, qm::cint::IntegralEnvironment &env,
                             const qm::AOBasis &aobasis,
                             const qm::AOBasis &auxbasis,
                             const ShellPairList &shellpairs,
                             int thread_id = 0) noexcept {
    auto nthreads = occ::parallel::get_num_threads();
    occ::qm::cint::Optimizer opt(env, Op::coulomb, 3);
    size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                     auxbasis.max_shell_size();
    auto buffer = std::make_unique<double[]>(bufsize);
    IntegralResult args;
    args.thread = thread_id;
    args.buffer = buffer.get();
    std::array<int, 3> shell_idx;
    const auto &first_bf_ao = aobasis.first_bf();
    const auto &first_bf_aux = auxbasis.first_bf();
    const int nsh_ao = aobasis.size();
    for (int auxP = 0; auxP < auxbasis.size(); auxP++) {
        if (auxP % nthreads != thread_id)
            continue;
        const auto &shauxP = auxbasis[auxP];
        args.bf[2] = first_bf_aux[auxP];
        args.shell[2] = auxP;
        for (int p = 0; p < aobasis.size(); p++) {
            args.bf[0] = first_bf_ao[p];
            args.shell[0] = p;
            const auto &shp = aobasis[p];
            const auto &plist = shellpairs.at(p);
            for (const int &q : plist) {
                args.bf[1] = first_bf_ao[q];
                args.shell[1] = q;
                const auto &shq = aobasis[q];
                shell_idx = {p, q, auxP + nsh_ao};
                args.dims = env.three_center_helper<Op::coulomb, kind>(
                    shell_idx, nullptr, buffer.get(), nullptr);
                if (args.dims[0] > -1) {
                    f(args);
                }
            }
        }
    }
}

void IntegralEngineDF::compute_stored_integrals() {
    if (m_integral_store.rows() == 0) {
        size_t nbf = m_ao_engine.nbf();
        size_t ndf = m_aux_engine.nbf();
        m_integral_store = Mat::Zero(nbf * nbf, ndf);
        auto lambda = [&](const IntegralResult &args) {
            size_t offset = 0;
            for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
                auto x =
                    Eigen::Map<Mat>(m_integral_store.col(i).data(), nbf, nbf);
                Eigen::Map<const Mat> buf_mat(args.buffer + offset,
                                              args.dims[0], args.dims[1]);
                x.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) =
                    buf_mat;
                if (args.bf[0] != args.bf[1])
                    x.block(args.bf[1], args.bf[0], args.dims[1],
                            args.dims[0]) = buf_mat.transpose();
                offset += args.dims[0] * args.dims[1];
            }
        };

        auto lambda2 = [&](int thread_id) {
            if (m_ao_engine.is_spherical()) {
                three_center_aux_kernel<ShellKind::Spherical>(
                    lambda, m_ao_engine.env(), m_ao_engine.aobasis(),
                    m_ao_engine.auxbasis(), m_ao_engine.shellpairs(),
                    thread_id);
            } else {
                three_center_aux_kernel<ShellKind::Cartesian>(
                    lambda, m_ao_engine.env(), m_ao_engine.aobasis(),
                    m_ao_engine.auxbasis(), m_ao_engine.shellpairs(),
                    thread_id);
            }
        };
        occ::parallel::parallel_do(lambda2);
    }
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_exchange_operator_kernel_r(IntegralEngine &engine,
                                      IntegralEngine &engine_aux,
                                      const MolecularOrbitals &mo,
                                      const Eigen::LLT<Mat> &Vsqrt_LLt) {
    const auto nthreads = occ::parallel::get_num_threads();
    size_t nmo = mo.Cocc.cols();
    const auto nbf = engine.aobasis().nbf();
    const auto ndf = engine.auxbasis().nbf();
    Mat K = Mat::Zero(nbf, nbf);
    std::vector<Mat> iuP(nmo * nthreads, Mat::Zero(nbf, ndf));
    Mat B(nbf, ndf);

    auto klambda = k_lambda_direct_r(iuP, mo);

    auto lambda = [&](int thread_id) {
        three_center_aux_kernel<kind>(klambda, engine.env(), engine.aobasis(),
                                      engine.auxbasis(), engine.shellpairs(),
                                      thread_id);
    };
    occ::parallel::parallel_do(lambda);

    for (size_t i = nmo; i < nmo * nthreads; i++) {
        iuP[i % nmo] += iuP[i];
    }

    for (size_t i = 0; i < nmo; i++) {
        B = Vsqrt_LLt.solve(iuP[i].transpose());
        K.noalias() += B.transpose() * B;
    }

    return 0.5 * (K + K.transpose());
}

Mat IntegralEngineDF::exchange(const MolecularOrbitals &mo) {
    bool direct = !use_stored_integrals();
    if (!direct) {
        compute_stored_integrals();
        return stored_exchange_kernel_r(m_integral_store, m_ao_engine.aobasis(),
                                        m_ao_engine.auxbasis(), mo, Vsqrt_LLt);
    } else if (m_ao_engine.is_spherical()) {
        return direct_exchange_operator_kernel_r<ShellKind::Spherical>(
            m_ao_engine, m_aux_engine, mo, Vsqrt_LLt);
    } else {
        return direct_exchange_operator_kernel_r<ShellKind::Cartesian>(
            m_ao_engine, m_aux_engine, mo, Vsqrt_LLt);
    }
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_r(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const Eigen::LLT<Mat> &V_LLt) {
    const auto nthreads = occ::parallel::get_num_threads();
    const auto nbf = engine.aobasis().nbf();
    const auto ndf = engine.auxbasis().nbf();
    std::vector<Vec> gg(nthreads, Vec::Zero(ndf));
    std::vector<Mat> JJ(nthreads, Mat::Zero(nbf, nbf));

    const auto &D = mo.D;
    auto glambda = g_lambda_direct_r(gg, mo);
    auto lambda = [&](int thread_id) {
        three_center_aux_kernel<kind>(glambda, engine.env(), engine.aobasis(),
                                      engine.auxbasis(), engine.shellpairs(),
                                      thread_id);
    };
    occ::parallel::parallel_do(lambda);

    for (int i = 1; i < nthreads; i++) {
        gg[0] += gg[i];
    }
    Vec d = V_LLt.solve(gg[0]);

    auto jlambda = j_lambda_direct_r(JJ, d);
    auto lambda2 = [&](int thread_id) {
        three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                      engine.auxbasis(), engine.shellpairs(),
                                      thread_id);
    };
    occ::parallel::parallel_do(lambda2);

    for (int i = 1; i < nthreads; i++) {
        JJ[0] += JJ[i];
    }
    return (JJ[0] + JJ[0].transpose());
}

Mat IntegralEngineDF::coulomb(const MolecularOrbitals &mo) {
    bool direct = !use_stored_integrals();
    if (!direct) {
        compute_stored_integrals();
        return stored_coulomb_kernel_r(m_integral_store, m_ao_engine.aobasis(),
                                       m_ao_engine.auxbasis(), mo, V_LLt);
    } else if (m_ao_engine.is_spherical()) {
        return direct_coulomb_operator_kernel_r<ShellKind::Spherical>(
            m_ao_engine, m_aux_engine, mo, V_LLt);
    } else {
        return direct_coulomb_operator_kernel_r<ShellKind::Cartesian>(
            m_ao_engine, m_aux_engine, mo, V_LLt);
    }
}

template <ShellKind kind = ShellKind::Cartesian>
std::pair<Mat, Mat> direct_coulomb_and_exchange_operator_kernel_r(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const Eigen::LLT<Mat> &V_LLt,
    const Eigen::LLT<Mat> &Vsqrt_LLt) {
    size_t nmo = mo.Cocc.cols();
    const auto nthreads = occ::parallel::get_num_threads();
    const auto nbf = engine.aobasis().nbf();
    const auto ndf = engine.auxbasis().nbf();

    std::vector<Vec> gg(nthreads, Vec::Zero(ndf));
    std::vector<Mat> JJ(nthreads, Mat::Zero(nbf, nbf));
    std::vector<Mat> KK(nthreads, Mat::Zero(nbf, nbf));

    std::vector<Mat> iuP(nmo, Mat::Zero(nbf, ndf));

    auto jk_lambda_1 = jk_lambda_direct_r(gg, iuP, mo);
    auto lambda = [&](int thread_id) {
        three_center_aux_kernel<kind>(jk_lambda_1, engine.env(),
                                      engine.aobasis(), engine.auxbasis(),
                                      engine.shellpairs(), thread_id);
    };
    occ::parallel::parallel_do(lambda);

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
    auto lambda2 = [&](int thread_id) {
        three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                      engine.auxbasis(), engine.shellpairs(),
                                      thread_id);
    };
    occ::parallel::parallel_do(lambda2);

    for (int i = 1; i < nthreads; i++) {
        JJ[0] += JJ[i];
        KK[0] += KK[i];
    }

    return {JJ[0] + JJ[0].transpose(), 0.5 * (KK[0] + KK[0].transpose())};
}

std::pair<Mat, Mat>
IntegralEngineDF::coulomb_and_exchange(const MolecularOrbitals &mo) {
    bool direct = !use_stored_integrals();
    if (!direct) {
        compute_stored_integrals();
        return {coulomb(mo), exchange(mo)};
    } else if (m_ao_engine.is_spherical()) {
        return direct_coulomb_and_exchange_operator_kernel_r<
            ShellKind::Spherical>(m_ao_engine, m_aux_engine, mo, V_LLt,
                                  Vsqrt_LLt);
    } else {
        return direct_coulomb_and_exchange_operator_kernel_r<
            ShellKind::Cartesian>(m_ao_engine, m_aux_engine, mo, V_LLt,
                                  Vsqrt_LLt);
    }
}

Mat IntegralEngineDF::fock_operator(const MolecularOrbitals &mo) {
    auto [J, K] = coulomb_and_exchange(mo);
    J.noalias() -= K;
    return J;
}

} // namespace occ::qm
