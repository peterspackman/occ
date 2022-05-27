#include <occ/qm/integral_engine.h>

namespace occ::qm {

class IntegralEngineDF {
  public:
    using ShellPairList = std::vector<std::vector<size_t>>;
    using ShellList = std::vector<OccShell>;
    using AtomList = std::vector<occ::core::Atom>;
    using ShellKind = OccShell::Kind;
    using Op = cint::Operator;
    using Buffer = std::vector<double>;
    using IntegralResult = IntegralEngine::IntegralResult<3>;

    IntegralEngineDF(const AtomList &atoms, const ShellList &ao,
                     const ShellList &df);

    template <ShellKind kind = ShellKind::Cartesian>
    Mat exchange_operator(const MolecularOrbitals &mo) const {
        const auto nthreads = occ::parallel::get_num_threads();
        size_t nmo = mo.Cocc.cols();
        const auto nbf = m_ao_env.nbf();
        const auto ndf = m_aux_env.nbf();
        Mat K = Mat::Zero(nbf, nbf);
        std::vector<Mat> iuP(nmo * nthreads);
        Mat B(nbf, ndf);
        for (auto &x : iuP) {
            x = Mat::Zero(nbf, ndf);
        }

        auto klambda = [&](const IntegralResult &args) {
            size_t offset = nmo * args.thread;
            auto f = [&offset, &nmo, &mo, &iuP](int bf_aux, int bf1, int bf2,
                                                const auto &buf_mat) {
                const auto n1 = buf_mat.rows();
                const auto n2 = buf_mat.cols();
                bool same_shell = bf1 == bf2;
                for (size_t i = 0; i < nmo; i++) {
                    auto &iuPx = iuP[offset + i];
                    auto c2 = mo.Cocc.block(bf1, i, n1, 1);
                    auto c3 = mo.Cocc.block(bf2, i, n2, 1);
                    iuPx.block(bf1, bf_aux, n1, 1) += buf_mat * c3;
                    if (!same_shell)
                        iuPx.block(bf2, bf_aux, n2, 1) +=
                            (buf_mat.transpose() * c2);
                }
            };
            inner_loop(f, args);
        };

        auto lambda = [&](int thread_id) {
            m_ao_env.evaluate_three_center_aux<kind>(klambda, thread_id);
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

    template <ShellKind kind = ShellKind::Cartesian>
    Mat coulomb_operator(const MolecularOrbitals &mo) const {
        const auto nthreads = occ::parallel::get_num_threads();
        std::vector<Vec> gg(nthreads);
        std::vector<Mat> JJ(nthreads);
        for (int i = 0; i < nthreads; i++) {
            gg[i] = Vec::Zero(m_aux_env.nbf());
            JJ[i] = Mat::Zero(m_ao_env.nbf(), m_ao_env.nbf());
        }

        const auto &D = mo.D;
        auto glambda = [&](const IntegralResult &args) {
            auto &g = gg[args.thread];
            auto f = [&g, &D](int bf_aux, int bf1, int bf2,
                              const auto &buf_mat) {
                g(bf_aux) +=
                    (D.block(bf1, bf2, buf_mat.rows(), buf_mat.cols()).array() *
                     buf_mat.array())
                        .sum();
                if (bf1 != bf2)
                    g(bf_aux) +=
                        (D.block(bf2, bf1, buf_mat.cols(), buf_mat.rows())
                             .array() *
                         buf_mat.transpose().array())
                            .sum();
            };
            inner_loop(f, args);
        };

        auto lambda = [&](int thread_id) {
            m_ao_env.evaluate_three_center_aux<kind>(glambda, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (int i = 1; i < nthreads; i++) {
            gg[0] += gg[i];
        }
        Vec d = V_LLt.solve(gg[0]);

        auto jlambda = [&](const IntegralResult &args) {
            auto &J = JJ[args.thread];
            auto f = [&d, &J](int bf_aux, int bf1, int bf2,
                              const auto &buf_mat) {
                J.block(bf1, bf2, buf_mat.rows(), buf_mat.cols()) +=
                    d(bf_aux) * buf_mat;
                if (bf1 != bf2)
                    J.block(bf2, bf1, buf_mat.cols(), buf_mat.rows()) +=
                        d(bf_aux) * buf_mat.transpose();
            };
            inner_loop(f, args);
        };
        auto lambda2 = [&](int thread_id) {
            m_ao_env.evaluate_three_center_aux<kind>(jlambda, thread_id);
        };
        occ::parallel::parallel_do(lambda2);

        for (int i = 1; i < nthreads; i++) {
            JJ[0] += JJ[i];
        }
        return (JJ[0] + JJ[0].transpose());
    }

  private:
    mutable IntegralEngine m_ao_env;  // engine with ao basis & aux basis
    mutable IntegralEngine m_aux_env; // engine with just aux basis
    Eigen::LLT<Mat> V_LLt, Vsqrt_LLt;

    template <typename Func>
    void inner_loop(Func &f,
                    const IntegralEngine::IntegralResult<3> &args) const {
        const auto incr = args.dims[0] * args.dims[1];
        const auto first_bf_aux = m_aux_env.first_bf()[args.shell[2]];
        const auto bf1 = m_ao_env.first_bf()[args.shell[0]];
        const auto bf2 = m_ao_env.first_bf()[args.shell[1]];

        for (size_t p = 0, offset = 0; p < args.dims[2]; p++, offset += incr) {
            const auto bf_aux = first_bf_aux + p;
            Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                          args.dims[1]);
            f(bf_aux, bf1, bf2, buf_mat);
        }
    }
};

} // namespace occ::qm
