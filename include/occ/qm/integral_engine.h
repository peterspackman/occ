#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/parallel.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/occshell.h>
#include <vector>

namespace occ::qm {

class IntegralEngine {
  public:
    template <size_t num_centers> struct IntegralResult {
        int thread{0};
        std::array<int, num_centers> shell{0};
        std::array<int, num_centers> bf{0};
        std::array<int, num_centers> dims{0};
        const double *buffer{nullptr};
    };

    using ShellPairList = std::vector<std::vector<size_t>>;
    using ShellList = std::vector<OccShell>;
    using ShellKind = OccShell::Kind;
    using IntEnv = cint::IntegralEnvironment;
    using Op = cint::Operator;
    using AtomList = std::vector<occ::core::Atom>;
    using Buffer = std::vector<double>;

    IntegralEngine(const AtomList &at, const ShellList &sh)
        : m_atoms(at), m_shells(sh), m_env(at, sh) {
        for (const auto &shell : m_shells) {
            m_first_bf.push_back(m_nbf);
            m_nbf += shell.size();
            m_nsh += 1;
            m_max_shell_size = std::max(m_max_shell_size, shell.size());
        }
        compute_m_shellpairs();
    }

    inline auto nbf() const { return m_nbf; }
    inline auto nbf_aux() const { return m_nbf_aux; }
    inline auto nsh() const { return m_nsh; }
    inline auto nsh_aux() const { return m_nsh_aux; }

    void set_auxiliary_basis(const ShellList &bs) {
        m_nbf_aux = 0;
        m_nsh_aux = 0;
        m_shells_aux.reserve(bs.size());
        m_sites_aux.reserve(bs.size());
        for (const auto &shell : bs) {
            m_shells_aux.push_back(shell);
            m_sites_aux.push_back(
                {0, shell.origin(0), shell.origin(1), shell.origin(2)});
            m_first_bf_aux.push_back(m_nbf);
            m_nbf_aux += shell.size();
            m_nsh_aux += 1;
            m_max_shell_size_aux = std::max(m_max_shell_size, shell.size());
        }
        AtomList combined_sites = m_atoms;
        combined_sites.insert(combined_sites.end(), m_sites_aux.begin(),
                              m_sites_aux.end());
        ShellList combined = m_shells;
        combined.insert(combined.end(), m_shells_aux.begin(),
                        m_shells_aux.end());
        m_env = IntEnv(combined_sites, combined);
    }

    bool have_auxiliary_basis() const { return m_nsh_aux > 0; }

    template <Op op, ShellKind kind, typename Lambda>
    void evaluate_two_center(Lambda &f, int thread_id = 0) const {
        auto nthreads = occ::parallel::get_num_threads();
        Buffer buffer(buffer_size_1e());
        for (int p = 0, pq = 0; p < m_nsh; p++) {
            int bf1 = m_first_bf[p];
            const auto &sh1 = m_shells[p];
            for (const int &q : m_shellpairs.at(p)) {
                if (pq++ % nthreads != thread_id)
                    continue;
                int bf2 = m_first_bf[q];
                const auto &sh2 = m_shells[q];
                std::array<int, 2> idxs{p, q};
                IntegralResult<2> args{
                    thread_id,
                    idxs,
                    {bf1, bf2},
                    m_env.two_center_helper<op, kind>(idxs, nullptr,
                                                      buffer.data(), nullptr),
                    buffer.data()};
                if (args.dims[0] > -1)
                    f(args);
            }
        }
    }

    template <Op op, ShellKind kind, typename Lambda>
    void evaluate_four_center(Lambda &f, int thread_id = 0) const {
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<double> buffer(buffer_size_2e());
        std::array<int, 4> shell_idx;
        std::array<int, 4> bf;
        for (int p = 0, pqrs = 0; p < m_nsh; p++) {
            const auto &sh1 = m_shells[p];
            bf[0] = m_first_bf[p];
            const auto &plist = m_shellpairs.at(p);
            for (const int &q : plist) {
                bf[1] = m_first_bf[q];
                const auto &sh2 = m_shells[q];
                for (int r = 0; r <= p; r++) {
                    const auto &sh3 = m_shells[r];
                    bf[2] = m_first_bf[r];
                    const auto s_max = (p == r) ? q : r;
                    for (const int s : m_shellpairs.at(r)) {
                        if (s > s_max)
                            break;
                        if (pqrs++ % nthreads != thread_id)
                            continue;
                        bf[3] = m_first_bf[s];
                        const auto &sh4 = m_shells[s];
                        shell_idx = {p, q, r, s};

                        IntegralResult<4> args{
                            thread_id, shell_idx, bf,
                            m_env.four_center_helper<Op::coulomb, kind>(
                                shell_idx, nullptr, buffer.data(), nullptr),
                            buffer.data()};
                        if (args.dims[0] > -1)
                            f(args);
                    }
                }
            }
        }
    }

    template <ShellKind kind, typename Lambda>
    void evaluate_three_center(Lambda &f, int thread_id = 0) const {
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<double> buffer(buffer_size_3e());
        IntegralResult<3> args;
        args.thread = thread_id;
        args.buffer = buffer.data();
        for (int auxP = 0; auxP < m_nsh_aux; auxP++) {
            if (auxP % nthreads != thread_id)
                continue;
            const auto &shauxP = m_shells_aux[auxP];
            args.bf[2] = m_first_bf_aux[auxP];
            args.shell[2] = auxP + m_nsh;
            for (int p = 0; p < m_nsh; p++) {
                args.bf[0] = m_first_bf[p];
                args.shell[0] = p;
                const auto &shp = m_shells[p];
                const auto &plist = m_shellpairs.at(p);
                for (const int &q : plist) {
                    args.bf[1] = m_first_bf[q];
                    args.shell[1] = q;
                    args.dims = m_env.three_center_helper<Op::coulomb, kind>(
                        args.shell, nullptr, buffer.data(), nullptr);
                    if (args.dims[0] > -1)
                        f(args);
                }
            }
        }
    }

    template <Op op, ShellKind kind = ShellKind::Cartesian>
    Mat one_electron_operator() const {
        auto nthreads = occ::parallel::get_num_threads();
        Mat result = Mat::Zero(m_nbf, m_nbf);
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(m_nbf, m_nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        auto f = [&results](const IntegralResult<2> &args) {
            auto &result = results[args.thread];
            Eigen::Map<const occ::Mat> tmp(args.buffer, args.dims[0],
                                           args.dims[1]);
            result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) =
                tmp;
            if (args.shell[0] != args.shell[1]) {
                result.block(args.bf[1], args.bf[0], args.dims[1],
                             args.dims[0]) = tmp.transpose();
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_two_center<op, kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }
        return results[0];
    }

    template <ShellKind kind = ShellKind::Cartesian>
    Mat fock_operator(const Mat &D) const {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(m_nbf, m_nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        auto f = [&D, &results](const IntegralResult<4> &args) {
            auto &g = results[args.thread];
            auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
            auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                    ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                    : 2;
            auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
            auto scale = pq_degree * rs_degree * pr_qs_degree;

            for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
                const auto bf3 = f3 + args.bf[3];
                for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
                    const auto bf2 = f2 + args.bf[2];
                    for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
                        const auto bf1 = f1 + args.bf[1];
                        for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
                            const auto bf0 = f0 + args.bf[0];
                            const auto value = args.buffer[f0123] * scale;
                            // J
                            g(bf0, bf1) += D(bf2, bf3) * value;
                            g(bf2, bf3) += D(bf0, bf1) * value;
                            // K
                            g(bf0, bf2) -= 0.25 * D(bf1, bf3) * value;
                            g(bf1, bf3) -= 0.25 * D(bf0, bf2) * value;
                            g(bf0, bf3) -= 0.25 * D(bf1, bf2) * value;
                            g(bf1, bf2) -= 0.25 * D(bf0, bf3) * value;
                        }
                    }
                }
            }
        };
        auto lambda = [&](int thread_id) {
            evaluate_four_center<op, kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }

        return 0.5 * (results[0] + results[0].transpose());
    }

    template <ShellKind kind = ShellKind::Cartesian>
    Vec electric_potential(const Mat &D, const Mat3N &points) {
        ShellList dummy_shells;
        dummy_shells.reserve(points.cols());
        for (size_t i = 0; i < points.cols(); i++) {
            dummy_shells.push_back(
                OccShell({1.0, {points(0, i), points(1, i), points(2, i)}}));
        }
        set_auxiliary_basis(dummy_shells);
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<Vec> results;
        results.emplace_back(Vec::Zero(points.cols()));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        size_t nsh = m_nsh;
        auto f = [nsh, &D, &results](const IntegralResult<3> &args) {
            auto &v = results[args.thread];
            auto scale = (args.shell[0] == args.shell[1]) ? 1 : 2;
            Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
            v(args.shell[2] - nsh) -=
                (D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                     .array() *
                 tmp.array())
                    .sum() *
                scale;
        };

        auto lambda = [&](int thread_id) {
            evaluate_three_center<kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; i++) {
            results[0] += results[i];
        }
        return 2 * results[0];
    }

    void compute_m_shellpairs(double threshold = 1e-12) {
        constexpr auto op = Op::overlap;
        m_shellpairs.resize(m_nsh);
        std::vector<double> buffer(buffer_size_1e());
        for (int p = 0; p < m_nsh; p++) {
            auto &plist = m_shellpairs[p];
            const auto &sh1 = m_shells[p];
            for (int q = 0; q <= p; q++) {
                if (m_shells[p].origin == m_shells[q].origin) {
                    plist.push_back(q);
                    continue;
                }
                const auto &sh2 = m_shells[q];
                std::array<int, 2> idxs{p, q};
                std::array<int, 2> dims =
                    m_env.two_center_helper<op, ShellKind::Spherical>(
                        idxs, nullptr, buffer.data(), nullptr);
                Eigen::Map<const occ::Mat> tmp(buffer.data(), dims[0], dims[1]);
                if (tmp.norm() >= threshold) {
                    plist.push_back(q);
                }
            }
        }
    }

    template <ShellKind kind = ShellKind::Cartesian> Mat schwarz() const {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        constexpr bool use_euclidean_norm{false};
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(m_nsh, m_nsh));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        auto f = [&results](const IntegralResult<4> &args) {
            auto &result = results[args.thread];
            auto N = args.dims[0] * args.dims[1];
            Eigen::Map<const occ::Mat> tmp(args.buffer, N, N);
            double sq_norm =
                use_euclidean_norm ? tmp.norm() : tmp.lpNorm<Eigen::Infinity>();
            double norm = std::sqrt(sq_norm);
            result(args.shell[0], args.shell[1]) = norm;
            result(args.shell[1], args.shell[0]) = norm;
        };

        auto lambda = [&](int thread_id) {
            std::vector<double> buffer(buffer_size_2e());
            for (int p = 0, pq = 0; p < m_nsh; p++) {
                int bf1 = m_first_bf[p];
                const auto &sh1 = m_shells[p];
                for (const int &q : m_shellpairs.at(p)) {
                    if (pq++ % nthreads != thread_id)
                        continue;
                    int bf2 = m_first_bf[q];
                    const auto &sh2 = m_shells[q];
                    std::array<int, 4> idxs{p, q, p, q};
                    IntegralResult<4> args{
                        thread_id,
                        idxs,
                        {bf1, bf2, bf1, bf2},
                        m_env.four_center_helper<op, kind>(
                            idxs, nullptr, buffer.data(), nullptr),
                        buffer.data()};
                    if (args.dims[0] > -1)
                        f(args);
                }
            }
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }

        return results[0];
    }

  private:
    size_t m_nbf{0}, m_nbf_aux{0};
    size_t m_nsh{0}, m_nsh_aux{0};
    size_t m_max_shell_size{0}, m_max_shell_size_aux{0};
    AtomList m_atoms, m_sites_aux;
    ShellList m_shells;
    ShellList m_shells_aux;
    std::vector<size_t> m_first_bf;
    std::vector<size_t> m_first_bf_aux;
    ShellPairList m_shellpairs;
    mutable IntEnv m_env;

    inline size_t buffer_size_1e() const {
        return m_max_shell_size * m_max_shell_size;
    }

    inline size_t buffer_size_3e() const {
        return m_max_shell_size_aux * buffer_size_1e();
    }

    inline size_t buffer_size_2e() const {
        return buffer_size_1e() * buffer_size_1e();
    }
};
} // namespace occ::qm
