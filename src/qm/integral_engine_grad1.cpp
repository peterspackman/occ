#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;


template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_grad(Lambda &f, cint::IntegralEnvironment &env,
                         const AOBasis &basis, int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<2>;
    occ::qm::cint::Optimizer opt(env, op, 2, 1);
    auto nthreads = occ::parallel::get_num_threads();
    auto bufsize = env.buffer_size_1e(op, 1);
    const auto nsh = basis.size();

    auto buffer = std::make_unique<double[]>(bufsize);
    const auto &first_bf = basis.first_bf();
    for (int p = 0, pq = 0; p < nsh; p++) {
        int bf1 = first_bf[p];
        const auto &sh1 = basis[p];
        for (int q = 0; q <= p; q++) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            const auto &sh2 = basis[q];
            std::array<int, 2> idxs{p, q};
            Result args{thread_id,
                        idxs,
                        {bf1, bf2},
                        env.two_center_helper_grad<op, kind>(
                            idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                        buffer.get()};
            if (args.dims[0] > -1)
                f(args);
        }
    }
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_with_shellpairs_grad(Lambda &f,
                                         cint::IntegralEnvironment &env,
                                         const AOBasis &basis,
                                         const ShellPairList &shellpairs,
                                         int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<2>;
    occ::qm::cint::Optimizer opt(env, op, 2, 1);
    auto nthreads = occ::parallel::get_num_threads();
    auto bufsize = env.buffer_size_1e(op, 1);

    auto buffer = std::make_unique<double[]>(bufsize);
    const auto &first_bf = basis.first_bf();
    for (int p = 0, pq = 0; p < basis.size(); p++) {
        int bf1 = first_bf[p];
        const auto &sh1 = basis[p];
        for (const int &q : shellpairs[p]) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            const auto &sh2 = basis[q];
            std::array<int, 2> idxs{p, q};
            Result args{thread_id,
                        idxs,
                        {bf1, bf2},
                        env.two_center_helper_grad<op, kind>(
                            idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                        buffer.get()};
            if (args.dims[0] > -1)
                f(args);

            if(p != q) {
                std::array<int, 2> idxs2{q, p};
                Result args2{thread_id,
                            idxs2,
                            {bf2, bf1},
                            env.two_center_helper_grad<op, kind>(
                                idxs2, opt.optimizer_ptr(), buffer.get(), nullptr),
                            buffer.get()};
                if (args2.dims[0] > -1)
                    f(args2);
            }

        }
    }
}

template <Op op, ShellKind kind = ShellKind::Cartesian>
MatTriple one_electron_operator_grad_kernel(const AOBasis &basis,
                                 cint::IntegralEnvironment &env,
                                 const ShellPairList &shellpairs) {
    using Result = IntegralEngine::IntegralResult<2>;
    auto nthreads = occ::parallel::get_num_threads();
    const auto nbf = basis.nbf();
    MatTriple result;
    result.x = Mat::Zero(nbf, nbf);
    result.y = Mat::Zero(nbf, nbf);
    result.z = Mat::Zero(nbf, nbf);

    std::vector<MatTriple> results;
    results.push_back(result);

    for (size_t i = 1; i < nthreads; i++) {
        results.push_back(results[0]);
    }

    auto f = [&results](const Result &args) {
        auto &result = results[args.thread];
        const auto num_elements = args.dims[0] * args.dims[1];
        Eigen::Map<const Mat>
          tmpx(args.buffer, args.dims[0], args.dims[1]),
          tmpy(args.buffer + num_elements, args.dims[0], args.dims[1]),
          tmpz(args.buffer + num_elements * 2, args.dims[0], args.dims[1]);

        result.x.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpx;
        result.y.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpy;
        result.z.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpz;
    };

    auto lambda = [&](int thread_id) {
        if (shellpairs.size() > 0) {
            evaluate_two_center_with_shellpairs_grad<op, kind>(
                f, env, basis, shellpairs, thread_id);
        } else {
            evaluate_two_center_grad<op, kind>(f, env, basis, thread_id);
        }
    };
    occ::parallel::parallel_do(lambda);

    for (auto i = 1; i < nthreads; ++i) {
        results[0].x.noalias() += results[i].x;
        results[0].y.noalias() += results[i].y;
        results[0].z.noalias() += results[i].z;
    }
    return results[0];
}

MatTriple IntegralEngine::one_electron_operator_grad(Op op,
                                          bool use_shellpair_list) const {
    bool spherical = is_spherical();
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    ShellPairList empty_shellpairs = {};
    const auto &shellpairs =
        use_shellpair_list ? m_shellpairs : empty_shellpairs;
    switch (op) {
    case Op::overlap: {
        if (spherical) {
            return one_electron_operator_grad_kernel<Op::overlap, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_grad_kernel<Op::overlap, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    case Op::nuclear: {
        if (spherical) {
            return one_electron_operator_grad_kernel<Op::nuclear, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_grad_kernel<Op::nuclear, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    case Op::kinetic: {
        if (spherical) {
            return one_electron_operator_grad_kernel<Op::kinetic, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_grad_kernel<Op::kinetic, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    case Op::coulomb: {
        if (spherical) {
            return one_electron_operator_grad_kernel<Op::coulomb, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_grad_kernel<Op::coulomb, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    default:
        throw std::runtime_error("Invalid operator for two-center integral");
        break;
    }
}

}
