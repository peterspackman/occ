#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace occ::qm {

namespace impl {

inline void j_inner_r_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
    J(bf0, bf1) += D(bf2, bf3) * value;
}

inline void k_inner_r_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K,
		           int bf0, int bf1, int bf2, int bf3, double value) {
    K(bf0, bf2) -= D(bf1, bf3) * value;
    K(bf0, bf3) -= D(bf1, bf2) * value;
}

inline void j_inner_g_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
    auto Jaa = occ::qm::block::aa(J);
    auto Jbb = occ::qm::block::bb(J);
    const auto Daa = occ::qm::block::aa(D);
    const auto Dbb = occ::qm::block::bb(D);
    Jaa(bf0, bf1) += Daa(bf2, bf3) * value;
    Jbb(bf0, bf1) += Dbb(bf2, bf3) * value;
}

inline void k_inner_g_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K,
		           int bf0, int bf1, int bf2, int bf3, double value) {
    auto Kaa = occ::qm::block::aa(K);
    auto Kab = occ::qm::block::ab(K);
    auto Kba = occ::qm::block::ba(K);
    auto Kbb = occ::qm::block::bb(K);
    const auto Daa = occ::qm::block::aa(D);
    const auto Dab = occ::qm::block::ab(D);
    const auto Dba = occ::qm::block::ba(D);
    const auto Dbb = occ::qm::block::bb(D);
    Kaa(bf0, bf2) -= Daa(bf1, bf3) * value;
    Kaa(bf0, bf3) -= Daa(bf1, bf2) * value;
    Kbb(bf0, bf2) -= Dbb(bf1, bf3) * value;
    Kbb(bf0, bf3) -= Dbb(bf1, bf2) * value;
    Kab(bf0, bf2) -= (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
    Kab(bf0, bf3) -= (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
    Kba(bf0, bf2) -= (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
    Kba(bf0, bf3) -= (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
}

inline void j_inner_u_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
			   int bf1, int bf2, int bf3, double value) {
    auto Ja = occ::qm::block::a(J);
    auto Jb = occ::qm::block::b(J);
    const auto Da = occ::qm::block::a(D);
    const auto Db = occ::qm::block::b(D);
    Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
}

inline void k_inner_u_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K,
	                   int bf0, int bf1, int bf2, int bf3, double value) {
    auto Ka = occ::qm::block::a(K);
    auto Kb = occ::qm::block::b(K);
    const auto Da = occ::qm::block::a(D);
    const auto Db = occ::qm::block::b(D);
    Ka(bf0, bf2) -= 2 * Da(bf1, bf3) * value;
    Ka(bf0, bf3) -= 2 * Da(bf1, bf2) * value;
    Kb(bf0, bf2) -= 2 * Db(bf1, bf3) * value;
    Kb(bf0, bf3) -= 2 * Db(bf1, bf2) * value;
}

template <occ::qm::SpinorbitalKind sk>
void delegate_j_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
                int bf2, int bf3, double value) {
    if constexpr (sk == SpinorbitalKind::Restricted) {
        j_inner_r_grad(D, J, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        j_inner_u_grad(D, J, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::General) {
        j_inner_g_grad(D, J, bf0, bf1, bf2, bf3, value);
    }
}

template <occ::qm::SpinorbitalKind sk>
void delegate_k_grad(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K, int bf0, int bf1,
                int bf2, int bf3, double value) {
    if constexpr (sk == SpinorbitalKind::Restricted) {
        k_inner_r_grad(D, K, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        k_inner_u_grad(D, K, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::General) {
        k_inner_g_grad(D, K, bf0, bf1, bf2, bf3, value);
    }
}

} // namespace impl


using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;


template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_grad(Lambda &f, IntEnv &env,
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
        for (int q = 0; q <= p; q++) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
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
void evaluate_two_center_with_shellpairs_grad(Lambda &f, IntEnv &env,
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
        for (const auto &q : shellpairs[p]) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            std::array<int, 2> idxs{p, static_cast<int>(q)};
            Result args{thread_id,
                        idxs,
                        {bf1, bf2},
                        env.two_center_helper_grad<op, kind>(
                            idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                        buffer.get()};
            if (args.dims[0] > -1)
                f(args);

            if(p != q) {
                std::array<int, 2> idxs2{static_cast<int>(q), p};
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
                                            IntEnv &env,
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

MatTriple IntegralEngine::rinv_operator_grad_atom(size_t atom_index, bool use_shellpair_list) const {
    const auto &atoms = m_aobasis.atoms();
    if(atom_index > atoms.size()) throw std::runtime_error("Invalid atom index for rinv operator");

    bool spherical = is_spherical();
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    ShellPairList empty_shellpairs = {};
    const auto &shellpairs =
        use_shellpair_list ? m_shellpairs : empty_shellpairs;

    std::array<double, 3> origin{atoms[atom_index].x, atoms[atom_index].y, atoms[atom_index].z};
    m_env.set_rinv_origin(origin);
    MatTriple result;

    if (spherical) {
	result = one_electron_operator_grad_kernel<Op::rinv, Sph>(
	    m_aobasis, m_env, shellpairs);
    } else {
	result = one_electron_operator_grad_kernel<Op::rinv, Cart>(
	    m_aobasis, m_env, shellpairs);
    }
    m_env.set_rinv_origin({0.0, 0.0, 0.0});
    return result;
}

// helper functions to make chained calls to std::max more clear
inline double max_of(double p, double q, double r) {
    return std::max(p, std::max(q, r));
}

inline double max_of(double p, double q, double r, double s) {
    return std::max(p, std::max(q, std::max(r, s)));
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center_grad(Lambda &f, IntEnv &env,
			       const AOBasis &basis, const ShellPairList &shellpairs,
			       const Mat &Dnorm = Mat(), const Mat &Schwarz = Mat(),
			       double precision = 1e-12, int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<4>;
    occ::timing::start(occ::timing::category::ints4c2e);
    auto nthreads = occ::parallel::get_num_threads();
    occ::qm::cint::Optimizer opt(env, Op::coulomb, 4, 1);
    auto buffer = std::make_unique<double[]>(env.buffer_size_2e(1));
    std::array<int, 4> shell_idx;
    std::array<int, 4> bf;

    const auto &first_bf = basis.first_bf();

    const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
    // <pq|rs>
    for (size_t p = 0, pqrs = 0; p < basis.size(); p++) {
        bf[0] = first_bf[p];
        for (const auto q : shellpairs[p]) {
            bf[1] = first_bf[q];
            const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;
            for (size_t r = 0; r < basis.size(); r++) {
                bf[2] = first_bf[r];
		for(const auto s: shellpairs[r]) {
                    if (pqrs++ % nthreads != thread_id)
                        continue;
		    const double norm_pqr =
			do_schwarz_screen
			    ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq)
			    : 0.0;
                    bf[3] = first_bf[s];
		    const double norm_pqrs =
                        do_schwarz_screen ? max_of(Dnorm(p, s), Dnorm(q, s),
                                                   Dnorm(r, s), norm_pqr)
                                          : 0.0;
                    if (do_schwarz_screen &&
                        norm_pqrs * Schwarz(p, q) * Schwarz(r, s) < precision)
                        continue;
                    shell_idx = {static_cast<int>(p), static_cast<int>(q),
                                 static_cast<int>(r), static_cast<int>(s)};
		    // PQ | RS and PQ | SR

		    {
			Result args{thread_id, shell_idx, bf,
				    env.four_center_helper_grad<Op::coulomb, kind>(
					    shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr
				    ),
				    buffer.get()};
			if (args.dims[0] > -1) {
			    f(args);
			}
		    }
		    // QP | RS and QP | SR
		    if(p != q) {
			std::array<int, 4> shell_idx2 = {
			    static_cast<int>(q), static_cast<int>(p),
			    static_cast<int>(r), static_cast<int>(s)
			};
			std::array<int, 4> bf2{
			    bf[1], bf[0], bf[2], bf[3]
			};
			Result args2{thread_id, shell_idx2, bf2,
				    env.four_center_helper_grad<Op::coulomb, kind>(
					    shell_idx2, opt.optimizer_ptr(), buffer.get(), nullptr
				    ),
				    buffer.get()};
			if (args2.dims[0] > -1) {
			    f(args2);
			}
		    }
                }
            }
        }
    }
    occ::timing::stop(occ::timing::category::ints4c2e);
}


template<class Func>
inline void four_center_inner_loop(Func &store, 
			           const IntegralEngine::IntegralResult<4> &args,
			           const Mat &D,
			           MatTriple &dest) {
        auto scale = (args.shell[2] == args.shell[3]) ? 1 : 2;

        const auto num_elements = args.dims[0] * args.dims[1] * args.dims[2] * args.dims[3];
	
        for (auto f3 = 0, f0123 = 0; f3 < args.dims[3]; ++f3) {
            const auto bf3 = f3 + args.bf[3];
            for (auto f2 = 0; f2 < args.dims[2]; ++f2) {
                const auto bf2 = f2 + args.bf[2];
                for (auto f1 = 0; f1 < args.dims[1]; ++f1) {
                    const auto bf1 = f1 + args.bf[1];
                    for (auto f0 = 0; f0 < args.dims[0]; ++f0, ++f0123) {
                        const auto bf0 = f0 + args.bf[0];
			// x
			store(D, dest.x, bf0, bf1, bf2, bf3,
			      args.buffer[f0123] * scale);
			// y 
			store(D, dest.y, bf0, bf1, bf2, bf3,
			      args.buffer[f0123 + num_elements] * scale);
			// z
			    
		        store(D, dest.z, bf0, bf1, bf2, bf3,
			      args.buffer[f0123 + 2 * num_elements] * scale);
                    }
                }
            }
        }
}

template <SpinorbitalKind sk>
std::vector<MatTriple> initialize_result_matrices(size_t nbf, size_t nthreads) {
    auto [rows, cols] = occ::qm::matrix_dimensions<sk>(nbf);
    std::vector<MatTriple> results(nthreads);
    for(auto &r: results) {
	r.x = Mat::Zero(rows, cols);
	r.y = Mat::Zero(rows, cols);
	r.z = Mat::Zero(rows, cols);
    }
    return results;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatTriple coulomb_kernel_grad(cint::IntegralEnvironment &env, const AOBasis &basis,
		        const ShellPairList &shellpairs, const MolecularOrbitals &mo,
		        double precision = 1e-12, const Mat &Schwarz = Mat()) {
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;

    const auto nbf = basis.nbf();

    auto results = initialize_result_matrices<sk>(nbf, nthreads);
    Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);


    const auto &D = mo.D;
    auto f = [&D, &results](const Result &args) {
        auto &dest = results[args.thread];
	four_center_inner_loop(impl::delegate_j_grad<sk>, args, D, dest);
    };
    auto lambda = [&](int thread_id) {
        evaluate_four_center_grad<op, kind>(f, env, basis, shellpairs, Dnorm,
					    Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);


    for (size_t i = 1; i < nthreads; i++) {
	results[0].x.noalias() += results[i].x;
	results[0].y.noalias() += results[i].y;
	results[0].z.noalias() += results[i].z;
    }

    results[0].scale_by(-2.0);
    return results[0];
}



MatTriple IntegralEngine::coulomb_grad(SpinorbitalKind sk, const MolecularOrbitals &mo,
				       const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return coulomb_kernel_grad<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return coulomb_kernel_grad<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_kernel_grad<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return coulomb_kernel_grad<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_kernel_grad<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return coulomb_kernel_grad<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }
    }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
JKTriple coulomb_exchange_kernel_grad(IntEnv &env, 
				      const AOBasis &basis,
				      const ShellPairList &shellpairs,
				      const MolecularOrbitals &mo,
		                      double precision = 1e-12,
			              const Mat &Schwarz = Mat()) {
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;

    const auto nbf = basis.nbf();

    auto jmats = initialize_result_matrices<sk>(nbf, nthreads);
    auto kmats = initialize_result_matrices<sk>(nbf, nthreads);
    Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

    const auto &D = mo.D;
    auto f = [&D, &jmats, &kmats](const Result &args) {
        auto &dest_j = jmats[args.thread];
	four_center_inner_loop(impl::delegate_j_grad<sk>, args, D, dest_j);

        auto &dest_k = kmats[args.thread];
	four_center_inner_loop(impl::delegate_k_grad<sk>, args, D, dest_k);
    };
    auto lambda = [&](int thread_id) {
        evaluate_four_center_grad<op, kind>(f, env, basis, shellpairs, Dnorm,
					    Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);


    for (size_t i = 1; i < nthreads; i++) {
	jmats[0].x.noalias() += jmats[i].x;
	jmats[0].y.noalias() += jmats[i].y;
	jmats[0].z.noalias() += jmats[i].z;
	kmats[0].x.noalias() += kmats[i].x;
	kmats[0].y.noalias() += kmats[i].y;
	kmats[0].z.noalias() += kmats[i].z;
    }

    jmats[0].scale_by(-2.0);
    kmats[0].scale_by(0.5);
    return {jmats[0], kmats[0]};
}

JKTriple IntegralEngine::coulomb_exchange_grad(
	SpinorbitalKind sk, const MolecularOrbitals &mo,
	const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return coulomb_exchange_kernel_grad<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return coulomb_exchange_kernel_grad<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_exchange_kernel_grad<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return coulomb_exchange_kernel_grad<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_exchange_kernel_grad<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return coulomb_exchange_kernel_grad<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }
    }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatTriple fock_kernel_grad(IntEnv &env, 
			   const AOBasis &basis,
			   const ShellPairList &shellpairs,
			   const MolecularOrbitals &mo,
		           double precision = 1e-12,
			   const Mat &Schwarz = Mat()) {
   auto [J, K] = coulomb_exchange_kernel_grad<sk, kind>(env, basis, shellpairs, mo, precision, Schwarz);
   J.x -= K.x;
   J.y -= K.y;
   J.z -= K.z;
   return J;
}

MatTriple IntegralEngine::fock_operator_grad(SpinorbitalKind sk, const MolecularOrbitals &mo,
					     const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return fock_kernel_grad<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return fock_kernel_grad<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return fock_kernel_grad<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return fock_kernel_grad<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return fock_kernel_grad<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                               m_precision, Schwarz);
        } else {
            return fock_kernel_grad<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                                m_precision, Schwarz);
        }
    }
}

}
