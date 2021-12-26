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


DFFockEngine::Policy policy_choice(const DFFockEngine &df) {
    if(df.memory_limit >= df.integral_storage_max_size()) {
        return DFFockEngine::Policy::Stored;
    }
    return DFFockEngine::Policy::Direct;
}

Mat DFFockEngine::compute_J(const MolecularOrbitals &mo, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if(policy == Policy::Direct) {
        return compute_J_direct(mo);
    }
    else {
        return compute_J_stored(mo);
    }
}

Mat DFFockEngine::compute_K(const MolecularOrbitals &mo, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if(policy == Policy::Direct) {
        return compute_K_direct(mo);
    }
    else {
        return compute_K_stored(mo);
    }
}

Mat DFFockEngine::compute_fock(const MolecularOrbitals &mo, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if(policy == Policy::Direct) {
        return compute_fock_direct(mo);
    }
    else {
        return compute_fock_stored(mo);
    }
}

std::pair<Mat, Mat> DFFockEngine::compute_JK(const MolecularOrbitals &mo, Policy policy) {
    policy = (policy == Policy::Choose) ? policy_choice(*this) : policy;
    if(policy == Policy::Direct) {
        return compute_JK_direct(mo);
    }
    else {
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

    three_center_integral_helper(lambda);
    m_have_integrals = true;
  }
}



Mat DFFockEngine::compute_J_direct(const MolecularOrbitals &mo) {

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
            Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
            g(i) += (mo.D.block(bf2, bf3, n2, n3).array() * buf_mat.array()).sum();
	    if(bf2 != bf3) 
		g(i) += (mo.D.block(bf3, bf2, n3, n2).array() * buf_mat.transpose().array()).sum();
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
            Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
            J.block(bf2, bf3, n2, n3) += d(i) * buf_mat;
	    if(bf2 != bf3) J.block(bf3, bf2, n3, n2) += d(i) * buf_mat.transpose();
            offset += n2 * n3;
        }
    };

    three_center_integral_helper(Jlambda);

    for (int i = 1; i < nthreads; i++)
        JJ[0] += JJ[i];
    return (JJ[0] + JJ[0].transpose());
}

Mat DFFockEngine::compute_K_direct(const MolecularOrbitals &mo) {
    using occ::parallel::nthreads;
    size_t nmo = mo.Cocc.cols();
    Mat K = Mat::Zero(nbf, nbf);
    std::vector<Mat> iuP(nmo * nthreads);
    Mat B(nbf, ndf);
    for(auto &x : iuP) x = Mat::Zero(nbf, ndf);

    auto lambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                       size_t n2, size_t bf3, size_t n3, const double *buf) {
	for(size_t i = 0; i < mo.Cocc.cols(); i++) {
	    auto &iuPx = iuP[nmo * thread_id + i];
	    auto c2 = mo.Cocc.block(bf2, i, n2, 1);
	    auto c3 = mo.Cocc.block(bf3, i, n3, 1);

	    size_t offset = 0;
	    for(size_t r = bf1; r < bf1 + n1; r++) {
		Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
		iuPx.block(bf2, r, n2, 1) += buf_mat * c3;
		if(bf2 != bf3)
		    iuPx.block(bf3, r, n3, 1) += (buf_mat.transpose() * c2);
		offset += n2 * n3;
	    }
	}
    };

    three_center_integral_helper(lambda);

    for(size_t i = nmo; i < nmo * nthreads; i++) {
	iuP[i % nmo] += iuP[i];
    }

    for(size_t i = 0; i < nmo; i++) {
	B = iuP[i] * Vinv_sqrt;
	K.noalias() += B * B.transpose();
    }

    return K;
}

std::pair<Mat,Mat> DFFockEngine::compute_JK_direct(const MolecularOrbitals &mo) {

    using occ::parallel::nthreads;
    size_t nmo = mo.Cocc.cols();

    std::vector<Vec> gg(nthreads);
    std::vector<Mat> JJ(nthreads);
    std::vector<Mat> KK(nthreads);

    std::vector<Mat> iuP(nmo);

    for(auto &x : iuP) x = Mat::Zero(nbf, ndf);

    for (int i = 0; i < nthreads; i++) {
        gg[i] = Vec::Zero(ndf);
        JJ[i] = Mat::Zero(nbf, nbf);
	KK[i] = Mat::Zero(nbf, nbf);
    }

    auto lambda1 = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                       size_t n2, size_t bf3, size_t n3, const double *buf) {
        auto &g = gg[thread_id];
        size_t offset = 0;

        for (size_t r = bf1; r < bf1 + n1; r++) {
            Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
            g(r) += (mo.D.block(bf2, bf3, n2, n3).array() * buf_mat.array()).sum();
	    if(bf2 != bf3) {
                g(r) += (mo.D.block(bf3, bf2, n3, n2).array() * buf_mat.transpose().array()).sum();
            }
            offset += n2 * n3;
        }

        for(size_t i = 0; i < mo.Cocc.cols(); i++) {
            auto &iuPx = iuP[i];
            auto c2 = mo.Cocc.block(bf2, i, n2, 1);
            auto c3 = mo.Cocc.block(bf3, i, n3, 1);

            size_t offset = 0;
	    Mat tmp1, tmp2;
	    // because we parallelize over bf1 i.e. r,
	    // there's no need for a mutex as only one thread writes each column
            for(size_t r = bf1; r < bf1 + n1; r++) {
                Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
                iuPx.block(bf2, r, n2, 1) += buf_mat * c3;
                if(bf2 != bf3) {
		    iuPx.block(bf3, r, n3, 1) += (buf_mat.transpose() * c2);
		}
                offset += n2 * n3;
            }
        }
    };

    three_center_integral_helper(lambda1);

    for (int i = 1; i < nthreads; i++)
        gg[0] += gg[i];

    auto klambda = [&](int thread_id) {
	Mat B(nbf, ndf);
	for(size_t i = 0; i < nmo; i++) {
	    if(i % nthreads != thread_id) continue;
	    B = iuP[i] * Vinv_sqrt;
	    KK[thread_id].noalias() += B * B.transpose();
	}
    };

    occ::parallel::parallel_do(klambda);


    occ::timing::start(occ::timing::category::la);
    Vec d = V_LLt.solve(gg[0]);
    occ::timing::stop(occ::timing::category::la);

    auto Jlambda = [&](int thread_id, size_t bf1, size_t n1, size_t bf2,
                       size_t n2, size_t bf3, size_t n3, const double *buf) {
        auto &J = JJ[thread_id];
        size_t offset = 0;
        for (size_t i = bf1; i < bf1 + n1; i++) {
            Eigen::Map<const MatRM> buf_mat(&buf[offset], n2, n3);
            J.block(bf2, bf3, n2, n3) += d(i) * buf_mat;
	    if(bf2 != bf3) J.block(bf3, bf2, n3, n2) += d(i) * buf_mat.transpose();
            offset += n2 * n3;
        }
    };

    three_center_integral_helper(Jlambda);

    for (int i = 1; i < nthreads; i++) {
        JJ[0] += JJ[i];
	KK[0] += KK[i];
    }

    return {JJ[0] + JJ[0].transpose(), KK[0]};
}

Mat DFFockEngine::compute_fock_direct(const MolecularOrbitals &mo) {
    Mat J, K;
    std::tie(J, K) = compute_JK_direct(mo);
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

std::pair<Mat, Mat> DFFockEngine::compute_JK_stored(const MolecularOrbitals &mo) {
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
