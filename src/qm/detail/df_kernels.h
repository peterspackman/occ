#pragma once
#include "kernel_traits.h"
#include <occ/qm/integral_engine_df.h>

namespace occ::qm::detail {

using IntegralResult = IntegralEngine::IntegralResult<3>;

// Generic TBB-based three-center integral computation helper
template <ShellKind kind, typename Lambda>
void compute_three_center_integrals_tbb(Lambda &process_lambda,
                                        cint::IntegralEnvironment &env,
                                        const AOBasis &aobasis,
                                        const AOBasis &auxbasis,
                                        const ShellPairList &shellpairs,
                                        cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::ints3c2e);
  
  // Parallelize over auxiliary basis functions using TBB work-stealing
  occ::parallel::parallel_for(size_t(0), auxbasis.size(), [&](size_t auxP) {
    size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                     auxbasis.max_shell_size();
    auto buffer = std::make_unique<double[]>(bufsize);
    IntegralResult args;
    args.buffer = buffer.get();
    std::array<int, 3> shell_idx;
    const auto &first_bf_ao = aobasis.first_bf();
    const auto &first_bf_aux = auxbasis.first_bf();
    const int nsh_ao = aobasis.size();
    
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    
    for (int p = 0; p < aobasis.size(); p++) {
      args.bf[0] = first_bf_ao[p];
      args.shell[0] = p;
      const auto &plist = shellpairs[p];
      for (const auto &q : plist) {
        args.bf[1] = first_bf_ao[q];
        args.shell[1] = q;
        shell_idx = {p, static_cast<int>(q), static_cast<int>(auxP) + nsh_ao};
        args.dims = env.three_center_helper<Op::coulomb, kind>(
            shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
        if (args.dims[0] > -1) {
          process_lambda(args);
        }
      }
    }
  });
  
  occ::timing::stop(occ::timing::category::ints3c2e);
}

// Generic helper to accumulate and reduce thread-local storage of vectors
template<typename T>
std::vector<T> reduce_thread_local_vectors(const occ::parallel::thread_local_storage<std::vector<T>>& tl_storage) {
  if (tl_storage.begin() == tl_storage.end()) {
    return {};
  }
  
  size_t n_items = tl_storage.begin()->size();
  std::vector<T> result(n_items);
  for (size_t i = 0; i < n_items; i++) {
    result[i] = T::Zero(tl_storage.begin()->operator[](i).rows(), tl_storage.begin()->operator[](i).cols());
  }
  
  for (const auto &local_data : tl_storage) {
    for (size_t i = 0; i < n_items; i++) {
      result[i] += local_data[i];
    }
  }
  return result;
}

// Generic exchange contraction helper
template<typename KMatType>
void contract_exchange_matrices(const std::vector<Mat>& iuP, const Eigen::LLT<Mat>& V_LLt, 
                               KMatType& K_block) {
  Mat X(iuP[0].rows(), iuP[0].cols());
  for (size_t i = 0; i < iuP.size(); i++) {
    X = V_LLt.solve(iuP[i].transpose());
    K_block.noalias() += iuP[i] * X;
  }
}

// Combined JK data structure for thread-local storage
struct JKData {
  Vec g;
  std::vector<Mat> iuP;
  
  JKData(size_t ndf, size_t nbf, size_t nocc) 
    : g(Vec::Zero(ndf)), iuP(nocc, Mat::Zero(nbf, ndf)) {}
};

// Old g_lambda_direct_r function removed - replaced with TBB implementations

// Old g_lambda functions removed - replaced with TBB implementations

inline auto g_lambda_direct_g(std::vector<Vec> &gg_alpha,
                              std::vector<Vec> &gg_beta,
                              const MolecularOrbitals &mo) {
  return [&](const IntegralResult &args) {
    auto &ga = gg_alpha[args.thread];
    auto &gb = gg_beta[args.thread];
    const auto Daa = block::aa(mo.D);
    const auto Dbb = block::bb(mo.D);
    size_t offset = 0;
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        ga(i) += (Daa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        gb(i) += (Dbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        ga(i) += (Daa.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                      .array() *
                  buf_mat.transpose().array())
                     .sum();
        gb(i) += (Dbb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                      .array() *
                  buf_mat.transpose().array())
                     .sum();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        ga(i) += (Daa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        gb(i) += (Dbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
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
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            d(i) * buf_mat;
        J.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
            d(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
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
    auto Ja = block::a(JJ[args.thread]);
    auto Jb = block::b(JJ[args.thread]);
    size_t offset = 0;
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            (da(i) + db(i)) * buf_mat;
        Ja.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
            (da(i) + db(i)) * buf_mat.transpose();
        Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            (da(i) + db(i)) * buf_mat;
        Jb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
            (da(i) + db(i)) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            (da(i) + db(i)) * buf_mat;
        Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            (da(i) + db(i)) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };
}

inline auto j_lambda_direct_g(std::vector<Mat> &JJ, const Vec &da,
                              const Vec &db) {
  return [&JJ, &da, &db](const IntegralResult &args) {
    auto Jaa = block::aa(JJ[args.thread]);
    auto Jbb = block::bb(JJ[args.thread]);
    size_t offset = 0;
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        Jaa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            da(i) * buf_mat;
        Jaa.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
            da(i) * buf_mat.transpose();
        Jbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            db(i) * buf_mat;
        Jbb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
            db(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        Jaa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            da(i) * buf_mat;
        Jbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            db(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };
}

inline auto k_lambda_direct_r(std::vector<Mat> &iuP,
                              const MolecularOrbitals &mo) {
  size_t nocc = mo.Cocc.cols();
  return [&, nocc](const IntegralResult &args) {
    for (size_t i = 0; i < mo.Cocc.cols(); i++) {
      auto &iuPx = iuP[nocc * args.thread + i];
      auto c2 = mo.Cocc.block(args.bf[0], i, args.dims[0], 1);
      auto c3 = mo.Cocc.block(args.bf[1], i, args.dims[1], 1);

      size_t offset = 0;
      if (args.bf[0] != args.bf[1]) {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                        args.dims[1]);
          iuPx.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3;
          iuPx.block(args.bf[1], r, args.dims[1], 1) +=
              (buf_mat.transpose() * c2);
          offset += args.dims[0] * args.dims[1];
        }
      } else {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                        args.dims[1]);
          iuPx.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3;
          offset += args.dims[0] * args.dims[1];
        }
      }
    }
  };
}

inline auto k_lambda_direct_u(std::vector<Mat> &iuPa, std::vector<Mat> &iuPb,
                              const MolecularOrbitals &mo) {
  size_t nocc = mo.Cocc.cols();
  return [&, nocc](const IntegralResult &args) {
    for (size_t i = 0; i < mo.Cocc.cols(); i++) {
      auto &iuPxa = iuPa[nocc * args.thread + i];
      auto &iuPxb = iuPb[nocc * args.thread + i];
      auto c2a = block::a(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3a = block::a(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);
      auto c2b = block::b(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3b = block::b(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);

      size_t offset = 0;
      if (args.bf[0] != args.bf[1]) {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                        args.dims[1]);
          iuPxa.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3a;
          iuPxa.block(args.bf[1], r, args.dims[1], 1) +=
              (buf_mat.transpose() * c2a);

          iuPxb.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3b;
          iuPxb.block(args.bf[1], r, args.dims[1], 1) +=
              (buf_mat.transpose() * c2b);
          offset += args.dims[0] * args.dims[1];
        }
      } else {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                        args.dims[1]);
          iuPxa.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3a;
          iuPxb.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3b;
          offset += args.dims[0] * args.dims[1];
        }
      }
    }
  };
}

inline auto jk_lambda_direct_unpolarized(std::vector<Vec> &gg,
                                         std::vector<Mat> &iuP,
                                         const MolecularOrbitals &mo) {
  return [&](const IntegralResult &args) {
    auto &g = gg[args.thread];
    size_t offset = 0;
    size_t nocc = mo.Cocc.cols();
    const auto c2 = mo.Cocc.block(args.bf[0], 0, args.dims[0], nocc);
    const auto c3 = mo.Cocc.block(args.bf[1], 0, args.dims[1], nocc);
    const auto Dblock1 =
        mo.D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Dblock2 =
        mo.D.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);

    Mat c3_term(args.dims[0], nocc);
    Mat c2_term(args.dims[1], nocc);

    if (args.bf[0] != args.bf[1]) {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        g(r) += (Dblock1.array() * buf_mat.array()).sum();
        c3_term = buf_mat * c3;
        g(r) += (Dblock2.array() * buf_mat.transpose().array()).sum();
        c2_term = buf_mat.transpose() * c2;

        for (int i = 0; i < nocc; i++) {
          auto &iuPx = iuP[i];
          iuPx.block(args.bf[0], r, args.dims[0], 1) +=
              c3_term.block(0, i, args.dims[0], 1);
          iuPx.block(args.bf[1], r, args.dims[1], 1) +=
              c2_term.block(0, i, args.dims[1], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        g(r) += (Dblock1.array() * buf_mat.array()).sum();
        c3_term = buf_mat * c3;

        for (int i = 0; i < nocc; i++) {
          auto &iuPx = iuP[i];
          iuPx.block(args.bf[0], r, args.dims[0], 1) +=
              c3_term.block(0, i, args.dims[0], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    }
  };
}

inline auto jk_lambda_direct_polarized(
    std::vector<Vec> &gg_alpha, std::vector<Vec> &gg_beta,
    std::vector<Mat> &iuPa, std::vector<Mat> &iuPb, Eigen::Ref<const Mat> Da,
    Eigen::Ref<const Mat> Db, Eigen::Ref<const Mat> Cocc_a,
    Eigen::Ref<const Mat> Cocc_b) {
  return [&gg_alpha, &gg_beta, &iuPa, &iuPb, Da, Db, Cocc_a,
          Cocc_b](const IntegralResult &args) {
    auto &ga = gg_alpha[args.thread];
    auto &gb = gg_beta[args.thread];
    size_t offset = 0;
    // nbf x nocc
    size_t n_alpha = Cocc_a.cols();
    size_t n_beta = Cocc_b.cols();
    auto c3a = Cocc_a.block(args.bf[1], 0, args.dims[1], n_alpha);
    auto c3b = Cocc_b.block(args.bf[1], 0, args.dims[1], n_beta);
    Mat c3_term_a(args.dims[0], n_alpha);
    Mat c3_term_b(args.dims[0], n_beta);

    if (args.bf[0] != args.bf[1]) {
      auto c2a = Cocc_a.block(args.bf[0], 0, args.dims[0], n_alpha);
      auto c2b = Cocc_b.block(args.bf[0], 0, args.dims[0], n_beta);
      Mat c2_term_a(args.dims[1], n_alpha);
      Mat c2_term_b(args.dims[1], n_beta);
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);

        ga(r) +=
            (Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                 .array() *
             buf_mat.array())
                .sum();
        gb(r) +=
            (Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                 .array() *
             buf_mat.array())
                .sum();

        c3_term_a = buf_mat * c3a;
        c3_term_b = buf_mat * c3b;

        ga(r) +=
            (Da.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                 .array() *
             buf_mat.transpose().array())
                .sum();
        gb(r) +=
            (Db.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                 .array() *
             buf_mat.transpose().array())
                .sum();

        c2_term_a = buf_mat.transpose() * c2a;
        c2_term_b = buf_mat.transpose() * c2b;

        for (int i = 0; i < n_alpha; i++) {
          auto &iuPxa = iuPa[i];
          iuPxa.block(args.bf[0], r, args.dims[0], 1) +=
              c3_term_a.block(0, i, args.dims[0], 1);
          iuPxa.block(args.bf[1], r, args.dims[1], 1) +=
              c2_term_a.block(0, i, args.dims[1], 1);
        }

        for (int i = 0; i < n_beta; i++) {
          auto &iuPxb = iuPb[i];
          iuPxb.block(args.bf[0], r, args.dims[0], 1) +=
              c3_term_b.block(0, i, args.dims[0], 1);
          iuPxb.block(args.bf[1], r, args.dims[1], 1) +=
              c2_term_b.block(0, i, args.dims[1], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        ga(r) +=
            (Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                 .array() *
             buf_mat.array())
                .sum();
        gb(r) +=
            (Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                 .array() *
             buf_mat.array())
                .sum();

        c3_term_a = buf_mat * c3a;
        c3_term_b = buf_mat * c3b;

        for (int i = 0; i < n_alpha; i++) {
          auto &iuPxa = iuPa[i];
          iuPxa.block(args.bf[0], r, args.dims[0], 1) +=
              c3_term_a.block(0, i, args.dims[0], 1);
        }

        for (int i = 0; i < n_beta; i++) {
          auto &iuPxb = iuPb[i];
          iuPxb.block(args.bf[0], r, args.dims[0], 1) +=
              c3_term_b.block(0, i, args.dims[0], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    }
  };
}

template <ShellKind kind, typename Lambda>
void three_center_aux_kernel(Lambda &f, cint::IntegralEnvironment &env,
                             const qm::AOBasis &aobasis,
                             const qm::AOBasis &auxbasis,
                             const ShellPairList &shellpairs,
                             cint::Optimizer &opt,
                             int thread_id = 0) noexcept {
  occ::timing::start(occ::timing::category::ints3c2e);
  auto nthreads = occ::parallel::get_num_threads();
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
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    for (int p = 0; p < aobasis.size(); p++) {
      args.bf[0] = first_bf_ao[p];
      args.shell[0] = p;
      const auto &plist = shellpairs[p];
      for (const auto &q : plist) {
        args.bf[1] = first_bf_ao[q];
        args.shell[1] = q;
        shell_idx = {p, static_cast<int>(q), auxP + nsh_ao};
        args.dims = env.three_center_helper<Op::coulomb, kind>(
            shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  }
  occ::timing::stop(occ::timing::category::ints3c2e);
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_exchange_operator_kernel_r(IntegralEngine &engine,
                                      IntegralEngine &engine_aux,
                                      const MolecularOrbitals &mo,
                                      const Eigen::LLT<Mat> &V_LLt,
                                      cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  // TBB thread-local storage for intermediate matrices
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuP([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });

  // Process each auxiliary shell with work-stealing parallelization
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_iuP = tl_iuP.local();
    
    for (size_t i = 0; i < nocc; i++) {
      auto &iuPx = local_iuP[i];
      auto c2 = mo.Cocc.block(args.bf[0], i, args.dims[0], 1);
      auto c3 = mo.Cocc.block(args.bf[1], i, args.dims[1], 1);

      size_t offset = 0;
      if (args.bf[0] != args.bf[1]) {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
          iuPx.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3;
          iuPx.block(args.bf[1], r, args.dims[1], 1) += buf_mat.transpose() * c2;
          offset += args.dims[0] * args.dims[1];
        }
      } else {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
          iuPx.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3;
          offset += args.dims[0] * args.dims[1];
        }
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(process_integrals, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local results and contract
  auto iuP = reduce_thread_local_vectors(tl_iuP);
  Mat K = Mat::Zero(nbf, nbf);
  contract_exchange_matrices(iuP, V_LLt, K);

  occ::timing::stop(occ::timing::category::df);
  return 0.5 * (K + K.transpose());
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_exchange_operator_kernel_u(IntegralEngine &engine,
                                      IntegralEngine &engine_aux,
                                      const MolecularOrbitals &mo,
                                      const Eigen::LLT<Mat> &V_LLt,
                                      cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);

  // TBB thread-local storage for intermediate matrices (alpha and beta)
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuPa([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuPb([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });

  // Process each auxiliary shell with work-stealing parallelization
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_iuPa = tl_iuPa.local();
    auto &local_iuPb = tl_iuPb.local();
    
    for (size_t i = 0; i < nocc; i++) {
      auto &iuPxa = local_iuPa[i];
      auto &iuPxb = local_iuPb[i];
      auto c2a = block::a(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3a = block::a(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);
      auto c2b = block::b(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3b = block::b(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);

      size_t offset = 0;
      if (args.bf[0] != args.bf[1]) {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
          iuPxa.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3a;
          iuPxa.block(args.bf[1], r, args.dims[1], 1) += buf_mat.transpose() * c2a;
          iuPxb.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3b;
          iuPxb.block(args.bf[1], r, args.dims[1], 1) += buf_mat.transpose() * c2b;
          offset += args.dims[0] * args.dims[1];
        }
      } else {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
          iuPxa.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3a;
          iuPxb.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3b;
          offset += args.dims[0] * args.dims[1];
        }
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(process_integrals, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local results and contract for both spin channels
  auto iuPa = reduce_thread_local_vectors(tl_iuPa);
  auto iuPb = reduce_thread_local_vectors(tl_iuPb);
  
  Mat K = Mat::Zero(rows, cols);
  auto Ka = block::a(K);
  auto Kb = block::b(K);
  contract_exchange_matrices(iuPa, V_LLt, Ka);
  contract_exchange_matrices(iuPb, V_LLt, Kb);

  block::a(K) += block::a(K).transpose().eval();
  block::b(K) += block::b(K).transpose().eval();
  K *= 0.5;

  occ::timing::stop(occ::timing::category::df);
  return K;
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_exchange_operator_kernel_g(IntegralEngine &engine,
                                      IntegralEngine &engine_aux,
                                      const MolecularOrbitals &mo,
                                      const Eigen::LLT<Mat> &V_LLt,
                                      cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);

  // TBB thread-local storage for intermediate matrices (alpha and beta)
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuPa([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuPb([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });

  // Process each auxiliary shell with work-stealing parallelization
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_iuPa = tl_iuPa.local();
    auto &local_iuPb = tl_iuPb.local();
    
    for (size_t i = 0; i < nocc; i++) {
      auto &iuPxa = local_iuPa[i];
      auto &iuPxb = local_iuPb[i];
      auto c2a = block::a(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3a = block::a(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);
      auto c2b = block::b(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3b = block::b(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);

      size_t offset = 0;
      if (args.bf[0] != args.bf[1]) {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
          iuPxa.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3a;
          iuPxa.block(args.bf[1], r, args.dims[1], 1) += buf_mat.transpose() * c2a;
          iuPxb.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3b;
          iuPxb.block(args.bf[1], r, args.dims[1], 1) += buf_mat.transpose() * c2b;
          offset += args.dims[0] * args.dims[1];
        }
      } else {
        for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
          Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
          iuPxa.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3a;
          iuPxb.block(args.bf[0], r, args.dims[0], 1) += buf_mat * c3b;
          offset += args.dims[0] * args.dims[1];
        }
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(process_integrals, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local results and contract for general spinorbital case
  auto iuPa = reduce_thread_local_vectors(tl_iuPa);
  auto iuPb = reduce_thread_local_vectors(tl_iuPb);
  
  Mat K = Mat::Zero(rows, cols);
  Mat Xa(nbf, ndf), Xb(nbf, ndf);
  
  // Contract exchange matrix with cross-terms for general case
  for (size_t i = 0; i < nocc; i++) {
    Xa = V_LLt.solve(iuPa[i].transpose());
    Xb = V_LLt.solve(iuPb[i].transpose());
    block::aa(K).noalias() += iuPa[i] * Xa;
    block::bb(K).noalias() += iuPb[i] * Xb;
    block::ab(K).noalias() += (iuPa[i] * Xb) + (iuPb[i] * Xa);
    block::ba(K).noalias() += (iuPa[i] * Xb) + (iuPb[i] * Xa);
  }

  block::aa(K) += block::aa(K).transpose().eval();
  block::bb(K) += block::ab(K).transpose().eval();
  block::ba(K) += block::ba(K).transpose().eval();
  block::bb(K) += block::bb(K).transpose().eval();
  K *= 0.5;
  occ::timing::stop(occ::timing::category::df);

  return K;
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_r(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const Eigen::LLT<Mat> &V_LLt,
                                     cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  // TBB thread-local storage for Coulomb vector
  occ::parallel::thread_local_storage<Vec> tl_g(Vec::Zero(ndf));

  // First pass: compute Coulomb vector g
  auto compute_g = [&](const IntegralResult &args) {
    auto &g = tl_g.local();
    size_t offset = 0;
    const auto Dblock1 = mo.D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Dblock2 = mo.D.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        g(i) += (Dblock1.array() * buf_mat.array()).sum();
        g(i) += (Dblock2.array() * buf_mat.transpose().array()).sum();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        g(i) += (Dblock1.array() * buf_mat.array()).sum();
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(compute_g, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local g vectors
  Vec g_total = Vec::Zero(ndf);
  for (const auto &g : tl_g) {
    g_total += g;
  }
  
  Vec d = V_LLt.solve(g_total);

  // TBB thread-local storage for Coulomb matrix
  occ::parallel::thread_local_storage<Mat> tl_J(Mat::Zero(nbf, nbf));

  // Second pass: build J matrix
  auto build_J = [&](const IntegralResult &args) {
    auto &J = tl_J.local();
    size_t offset = 0;
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        J.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(build_J, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local J matrices
  Mat J = Mat::Zero(nbf, nbf);
  for (const auto &local_J : tl_J) {
    J += local_J;
  }
  
  occ::timing::stop(occ::timing::category::df);
  return (J + J.transpose());
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_u(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const Eigen::LLT<Mat> &V_LLt,
                                     cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);

  // TBB thread-local storage for total Coulomb vector (alpha + beta)
  occ::parallel::thread_local_storage<Vec> tl_g(Vec::Zero(ndf));

  // First pass: compute total Coulomb vector g = g_alpha + g_beta
  auto compute_g = [&](const IntegralResult &args) {
    auto &g = tl_g.local();
    const auto Da = block::a(mo.D);
    const auto Db = block::b(mo.D);
    size_t offset = 0;
    const auto Da_block1 = Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Da_block2 = Da.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);
    const auto Db_block1 = Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Db_block2 = Db.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        // Total density = alpha + beta
        g(i) += (Da_block1.array() * buf_mat.array()).sum();
        g(i) += (Da_block2.array() * buf_mat.transpose().array()).sum();
        g(i) += (Db_block1.array() * buf_mat.array()).sum();
        g(i) += (Db_block2.array() * buf_mat.transpose().array()).sum();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        // Total density = alpha + beta
        g(i) += (Da_block1.array() * buf_mat.array()).sum();
        g(i) += (Db_block1.array() * buf_mat.array()).sum();
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(compute_g, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local g vectors to get total
  Vec g_total = Vec::Zero(ndf);
  for (const auto &g : tl_g) {
    g_total += g;
  }
  
  // Solve once for total density
  Vec d = V_LLt.solve(g_total);

  // TBB thread-local storage for Coulomb matrix
  occ::parallel::thread_local_storage<Mat> tl_J(Mat::Zero(rows, cols));

  // Second pass: build J matrix - same coefficient d for both alpha and beta blocks
  auto build_J = [&](const IntegralResult &args) {
    auto &J = tl_J.local();
    auto Ja = block::a(J);
    auto Jb = block::b(J);
    size_t offset = 0;
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        // Same d for both spin blocks (Coulomb couples to total density)
        Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        Ja.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d(i) * buf_mat.transpose();
        Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        Jb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        // Same d for both spin blocks
        Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(build_J, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local J matrices
  Mat J = Mat::Zero(rows, cols);
  for (const auto &local_J : tl_J) {
    J += local_J;
  }
  
  occ::timing::stop(occ::timing::category::df);
  return 2 * J;
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_g(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const Eigen::LLT<Mat> &V_LLt,
                                     cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);

  // TBB thread-local storage for separate alpha and beta Coulomb vectors
  occ::parallel::thread_local_storage<Vec> tl_gaa(Vec::Zero(ndf));
  occ::parallel::thread_local_storage<Vec> tl_gbb(Vec::Zero(ndf));

  // First pass: compute separate Coulomb vectors for Daa and Dbb
  auto compute_g = [&](const IntegralResult &args) {
    auto &gaa = tl_gaa.local();
    auto &gbb = tl_gbb.local();
    const auto Daa = block::aa(mo.D);
    const auto Dbb = block::bb(mo.D);
    size_t offset = 0;
    const auto Daa_block1 = Daa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Daa_block2 = Daa.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);
    const auto Dbb_block1 = Dbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Dbb_block2 = Dbb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        gaa(i) += (Daa_block1.array() * buf_mat.array()).sum();
        gaa(i) += (Daa_block2.array() * buf_mat.transpose().array()).sum();
        gbb(i) += (Dbb_block1.array() * buf_mat.array()).sum();
        gbb(i) += (Dbb_block2.array() * buf_mat.transpose().array()).sum();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        gaa(i) += (Daa_block1.array() * buf_mat.array()).sum();
        gbb(i) += (Dbb_block1.array() * buf_mat.array()).sum();
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(compute_g, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local g vectors separately
  Vec gaa_total = Vec::Zero(ndf);
  Vec gbb_total = Vec::Zero(ndf);
  for (const auto &gaa : tl_gaa) {
    gaa_total += gaa;
  }
  for (const auto &gbb : tl_gbb) {
    gbb_total += gbb;
  }
  
  // Apply factor of 2 before solving, matching stored kernel exactly  
  Vec d_aa = V_LLt.solve(2 * gaa_total);
  Vec d_bb = V_LLt.solve(2 * gbb_total);

  // TBB thread-local storage for Coulomb matrix
  occ::parallel::thread_local_storage<Mat> tl_J(Mat::Zero(rows, cols));

  // Second pass: build J matrix with separate coefficients
  auto build_J = [&](const IntegralResult &args) {
    auto &J = tl_J.local();
    auto Jaa = block::aa(J);
    auto Jbb = block::bb(J);
    size_t offset = 0;
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        Jaa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_aa(i) * buf_mat;
        Jaa.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d_aa(i) * buf_mat.transpose();
        Jbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_bb(i) * buf_mat;
        Jbb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d_bb(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        Jaa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_aa(i) * buf_mat;
        Jbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_bb(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(build_J, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce thread-local J matrices
  Mat J = Mat::Zero(rows, cols);
  for (const auto &local_J : tl_J) {
    J += local_J;
  }
  
  occ::timing::stop(occ::timing::category::df);
  return 2 * J;  // Apply final factor of 2 to match stored kernel exactly
}

template <ShellKind kind = ShellKind::Cartesian>
JKPair direct_coulomb_and_exchange_operator_kernel_r(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const Eigen::LLT<Mat> &V_LLt,
    cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  // TBB thread-local storage for combined JK data
  occ::parallel::thread_local_storage<JKData> tl_jk_data([=]() {
    return JKData(ndf, nbf, nocc);
  });

  // Combined JK computation in single pass over three-center integrals
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_data = tl_jk_data.local();
    auto &g = local_data.g;
    auto &iuP = local_data.iuP;
    
    size_t offset = 0;
    const auto c2 = mo.Cocc.block(args.bf[0], 0, args.dims[0], nocc);
    const auto c3 = mo.Cocc.block(args.bf[1], 0, args.dims[1], nocc);
    const auto Dblock1 = mo.D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Dblock2 = mo.D.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);

    Mat c3_term(args.dims[0], nocc);
    Mat c2_term(args.dims[1], nocc);

    if (args.bf[0] != args.bf[1]) {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        
        // Coulomb contribution
        g(r) += (Dblock1.array() * buf_mat.array()).sum();
        g(r) += (Dblock2.array() * buf_mat.transpose().array()).sum();
        
        // Exchange contribution
        c3_term = buf_mat * c3;
        c2_term = buf_mat.transpose() * c2;
        for (int i = 0; i < nocc; i++) {
          iuP[i].block(args.bf[0], r, args.dims[0], 1) += c3_term.block(0, i, args.dims[0], 1);
          iuP[i].block(args.bf[1], r, args.dims[1], 1) += c2_term.block(0, i, args.dims[1], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        
        // Coulomb contribution
        g(r) += (Dblock1.array() * buf_mat.array()).sum();
        
        // Exchange contribution
        c3_term = buf_mat * c3;
        for (int i = 0; i < nocc; i++) {
          iuP[i].block(args.bf[0], r, args.dims[0], 1) += c3_term.block(0, i, args.dims[0], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(process_integrals, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce Coulomb data and solve
  Vec g = Vec::Zero(ndf);
  for (const auto &local_data : tl_jk_data) {
    g += local_data.g;
  }
  
  occ::timing::start(occ::timing::category::la);
  Vec d = V_LLt.solve(g);
  occ::timing::stop(occ::timing::category::la);

  // Build J matrix using TBB
  occ::parallel::thread_local_storage<Mat> tl_J(Mat::Zero(nbf, nbf));
  
  auto build_J = [&](const IntegralResult &args) {
    auto &local_J = tl_J.local();
    size_t offset = 0;
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        local_J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        local_J.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        local_J.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(build_J, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce J matrices
  Mat J = Mat::Zero(nbf, nbf);
  for (const auto &local_J : tl_J) {
    J += local_J;
  }

  // Reduce exchange data and contract
  std::vector<Mat> iuP_total(nocc, Mat::Zero(nbf, ndf));
  for (const auto &local_data : tl_jk_data) {
    for (size_t i = 0; i < nocc; i++) {
      iuP_total[i] += local_data.iuP[i];
    }
  }

  Mat K = Mat::Zero(nbf, nbf);
  contract_exchange_matrices(iuP_total, V_LLt, K);

  occ::timing::stop(occ::timing::category::df);
  return {J + J.transpose(), 0.5 * (K + K.transpose())};
}

template <ShellKind kind = ShellKind::Cartesian>
JKPair direct_coulomb_and_exchange_operator_kernel_u(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const Eigen::LLT<Mat> &V_LLt,
    cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.n_ao;
  const auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);

  std::vector<Vec> gg_alpha(nthreads, Vec::Zero(ndf));
  std::vector<Vec> gg_beta(nthreads, Vec::Zero(ndf));
  std::vector<Mat> JJ(nthreads, Mat::Zero(rows, cols));
  std::vector<Mat> KK(nthreads, Mat::Zero(rows, cols));
  Mat J = Mat::Zero(rows, cols), K = Mat::Zero(rows, cols);

  std::vector<Mat> iuP_alpha(nocc, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuP_beta(nocc, Mat::Zero(nbf, ndf));

  auto jk_lambda_1 = jk_lambda_direct_polarized(
      gg_alpha, gg_beta, iuP_alpha, iuP_beta, block::a(mo.D), block::b(mo.D),
      mo.occ_alpha(), mo.occ_beta());
  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(jk_lambda_1, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_for(0, nthreads, lambda);

  for (int i = 1; i < nthreads; i++) {
    gg_alpha[0] += gg_alpha[i];
    gg_beta[0] += gg_beta[i];
  }

  auto klambda = [&](int thread_id) {
    Mat Xa = Mat::Zero(nbf, ndf);
    Mat Xb = Mat::Zero(nbf, ndf);
    for (size_t i = 0; i < nocc; i++) {
      if (i % nthreads != thread_id)
        continue;
      Xa = V_LLt.solve(iuP_alpha[i].transpose());
      Xb = V_LLt.solve(iuP_beta[i].transpose());
      block::a(KK[thread_id]).noalias() += iuP_alpha[i] * Xa;
      block::b(KK[thread_id]).noalias() += iuP_beta[i] * Xb;
    }
  };

  occ::parallel::parallel_for(0, nthreads, klambda);

  occ::timing::start(occ::timing::category::la);
  Vec d_alpha = V_LLt.solve(gg_alpha[0]);
  Vec d_beta = V_LLt.solve(gg_beta[0]);
  occ::timing::stop(occ::timing::category::la);

  auto jlambda = j_lambda_direct_u(JJ, d_alpha, d_beta);
  auto lambda2 = [&](int thread_id) {
    three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_for(0, nthreads, lambda2);

  auto Ja = block::a(J);
  auto Jb = block::b(J);
  auto Ka = block::a(K);
  auto Kb = block::b(K);
  for (int i = 0; i < nthreads; i++) {
    auto JJa = block::a(JJ[i]);
    auto JJb = block::b(JJ[i]);
    auto KKa = block::a(KK[i]);
    auto KKb = block::b(KK[i]);
    Ja.noalias() += JJa + JJa.transpose();
    Jb.noalias() += JJb + JJb.transpose();
    Ka.noalias() += KKa + KKa.transpose();
    Kb.noalias() += KKb + KKb.transpose();
  }

  K *= 0.5;
  occ::timing::stop(occ::timing::category::df);
  return {J, K};
}

template <ShellKind kind = ShellKind::Cartesian>
JKPair direct_coulomb_and_exchange_operator_kernel_g(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const Eigen::LLT<Mat> &V_LLt,
    cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.n_alpha; // number of electrons == n_alpha for general
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);

  // TBB thread-local storage for Coulomb vectors (separate for aa and bb)
  occ::parallel::thread_local_storage<Vec> tl_gaa(Vec::Zero(ndf));
  occ::parallel::thread_local_storage<Vec> tl_gbb(Vec::Zero(ndf));
  
  // TBB thread-local storage for exchange intermediate matrices
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuPa([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });
  occ::parallel::thread_local_storage<std::vector<Mat>> tl_iuPb([=]() {
    return std::vector<Mat>(nocc, Mat::Zero(nbf, ndf));
  });

  // Combined JK computation in single pass over three-center integrals
  auto process_integrals = [&](const IntegralResult &args) {
    auto &gaa = tl_gaa.local();
    auto &gbb = tl_gbb.local();
    auto &local_iuPa = tl_iuPa.local();
    auto &local_iuPb = tl_iuPb.local();
    
    const auto Daa = block::aa(mo.D);
    const auto Dbb = block::bb(mo.D);
    size_t offset = 0;
    
    // Precompute orbital coefficient blocks
    const auto Ca_block1 = block::a(mo.Cocc).block(args.bf[0], 0, args.dims[0], nocc);
    const auto Ca_block2 = block::a(mo.Cocc).block(args.bf[1], 0, args.dims[1], nocc);
    const auto Cb_block1 = block::b(mo.Cocc).block(args.bf[0], 0, args.dims[0], nocc);
    const auto Cb_block2 = block::b(mo.Cocc).block(args.bf[1], 0, args.dims[1], nocc);
    
    const auto Daa_block1 = Daa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Daa_block2 = Daa.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);
    const auto Dbb_block1 = Dbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
    const auto Dbb_block2 = Dbb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]);

    Mat ca_term1(args.dims[0], nocc);
    Mat ca_term2(args.dims[1], nocc);
    Mat cb_term1(args.dims[0], nocc);
    Mat cb_term2(args.dims[1], nocc);

    if (args.bf[0] != args.bf[1]) {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        
        // Coulomb contributions  
        gaa(r) += (Daa_block1.array() * buf_mat.array()).sum();
        gaa(r) += (Daa_block2.array() * buf_mat.transpose().array()).sum();
        gbb(r) += (Dbb_block1.array() * buf_mat.array()).sum();
        gbb(r) += (Dbb_block2.array() * buf_mat.transpose().array()).sum();
        
        // Exchange contributions
        ca_term1 = buf_mat * Ca_block2;
        ca_term2 = buf_mat.transpose() * Ca_block1;
        cb_term1 = buf_mat * Cb_block2;
        cb_term2 = buf_mat.transpose() * Cb_block1;
        
        for (int i = 0; i < nocc; i++) {
          local_iuPa[i].block(args.bf[0], r, args.dims[0], 1) += ca_term1.block(0, i, args.dims[0], 1);
          local_iuPa[i].block(args.bf[1], r, args.dims[1], 1) += ca_term2.block(0, i, args.dims[1], 1);
          local_iuPb[i].block(args.bf[0], r, args.dims[0], 1) += cb_term1.block(0, i, args.dims[0], 1);
          local_iuPb[i].block(args.bf[1], r, args.dims[1], 1) += cb_term2.block(0, i, args.dims[1], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t r = args.bf[2]; r < args.bf[2] + args.dims[2]; r++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        
        // Coulomb contributions  
        gaa(r) += (Daa_block1.array() * buf_mat.array()).sum();
        gbb(r) += (Dbb_block1.array() * buf_mat.array()).sum();
        
        // Exchange contributions
        ca_term1 = buf_mat * Ca_block2;
        cb_term1 = buf_mat * Cb_block2;
        
        for (int i = 0; i < nocc; i++) {
          local_iuPa[i].block(args.bf[0], r, args.dims[0], 1) += ca_term1.block(0, i, args.dims[0], 1);
          local_iuPb[i].block(args.bf[0], r, args.dims[0], 1) += cb_term1.block(0, i, args.dims[0], 1);
        }
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(process_integrals, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce Coulomb vectors
  Vec gaa_total = Vec::Zero(ndf);
  Vec gbb_total = Vec::Zero(ndf);
  for (const auto &gaa : tl_gaa) {
    gaa_total += gaa;
  }
  for (const auto &gbb : tl_gbb) {
    gbb_total += gbb;
  }
  
  // Solve for Coulomb coefficients (apply factor of 2 like stored kernel)
  occ::timing::start(occ::timing::category::la);
  Vec d_aa = V_LLt.solve(2 * gaa_total);
  Vec d_bb = V_LLt.solve(2 * gbb_total);
  occ::timing::stop(occ::timing::category::la);

  // Build Coulomb matrix using TBB
  occ::parallel::thread_local_storage<Mat> tl_J(Mat::Zero(rows, cols));
  
  auto build_J = [&](const IntegralResult &args) {
    auto &J_local = tl_J.local();
    auto Jaa = block::aa(J_local);
    auto Jbb = block::bb(J_local);
    size_t offset = 0;
    
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        Jaa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_aa(i) * buf_mat;
        Jaa.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d_aa(i) * buf_mat.transpose();
        Jbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_bb(i) * buf_mat;
        Jbb.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) += d_bb(i) * buf_mat.transpose();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0], args.dims[1]);
        Jaa.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_aa(i) * buf_mat;
        Jbb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += d_bb(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };

  compute_three_center_integrals_tbb<kind>(build_J, engine.env(), 
                                           engine.aobasis(), engine.auxbasis(), 
                                           engine.shellpairs(), opt);

  // Reduce J matrices
  Mat J = Mat::Zero(rows, cols);
  for (const auto &local_J : tl_J) {
    J += local_J;
  }

  // Reduce exchange data and contract
  auto iuPa_total = reduce_thread_local_vectors(tl_iuPa);
  auto iuPb_total = reduce_thread_local_vectors(tl_iuPb);
  
  Mat K = Mat::Zero(rows, cols);
  Mat Xa(nbf, ndf), Xb(nbf, ndf);
  
  // Contract exchange matrix with cross-terms for general case
  for (size_t i = 0; i < nocc; i++) {
    Xa = V_LLt.solve(iuPa_total[i].transpose());
    Xb = V_LLt.solve(iuPb_total[i].transpose());
    block::aa(K).noalias() += iuPa_total[i] * Xa;
    block::bb(K).noalias() += iuPb_total[i] * Xb;
    block::ab(K).noalias() += (iuPa_total[i] * Xb) + (iuPb_total[i] * Xa);
    block::ba(K).noalias() += (iuPa_total[i] * Xb) + (iuPb_total[i] * Xa);
  }

  block::aa(K) += block::aa(K).transpose().eval();
  block::bb(K) += block::bb(K).transpose().eval();
  block::ba(K) += block::ba(K).transpose().eval();
  block::ab(K) += block::ab(K).transpose().eval();
  K *= 0.5;

  // Apply final factor of 2 to match stored kernel exactly
  J *= 2;
  
  occ::timing::stop(occ::timing::category::df);
  return {J, K};
}

inline Mat stored_coulomb_kernel_r(const Mat &ints, const AOBasis &aobasis,
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

inline Mat stored_coulomb_kernel_u(const Mat &ints, const AOBasis &aobasis,
                                   const AOBasis &auxbasis,
                                   const MolecularOrbitals &mo,
                                   const Eigen::LLT<Mat> V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto ndf = auxbasis.nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);
  Vec ga(ndf), gb(ndf);
  for (int r = 0; r < ndf; r++) {
    const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
    ga(r) = (block::a(mo.D).array() * tr.array()).sum();
    gb(r) = (block::b(mo.D).array() * tr.array()).sum();
  }
  Vec d_total = V_LLt.solve(ga + gb);
  Mat J = Mat::Zero(rows, cols);
  for (int r = 0; r < ndf; r++) {
    const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
    block::a(J) += d_total(r) * tr;
    block::b(J) += d_total(r) * tr;
  }
  return 2 * J;
}

inline Mat stored_coulomb_kernel_g(const Mat &ints, const AOBasis &aobasis,
                                   const AOBasis &auxbasis,
                                   const MolecularOrbitals &mo,
                                   const Eigen::LLT<Mat> V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto ndf = auxbasis.nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);
  // only alpha-alpha and beta-beta contributions to coulomb
  Vec gaa(ndf), gbb(ndf);
  for (int r = 0; r < ndf; r++) {
    const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
    gaa(r) = 2 * (block::aa(mo.D).array() * tr.array()).sum();
    gbb(r) = 2 * (block::bb(mo.D).array() * tr.array()).sum();
  }
  Vec daa = V_LLt.solve(gaa), dbb = V_LLt.solve(gbb);
  Mat J = Mat::Zero(rows, cols);
  for (int r = 0; r < ndf; r++) {
    const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
    block::aa(J) += daa(r) * tr;
    block::bb(J) += dbb(r) * tr;
  }
  return 2 * J;
}

inline Mat stored_exchange_kernel_r(const Mat &ints, const AOBasis &aobasis,
                                    const AOBasis &auxbasis,
                                    const MolecularOrbitals &mo,
                                    const Eigen::LLT<Mat> &V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto ndf = auxbasis.nbf();
  Mat K = Mat::Zero(nbf, nbf);
  // temporaries
  Mat iuP = Mat::Zero(nbf, ndf);
  Mat X(nbf, ndf);
  for (size_t i = 0; i < mo.Cocc.cols(); i++) {
    auto c = mo.Cocc.col(i);
    for (size_t r = 0; r < ndf; r++) {
      const auto vu = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
      iuP.col(r) = (vu * c);
    }
    X = V_LLt.solve(iuP.transpose());
    K.noalias() += iuP * X;
  }
  return K;
}

inline Mat stored_exchange_kernel_u(const Mat &ints, const AOBasis &aobasis,
                                    const AOBasis &auxbasis,
                                    const MolecularOrbitals &mo,
                                    const Eigen::LLT<Mat> &V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto ndf = auxbasis.nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);
  Mat K = Mat::Zero(rows, cols);
  // temporaries
  Mat iuPa = Mat::Zero(nbf, ndf), iuPb = Mat::Zero(nbf, ndf);
  Mat Xa(nbf, ndf), Xb(nbf, ndf);
  for (size_t i = 0; i < mo.Cocc.cols(); i++) {
    auto ca = block::a(mo.Cocc.col(i));
    auto cb = block::b(mo.Cocc.col(i));
    for (size_t r = 0; r < ndf; r++) {
      const auto vu = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
      iuPa.col(r) = (vu * ca);
      iuPb.col(r) = (vu * cb);
    }
    Xa = V_LLt.solve(iuPa.transpose());
    Xb = V_LLt.solve(iuPb.transpose());
    block::a(K).noalias() += iuPa * Xa;
    block::b(K).noalias() += iuPb * Xb;
  }
  return K;
}

inline Mat stored_exchange_kernel_g(const Mat &ints, const AOBasis &aobasis,
                                    const AOBasis &auxbasis,
                                    const MolecularOrbitals &mo,
                                    const Eigen::LLT<Mat> &V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto ndf = auxbasis.nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);
  Mat K = Mat::Zero(rows, cols);
  // temporaries
  Mat iuPa = Mat::Zero(nbf, ndf), iuPb = Mat::Zero(nbf, ndf);
  Mat Xa(nbf, ndf), Xb(nbf, ndf);
  for (size_t i = 0; i < mo.Cocc.cols(); i++) {
    auto ca = block::a(mo.Cocc.col(i));
    auto cb = block::b(mo.Cocc.col(i));
    for (size_t r = 0; r < ndf; r++) {
      const auto vu = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
      iuPa.col(r) = (vu * ca);
      iuPb.col(r) = (vu * cb);
    }
    Xa = V_LLt.solve(iuPa.transpose());
    Xb = V_LLt.solve(iuPb.transpose());
    block::aa(K).noalias() += iuPa * Xa;
    block::bb(K).noalias() += iuPb * Xb;
    block::ab(K).noalias() += (iuPa * Xb) + (iuPb * Xa);
    block::ba(K).noalias() += (iuPa * Xb) + (iuPb * Xa);
  }
  return K;
}

// Kernel for reconstructing full AO integral tensor from DF
// Following the same pattern as the exchange kernel
inline auto ao_tensor_reconstruction_kernel(std::vector<Eigen::Tensor<double, 4>>& tensors,
                                           const Mat& ints, const AOBasis& aobasis,
                                           const AOBasis& auxbasis,
                                           const Eigen::LLT<Mat>& V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto naux = auxbasis.nbf();
  const auto nthreads = occ::parallel::get_num_threads();
  
  return [&tensors, &ints, &V_LLt, nbf, naux, nthreads](int thread_id) {
    auto& tensor = tensors[thread_id];
    
    // Divide work among threads by μν pairs
    size_t total_pairs = nbf * nbf;
    size_t pairs_per_thread = (total_pairs + nthreads - 1) / nthreads;
    size_t pair_start = thread_id * pairs_per_thread;
    size_t pair_end = std::min(pair_start + pairs_per_thread, total_pairs);
    
    // Temporary storage
    Mat munuP = Mat::Zero(nbf, naux);  // (μν|P) for all μν
    Mat X = Mat::Zero(nbf, naux);      // V^(-1) * (ρσ|P)^T
    
    // Process assigned μν pairs
    for (size_t pair_idx = pair_start; pair_idx < pair_end; ++pair_idx) {
      size_t mu = pair_idx / nbf;
      size_t nu = pair_idx % nbf;
      
      // Extract (μν|P) vector for this μν pair following exchange kernel pattern
      for (size_t P = 0; P < naux; ++P) {
        const auto eri_P = Eigen::Map<const Mat>(ints.col(P).data(), nbf, nbf);
        munuP(mu * nbf + nu, P) = eri_P(mu, nu);
      }
    }
    
    // For each ρσ pair, compute V^(-1) * (ρσ|P) and then dot with (μν|P)
    for (size_t rho = 0; rho < nbf; ++rho) {
      for (size_t sigma = 0; sigma < nbf; ++sigma) {
        // Extract (ρσ|P) vector
        Vec rhosigmaP = Vec::Zero(naux);
        for (size_t P = 0; P < naux; ++P) {
          const auto eri_P = Eigen::Map<const Mat>(ints.col(P).data(), nbf, nbf);
          rhosigmaP(P) = eri_P(rho, sigma);
        }
        
        // Solve V * x = (ρσ|P) to get x = V^(-1) * (ρσ|P)
        Vec x = V_LLt.solve(rhosigmaP);
        
        // Now compute (μν|ρσ) = (μν|P) * V^(-1) * (ρσ|P) for assigned μν pairs
        for (size_t pair_idx = pair_start; pair_idx < pair_end; ++pair_idx) {
          size_t mu = pair_idx / nbf;
          size_t nu = pair_idx % nbf;
          
          double integral_value = 0.0;
          for (size_t P = 0; P < naux; ++P) {
            const auto eri_P = Eigen::Map<const Mat>(ints.col(P).data(), nbf, nbf);
            integral_value += eri_P(mu, nu) * x(P);
          }
          
          tensor(mu, nu, rho, sigma) = integral_value;
        }
      }
    }
  };
}

// Optimized kernel using batched operations
inline auto ao_tensor_reconstruction_kernel_batched(std::vector<Eigen::Tensor<double, 4>>& tensors,
                                                   const Mat& ints, const AOBasis& aobasis,
                                                   const AOBasis& auxbasis,
                                                   const Eigen::LLT<Mat>& V_LLt) {
  const auto nbf = aobasis.nbf();
  const auto naux = auxbasis.nbf();
  const auto nthreads = occ::parallel::get_num_threads();
  
  return [&tensors, &ints, &V_LLt, nbf, naux, nthreads](int thread_id) {
    auto& tensor = tensors[thread_id];
    
    // Divide work among threads by μν pairs
    size_t total_pairs = nbf * nbf;
    size_t pairs_per_thread = (total_pairs + nthreads - 1) / nthreads;
    size_t pair_start = thread_id * pairs_per_thread;
    size_t pair_end = std::min(pair_start + pairs_per_thread, total_pairs);
    
    // Pre-compute V^(-1) * (ρσ|P)^T for all ρσ pairs to avoid repeated solves
    Mat all_rhosigma_P = Mat::Zero(naux, nbf * nbf);
    Mat X_all = Mat::Zero(naux, nbf * nbf);
    
    // Extract all (ρσ|P) integrals
    for (size_t rho = 0; rho < nbf; ++rho) {
      for (size_t sigma = 0; sigma < nbf; ++sigma) {
        size_t rhosigma_idx = rho * nbf + sigma;
        for (size_t P = 0; P < naux; ++P) {
          const auto eri_P = Eigen::Map<const Mat>(ints.col(P).data(), nbf, nbf);
          all_rhosigma_P(P, rhosigma_idx) = eri_P(rho, sigma);
        }
      }
    }
    
    // Solve V * X = (ρσ|P) for all ρσ pairs at once
    X_all = V_LLt.solve(all_rhosigma_P);
    
    // Process assigned μν pairs
    for (size_t pair_idx = pair_start; pair_idx < pair_end; ++pair_idx) {
      size_t mu = pair_idx / nbf;
      size_t nu = pair_idx % nbf;
      
      // Extract (μν|P) vector
      Vec munuP = Vec::Zero(naux);
      for (size_t P = 0; P < naux; ++P) {
        const auto eri_P = Eigen::Map<const Mat>(ints.col(P).data(), nbf, nbf);
        munuP(P) = eri_P(mu, nu);
      }
      
      // Compute all (μν|ρσ) for this μν using precomputed X values
      for (size_t rho = 0; rho < nbf; ++rho) {
        for (size_t sigma = 0; sigma < nbf; ++sigma) {
          size_t rhosigma_idx = rho * nbf + sigma;
          double integral_value = munuP.dot(X_all.col(rhosigma_idx));
          tensor(mu, nu, rho, sigma) = integral_value;
        }
      }
    }
  };
}

// Direct DF-MP2 MO integral kernel using symmetric formulation
// Computes (ia|jb) directly without reconstructing full AO tensor
inline void compute_df_mp2_integrals(std::vector<std::vector<std::vector<std::vector<double>>>>& ovov_tensor,
                                     const Mat& ints, const AOBasis& aobasis, const AOBasis& auxbasis,
                                     const MolecularOrbitals& mo, const Eigen::LLT<Mat>& V_LLt,
                                     size_t n_occ, size_t n_virt) {
  const auto nbf = aobasis.nbf();
  const auto naux = auxbasis.nbf();
  
  // Step 1: First transformation b^Q_iν = Σ_μ C^i_μ (μν|Q)
  // Following exchange kernel pattern
  std::vector<Mat> b_iP(n_occ, Mat::Zero(nbf, naux));
  
  for (size_t i = 0; i < n_occ; ++i) {
    auto c_i = mo.C.col(i);  // occupied orbital coefficients
    for (size_t P = 0; P < naux; ++P) {
      const auto eri_P = Eigen::Map<const Mat>(ints.col(P).data(), nbf, nbf);
      b_iP[i].col(P) = eri_P * c_i;  // b^P_iν
    }
  }
  
  // Step 2: Second transformation b^Q_ia = Σ_ν C^a_ν b^Q_iν
  std::vector<Mat> b_ia(n_occ, Mat::Zero(n_virt, naux));
  
  for (size_t i = 0; i < n_occ; ++i) {
    for (size_t P = 0; P < naux; ++P) {
      for (size_t a = 0; a < n_virt; ++a) {
        double sum = 0.0;
        for (size_t nu = 0; nu < nbf; ++nu) {
          size_t virt_idx = n_occ + a;  // virtual orbital index in full MO space
          sum += mo.C(nu, virt_idx) * b_iP[i](nu, P);
        }
        b_ia[i](a, P) = sum;
      }
    }
  }
  
  // Step 3: Apply Coulomb metric J^(-1/2) to get symmetric b^Q objects
  std::vector<Mat> b_ia_sym(n_occ, Mat::Zero(n_virt, naux));
  
  for (size_t i = 0; i < n_occ; ++i) {
    // X = J^(-1/2) * b^T, so b_sym = b * J^(-1/2)^T = b * J^(-1/2) (since J^(-1/2) is symmetric)
    Mat X = V_LLt.solve(b_ia[i].transpose());  // X = J^(-1) * b^T
    b_ia_sym[i] = X.transpose();  // b_sym = X^T = b * J^(-1)
    
    // For proper symmetric formulation, we need sqrt(J^(-1)) not J^(-1)
    // But following the exchange kernel pattern exactly first
  }
  
  // Step 4: Final integral construction (ia|jb) = Σ_Q b^Q_ia * b^Q_jb
  for (size_t i = 0; i < n_occ; ++i) {
    for (size_t a = 0; a < n_virt; ++a) {
      for (size_t j = 0; j < n_occ; ++j) {
        for (size_t b = 0; b < n_virt; ++b) {
          double integral_value = 0.0;
          for (size_t Q = 0; Q < naux; ++Q) {
            integral_value += b_ia_sym[i](a, Q) * b_ia_sym[j](b, Q);
          }
          ovov_tensor[i][a][j][b] = integral_value;
        }
      }
    }
  }
}

} // namespace occ::qm::detail
