#pragma once
#include <Eigen/Cholesky>
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
void contract_exchange_matrices(const std::vector<Mat>& iuP, const CoulombMetric& V_LLt, 
                               KMatType& K_block) {
  // One entry per occupied orbital, so a system with no occupied orbitals at
  // all leaves nothing to contract -- the very first Fock build of a guess
  // that supplied a density but no orbitals. (An empty beta channel does not
  // reach here: the unrestricted kernel sizes both spin workspaces from
  // `mo.Cocc.cols()`, so a beta-empty system gets a zero-filled `iuPb`, not an
  // empty one.)
  if (iuP.empty())
    return;
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
                             const gto::AOBasis &aobasis,
                             const gto::AOBasis &auxbasis,
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
                                      const CoulombMetric &V_LLt,
                                      cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  // One shared workspace, not one per thread: the parallel loop runs over
  // auxiliary shells and every write below lands in the column indexed by an
  // auxiliary function, so two threads never touch the same element. These
  // are nocc x nbf x ndf, the largest allocation in the build, so duplicating
  // them per thread cost both the footprint and the time to zero and reduce
  // them. `direct_coulomb_and_exchange_operator_kernel_u` already shares its
  // copy for the same reason.
  std::vector<Mat> iuP(nocc, Mat::Zero(nbf, ndf));

  // Process each auxiliary shell with work-stealing parallelization
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_iuP = iuP;
    
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

  Mat K = Mat::Zero(nbf, nbf);
  contract_exchange_matrices(iuP, V_LLt, K);

  occ::timing::stop(occ::timing::category::df);
  return 0.5 * (K + K.transpose());
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_exchange_operator_kernel_u(IntegralEngine &engine,
                                      IntegralEngine &engine_aux,
                                      const MolecularOrbitals &mo,
                                      const CoulombMetric &V_LLt,
                                      cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);

  // TBB thread-local storage for intermediate matrices (alpha and beta)
  // One shared workspace, not one per thread: the parallel loop runs over
  // auxiliary shells and every write below lands in the column indexed by an
  // auxiliary function, so two threads never touch the same element. These
  // are nocc x nbf x ndf, the largest allocation in the build, so duplicating
  // them per thread cost both the footprint and the time to zero and reduce
  // them. `direct_coulomb_and_exchange_operator_kernel_u` already shares its
  // copy for the same reason.
  std::vector<Mat> iuPa(nocc, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuPb(nocc, Mat::Zero(nbf, ndf));

  // Process each auxiliary shell with work-stealing parallelization
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_iuPa = iuPa;
    auto &local_iuPb = iuPb;
    
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
                                      const CoulombMetric &V_LLt,
                                      cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);

  // TBB thread-local storage for intermediate matrices (alpha and beta)
  // One shared workspace, not one per thread: the parallel loop runs over
  // auxiliary shells and every write below lands in the column indexed by an
  // auxiliary function, so two threads never touch the same element. These
  // are nocc x nbf x ndf, the largest allocation in the build, so duplicating
  // them per thread cost both the footprint and the time to zero and reduce
  // them. `direct_coulomb_and_exchange_operator_kernel_u` already shares its
  // copy for the same reason.
  std::vector<Mat> iuPa(nocc, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuPb(nocc, Mat::Zero(nbf, ndf));

  // Process each auxiliary shell with work-stealing parallelization
  auto process_integrals = [&](const IntegralResult &args) {
    auto &local_iuPa = iuPa;
    auto &local_iuPb = iuPb;
    
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
  
  Mat K = Mat::Zero(rows, cols);
  Mat Xa(nbf, ndf), Xb(nbf, ndf);
  
  // iuPa/iuPb hold both halves of every canonical pair, so each block is
  // complete. block::ab is the lower-left (beta rows, alpha columns), so
  // K_ab = sum_i (μ i_b|P) V^-1 (P|ν i_a); it is not symmetric, and K_ba is
  // its transpose. Only the diagonal blocks are symmetrised, to drop
  // rounding asymmetry.
  for (size_t i = 0; i < nocc; i++) {
    Xa = V_LLt.solve(iuPa[i].transpose());
    Xb = V_LLt.solve(iuPb[i].transpose());
    block::aa(K).noalias() += iuPa[i] * Xa;
    block::bb(K).noalias() += iuPb[i] * Xb;
    block::ab(K).noalias() += iuPb[i] * Xa;
  }

  block::aa(K) = 0.5 * (block::aa(K) + block::aa(K).transpose()).eval();
  block::bb(K) = 0.5 * (block::bb(K) + block::bb(K).transpose()).eval();
  block::ba(K) = block::ab(K).transpose();
  occ::timing::stop(occ::timing::category::df);

  return K;
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_r(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const CoulombMetric &V_LLt,
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
                                     const CoulombMetric &V_LLt,
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
                                     const CoulombMetric &V_LLt,
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
  
  // Both diagonal blocks see the Coulomb field of the whole density,
  // Daa + Dbb (as `direct_coulomb_operator_kernel_u` does for Da + Db)
  Vec d_aa = V_LLt.solve(gaa_total + gbb_total);
  const Vec &d_bb = d_aa;

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
    const MolecularOrbitals &mo, const CoulombMetric &V_LLt,
    cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  // One shared workspace, not one per thread.
  //
  // The parallel loop runs over auxiliary shells and every accumulation here
  // is indexed by an auxiliary basis function, so two threads never touch the
  // same element of `g` or the same column of `iuP`. Duplicating them per
  // thread and reducing afterwards was pure overhead: `iuP` is
  // nocc x nbf x ndf, the largest allocation in the build (1.2 GiB for a
  // 50-atom dimer in a JK-fitting basis), so the copies dominated both the
  // footprint and the time spent zeroing and summing them.
  JKData shared(ndf, nbf, nocc);
  Vec &g = shared.g;
  std::vector<Mat> &iuP = shared.iuP;

  // Combined JK computation in single pass over three-center integrals
  auto process_integrals = [&](const IntegralResult &args) {
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

  Mat K = Mat::Zero(nbf, nbf);
  contract_exchange_matrices(iuP, V_LLt, K);

  occ::timing::stop(occ::timing::category::df);
  return {J + J.transpose(), 0.5 * (K + K.transpose())};
}

template <ShellKind kind = ShellKind::Cartesian>
JKPair direct_coulomb_and_exchange_operator_kernel_u(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const CoulombMetric &V_LLt,
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
    const MolecularOrbitals &mo, const CoulombMetric &V_LLt,
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
  // One shared workspace, not one per thread: the parallel loop runs over
  // auxiliary shells and every write below lands in the column indexed by an
  // auxiliary function, so two threads never touch the same element. These
  // are nocc x nbf x ndf, the largest allocation in the build, so duplicating
  // them per thread cost both the footprint and the time to zero and reduce
  // them. `direct_coulomb_and_exchange_operator_kernel_u` already shares its
  // copy for the same reason.
  std::vector<Mat> iuPa(nocc, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuPb(nocc, Mat::Zero(nbf, ndf));

  // Combined JK computation in single pass over three-center integrals
  auto process_integrals = [&](const IntegralResult &args) {
    auto &gaa = tl_gaa.local();
    auto &gbb = tl_gbb.local();
    auto &local_iuPa = iuPa;
    auto &local_iuPb = iuPb;
    
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
  
  // Both diagonal blocks see the Coulomb field of the whole density,
  // Daa + Dbb (as `direct_coulomb_and_exchange_operator_kernel_u` does)
  occ::timing::start(occ::timing::category::la);
  Vec d_aa = V_LLt.solve(gaa_total + gbb_total);
  const Vec &d_bb = d_aa;
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

  const auto &iuPa_total = iuPa;
  const auto &iuPb_total = iuPb;

  Mat K = Mat::Zero(rows, cols);
  Mat Xa(nbf, ndf), Xb(nbf, ndf);
  
  // iuPa/iuPb hold both halves of every canonical pair, so each block is
  // complete. block::ab is the lower-left (beta rows, alpha columns), so
  // K_ab = sum_i (μ i_b|P) V^-1 (P|ν i_a); it is not symmetric, and K_ba is
  // its transpose. Only the diagonal blocks are symmetrised, to drop
  // rounding asymmetry.
  for (size_t i = 0; i < nocc; i++) {
    Xa = V_LLt.solve(iuPa_total[i].transpose());
    Xb = V_LLt.solve(iuPb_total[i].transpose());
    block::aa(K).noalias() += iuPa_total[i] * Xa;
    block::bb(K).noalias() += iuPb_total[i] * Xb;
    block::ab(K).noalias() += iuPb_total[i] * Xa;
  }

  block::aa(K) = 0.5 * (block::aa(K) + block::aa(K).transpose()).eval();
  block::bb(K) = 0.5 * (block::bb(K) + block::bb(K).transpose()).eval();
  block::ba(K) = block::ab(K).transpose();

  // Apply final factor of 2 to match stored kernel exactly
  J *= 2;
  
  occ::timing::stop(occ::timing::category::df);
  return {J, K};
}

} // namespace occ::qm::detail
