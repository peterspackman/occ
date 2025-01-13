#pragma once
#include "kernel_traits.h"
#include <occ/qm/integral_engine_df.h>

namespace occ::qm::detail {

using IntegralResult = IntegralEngine::IntegralResult<3>;

inline auto g_lambda_direct_r(std::vector<Vec> &gg,
                              const MolecularOrbitals &mo) {
  return [&](const IntegralResult &args) {
    auto &g = gg[args.thread];
    size_t offset = 0;
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        g(i) += (mo.D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                     .array() *
                 buf_mat.array())
                    .sum();
        g(i) += (mo.D.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                     .array() *
                 buf_mat.transpose().array())
                    .sum();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        g(i) += (mo.D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
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
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        ga(i) += (Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        gb(i) += (Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        ga(i) += (Da.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                      .array() *
                  buf_mat.transpose().array())
                     .sum();
        gb(i) += (Db.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                      .array() *
                  buf_mat.transpose().array())
                     .sum();
        offset += args.dims[0] * args.dims[1];
      }
    } else {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        ga(i) += (Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        gb(i) += (Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                      .array() *
                  buf_mat.array())
                     .sum();
        offset += args.dims[0] * args.dims[1];
      }
    }
  };
}

inline auto g_lambda_direct_g(std::vector<Vec> &gg_alpha,
                              std::vector<Vec> &gg_beta,
                              const MolecularOrbitals &mo) {
  return [&](const IntegralResult &args) {
    auto &ga = gg_alpha[args.thread];
    auto &gb = gg_beta[args.thread];
    const auto Daa = qm::block::aa(mo.D);
    const auto Dbb = qm::block::bb(mo.D);
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
    auto Ja = qm::block::a(JJ[args.thread]);
    auto Jb = qm::block::b(JJ[args.thread]);
    size_t offset = 0;
    if (args.bf[0] != args.bf[1]) {
      for (size_t i = args.bf[2]; i < args.bf[2] + args.dims[2]; i++) {
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
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
        Eigen::Map<const Mat> buf_mat(args.buffer + offset, args.dims[0],
                                      args.dims[1]);
        Ja.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            da(i) * buf_mat;
        Jb.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
            db(i) * buf_mat;
        offset += args.dims[0] * args.dims[1];
      }
    }
  };
}

inline auto j_lambda_direct_g(std::vector<Mat> &JJ, const Vec &da,
                              const Vec &db) {
  return [&JJ, &da, &db](const IntegralResult &args) {
    auto Jaa = qm::block::aa(JJ[args.thread]);
    auto Jbb = qm::block::bb(JJ[args.thread]);
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
      auto c2a = qm::block::a(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3a = qm::block::a(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);
      auto c2b = qm::block::b(mo.Cocc).block(args.bf[0], i, args.dims[0], 1);
      auto c3b = qm::block::b(mo.Cocc).block(args.bf[1], i, args.dims[1], 1);

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
    size_t n_beta = Cocc_a.cols();
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
            2 * (Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                     .array() *
                 buf_mat.array())
                    .sum();
        gb(r) +=
            2 * (Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                     .array() *
                 buf_mat.array())
                    .sum();

        c3_term_a = buf_mat * c3a;
        c3_term_b = buf_mat * c3b;

        ga(r) +=
            2 * (Da.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
                     .array() *
                 buf_mat.transpose().array())
                    .sum();
        gb(r) +=
            2 * (Db.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0])
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
            2 * (Da.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                     .array() *
                 buf_mat.array())
                    .sum();
        gb(r) +=
            2 * (Db.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
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
void three_center_aux_kernel(Lambda &f, qm::cint::IntegralEnvironment &env,
                             const qm::AOBasis &aobasis,
                             const qm::AOBasis &auxbasis,
                             const ShellPairList &shellpairs,
                             occ::qm::cint::Optimizer &opt,
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
                                      occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nthreads = occ::parallel::get_num_threads();
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  Mat K = Mat::Zero(nbf, nbf);
  std::vector<Mat> iuP(nocc * nthreads, Mat::Zero(nbf, ndf));
  Mat X(nbf, ndf);

  auto klambda = k_lambda_direct_r(iuP, mo);

  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(klambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (size_t i = nocc; i < nocc * nthreads; i++) {
    iuP[i % nocc] += iuP[i];
  }

  for (size_t i = 0; i < nocc; i++) {
    X = V_LLt.solve(iuP[i].transpose());
    K.noalias() += iuP[i] * X;
  }

  occ::timing::stop(occ::timing::category::df);
  return 0.5 * (K + K.transpose());
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_exchange_operator_kernel_u(IntegralEngine &engine,
                                      IntegralEngine &engine_aux,
                                      const MolecularOrbitals &mo,
                                      const Eigen::LLT<Mat> &V_LLt,
                                      occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nthreads = occ::parallel::get_num_threads();
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);
  Mat K = Mat::Zero(rows, cols);
  std::vector<Mat> iuPa(nocc * nthreads, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuPb(nocc * nthreads, Mat::Zero(nbf, ndf));
  Mat Xa(nbf, ndf), Xb(nbf, ndf);

  auto klambda = k_lambda_direct_u(iuPa, iuPb, mo);

  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(klambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (size_t i = nocc; i < nocc * nthreads; i++) {
    iuPa[i % nocc] += iuPa[i];
    iuPb[i % nocc] += iuPb[i];
  }

  for (size_t i = 0; i < nocc; i++) {
    Xa = V_LLt.solve(iuPa[i].transpose());
    Xb = V_LLt.solve(iuPb[i].transpose());
    block::a(K).noalias() += iuPa[i] * Xa;
    block::b(K).noalias() += iuPb[i] * Xb;
  }

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
                                      occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nthreads = occ::parallel::get_num_threads();
  size_t nocc = mo.Cocc.cols();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);
  Mat K = Mat::Zero(rows, cols);
  std::vector<Mat> iuPa(nocc * nthreads, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuPb(nocc * nthreads, Mat::Zero(nbf, ndf));
  Mat Xa(nbf, ndf), Xb(nbf, ndf);

  auto klambda = k_lambda_direct_u(iuPa, iuPb, mo);

  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(klambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (size_t i = nocc; i < nocc * nthreads; i++) {
    iuPa[i % nocc] += iuPa[i];
    iuPb[i % nocc] += iuPb[i];
  }

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
                                     occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  std::vector<Vec> gg(nthreads, Vec::Zero(ndf));
  std::vector<Mat> JJ(nthreads, Mat::Zero(nbf, nbf));

  auto glambda = g_lambda_direct_r(gg, mo);
  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(glambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
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
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda2);

  for (int i = 1; i < nthreads; i++) {
    JJ[0] += JJ[i];
  }
  occ::timing::stop(occ::timing::category::df);
  return (JJ[0] + JJ[0].transpose());
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_u(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const Eigen::LLT<Mat> &V_LLt,
                                     occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::Unrestricted>(nbf);
  std::vector<Vec> gg_alpha(nthreads, Vec::Zero(ndf));
  std::vector<Vec> gg_beta(nthreads, Vec::Zero(ndf));
  std::vector<Mat> JJ(nthreads, Mat::Zero(rows, cols));

  auto glambda = g_lambda_direct_u(gg_alpha, gg_beta, mo);
  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(glambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (int i = 1; i < nthreads; i++) {
    gg_alpha[0] += gg_alpha[i];
    gg_beta[0] += gg_beta[i];
  }
  Vec d_alpha = V_LLt.solve(gg_alpha[0]);
  Vec d_beta = V_LLt.solve(gg_beta[0]);

  auto jlambda = j_lambda_direct_u(JJ, d_alpha, d_beta);
  auto lambda2 = [&](int thread_id) {
    three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda2);

  Mat J = Mat::Zero(rows, cols);
  auto Ja = block::a(J);
  auto Jb = block::b(J);
  for (const auto &part : JJ) {
    auto part_a = block::a(part);
    auto part_b = block::b(part);
    Ja.noalias() += part_a + part_a.transpose();
    Jb.noalias() += part_b + part_b.transpose();
  }

  J *= 2;

  occ::timing::stop(occ::timing::category::df);
  return J;
}

template <ShellKind kind = ShellKind::Cartesian>
Mat direct_coulomb_operator_kernel_g(IntegralEngine &engine,
                                     IntegralEngine &engine_aux,
                                     const MolecularOrbitals &mo,
                                     const Eigen::LLT<Mat> &V_LLt,
                                     occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  const auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();
  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);
  std::vector<Vec> gg_alpha(nthreads, Vec::Zero(ndf));
  std::vector<Vec> gg_beta(nthreads, Vec::Zero(ndf));
  std::vector<Mat> JJ(nthreads, Mat::Zero(rows, cols));

  auto glambda = g_lambda_direct_g(gg_alpha, gg_beta, mo);
  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(glambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (int i = 1; i < nthreads; i++) {
    gg_alpha[0] += gg_alpha[i];
    gg_beta[0] += gg_beta[i];
  }
  Vec d_alpha = V_LLt.solve(gg_alpha[0]);
  Vec d_beta = V_LLt.solve(gg_beta[0]);

  auto jlambda = j_lambda_direct_g(JJ, d_alpha, d_beta);
  auto lambda2 = [&](int thread_id) {
    three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda2);

  Mat J = Mat::Zero(rows, cols);
  auto Jaa = block::aa(J);
  auto Jbb = block::bb(J);
  for (const auto &part : JJ) {
    auto part_aa = block::aa(part);
    auto part_bb = block::bb(part);
    Jaa.noalias() += part_aa + part_aa.transpose();
    Jbb.noalias() += part_bb + part_bb.transpose();
  }

  J *= 2;

  occ::timing::stop(occ::timing::category::df);
  return J;
}

template <ShellKind kind = ShellKind::Cartesian>
JKPair direct_coulomb_and_exchange_operator_kernel_r(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const Eigen::LLT<Mat> &V_LLt,
    occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.Cocc.cols();
  const auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  std::vector<Vec> gg(nthreads, Vec::Zero(ndf));
  std::vector<Mat> JJ(nthreads, Mat::Zero(nbf, nbf));
  std::vector<Mat> KK(nthreads, Mat::Zero(nbf, nbf));

  std::vector<Mat> iuP(nocc, Mat::Zero(nbf, ndf));

  auto jk_lambda_1 = jk_lambda_direct_unpolarized(gg, iuP, mo);
  auto lambda = [&jk_lambda_1, &engine, &opt](int thread_id) {
    three_center_aux_kernel<kind>(jk_lambda_1, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (int i = 1; i < nthreads; i++)
    gg[0] += gg[i];

  auto klambda = [&, V_LLt](int thread_id) {
    Mat X(nbf, ndf);
    for (size_t i = 0; i < nocc; i++) {
      if (i % nthreads != thread_id)
        continue;
      X = V_LLt.solve(iuP[i].transpose());
      KK[thread_id].noalias() += iuP[i] * X;
    }
  };
  occ::parallel::parallel_do(klambda);

  occ::timing::start(occ::timing::category::la);
  Vec d = V_LLt.solve(gg[0]);
  occ::timing::stop(occ::timing::category::la);

  auto jlambda = j_lambda_direct_r(JJ, d);
  auto lambda2 = [&](int thread_id) {
    three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda2);

  for (int i = 1; i < nthreads; i++) {
    JJ[0] += JJ[i];
    KK[0] += KK[i];
  }

  occ::timing::stop(occ::timing::category::df);
  return {JJ[0] + JJ[0].transpose(), 0.5 * (KK[0] + KK[0].transpose())};
}

template <ShellKind kind = ShellKind::Cartesian>
JKPair direct_coulomb_and_exchange_operator_kernel_u(
    IntegralEngine &engine, IntegralEngine &engine_aux,
    const MolecularOrbitals &mo, const Eigen::LLT<Mat> &V_LLt,
    occ::qm::cint::Optimizer &opt) {
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
  occ::parallel::parallel_do(lambda);

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

  occ::parallel::parallel_do(klambda);

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
  occ::parallel::parallel_do(lambda2);

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
    occ::qm::cint::Optimizer &opt) {
  occ::timing::start(occ::timing::category::df);
  size_t nocc = mo.n_alpha; // number of electrons == n_alpha for general
  const auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = engine.aobasis().nbf();
  const auto ndf = engine.auxbasis().nbf();

  const auto [rows, cols] =
      occ::qm::matrix_dimensions<occ::qm::SpinorbitalKind::General>(nbf);

  std::vector<Vec> gg_alpha(nthreads, Vec::Zero(ndf));
  std::vector<Vec> gg_beta(nthreads, Vec::Zero(ndf));
  std::vector<Mat> JJ(nthreads, Mat::Zero(rows, cols));
  std::vector<Mat> KK(nthreads, Mat::Zero(rows, cols));
  Mat J = Mat::Zero(rows, cols), K = Mat::Zero(rows, cols);

  std::vector<Mat> iuP_alpha(nocc, Mat::Zero(nbf, ndf));
  std::vector<Mat> iuP_beta(nocc, Mat::Zero(nbf, ndf));

  auto jk_lambda_1 = jk_lambda_direct_polarized(
      gg_alpha, gg_beta, iuP_alpha, iuP_beta, block::aa(mo.D), block::bb(mo.D),
      mo.occ_alpha(), mo.occ_beta());
  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(jk_lambda_1, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (int i = 1; i < nthreads; i++) {
    gg_alpha[0] += gg_alpha[i];
    gg_beta[0] += gg_beta[i];
  }

  auto klambda = [&iuP_alpha, &iuP_beta, &KK, &V_LLt, nthreads, nocc, nbf,
                  ndf](int thread_id) {
    auto &Kpart = KK[thread_id];
    auto aa_part = block::aa(Kpart);
    auto ab_part = block::ab(Kpart);
    auto ba_part = block::ba(Kpart);
    auto bb_part = block::bb(Kpart);
    Mat Xa = Mat::Zero(nbf, ndf);
    Mat Xb = Mat::Zero(nbf, ndf);
    for (size_t i = 0; i < nocc; i++) {
      if (i % nthreads != thread_id)
        continue;
      Xa = V_LLt.solve(iuP_alpha[i].transpose());
      Xb = V_LLt.solve(iuP_beta[i].transpose());
      aa_part.noalias() += iuP_alpha[i] * Xa;
      bb_part.noalias() += iuP_beta[i] * Xb;
      ab_part.noalias() += (iuP_alpha[i] * Xb) + (iuP_beta[i] * Xa);
      ba_part.noalias() += (iuP_alpha[i] * Xb) + (iuP_beta[i] * Xa);
    }
  };

  occ::parallel::parallel_do(klambda);

  occ::timing::start(occ::timing::category::la);
  Vec d_alpha = V_LLt.solve(gg_alpha[0]);
  Vec d_beta = V_LLt.solve(gg_beta[0]);
  occ::timing::stop(occ::timing::category::la);

  auto jlambda = j_lambda_direct_g(JJ, d_alpha, d_beta);
  auto lambda2 = [&](int thread_id) {
    three_center_aux_kernel<kind>(jlambda, engine.env(), engine.aobasis(),
                                  engine.auxbasis(), engine.shellpairs(), opt,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda2);

  auto Jaa = block::aa(J);
  auto Jbb = block::bb(J);
  auto Kaa = block::aa(K);
  auto Kab = block::ab(K);
  auto Kba = block::ba(K);
  auto Kbb = block::bb(K);
  for (int i = 0; i < nthreads; i++) {
    auto JJaa = block::aa(JJ[i]);
    auto JJbb = block::bb(JJ[i]);

    auto KKaa = block::aa(KK[i]);
    auto KKab = block::ab(KK[i]);
    auto KKba = block::ba(KK[i]);
    auto KKbb = block::bb(KK[i]);
    Jaa.noalias() += JJaa + JJaa.transpose();
    Jbb.noalias() += JJbb + JJbb.transpose();

    Kaa.noalias() += KKaa + KKaa.transpose();
    Kab.noalias() += KKab + KKab.transpose();
    Kba.noalias() += KKba + KKba.transpose();
    Kbb.noalias() += KKbb + KKbb.transpose();
  }

  // can move a factor of 2 in the jk_lambda_g or in g_alpha/beta but this
  // saves flops (tiny)
  K *= 0.5;
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
    ga(r) = 2 * (block::a(mo.D).array() * tr.array()).sum();
    gb(r) = 2 * (block::b(mo.D).array() * tr.array()).sum();
  }
  Vec da = V_LLt.solve(ga), db = V_LLt.solve(gb);
  Mat J = Mat::Zero(rows, cols);
  for (int r = 0; r < ndf; r++) {
    const auto tr = Eigen::Map<const Mat>(ints.col(r).data(), nbf, nbf);
    block::a(J) += da(r) * tr;
    block::b(J) += db(r) * tr;
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
} // namespace occ::qm::detail
