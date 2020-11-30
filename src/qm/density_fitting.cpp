#include <tonto/qm/density_fitting.h>
#include <tonto/core/parallel.h>
#include <libint2.hpp>

namespace tonto::df {

tonto::MatRM DFFockEngine::compute_2body_fock_dfC(const tonto::MatRM& Cocc) {

  using tonto::parallel::nthreads;

  const size_t n = obs.nbf();
  const size_t ndf = dfbs.nbf();

  // using first time? compute 3-center ints and transform to inv sqrt
  // representation
  if (xyK.size() == 0) {

    const auto nshells = obs.size();
    const auto nshells_df = dfbs.size();
    const auto& unitshell = libint2::Shell::unit();

    // construct the 2-electron 3-center repulsion integrals engine
    // since the code assumes (xx|xs) braket, and Engine/libint only produces
    // (xs|xx), use 4-center engine
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] = libint2::Engine(libint2::Operator::coulomb,
                                 std::max(obs.max_nprim(), dfbs.max_nprim()),
                                 std::max(obs.max_l(), dfbs.max_l()), 0);
    engines[0].set(libint2::BraKet::xs_xx);
    for (size_t i = 1; i != nthreads; ++i) {
      engines[i] = engines[0];
    }

    auto shell2bf = obs.shell2bf();
    auto shell2bf_df = dfbs.shell2bf();

    std::array<size_t, 3> Zxy_dims{ndf, n, n};
    std::vector<double> Zxy(ndf * n * n);

    auto lambda = [&](int thread_id) {

      auto& engine = engines[thread_id];
      const auto& results = engine.results();

      // loop over permutationally-unique set of shells
      for (auto s1 = 0l, s123 = 0l; s1 != nshells_df; ++s1) {
        auto bf1_first = shell2bf_df[s1];  // first basis function in this shell
        auto n1 = dfbs[s1].size();  // number of basis functions in this shell

        for (auto s2 = 0; s2 != nshells; ++s2) {
          auto bf2_first = shell2bf[s2];
          auto n2 = obs[s2].size();
          const auto n12 = n1 * n2;

          for (auto s3 = 0; s3 != nshells; ++s3, ++s123) {
            if (s123 % nthreads != thread_id) continue;

            auto bf3_first = shell2bf[s3];
            auto n3 = obs[s3].size();
            const auto n123 = n12 * n3;

            engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xx, 0>(
                dfbs[s1], unitshell, obs[s2], obs[s3]);
            const auto* buf = results[0];
            if (buf == nullptr)
              continue;

            size_t offset = 0;
            for(size_t i = bf1_first; i < bf1_first + n1; i++) {
                for(size_t j = bf2_first; j < bf2_first + n2; j++) {
                    for(size_t k = bf3_first; k < bf3_first + n3; k++) {
                        Zxy[k + n * (j + n * i)] = buf[offset];
                        offset++;
                    }
                }
            }
          }  // s3
        }    // s2
      }      // s1

    };  // lambda

    tonto::parallel::parallel_do(lambda);

    tonto::MatRM V = tonto::ints::compute_2body_2index_ints(dfbs);
    Eigen::LLT<tonto::MatRM> V_LLt(V);
    tonto::MatRM I = tonto::MatRM::Identity(ndf, ndf);
    auto L = V_LLt.matrixL();
    tonto::MatRM V_L = L;
    tonto::MatRM Linv_t = L.solve(I).transpose();
    // check
    //  std::cout << "||V - L L^t|| = " << (V - V_L * V_L.transpose()).norm() <<
    //  std::endl;
    //  std::cout << "||I - L L^-1|| = " << (I - V_L *
    //  Linv_t.transpose()).norm() << std::endl;
    //  std::cout << "||V^-1 - L^-1^t L^-1|| = " << (V.inverse() - Linv_t *
    //  Linv_t.transpose()).norm() << std::endl;

    std::array<size_t, 2> K_dims{ndf, ndf};
    std::vector<double> K(ndf * ndf);
    std::copy(Linv_t.data(), Linv_t.data() + ndf * ndf, K.data());

    xyK_dims = {n, n, ndf};
    xyK.resize(n * n * ndf);
    std::fill(xyK.begin(), xyK.end(), 0.0);
    // xyK(j,k,l) = Zxy(i,j,k) * K(i,l)
    for(size_t i = 0; i < K_dims[0]; i++) {
        for(size_t j = 0; j < Zxy_dims[1]; j++) {
            size_t jZxy = i * Zxy_dims[1] + j;
            for(size_t k = 0; k < Zxy_dims[2]; k++) {
                size_t kxyK = j * xyK_dims[1] + k;
                size_t kZxy = jZxy * Zxy_dims[2] + k;
                for(size_t l = 0; l < K_dims[1]; l++) {
                    size_t lxyK = kxyK * xyK_dims[2] + l;
                    size_t lK = i * K_dims[1] + l;
                    xyK[lxyK] = xyK[lxyK] + Zxy[kZxy] * K[lK];
                }
            }
        }
    }
    Zxy.clear();
  }  // if (xyK.size() == 0)

  // compute exchange
  const size_t nocc = Cocc.cols();
  const double * Co = Cocc.data();
  std::array<size_t, 2> Co_dims{n, nocc};
  std::array<size_t, 3> xiK_dims{xyK_dims[0], Co_dims[1], xyK_dims[2]};
  std::vector<double> xiK(xiK_dims[0] * xiK_dims[1] * xiK_dims[2], 0.0);

  // xiK(i,l,k) = xyK(i,j,k) * Co(j,l)
  for(size_t i = 0; i < xyK_dims[0]; i++) {
      for(size_t j = 0; j < Co_dims[0]; j++) {
          size_t jxyK = i * xyK_dims[1] + j;
          for(size_t l = 0; l < Co_dims[1]; l++) {
              size_t lxiK = i * xiK_dims[1] + l;
              size_t lCo = j * Co_dims[1] + l;
              for(size_t k = 0; k < xyK_dims[2]; k++) {
                  size_t kxiK = lxiK * xiK_dims[2] + k;
                  size_t kxyK = jxyK * xyK_dims[2] + k;
                  xiK[kxiK] = xiK[kxiK] + xyK[kxyK] * Co[lCo];
              }
          }
      }
  }

  std::array<size_t, 2> G_dims{xiK_dims[0], xiK_dims[0]};
  std::vector<double> G(G_dims[0] * G_dims[1]);
  //exchange
  // G(i,l) = xiK(i,j,k) * xiK(l,j,k)
  for(size_t i = 0; i < xiK_dims[0]; i++) {
      for(size_t l = 0; l < xiK_dims[0]; l++) {
          size_t lG = i * G_dims[1] + l;
          double tjG_val = 0.0;
          for(size_t j = 0; j < xiK_dims[1]; j++) {
              size_t jxiK = i * xiK_dims[1] + j;
              size_t jxiK0 = l * xiK_dims[1] + j;
              for(size_t k = 0; k < xiK_dims[2]; k++) {
                  size_t kxiK = jxiK * xiK_dims[2] + k;
                  size_t kxiK0 = jxiK0 * xiK_dims[2] + k;
                  tjG_val += xiK[kxiK] * xiK[kxiK0];
              }
          }
          G[lG] = tjG_val;
      }
  }

  //coulomb
  //J(k) = xiK(i,j,k) * Co(i,j)
  size_t J_dim = xiK_dims[2];
  std::vector<double> J(J_dim, 0);
  for(size_t i = 0; i < Co_dims[0]; i++) {
      for(size_t j = 0; j < Co_dims[1]; j++) {
          size_t jxiK = i * xiK_dims[1] + j;
          size_t jCo = i * Co_dims[1] + j;
          for(size_t k = 0; k < xiK_dims[2]; k++) {
              size_t kxiK = jxiK * xiK_dims[2] + k;
              J[k] = J[k] + xiK[kxiK] * Co[jCo];
          }
      }
  }
  // G(i,j) = 2 * xyK(i,j,k) * J(k) - G(i,j)
  for(size_t i = 0; i < G_dims[0]; i++) {
      for(size_t j = 0; j < G_dims[1]; j++) {
          size_t jG = i * G_dims[1] + j;
          size_t jxyK = i * xyK_dims[1] + j;
          size_t jG0 = i * G_dims[1] + j;
          double tk_val = 0.0;
          for(size_t k = 0; k < J_dim; k++) {
              size_t kxyK = jxyK * xyK_dims[2] + k;
              tk_val += (2 * xyK[kxyK] * J[k]);
          }
          G[jG] = tk_val - G[jG0];
      }
  }
  // copy result to an Eigen::Matrix
  tonto::MatRM result(n, n);
  std::copy(G.data(), G.data() + G.size(), result.data());
  return result;
}


}
