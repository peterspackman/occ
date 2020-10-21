#include "density_fitting.h"
#include "parallel.h"
#include <libint2.hpp>

namespace tonto::df {

tonto::MatRM DFFockEngine::compute_2body_fock_dfC(const tonto::MatRM& Cocc) {

  using tonto::parallel::nthreads;

  const auto n = obs.nbf();
  const auto ndf = dfbs.nbf();

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

    Eigen::Tensor<double, 3> Zxy{ndf, n, n};

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

            Eigen::Sizes<3> lower_bound(bf1_first, bf2_first, bf3_first);
            Eigen::Sizes<3> nbf123(n1, n2, n3);
            Zxy.slice(lower_bound, nbf123) = Eigen::TensorMap<const Eigen::Tensor<const double, 3>>(buf, n1, n2, n3);
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

    Eigen::Tensor<double, 2> K(ndf, ndf);
    std::copy(Linv_t.data(), Linv_t.data() + ndf * ndf, K.data());

    xyK = Eigen::Tensor<double, 3>(n, n, ndf);

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(0, 0)};
    xyK = Zxy.contract(K, product_dims);
    Zxy = Eigen::Tensor<double, 3>(0, 0, 0);  // release memory
  }  // if (xyK.size() == 0)

  // compute exchange
  const auto nocc = Cocc.cols();
  Eigen::Tensor<double, 2> Co{n, nocc};
  std::copy(Cocc.data(), Cocc.data() + n * nocc, Co.data());
  Eigen::array<Eigen::IndexPair<int>, 1> dims1 = {Eigen::IndexPair<int>(1, 1)};
  Eigen::array<int, 2> shuf1({1, 2});
  Eigen::Tensor<double, 3> xiK = xyK.contract(Co, dims1).shuffle(shuf1);

  //exchange
  Eigen::array<Eigen::IndexPair<int>, 2> dims2 = {Eigen::IndexPair<int>({1, 1}), Eigen::IndexPair<int>({2, 2})};
  Eigen::Tensor<double, 2> G = xiK.contract(xiK, dims2);

  //coulomb
  Eigen::array<Eigen::IndexPair<int>, 2> dims3 = {Eigen::IndexPair<int>({0, 0}), Eigen::IndexPair<int>({1, 1})};
  Eigen::Tensor<double, 1> Jtmp = xiK.contract(Co, dims3);
  Eigen::array<Eigen::IndexPair<int>, 1> dims4 = {Eigen::IndexPair<int>({2, 0})};
  G = G - 2 * xyK.contract(Jtmp, dims4);

  // copy result to an Eigen::Matrix
  tonto::MatRM result(n, n);
  std::copy(G.data(), G.data() + G.size(), result.data());
  return result;
}


}
