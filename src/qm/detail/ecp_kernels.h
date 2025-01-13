#pragma once
#if HAVE_ECPINT

#include "kernel_traits.h"
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

template <typename Lambda>
void evaluate_two_center_ecp_with_shellpairs(
    Lambda &f, const std::vector<libecpint::GaussianShell> &shells,
    const std::vector<libecpint::ECP> &ecps, int lmax1, int lmax2,
    const ShellPairList &shellpairs, int thread_id = 0) {
  // TODO maybe share this? not sure if it's thread safe.
  libecpint::ECPIntegral ecp_integral(lmax1, lmax2, 0);
  using Result = IntegralEngine::IntegralResult<2>;
  auto nthreads = occ::parallel::get_num_threads();

  std::array<int, 2> dims;
  std::array<int, 2> bfs{-1, -1}; // ignore this

  libecpint::TwoIndex<double> tmp;
  for (int p = 0, pq = 0; p < shells.size(); p++) {
    const auto &sh1 = shells[p];

    dims[0] = sh1.ncartesian();

    for (const auto &q : shellpairs[p]) {
      if (pq++ % nthreads != thread_id)
        continue;
      const auto &sh2 = shells[q];
      dims[1] = sh2.ncartesian();
      libecpint::TwoIndex<double> buffer(dims[0], dims[1], 0.0);

      std::array<int, 2> idxs{p, static_cast<int>(q)};
      for (const auto &U : ecps) {
        ecp_integral.compute_shell_pair(U, sh1, sh2, tmp);
        buffer.add(tmp);
      }
      Result args{thread_id, idxs, bfs, dims, buffer.data.data()};
      f(args);
    }
  }
}

template <ShellKind kind = ShellKind::Cartesian>
Mat ecp_operator_kernel(const AOBasis &aobasis,
                        const std::vector<libecpint::GaussianShell> &aoshells,
                        const std::vector<libecpint::ECP> &ecps, int ao_max_l,
                        int ecp_max_l, const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<2>;
  auto nthreads = occ::parallel::get_num_threads();

  std::vector<Mat> results;
  results.emplace_back(Mat::Zero(aobasis.nbf(), aobasis.nbf()));
  for (size_t i = 1; i < nthreads; i++) {
    results.push_back(results[0]);
  }

  if constexpr (kind == ShellKind::Spherical) {
    std::vector<Mat> cart2sph;
    for (int i = 0; i <= ao_max_l; i++) {
      cart2sph.push_back(
          occ::gto::cartesian_to_spherical_transformation_matrix(i));
    }

    auto f = [&results, &aobasis, &cart2sph](const Result &args) {
      auto &result = results[args.thread];
      Eigen::Map<const occ::MatRM> tmp(args.buffer, args.dims[0], args.dims[1]);
      const int bf0 = aobasis.first_bf()[args.shell[0]];
      const int bf1 = aobasis.first_bf()[args.shell[1]];
      const int dim0 = aobasis[args.shell[0]].size();
      const int dim1 = aobasis[args.shell[1]].size();
      const int l0 = aobasis[args.shell[0]].l;
      const int l1 = aobasis[args.shell[1]].l;
      result.block(bf0, bf1, dim0, dim1) =
          cart2sph[l0] * tmp * cart2sph[l1].transpose();
      if (args.shell[0] != args.shell[1]) {
        result.block(bf1, bf0, dim1, dim0) =
            result.block(bf0, bf1, dim0, dim1).transpose();
      }
    };
    auto lambda = [&](int thread_id) {
      evaluate_two_center_ecp_with_shellpairs(f, aoshells, ecps, ao_max_l,
                                              ecp_max_l, shellpairs, thread_id);
    };
    occ::parallel::parallel_do(lambda);

  } else {
    auto f = [&results, &aobasis](const Result &args) {
      auto &result = results[args.thread];
      Eigen::Map<const occ::MatRM> tmp(args.buffer, args.dims[0], args.dims[1]);
      const int bf0 = aobasis.first_bf()[args.shell[0]];
      const int bf1 = aobasis.first_bf()[args.shell[1]];

      result.block(bf0, bf1, args.dims[0], args.dims[1]) = tmp;
      if (args.shell[0] != args.shell[1]) {
        result.block(bf1, bf0, args.dims[1], args.dims[0]) = tmp.transpose();
      }
    };
    auto lambda = [&](int thread_id) {
      evaluate_two_center_ecp_with_shellpairs(f, aoshells, ecps, ao_max_l,
                                              ecp_max_l, shellpairs, thread_id);
    };
    occ::parallel::parallel_do(lambda);
  };

  for (auto i = 1; i < nthreads; ++i) {
    results[0].noalias() += results[i];
  }
  return results[0];
}

} // namespace occ::qm::detail
#endif
