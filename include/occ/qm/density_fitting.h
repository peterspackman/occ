#pragma once
#include <occ/qm/ints.h>
#include <unsupported/Eigen/CXX11/Tensor>


namespace occ::df {
using occ::qm::BasisSet;

struct DFFockEngine {
  BasisSet obs;
  BasisSet dfbs;
  DFFockEngine(const BasisSet& _obs, const BasisSet& _dfbs)
      : obs(_obs), dfbs(_dfbs) {}

  std::array<size_t, 3> xyK_dims;
  std::vector<double> xyK;

  // a DF-based builder, using coefficients of occupied MOs
  Mat compute_2body_fock_dfC(const Mat& Cocc);
};

}
