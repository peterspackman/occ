#pragma once
#include "ints.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace tonto::df {
using libint2::BasisSet;

struct DFFockEngine {
  const BasisSet& obs;
  const BasisSet& dfbs;
  DFFockEngine(const BasisSet& _obs, const BasisSet& _dfbs)
      : obs(_obs), dfbs(_dfbs) {}

  Eigen::Tensor<double, 3> xyK;

  // a DF-based builder, using coefficients of occupied MOs
  tonto::MatRM compute_2body_fock_dfC(const tonto::MatRM& Cocc);
};

}
