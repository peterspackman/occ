#pragma once
#include <tonto/qm/ints.h>
#include <unsupported/Eigen/CXX11/Tensor>


namespace tonto::df {
using tonto::qm::BasisSet;

struct DFFockEngine {
  BasisSet obs;
  BasisSet dfbs;
  DFFockEngine(const BasisSet& _obs, const BasisSet& _dfbs)
      : obs(_obs), dfbs(_dfbs) {}

  std::array<size_t, 3> xyK_dims;
  std::vector<double> xyK;

  // a DF-based builder, using coefficients of occupied MOs
  tonto::MatRM compute_2body_fock_dfC(const tonto::MatRM& Cocc);
};

}
