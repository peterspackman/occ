#pragma once
#include <trajan/core/trajectory.h>

namespace trajan::core {

class Analysis {
public:
  virtual void analyze(const trajan::core::Trajectory &trajectory) = 0;
};
}; // namespace trajan::core
