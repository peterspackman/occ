#pragma once
#include <occ/core/diis.h>

namespace occ::qm {

class CDIIS : public occ::core::diis::DIIS {
public:
  Mat update(const Mat &overlap, const Mat &D, const Mat &F);
  double max_error() const { return m_max_error; }
  double min_error() const { return m_min_error; }

private:
  double m_max_error{0.0};
  double m_min_error{0.0};
};

} // namespace occ::qm
