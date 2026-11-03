#pragma once
#include <occ/core/diis.h>

namespace occ::qm {

class CDIIS : public occ::core::diis::DIIS {
public:
  // Extrapolate as soon as there are two vectors. A later start leaves the
  // first cycles after every reset (e.g. a COSX grid switch) as plain
  // Roothaan steps, which drift until extrapolation begins.
  explicit CDIIS(size_t start = 2, size_t subspace = 8)
      : occ::core::diis::DIIS(start, subspace) {}
  Mat update(const Mat &overlap, const Mat &D, const Mat &F);
  double max_error() const { return m_max_error; }
  double min_error() const { return m_min_error; }

private:
  double m_max_error{0.0};
  double m_min_error{0.0};
};

} // namespace occ::qm
