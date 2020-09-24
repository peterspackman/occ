#pragma once
#include <omp.h>

namespace craso::parallel {
inline int nthreads = 1;

/// fires off \c nthreads instances of lambda in parallel
template <typename Lambda> void parallel_do(Lambda &lambda) {
#pragma omp parallel
  {
    auto thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
}

} // namespace craso::parallel