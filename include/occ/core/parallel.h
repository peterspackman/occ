#pragma once
#include <occ/core/timings.h>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace occ::parallel {
inline int nthreads = 1;

inline void set_num_threads(int threads) {
  nthreads = threads;
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
}

inline int get_num_threads() { return nthreads; }

template <typename Lambda> void parallel_do(Lambda &lambda) {

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
  {
    auto thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
#else
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < nthreads; thread_id++) {
    if (thread_id != 0)
      threads.push_back(std::thread(lambda, thread_id));
    else
      lambda(thread_id);
  }
  for (auto &thread : threads)
    thread.join();
#endif
}

template <typename Lambda>
void parallel_do_timed(Lambda &lambda, occ::timing::category timing_category) {
  occ::timing::start(timing_category);
  parallel_do(lambda);
  occ::timing::stop(timing_category);
}

} // namespace occ::parallel
