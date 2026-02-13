#pragma once
#include <algorithm>
#include <memory>
#include <occ/core/timings.h>
#include <thread>
#include <type_traits>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>

namespace occ::parallel {
inline int nthreads = 1;

// Proper TBB global control management
inline std::unique_ptr<tbb::global_control>& get_tbb_control() {
  static std::unique_ptr<tbb::global_control> control;
  return control;
}

inline void shutdown_tbb() {
  get_tbb_control().reset();
}

inline void set_num_threads(int threads) {
  nthreads = threads;
  // Reset any existing control before creating a new one
  auto& control = get_tbb_control();
  control.reset(new tbb::global_control(
      tbb::global_control::max_allowed_parallelism, nthreads));
}

inline int get_num_threads() { return nthreads; }

// Efficient parallel for loop with automatic load balancing
template <typename Lambda>
void parallel_for(size_t begin, size_t end, Lambda &&lambda,
                  size_t grainsize = 0) {
  if (grainsize == 0) {
    // Use grainsize=1 to let TBB's work stealing handle load balancing
    // TBB is smart about chunking internally
    grainsize = 1;
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, grainsize),
                    [&lambda](const tbb::blocked_range<size_t> &range) {
                      for (size_t i = range.begin(); i != range.end(); ++i) {
                        lambda(i);
                      }
                    });
}

// Parallel reduction with custom reduction operation
template <typename T, typename Lambda, typename Reduction>
T parallel_reduce(size_t begin, size_t end, T init, Lambda &&lambda,
                  Reduction &&reduction, size_t grainsize = 0) {
  if (grainsize == 0) {
    grainsize = std::max(size_t(1), (end - begin) / (nthreads * 4));
  }
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(begin, end, grainsize), init,
      [&lambda](const tbb::blocked_range<size_t> &range, T value) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          value = lambda(i, value);
        }
        return value;
      },
      reduction);
}

// 2D parallel for loop for nested loops (useful for shell pairs)
template <typename Lambda>
void parallel_for_2d(size_t rows_begin, size_t rows_end, size_t cols_begin,
                     size_t cols_end, Lambda &&lambda, size_t row_grainsize = 0,
                     size_t col_grainsize = 0) {
  if (row_grainsize == 0) {
    row_grainsize =
        std::max(size_t(1), (rows_end - rows_begin) / (nthreads * 2));
  }
  if (col_grainsize == 0) {
    col_grainsize =
        std::max(size_t(1), (cols_end - cols_begin) / (nthreads * 2));
  }
  tbb::parallel_for(
      tbb::blocked_range2d<size_t>(rows_begin, rows_end, row_grainsize,
                                   cols_begin, cols_end, col_grainsize),
      [&lambda](const tbb::blocked_range2d<size_t> &range) {
        for (size_t i = range.rows().begin(); i != range.rows().end(); ++i) {
          for (size_t j = range.cols().begin(); j != range.cols().end(); ++j) {
            lambda(i, j);
          }
        }
      });
}

// Dynamic task group for irregular parallelism
template <typename Lambda>
void parallel_invoke(Lambda &&lambda1, Lambda &&lambda2) {
  tbb::parallel_invoke(lambda1, lambda2);
}

template <typename... Lambdas> void parallel_invoke(Lambdas &&...lambdas) {
  tbb::parallel_invoke(std::forward<Lambdas>(lambdas)...);
}

// Thread-local storage type alias
template <typename T>
using thread_local_storage = tbb::enumerable_thread_specific<T>;

// Helper template for parallel accumulation pattern
// This encapsulates the common pattern of:
// 1. Create thread-local accumulators
// 2. Process items in parallel
// 3. Reduce/combine results
template <typename T, typename WorkItems, typename ProcessFunc,
          typename CombineFunc>
T parallel_accumulate(const WorkItems &items, T init_value,
                      ProcessFunc &&process, CombineFunc &&combine) {
  thread_local_storage<T> local_results(init_value);

  parallel_for(size_t(0), items.size(), [&](size_t i) {
    auto &local = local_results.local();
    process(items[i], local, i);
  });

  T result = init_value;
  for (const auto &local : local_results) {
    result = combine(result, local);
  }
  return result;
}

// Specialization for index-based parallel accumulation
template <typename T, typename ProcessFunc, typename CombineFunc>
T parallel_accumulate_indexed(size_t num_items, T init_value,
                              ProcessFunc &&process, CombineFunc &&combine) {
  thread_local_storage<T> local_results(init_value);

  parallel_for(size_t(0), num_items, [&](size_t i) {
    auto &local = local_results.local();
    process(i, local);
  });

  T result = init_value;
  for (const auto &local : local_results) {
    result = combine(result, local);
  }
  return result;
}

} // namespace occ::parallel
