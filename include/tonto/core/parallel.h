#pragma once
#include <omp.h>

namespace tonto::parallel {
inline int nthreads = 1;

template<typename Lambda>
void parallel_do(Lambda &lambda) {
    omp_set_num_threads(nthreads);
    #pragma omp parallel
    {
        auto thread_id = omp_get_thread_num();
        lambda(thread_id);
    }
}

}
