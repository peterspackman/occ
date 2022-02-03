#pragma once
#include <thread>
#include <vector>

namespace occ::parallel {
inline int nthreads = 1;

template <typename Lambda> void parallel_do(Lambda &lambda) {
    std::vector<std::thread> threads;
    for (int thread_id = 0; thread_id < nthreads; thread_id++) {
        if (thread_id != 0)
            threads.push_back(std::move(std::thread(lambda, thread_id)));
        else
            lambda(thread_id);
    }
    for (auto &thread: threads)
        thread.join();
}

} // namespace occ::parallel
