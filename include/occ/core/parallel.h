#pragma once
#include <thread>
#include <vector>

namespace occ::parallel {
inline int nthreads = 1;

template<typename Lambda>
void parallel_do(Lambda &lambda) {
    std::vector<std::thread> threads;
    for(int thread_id = 0; thread_id < nthreads; thread_id++) {
        if(thread_id != nthreads - 1) threads.push_back(std::thread(lambda, thread_id));
        else lambda(thread_id);
    }
    for(int thread_id = 0; thread_id < threads.size(); thread_id++) threads[thread_id].join();
}

}
