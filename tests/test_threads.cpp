// Linked into every test executable. When OCC_TEST_THREADS is set (ctest
// sets it, see CMakeLists.txt) each test process caps its parallelism, so
// `ctest -jN` runs N processes instead of N processes that each claim every
// core.
#include <algorithm>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <cstdlib>
#include <occ/core/parallel.h>

namespace {

class ThreadLimit : public Catch::EventListenerBase {
public:
  using Catch::EventListenerBase::EventListenerBase;
  void testRunStarting(const Catch::TestRunInfo &) override {
    if (const char *n = std::getenv("OCC_TEST_THREADS"))
      occ::parallel::set_num_threads(std::max(1, std::atoi(n)));
  }
};

} // namespace

CATCH_REGISTER_LISTENER(ThreadLimit)
