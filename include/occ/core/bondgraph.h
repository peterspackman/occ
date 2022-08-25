#pragma once
#include <occ/core/graph.h>

namespace occ::core::graph {

struct PeriodicEdge {
    double dist{0.0};
    size_t source{0}, target{0};
    size_t source_asym_idx{0}, target_asym_idx{0};
    int h{0}, k{0}, l{0};
};
struct PeriodicVertex {
    size_t uc_idx{0};
};

using PeriodicBondGraph = Graph<PeriodicVertex, PeriodicEdge>;
} // namespace occ::core::graph
