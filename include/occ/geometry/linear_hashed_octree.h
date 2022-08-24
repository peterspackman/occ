#pragma once
#include <deque>
#include <occ/3rdparty/parallel_hashmap/phmap.h>
#include <occ/geometry/morton.h>
#include <optional>
#include <vector>

namespace occ::geometry {

template <typename N> struct LinearHashedOctree {
    using NodeMap = phmap::flat_hash_map<MIndex, N>;
    NodeMap nodes;
    std::vector<MIndex> leaves;

    LinearHashedOctree() : nodes{}, leaves{} {}

    template <class R, class C>
    void build(R &should_refine, C &construct_node) {
        std::deque<MIndex> queue;
        queue.push_back(MIndex{});

        while (!queue.empty()) {
            auto key = queue.front();
            queue.pop_front();
            auto node = construct_node(key);

            if (should_refine(key, node)) {
                for (uint_fast8_t i = 0; i < 8; ++i)
                    queue.push_back(key.child(i));
            } else {
                leaves.push_back(key);
            }
            nodes[key] = node;
        }
    }

    template <class V> void visit_leaves(V &visitor) const {
        for (const auto &key : leaves) {
            visitor(key);
        }
    }

    inline std::optional<N> get(MIndex key) const {
        auto search = nodes.find(key);
        if (search == nodes.end())
            return std::nullopt;
        return search->second;
    }
};

} // namespace occ::geometry
