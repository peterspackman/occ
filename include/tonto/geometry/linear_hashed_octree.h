#pragma once
#include <vector>
#include <deque>
#include <tonto/geometry/morton.h>
#include <tonto/3rdparty/robin_hood.h>

namespace tonto::geometry
{

template<typename N>
struct LinearHashedOctree
{
    using NodeMap = robin_hood::unordered_flat_map<MIndex, N, MIndexHash>;
    NodeMap nodes;
    std::vector<MIndex> leaves;

    LinearHashedOctree() : nodes{}, leaves{} {}

    template<class R, class C>
    void build(R &should_refine, C &construct_node)
    {
        std::deque<MIndex> queue;
        queue.push_back(MIndex{});

        while (!queue.empty())
        {
            auto key = queue.front();
            queue.pop_front();
            auto node = construct_node(key);
            if(should_refine(key, node))
            {
                for(uint_fast8_t i = 0; i < 8; ++i) queue.push_back(key.child(i));
            }
            else
            {
                leaves.push_back(key);
            }
            nodes[key] = node;
        }
    }

    template<class V>
    void visit_leaves(V &visitor) const
    {
        for(auto &key: leaves)
        {
            visitor(key);
        }
    }

    inline std::optional<N> get(MIndex key) const
    {
        if (nodes.contains(key)) return nodes.at(key);
        else return std::nullopt;
    }
};

}
