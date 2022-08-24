#pragma once
#include <occ/3rdparty/parallel_hashmap/phmap.h>
#include <optional>
#include <queue>
#include <stack>

namespace occ::core::graph {

template <typename VertexType, typename EdgeType> class Graph {

  public:
    using VertexDescriptor = std::size_t;
    using EdgeDescriptor = std::size_t;

    using Edges = phmap::flat_hash_map<EdgeDescriptor, EdgeType>;
    using Vertices = phmap::flat_hash_map<VertexDescriptor, VertexType>;

    using NeighborList = phmap::flat_hash_map<VertexDescriptor, EdgeDescriptor>;
    using AdjacencyList = phmap::flat_hash_map<VertexDescriptor, NeighborList>;

    Graph() = default;

    inline size_t size() const { return m_vertices.size(); }

    VertexDescriptor add_vertex(const VertexType &vertex) {
        VertexType v = vertex;
        return add_vertex(std::move(v));
    }

    VertexDescriptor add_vertex(VertexType &&vertex) {
        m_vertices.insert({m_current_vertex_descriptor, vertex});
        m_adjacency_list[m_current_vertex_descriptor] = {};
        return m_current_vertex_descriptor++;
    }

    EdgeDescriptor add_edge(VertexDescriptor source, VertexDescriptor target,
                            const EdgeType &edge) {
        EdgeType e = edge;
        return add_edge(source, target, std::move(e));
    }

    EdgeDescriptor add_edge(VertexDescriptor source, VertexDescriptor target,
                            EdgeType &&edge) {
        auto s = m_adjacency_list.find(source);

        if (s == m_adjacency_list.end()) {
            throw std::runtime_error("No such source vertex");
        }

        auto t = m_adjacency_list.find(target);
        if (t == m_adjacency_list.end()) {
            throw std::runtime_error("No such target vertex");
        }

        s->second.insert({target, m_current_edge_descriptor});
        t->second.insert({source, m_current_edge_descriptor});
        m_edges.insert({m_current_edge_descriptor, edge});
        return m_current_edge_descriptor++;
    }

    const auto &vertices() const { return m_vertices; }
    const auto &edges() const { return m_edges; }
    const auto &adjacency_list() const { return m_adjacency_list; }

    const auto vertex(VertexDescriptor v) const { return m_vertices.find(v); }
    const auto edge(EdgeDescriptor e) const { return m_edges.find(e); }
    const auto neighbors(VertexDescriptor v) const {
        return m_adjacency_list.find(v);
    }
    auto vertex(VertexDescriptor v) { return m_vertices.find(v); }
    auto edge(EdgeDescriptor e) { return m_edges.find(e); }
    auto neighbors(VertexDescriptor v) { return m_adjacency_list.find(v); }

    template <typename T>
    void depth_first_traversal(VertexDescriptor source, T &func) {
        phmap::flat_hash_set<VertexDescriptor> visited;
        std::stack<VertexDescriptor> store;
        store.push(source);
        while (!store.empty()) {
            const auto s = store.top();
            store.pop();
            if (visited.contains(s))
                continue;

            visited.insert(s);
            func(s);
            for (const auto &kv : m_adjacency_list[s]) {
                store.push(kv.first);
            }
        }
    }

    template <typename T>
    void breadth_first_traversal(VertexDescriptor source, T &func) {
        phmap::flat_hash_set<VertexDescriptor> visited;
        std::queue<VertexDescriptor> store;
        store.push(source);
        while (!store.empty()) {
            const auto s = store.front();
            store.pop();
            if (visited.contains(s))
                continue;

            visited.insert(s);
            func(s);
            for (const auto &kv : m_adjacency_list[s]) {
                store.push(kv.first);
            }
        }
    }

  private:
    VertexDescriptor m_current_vertex_descriptor{0};
    EdgeDescriptor m_current_edge_descriptor{0};
    AdjacencyList m_adjacency_list;
    Vertices m_vertices;
    Edges m_edges;
};

} // namespace occ::core::graph
