#pragma once
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/connected_components.hpp>

namespace craso::graph {

template<typename Vertex, typename Edge>
class BondGraph {
public:
    using GraphContainer = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, Vertex, Edge>;
    using vertex_t = typename boost::graph_traits<GraphContainer>::vertex_descriptor;
    using edge_t = typename boost::graph_traits<GraphContainer>::edge_descriptor;
    using edge_pair = std::pair<edge_t, edge_t>;
    using vertex_iter = typename boost::graph_traits<GraphContainer>::vertex_iterator;
    using edge_iter = typename boost::graph_traits<GraphContainer>::edge_iterator;
    using adjacency_iter = typename boost::graph_traits<GraphContainer>::adjacency_iterator;
    using degree_t = typename boost::graph_traits<GraphContainer>::degree_size_type;
    using adjacency_vertex_range_t = std::pair<adjacency_iter, adjacency_iter>;
    using vertex_range_t = std::pair<vertex_iter, vertex_iter>;
    using edge_range_t = std::pair<edge_iter, edge_iter>;
    using edge_property_t = typename boost::edge_bundle_type<GraphContainer>::type;
    using vertex_property_t = typename boost::vertex_bundle_type<GraphContainer>::type;
    BondGraph() {}

    void clear() { m_graph.clear(); }

    vertex_t add_vertex(const vertex_property_t& v) {
        return boost::add_vertex(v, m_graph);
    }

    std::optional<edge_t> add_edge(vertex_t i, vertex_t j, const edge_property_t& edge) {
        bool b; edge_t e;
        std::tie(e, b) = boost::add_edge(
            i, j, edge, m_graph
        );
        if(b) return e;
        else { return std::nullopt;}
    }

    bool is_connected(vertex_t i, vertex_t j) const
    {
        auto [e, b] = boost::edge(i, j, m_graph);
        return b;
    }

    size_t num_edges() const { return boost::num_edges(m_graph); }
    size_t num_vertices() const { return boost::num_vertices(m_graph); }
    vertex_range_t vertices() const { return boost::vertices(m_graph); }
    edge_range_t edges() const { return boost::edges(m_graph); }

    std::vector<Vertex> neighbor_list(vertex_t i) const {
        std::vector<Vertex> n;
        typename boost::graph_traits<GraphContainer>::adjacency_iterator vi, vi_end;
        for(std::tie(vi, vi_end) = boost::adjacent_vertices(i, m_graph); vi != vi_end; ++vi)
        {
            n.push_back(m_graph[*vi]);
        }
        return n;
    }

    std::optional<Edge> edge(vertex_t i, vertex_t j) {
        auto [e, b] = boost::edge(i, j, m_graph);
        if(b) return m_graph[e];
        else { return std::nullopt; }
    }

    auto& graph() { return m_graph; }

    auto connected_components() const {
        std::vector<int> component(boost::num_vertices(m_graph));
        int num_components = boost::connected_components(m_graph, &component[0]);
        return std::make_pair(num_components, component);
    }

private:
    GraphContainer m_graph;
};

struct PeriodicEdge {
    double dist{0.0};
    size_t source{0}, target{0};
    size_t source_asym_idx{0}, target_asym_idx{0};
    int h{0}, k{0}, l{0};
};
struct PeriodicVertex {
    size_t uc_idx{0};
};

using PeriodicBondGraph = BondGraph<PeriodicVertex, PeriodicEdge>;
}
