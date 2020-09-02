#pragma once
#include <QVector>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/connected_components.hpp>

class PeriodicBondGraph {
public:
    using EIndex = size_t;
    using VIndex = size_t;

    struct Edge {
        double dist{0.0};
        size_t source{0}, target{0};
        size_t source_asym_idx{0}, target_asym_idx{0};
        int h{0}, k{0}, l{0};
    };
    struct Vertex {
        VIndex uc_idx{0};
    };
    using GraphContainer = boost::adjacency_list<boost::setS, boost::vecS, boost::directedS, Vertex, Edge>;
    using vertex_t = boost::graph_traits<GraphContainer>::vertex_descriptor;
    using edge_t = boost::graph_traits<GraphContainer>::edge_descriptor;
    using edge_pair = std::pair<edge_t, edge_t>;
    using vertex_iter = boost::graph_traits<GraphContainer>::vertex_iterator;
    using edge_iter = boost::graph_traits<GraphContainer>::edge_iterator;
    using adjacency_iter = boost::graph_traits<GraphContainer>::adjacency_iterator;
    using degree_t = boost::graph_traits<GraphContainer>::degree_size_type;
    using adjacency_vertex_range_t = std::pair<adjacency_iter, adjacency_iter>;
    using vertex_range_t = std::pair<vertex_iter, vertex_iter>;
    using edge_range_t = std::pair<edge_iter, edge_iter>;

    PeriodicBondGraph();

    void clear() { m_graph.clear(); m_vertices.clear(); m_edges.clear(); }

    void addBond(const Edge&);
    bool isBonded(VIndex i, VIndex j) const;
    size_t numEdges() const { return m_edges.size(); }
    size_t numVertices() const { return m_vertices.size(); }
    inline vertex_t vertexHandle(VIndex i) const { return m_vertices[i]; }
    vertex_range_t vertices() const { return boost::vertices(m_graph); }
    edge_range_t edges() const { return boost::edges(m_graph); }
    QVector<VIndex> neighbors(VIndex i) const;
    std::optional<Edge> edge(VIndex i, VIndex j) {
        auto [e, b] = boost::edge(m_vertices[i], m_vertices[j], m_graph);
        if(b) return m_graph[e];
        else { return std::nullopt; }
    }
    auto& graph() { return m_graph; }

    auto connectedComponents() const {
        QVector<int> component(boost::num_vertices(m_graph));
        int num_components = boost::connected_components(m_graph, &component[0]);
        return std::make_pair(num_components, component);
    }
private:
    QVector<vertex_t> m_vertices;
    QVector<edge_t> m_edges;
    GraphContainer m_graph;
};
