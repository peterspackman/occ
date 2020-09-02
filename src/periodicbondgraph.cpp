#include "periodicbondgraph.h"

PeriodicBondGraph::PeriodicBondGraph()
{
}

void PeriodicBondGraph::addBond(const Edge &new_edge) {
    size_t n = std::max(new_edge.source, new_edge.target);
    for(size_t i = numVertices(); i <= n; i++) {
        vertex_t v = boost::add_vertex(m_graph);
        m_vertices.push_back(v);
        m_graph[v].uc_idx = i;
    }
    vertex_t l = m_vertices[new_edge.source], r = m_vertices[new_edge.target];
    edge_t e; bool b;

    std::tie(e, b) = boost::add_edge(l, r, m_graph);
    auto& e1 = m_graph[e];
    e1.dist = new_edge.dist;
    e1.source = new_edge.source;
    e1.target = new_edge.target;
    e1.source_asym_idx = new_edge.source_asym_idx;
    e1.target_asym_idx = new_edge.target_asym_idx;
    e1.h = new_edge.h;
    e1.k = new_edge.k;
    e1.l = new_edge.l;

    std::tie(e, b) = boost::add_edge(r, l, m_graph);
    auto& e2 = m_graph[e];
    e2.dist = new_edge.dist;
    e2.source = new_edge.target;
    e2.target = new_edge.source;
    e2.source_asym_idx = new_edge.target_asym_idx;
    e2.target_asym_idx = new_edge.source_asym_idx;
    e2.h = - new_edge.h;
    e2.k = - new_edge.k;
    e2.l = - new_edge.l;
}

bool PeriodicBondGraph::isBonded(VIndex i, VIndex j) const
{
    edge_t e; bool b;
    std::tie(e, b) = boost::edge(m_vertices[i], m_vertices[j], m_graph);
    return b;
}

QVector<PeriodicBondGraph::VIndex> PeriodicBondGraph::neighbors(VIndex i) const
{
    QVector<size_t> n;
    typename boost::graph_traits<GraphContainer>::adjacency_iterator vi, vi_end;
    for(std::tie(vi, vi_end) = boost::adjacent_vertices(m_vertices[i], m_graph); vi != vi_end; ++vi)
    {
        n.push_back(m_graph[*vi].uc_idx);
    }
    return n;
}
