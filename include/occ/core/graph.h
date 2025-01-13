#pragma once
#include <ankerl/unordered_dense.h>
#include <optional>
#include <queue>
#include <stack>

namespace occ::core::graph {

template <typename VertexType, typename EdgeType> class Graph {

public:
  using VertexDescriptor = std::size_t;
  using EdgeDescriptor = std::size_t;

  using Edges = ankerl::unordered_dense::map<EdgeDescriptor, EdgeType>;
  using Vertices = ankerl::unordered_dense::map<VertexDescriptor, VertexType>;

  using NeighborList =
      ankerl::unordered_dense::map<VertexDescriptor, EdgeDescriptor>;
  using AdjacencyList =
      ankerl::unordered_dense::map<VertexDescriptor, NeighborList>;

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
                          const EdgeType &edge, bool bidirectional = false) {
    EdgeType e = edge;
    return add_edge(source, target, std::move(e), bidirectional);
  }

  EdgeDescriptor add_edge(VertexDescriptor source, VertexDescriptor target,
                          EdgeType &&edge, bool bidirectional = false) {
    auto &s_neighbors = m_adjacency_list[source];
    s_neighbors.insert({target, m_current_edge_descriptor});
    if (bidirectional) {
      auto &t_neighbors = m_adjacency_list[target];
      t_neighbors.insert({source, m_current_edge_descriptor});
    }
    m_edges.insert({m_current_edge_descriptor, edge});
    return m_current_edge_descriptor++;
  }

  const EdgeType &edge(EdgeDescriptor e) const { return m_edges[e]; }
  const VertexType &vertex(VertexDescriptor v) const { return m_vertices[v]; }

  const auto &vertices() const { return m_vertices; }
  const auto &edges() const { return m_edges; }
  const auto &adjacency_list() const { return m_adjacency_list; }

  const auto neighbors(VertexDescriptor v) const {
    return m_adjacency_list.find(v);
  }
  auto vertex(VertexDescriptor v) { return m_vertices.find(v); }
  auto edge(EdgeDescriptor e) { return m_edges.find(e); }
  auto neighbors(VertexDescriptor v) { return m_adjacency_list.find(v); }

  bool is_connected(VertexDescriptor i, VertexDescriptor j) const {
    if (m_adjacency_list.at(i).contains(j))
      return true;
    if (m_adjacency_list.at(j).contains(i))
      return true;
    return false;
  }

  template <typename T>
  void depth_first_traversal(VertexDescriptor source, T &func) const {
    ankerl::unordered_dense::set<VertexDescriptor> visited;
    std::stack<VertexDescriptor> store;
    store.push(source);
    while (!store.empty()) {
      auto s = store.top();
      store.pop();
      if (visited.contains(s))
        continue;

      visited.insert(s);
      func(s);
      for (const auto &kv : m_adjacency_list.at(s)) {
        store.push(kv.first);
      }
    }
  }

  template <typename T>
  void breadth_first_traversal(VertexDescriptor source, T &func) const {
    ankerl::unordered_dense::set<VertexDescriptor> visited;
    std::queue<VertexDescriptor> store;
    store.push(source);
    while (!store.empty()) {
      auto s = store.front();
      store.pop();
      if (visited.contains(s))
        continue;

      visited.insert(s);
      func(s);
      for (const auto &kv : m_adjacency_list.at(s)) {
        store.push(kv.first);
      }
    }
  }

  template <typename T>
  void breadth_first_traversal_with_edge(VertexDescriptor source,
                                         T &func) const {
    ankerl::unordered_dense::set<VertexDescriptor> visited;
    std::queue<std::tuple<VertexDescriptor, VertexDescriptor, EdgeDescriptor>>
        store;
    store.push({source, source, 0});
    while (!store.empty()) {
      auto [s, predecessor, edge] = store.front();
      store.pop();
      if (visited.contains(s))
        continue;

      visited.insert(s);
      func(s, predecessor, edge);
      for (const auto &kv : m_adjacency_list.at(s)) {
        store.push({kv.first, s, kv.second});
      }
    }
  }

  template <typename T> void connected_component_traversal(T &func) {
    ankerl::unordered_dense::set<VertexDescriptor> visited;
    size_t current_component{0};
    auto call_with_component = [&current_component,
                                &func](const VertexDescriptor &desc) {
      func(desc, current_component);
    };

    for (const auto &v : m_vertices) {
      if (visited.contains(v.first))
        continue;
      breadth_first_traversal(v.first, call_with_component);
      current_component++;
    }
  }

  auto connected_components() const {
    ankerl::unordered_dense::map<VertexDescriptor, size_t> components;
    size_t current_component{0};
    auto set_component = [&current_component,
                          &components](const VertexDescriptor &desc) {
      components.insert({desc, current_component});
    };
    for (const auto &v : m_vertices) {
      if (components.find(v.first) != components.end())
        continue;
      depth_first_traversal(v.first, set_component);
      current_component++;
    }
    return components;
  }

  size_t num_edges() const { return m_edges.size(); }
  size_t num_vertices() const { return m_vertices.size(); }

private:
  VertexDescriptor m_current_vertex_descriptor{0};
  EdgeDescriptor m_current_edge_descriptor{0};
  AdjacencyList m_adjacency_list;
  Vertices m_vertices;
  Edges m_edges;
};

} // namespace occ::core::graph
