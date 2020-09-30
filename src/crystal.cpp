#include "crystal.h"
#include "element.h"
#include "kdtree.h"
#include <Eigen/Dense>
#include <iostream>

namespace tonto::crystal {

using tonto::graph::BondGraph;

Crystal::Crystal(const AsymmetricUnit &asym, const SpaceGroup &sg,
                 const UnitCell &uc)
    : m_asymmetric_unit(asym), m_space_group(sg), m_unit_cell(uc) {}

std::string AsymmetricUnit::chemical_formula() const {
  std::vector<tonto::chem::Element> els;
  for (int i = 0; i < atomic_numbers.size(); i++) {
    els.push_back(tonto::chem::Element(atomic_numbers[i]));
  }
  return tonto::chem::chemical_formula(els);
}

Eigen::VectorXd AsymmetricUnit::covalent_radii() const {
  Eigen::VectorXd result(atomic_numbers.size());
  for (int i = 0; i < atomic_numbers.size(); i++) {
    result(i) = tonto::chem::Element(atomic_numbers(i)).covalentRadius();
  }
  return result;
}

const AtomSlab &Crystal::unit_cell_atoms() const {
  if (!m_unit_cell_atoms_needs_update)
    return m_unit_cell_atoms;
  // TODO merge sites
  const auto &pos = m_asymmetric_unit.positions;
  const auto &atoms = m_asymmetric_unit.atomic_numbers;
  const int natom = num_sites();
  const int nsymops = symmetry_operations().size();
  Eigen::VectorXd occupation =
      m_asymmetric_unit.occupations.replicate(nsymops, 1);
  Eigen::VectorXi uc_nums = atoms.replicate(nsymops, 1);
  Eigen::VectorXi asym_idx =
      Eigen::VectorXi::LinSpaced(natom, 0, natom).replicate(nsymops, 1);
  Eigen::VectorXi sym;
  Eigen::Matrix3Xd uc_pos;
  std::tie(sym, uc_pos) = m_space_group.apply_all_symmetry_operations(pos);
  uc_pos = uc_pos.unaryExpr([](const double x) { return fmod(x + 7.0, 1.0); });
  m_unit_cell_atoms = AtomSlab{uc_pos, m_unit_cell.to_cartesian(uc_pos),
                               asym_idx, uc_nums, sym};
  m_unit_cell_atoms_needs_update = false;
  return m_unit_cell_atoms;
}

AtomSlab Crystal::slab(const HKL &lower, const HKL &upper) const {
  int ncells = (upper.h - lower.h + 1) * (upper.k - lower.k + 1) *
               (upper.l - lower.l + 1);
  const AtomSlab &uc_atoms = unit_cell_atoms();
  const size_t n_uc = uc_atoms.size();
  AtomSlab result;
  const int rows = uc_atoms.frac_pos.rows();
  const int cols = uc_atoms.frac_pos.cols();
  result.frac_pos.resize(3, ncells * n_uc);
  result.frac_pos.block(0, 0, rows, cols) = uc_atoms.frac_pos;
  result.asym_idx = uc_atoms.asym_idx.replicate(ncells, 1);
  result.symop = uc_atoms.symop.replicate(ncells, 1);
  result.atomic_numbers = uc_atoms.atomic_numbers.replicate(ncells, 1);
  int offset = n_uc;
  for (int h = lower.h; h <= upper.h; h++) {
    for (int k = lower.k; k <= upper.k; k++) {
      for (int l = lower.l; l <= upper.l; l++) {
        if (h == 0 && k == 0 && l == 0)
          continue;
        auto tmp = uc_atoms.frac_pos;
        tmp.colwise() +=
            Eigen::Vector3d{static_cast<double>(h), static_cast<double>(k),
                            static_cast<double>(l)};
        result.frac_pos.block(0, offset, rows, cols) = tmp;
        offset += n_uc;
      }
    }
  }
  result.cart_pos = to_cartesian(result.frac_pos);
  return result;
}

const PeriodicBondGraph &Crystal::unit_cell_connectivity() const {
  if (!m_unit_cell_connectivity_needs_update)
    return m_bond_graph;
  auto s = slab({-1, -1, -1}, {1, 1, 1});
  size_t n_asym = num_sites();
  size_t n_uc = n_asym * m_space_group.symmetry_operations().size();
  cx::KDTree<double> tree(s.cart_pos.rows(), s.cart_pos, cx::max_leaf);
  tree.index->buildIndex();
  auto covalent_radii = m_asymmetric_unit.covalent_radii();
  double max_cov = covalent_radii.maxCoeff();
  std::vector<std::pair<size_t, double>> idxs_dists;
  nanoflann::RadiusResultSet results((max_cov * 2 + 0.4) * (max_cov * 2 + 0.4),
                                     idxs_dists);

  for (size_t i = 0; i < n_uc; i++) {
    m_bond_graph_vertices.push_back(
        m_bond_graph.add_vertex(tonto::graph::PeriodicVertex{i}));
  }

  for (size_t uc_idx_l = 0; uc_idx_l < n_uc; uc_idx_l++) {
    double *q = s.cart_pos.col(uc_idx_l).data();
    size_t asym_idx_l = uc_idx_l % n_asym;
    double cov_a = covalent_radii(asym_idx_l);
    tree.index->findNeighbors(results, q, nanoflann::SearchParams());
    for (const auto &r : idxs_dists) {
      size_t idx;
      double d;
      std::tie(idx, d) = r;
      if (idx == uc_idx_l)
        continue;
      size_t uc_idx_r = idx % n_uc;
      if (uc_idx_r < uc_idx_l)
        continue;
      size_t asym_idx_r = uc_idx_r % n_asym;
      double cov_b = covalent_radii(asym_idx_r);
      if (d < ((cov_a + cov_b + 0.4) * (cov_a + cov_b + 0.4))) {
        auto pos = s.frac_pos.col(idx);
        tonto::graph::PeriodicEdge left_right{sqrt(d),
                                              uc_idx_l,
                                              uc_idx_r,
                                              asym_idx_l,
                                              asym_idx_r,
                                              static_cast<int>(floor(pos(0))),
                                              static_cast<int>(floor(pos(1))),
                                              static_cast<int>(floor(pos(2)))};
        m_bond_graph.add_edge(m_bond_graph_vertices[uc_idx_l],
                              m_bond_graph_vertices[uc_idx_r], left_right);
        tonto::graph::PeriodicEdge right_left{sqrt(d),
                                              uc_idx_r,
                                              uc_idx_l,
                                              asym_idx_r,
                                              asym_idx_l,
                                              -static_cast<int>(floor(pos(0))),
                                              -static_cast<int>(floor(pos(1))),
                                              -static_cast<int>(floor(pos(2)))};
        m_bond_graph.add_edge(m_bond_graph_vertices[uc_idx_r],
                              m_bond_graph_vertices[uc_idx_l], right_left);
      }
    }
    results.clear();
  }
  return m_bond_graph;
}

const std::vector<tonto::chem::Molecule> &Crystal::unit_cell_molecules() const {

  auto g = unit_cell_connectivity();
  auto atoms = unit_cell_atoms();
  auto [n, components] = g.connected_components();
  std::vector<HKL> shifts_vec(components.size());
  std::vector<int> predecessors(components.size());
  std::vector<std::vector<int>> groups(n);
  for (size_t i = 0; i < components.size(); i++) {
    predecessors[i] = -1;
    groups[components[i]].push_back(i);
  }

  struct Vis : public boost::default_bfs_visitor {
    Vis(std::vector<HKL> &hkl, std::vector<int> &pred)
        : m_hkl(hkl), m_p(pred) {}
    void tree_edge(PeriodicBondGraph::edge_t e,
                   const PeriodicBondGraph::GraphContainer &g) {
      m_p[e.m_target] = e.m_source;
      auto prop = g[e];
      auto hkls = m_hkl[e.m_source];
      m_hkl[e.m_target].h = hkls.h + prop.h;
      m_hkl[e.m_target].k = hkls.k + prop.k;
      m_hkl[e.m_target].l = hkls.l + prop.l;
    }
    std::vector<HKL> &m_hkl;
    std::vector<int> &m_p;
  };

  for (const auto &group : groups) {
    auto root = group[0];
    Eigen::VectorXi atomic_numbers(group.size());
    Eigen::Matrix3Xd positions(3, group.size());
    Eigen::Matrix3Xd shifts(3, group.size());
    shifts.setZero();
    boost::breadth_first_search(g.graph(), m_bond_graph_vertices[root],
                                boost::visitor(Vis(shifts_vec, predecessors)));
    std::vector<std::pair<size_t, size_t>> bonds;
    for (size_t i = 0; i < group.size(); i++) {
      size_t uc_idx = group[i];
      atomic_numbers(i) = atoms.atomic_numbers(uc_idx);
      positions.col(i) = atoms.frac_pos.col(uc_idx);
      shifts(0, i) = shifts_vec[uc_idx].h;
      shifts(1, i) = shifts_vec[uc_idx].k;
      shifts(2, i) = shifts_vec[uc_idx].l;
      for (const auto &n : g.neighbor_list(m_bond_graph_vertices[uc_idx])) {
        size_t group_idx = std::distance(
            group.begin(), std::find(group.begin(), group.end(), n.uc_idx));
        bonds.push_back(std::pair(i, group_idx));
      }
    }
    positions += shifts;
    tonto::chem::Molecule m(atomic_numbers, to_cartesian(positions));
    m.set_bonds(bonds);
    m_unit_cell_molecules.push_back(m);
  }
  return m_unit_cell_molecules;
}

} // namespace tonto::crystal
