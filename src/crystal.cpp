#include "crystal.h"
#include "element.h"
#include <Eigen/Dense>
#include <QDebug>
#include "kdtree.h"
#include <iostream>

Crystal::Crystal(const AsymmetricUnit& asym, const SpaceGroup& sg, const UnitCell& uc) :
    m_asymmetricUnit(asym), m_spaceGroup(sg), m_unitCell(uc)
{

}

QString AsymmetricUnit::chemicalFormula() const
{
    QVector<Elements::Element> els;
    for(int i = 0; i < atomicNumbers.size(); i++) {
        els.push_back(Elements::Element(atomicNumbers[i]));
    }
    return Elements::chemicalFormula(els);
}

Eigen::VectorXd AsymmetricUnit::covalentRadii() const
{
    Eigen::VectorXd result(atomicNumbers.size());
    for(int i = 0; i < atomicNumbers.size(); i++) {
        result(i) = Elements::Element(atomicNumbers(i)).covalentRadius();
    }
    return result;
}

const AtomSlab& Crystal::unitCellAtoms() const
{
    if(!m_unitCellAtomsNeedUpdate) return m_unitCellAtoms;
    // TODO merge sites
    const auto& pos = m_asymmetricUnit.positions;
    const auto& atoms = m_asymmetricUnit.atomicNumbers;
    const int natom = numSites();
    const int nsymops = symmetryOperations().length();
    Eigen::VectorXd occupation = m_asymmetricUnit.occupations.replicate(nsymops, 1);
    Eigen::VectorXi uc_nums = atoms.replicate(nsymops, 1);
    Eigen::VectorXi asymIndex = Eigen::VectorXi::LinSpaced(natom, 0, natom).replicate(nsymops, 1);
    Eigen::VectorXi sym;
    Eigen::Matrix3Xd uc_pos;
    std::tie(sym, uc_pos) = m_spaceGroup.applyAllSymmetryOperations(pos);
    uc_pos = uc_pos.unaryExpr([](const double x) { return fmod(x + 7.0, 1.0); });
    m_unitCellAtoms = AtomSlab{uc_pos, m_unitCell.toCartesian(uc_pos), asymIndex, uc_nums, sym};
    m_unitCellAtomsNeedUpdate = false;
    return m_unitCellAtoms;
}

AtomSlab Crystal::slab(const HKL &lower, const HKL &upper) const
{
    int ncells = (upper.h - lower.h + 1) * (upper.k - lower.k + 1) * (upper.l - lower.l + 1);
    const AtomSlab& uc_atoms = unitCellAtoms();
    const size_t n_uc = uc_atoms.size();
    AtomSlab result;
    const int rows = uc_atoms.fracPos.rows();
    const int cols = uc_atoms.fracPos.cols();
    result.fracPos.resize(3, ncells * n_uc);
    result.fracPos.block(0, 0, rows, cols) = uc_atoms.fracPos;
    result.asymIndex = uc_atoms.asymIndex.replicate(ncells, 1);
    result.symop = uc_atoms.symop.replicate(ncells, 1);
    result.atomicNumbers = uc_atoms.atomicNumbers.replicate(ncells, 1);
    int offset = n_uc;
    for(int h = lower.h; h <= upper.h; h++)
    {
        for(int k = lower.k; k <= upper.k; k++)
        {
            for(int l = lower.l; l <= upper.l; l++)
            {
                if(h == 0 && k == 0 && l == 0) continue;
                auto tmp = uc_atoms.fracPos;
                tmp.colwise() += Eigen::Vector3d{
                    static_cast<double>(h),
                    static_cast<double>(k),
                    static_cast<double>(l)
                };
                result.fracPos.block(0, offset, rows, cols) = tmp;
                offset += n_uc;
            }
        }
    }
    result.cartPos = toCartesian(result.fracPos);
    return result;
}

const PeriodicBondGraph& Crystal::unitCellConnectivity() const
{
    if(!m_unitCellConnectivityNeedUpdate) return m_bondGraph;
    auto s = slab({-1, -1, -1}, {1, 1, 1});
    size_t n_asym = numSites();
    size_t n_uc = n_asym * m_spaceGroup.symmetryOperations().size();
    cx::KDTree<double> tree(s.cartPos.rows(), s.cartPos, cx::max_leaf);
    tree.index->buildIndex();
    auto covalent_radii = m_asymmetricUnit.covalentRadii();
    double max_cov = covalent_radii.maxCoeff();
    std::vector<std::pair<size_t, double>> idxs_dists;
    nanoflann::RadiusResultSet results((max_cov * 2 + 0.4) * (max_cov * 2 + 0.4), idxs_dists);

    for(size_t uc_idx_l = 0; uc_idx_l < n_uc; uc_idx_l++) {
        double *q = s.cartPos.col(uc_idx_l).data();
        size_t asym_idx_l = uc_idx_l % n_asym;
        double cov_a = covalent_radii(asym_idx_l);
        tree.index->findNeighbors(results, q, nanoflann::SearchParams());
        for(const auto& r: idxs_dists) {
            size_t idx;
            double d;
            std::tie(idx, d) = r;
            if(idx == uc_idx_l) continue;
            size_t uc_idx_r = idx % n_uc;
            if(uc_idx_r < uc_idx_l) continue;
            size_t asym_idx_r = uc_idx_r % n_asym;
            double cov_b = covalent_radii(asym_idx_r);
            if(d < ((cov_a + cov_b + 0.4) * (cov_a + cov_b + 0.4))) {
                auto pos = s.fracPos.col(idx);

                m_bondGraph.addBond(PeriodicBondGraph::Edge{
                                        sqrt(d),
                                        uc_idx_l, uc_idx_r,
                                        asym_idx_l, asym_idx_r,
                                        static_cast<int>(floor(pos(0))),
                                        static_cast<int>(floor(pos(1))),
                                        static_cast<int>(floor(pos(2)))
                                    });
            }
        }
        results.clear();
    }
    qDebug() << "n_uc:" << n_uc;
    qDebug() << "n_asym:" << n_asym;
    qDebug() << "n_bonds:" << m_bondGraph.numEdges();
    for(const auto& x: m_bondGraph.neighbors(0))
    {
        qDebug() << "Neighbours: 0" << x;
    }
    return m_bondGraph;
}

const QVector<Molecule>& Crystal::unitCellMolecules() const {

    auto g = unitCellConnectivity();
    auto atoms = unitCellAtoms();
    auto [n, components] = g.connectedComponents();
    qDebug() << "components:\n" << components;
    QVector<HKL> shifts_vec(components.size());
    QVector<int> predecessors(components.size());
    QVector<QVector<int>> groups(n);
    for(int i = 0; i < components.size(); i++) {
        predecessors[i] = -1;
        groups[components[i]].push_back(i);
    }

    struct Vis : public boost::default_bfs_visitor
    {
        Vis(QVector<HKL>& hkl, QVector<int>& pred) : m_hkl(hkl), m_p(pred) {}
        void tree_edge(PeriodicBondGraph::edge_t e, const PeriodicBondGraph::GraphContainer& g)
        {
            m_p[e.m_target] = e.m_source;
            auto prop = g[e];
            auto hkls = m_hkl[e.m_source];
            m_hkl[e.m_target].h = hkls.h + prop.h;
            m_hkl[e.m_target].k = hkls.k + prop.k;
            m_hkl[e.m_target].l = hkls.l + prop.l;
        }
        QVector<HKL>& m_hkl;
        QVector<int>& m_p;
    };

    for(const auto& group : groups) {
        auto root = group[0];
        Eigen::VectorXi atomicNumbers(group.size());
        Eigen::Matrix3Xd positions(3, group.size());
        Eigen::Matrix3Xd shifts(3, group.size());
        shifts.setZero();
        boost::breadth_first_search(
              g.graph(), g.vertexHandle(root),
              boost::visitor(Vis(shifts_vec, predecessors))
        );
        QVector<QPair<size_t, size_t>> bonds;
        for(int i = 0; i < group.size(); i++) {
            size_t uc_idx = group[i];
            atomicNumbers(i) = atoms.atomicNumbers(uc_idx);
            positions.col(i) = atoms.fracPos.col(uc_idx);
            shifts(0, i) = shifts_vec[uc_idx].h;
            shifts(1, i) = shifts_vec[uc_idx].k;
            shifts(2, i) = shifts_vec[uc_idx].l;
            for(const auto& n: g.neighbors(uc_idx)) {
                size_t group_idx = std::distance(group.begin(), std::find(group.begin(), group.end(), n));
                bonds.push_back(QPair(i, group_idx));
            }
        }
        positions += shifts;
        Molecule m(atomicNumbers, toCartesian(positions));
        m.setBonds(bonds);
        m_unitCellMolecules.append(m);
    }
    for(const auto& m : m_unitCellMolecules) {
        qDebug() << "Molecule:";
        m.dump();
    }
    return m_unitCellMolecules;
}
