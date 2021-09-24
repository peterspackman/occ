#include <occ/core/linear_algebra.h>
#include <fmt/core.h>
#include <iostream>
#include <occ/core/element.h>
#include <occ/core/kdtree.h>
#include <occ/crystal/crystal.h>

namespace occ::crystal {

using occ::graph::BondGraph;

// Asymmetric unit
AsymmetricUnit::AsymmetricUnit(const Mat3N &frac_pos, const IVec &nums)
    : positions(frac_pos), atomic_numbers(nums), occupations(nums.rows()),
      charges(nums.rows()) {
    occupations.setConstant(1.0);
    charges = atomic_numbers.cast<double>();
    generate_default_labels();
}

AsymmetricUnit::AsymmetricUnit(const Mat3N &frac_pos, const IVec &nums,
                               const std::vector<std::string> &site_labels)
    : positions(frac_pos), atomic_numbers(nums), labels(site_labels),
      occupations(nums.rows()), charges(nums.rows()) {
    charges = atomic_numbers.cast<double>();
    occupations.setConstant(1.0);
}

void AsymmetricUnit::generate_default_labels() {
    occ::IVec counts(atomic_numbers.maxCoeff() + 1);
    counts.setConstant(1);
    labels.clear();
    labels.reserve(size());
    for (size_t i = 0; i < size(); i++) {
        auto num = atomic_numbers(i);
        auto symbol = occ::chem::Element(num).symbol();
        labels.push_back(fmt::format("{}{}", symbol, counts(num)++));
    }
}

Eigen::VectorXd AsymmetricUnit::covalent_radii() const {
    Eigen::VectorXd result(atomic_numbers.size());
    for (int i = 0; i < atomic_numbers.size(); i++) {
        result(i) = occ::chem::Element(atomic_numbers(i)).covalentRadius();
    }
    return result;
}

std::string AsymmetricUnit::chemical_formula() const {
    std::vector<occ::chem::Element> els;
    for (int i = 0; i < atomic_numbers.size(); i++) {
        els.push_back(occ::chem::Element(atomic_numbers[i]));
    }
    return occ::chem::chemical_formula(els);
}

// Atom Slab
const AtomSlab &Crystal::unit_cell_atoms() const {
    if (m_unit_cell_atoms_needs_update)
        update_unit_cell_atoms();
    return m_unit_cell_atoms;
}

void Crystal::update_unit_cell_atoms() const {
    // TODO merge sites
    constexpr double merge_tolerance = 1e-2;
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
    uc_pos =
        uc_pos.unaryExpr([](const double x) { return fmod(x + 7.0, 1.0); });

    occ::MaskArray mask(uc_pos.cols());
    mask.setConstant(false);

    for (size_t i = 0; i < uc_pos.cols(); i++) {
        if ((mask(i)))
            continue;
        occ::Vec3 p = uc_pos.col(i);
        for (size_t j = i + 1; j < uc_pos.cols(); j++) {
            double dist = (uc_pos.col(j) - p).norm();
            if (dist < merge_tolerance)
                mask(j) = true;
            if (occupation.rows() > 0)
                occupation(i) += occupation(j);
        }
    }
    Eigen::VectorXi idxs(uc_pos.cols() - mask.count());
    size_t n = 0;
    occ::Mat3N uc_pos_masked(3, idxs.rows());
    for (size_t i = 0; i < uc_pos.cols(); i++) {
        if (!mask(i)) {
            idxs(n) = i;
            uc_pos_masked.col(n) = uc_pos.col(i);
            n++;
        }
    }
    m_unit_cell_atoms = AtomSlab{
        uc_pos_masked, m_unit_cell.to_cartesian(uc_pos_masked),
        idxs.unaryExpr(asym_idx), idxs.unaryExpr(uc_nums), idxs.unaryExpr(sym)};
    m_unit_cell_atoms_needs_update = false;
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
                tmp.colwise() += Eigen::Vector3d{static_cast<double>(h),
                                                 static_cast<double>(k),
                                                 static_cast<double>(l)};
                result.frac_pos.block(0, offset, rows, cols) = tmp;
                offset += n_uc;
            }
        }
    }
    result.cart_pos = to_cartesian(result.frac_pos);
    return result;
}

Crystal::Crystal(const AsymmetricUnit &asym, const SpaceGroup &sg,
                 const UnitCell &uc)
    : m_asymmetric_unit(asym), m_space_group(sg), m_unit_cell(uc) {}

const PeriodicBondGraph &Crystal::unit_cell_connectivity() const {
    if (m_unit_cell_connectivity_needs_update)
        update_unit_cell_connectivity();
    return m_bond_graph;
}

void Crystal::update_unit_cell_connectivity() const {
    auto s = slab({-1, -1, -1}, {1, 1, 1});
    size_t n_asym = num_sites();
    size_t n_uc = m_unit_cell_atoms.size();
    occ::Mat3N cart_uc_pos = s.cart_pos.block(0, 0, 3, n_uc);
    occ::Mat3N cart_neighbor_pos =
        s.cart_pos.block(0, n_uc, 3, s.cart_pos.cols() - n_uc);
    cx::KDTree<double> tree(cart_neighbor_pos.rows(), cart_neighbor_pos,
                            cx::max_leaf);
    tree.index->buildIndex();
    auto covalent_radii = m_asymmetric_unit.covalent_radii();
    double max_cov = covalent_radii.maxCoeff();
    std::vector<std::pair<size_t, double>> idxs_dists;
    double max_dist = (max_cov * 2 + 0.4) * (max_cov * 2 + 0.4);
    nanoflann::RadiusResultSet results(max_dist, idxs_dists);

    for (size_t i = 0; i < n_uc; i++) {
        m_bond_graph_vertices.push_back(
            m_bond_graph.add_vertex(occ::graph::PeriodicVertex{i}));
    }

    size_t num_connections = 0;
    auto add_edge = [&](double d, size_t uc_l, size_t uc_r, size_t asym_l,
                        size_t asym_r, const occ::Vec3 &pos) {
        int h = static_cast<int>(floor(pos(0)));
        int k = static_cast<int>(floor(pos(1)));
        int l = static_cast<int>(floor(pos(2)));
        occ::graph::PeriodicEdge left_right{sqrt(d), uc_l, uc_r, asym_l,
                                            asym_r,  h,    k,    l};
        m_bond_graph.add_edge(m_bond_graph_vertices[uc_l],
                              m_bond_graph_vertices[uc_r], left_right);
        occ::graph::PeriodicEdge right_left{sqrt(d), uc_r, uc_l, asym_r,
                                            asym_l,  -h,   -k,   -l};
        m_bond_graph.add_edge(m_bond_graph_vertices[uc_r],
                              m_bond_graph_vertices[uc_l], right_left);
        num_connections++;
    };

    for (size_t uc_idx_l = 0; uc_idx_l < n_uc; uc_idx_l++) {
        size_t asym_idx_l = m_unit_cell_atoms.asym_idx(uc_idx_l);
        double cov_a = covalent_radii(asym_idx_l);
        for (size_t uc_idx_r = uc_idx_l + 1; uc_idx_r < n_uc; uc_idx_r++) {
            size_t asym_idx_r = m_unit_cell_atoms.asym_idx(uc_idx_r);
            double cov_b = covalent_radii(asym_idx_r);
            double d = (cart_uc_pos.col(uc_idx_l) - cart_uc_pos.col(uc_idx_r))
                           .squaredNorm();
            double threshold = (cov_a + cov_b + 0.4) * (cov_a + cov_b + 0.4);
            if (d < threshold) {
                add_edge(d, uc_idx_l, uc_idx_r, asym_idx_l, asym_idx_r,
                         m_unit_cell_atoms.frac_pos.col(uc_idx_r));
            }
        }
        double *q = cart_uc_pos.col(uc_idx_l).data();
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
            size_t asym_idx_r = m_unit_cell_atoms.asym_idx(uc_idx_r);
            double cov_b = covalent_radii(asym_idx_r);
            if (d < ((cov_a + cov_b + 0.4) * (cov_a + cov_b + 0.4))) {
                auto pos = s.frac_pos.col(idx + n_uc);
                add_edge(d, uc_idx_l, uc_idx_r, asym_idx_l, asym_idx_r, pos);
            }
        }
        results.clear();
    }
    m_unit_cell_connectivity_needs_update = false;
}

const std::vector<occ::chem::Molecule> &Crystal::unit_cell_molecules() const {
    if (m_unit_cell_molecules_needs_update)
        update_unit_cell_molecules();
    return m_unit_cell_molecules;
}

void Crystal::update_unit_cell_molecules() const {
    m_unit_cell_molecules.clear();
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

    size_t uc_mol_idx = 0;
    for (const auto &group : groups) {
        auto root = group[0];
        Eigen::VectorXi atomic_numbers(group.size());
        occ::IVec uc_idxs(group.size()), asym_idxs(group.size()),
            symops(group.size());
        Eigen::Matrix3Xd positions(3, group.size());
        Eigen::Matrix3Xd shifts(3, group.size());
        shifts.setZero();
        boost::breadth_first_search(
            g.graph(), m_bond_graph_vertices[root],
            boost::visitor(Vis(shifts_vec, predecessors)));
        std::vector<std::pair<size_t, size_t>> bonds;
        for (size_t i = 0; i < group.size(); i++) {
            size_t uc_idx = group[i];
            uc_idxs(i) = uc_idx;
            asym_idxs(i) = atoms.asym_idx(uc_idx);
            symops(i) = atoms.symop(uc_idx);
            atomic_numbers(i) = atoms.atomic_numbers(uc_idx);
            positions.col(i) = atoms.frac_pos.col(uc_idx);
            shifts(0, i) = shifts_vec[uc_idx].h;
            shifts(1, i) = shifts_vec[uc_idx].k;
            shifts(2, i) = shifts_vec[uc_idx].l;
            for (const auto &n :
                 g.neighbor_list(m_bond_graph_vertices[uc_idx])) {
                size_t group_idx = std::distance(
                    group.begin(),
                    std::find(group.begin(), group.end(), n.uc_idx));
                bonds.push_back(std::pair(i, group_idx));
            }
        }
        positions += shifts;
        occ::chem::Molecule m(atomic_numbers, to_cartesian(positions));
        m.set_bonds(bonds);
        m.set_unit_cell_idx(uc_idxs);
        m.set_asymmetric_unit_idx(asym_idxs);
        m.set_asymmetric_unit_symop(symops);
        m.set_unit_cell_molecule_idx(uc_mol_idx);
        m_unit_cell_molecules.push_back(m);
        uc_mol_idx++;
    }
    m_unit_cell_molecules_needs_update = false;
}

const std::vector<occ::chem::Molecule> &
Crystal::symmetry_unique_molecules() const {
    if (m_symmetry_unique_molecules_needs_update)
        update_symmetry_unique_molecules();
    return m_symmetry_unique_molecules;
}

void Crystal::update_symmetry_unique_molecules() const {
    const auto &uc_molecules = unit_cell_molecules();
    auto asym_atoms_found = std::vector<bool>(m_asymmetric_unit.size());
    m_symmetry_unique_molecules.clear();

    // sort by proportion of identity symop
    std::vector<size_t> indexes(uc_molecules.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    auto sort_func = [&uc_molecules](size_t a, size_t b) {
        const auto &symops_a = uc_molecules[a].asymmetric_unit_symop();
        size_t n_a = symops_a.rows();
        const auto &symops_b = uc_molecules[b].asymmetric_unit_symop();
        size_t n_b = symops_b.rows();
        double pct_a =
            std::count(symops_a.data(), symops_a.data() + n_a, 16484) * 1.0 /
            n_a;
        double pct_b =
            std::count(symops_b.data(), symops_b.data() + n_b, 16484) * 1.0 /
            n_b;
        return pct_a > pct_b;
    };
    std::sort(indexes.begin(), indexes.end(), sort_func);

    int num_found = 0;

    for (const auto &idx : indexes) {
        const auto &mol = uc_molecules[idx];
        const auto &asym_atoms_in_group = mol.asymmetric_unit_idx();
        bool all_found = true;
        for (size_t i = 0; i < asym_atoms_in_group.rows(); i++) {
            if (!asym_atoms_found[asym_atoms_in_group(i)]) {
                all_found = false;
                break;
            }
        }
        if (all_found)
            continue;
        else {
            for (size_t i = 0; i < asym_atoms_in_group.rows(); i++)
                asym_atoms_found[asym_atoms_in_group(i)] = true;
        }
        m_symmetry_unique_molecules.push_back(mol);
        m_symmetry_unique_molecules[num_found].set_asymmetric_molecule_idx(
            num_found);
        num_found++;
        if (std::all_of(asym_atoms_found.begin(), asym_atoms_found.end(),
                        [](bool v) { return v; }))
            break;
    }

    // now populate unit_cell_molecules
    for (auto &uc_mol : m_unit_cell_molecules) {
        if (uc_mol.asymmetric_molecule_idx() >= 0)
            continue;
        const auto uc_mol_asym = uc_mol.asymmetric_unit_idx();
        const auto uc_mol_size = uc_mol.size();
        for (const auto &asym_mol : m_symmetry_unique_molecules) {
            const auto asym_mol_size = asym_mol.size();
            if (asym_mol_size != uc_mol_size)
                continue;
            const auto asym_mol_asym = asym_mol.asymmetric_unit_idx();
            if ((uc_mol_asym.array() == asym_mol_asym.array()).all()) {
                uc_mol.set_asymmetric_molecule_idx(
                    asym_mol.asymmetric_molecule_idx());
                break;
            }
        }
    }
    m_symmetry_unique_molecules_needs_update = false;
}

CrystalDimers Crystal::symmetry_unique_dimers(double radius) const {
    using occ::chem::Dimer;
    CrystalDimers result;
    auto &dimers = result.unique_dimers;
    auto &mol_nbs = result.molecule_neighbors;

    HKL upper{std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
              std::numeric_limits<int>::min()};
    HKL lower{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
              std::numeric_limits<int>::max()};
    occ::Vec3 frac_radius = radius * 2 / m_unit_cell.lengths().array();

    for (size_t i = 0; i < m_asymmetric_unit.size(); i++) {
        const auto &pos = m_asymmetric_unit.positions.col(i);
        upper.h =
            std::max(upper.h, static_cast<int>(ceil(pos(0) + frac_radius(0))));
        upper.k =
            std::max(upper.k, static_cast<int>(ceil(pos(1) + frac_radius(1))));
        upper.l =
            std::max(upper.l, static_cast<int>(ceil(pos(2) + frac_radius(2))));

        lower.h =
            std::min(lower.h, static_cast<int>(floor(pos(0) - frac_radius(0))));
        lower.k =
            std::min(lower.k, static_cast<int>(floor(pos(1) - frac_radius(1))));
        lower.l =
            std::min(lower.l, static_cast<int>(floor(pos(2) - frac_radius(2))));
    }

    const auto &uc_mols = unit_cell_molecules();
    const auto &asym_mols = symmetry_unique_molecules();
    mol_nbs.resize(asym_mols.size());
    result.unique_dimer_idx.resize(asym_mols.size());

    for (int h = lower.h; h <= upper.h; h++) {
        for (int k = lower.k; k <= upper.k; k++) {
            for (int l = lower.l; l <= upper.l; l++) {
                occ::Vec3 cart_shift = to_cartesian(
                    occ::Vec3{static_cast<double>(h), static_cast<double>(k),
                              static_cast<double>(l)});
                for (const auto &asym_mol : asym_mols) {
                    int asym_idx_a = asym_mol.asymmetric_molecule_idx();
                    for (const auto &uc_mol : uc_mols) {
                        auto mol_translated = uc_mol.translated(cart_shift);
                        mol_translated.set_cell_shift({h, k, l});
                        double distance =
                            std::get<2>(asym_mol.nearest_atom(mol_translated));
                        if ((distance < radius) && (distance > 1e-1)) {
                            Dimer d(asym_mol, mol_translated);
                            mol_nbs[asym_idx_a].push_back(d);
                            if (std::any_of(
                                    dimers.begin(), dimers.end(),
                                    [&d](const Dimer &d2) { return d == d2; }))
                                continue;
                            dimers.push_back(d);
                        }
                    }
                }
            }
        }
    }

    auto sort_func = [](const Dimer &a, const Dimer &b) {
        return a.nearest_distance() < b.nearest_distance();
    };

    std::sort(dimers.begin(), dimers.end(), sort_func);
    size_t nb_idx = 0;
    for (auto &vec : mol_nbs) {
        std::sort(vec.begin(), vec.end(), sort_func);
        for (const auto &d : vec) {
            size_t idx = std::distance(
                dimers.begin(), std::find(dimers.begin(), dimers.end(), d));
            result.unique_dimer_idx[nb_idx].push_back(idx);
        }
        nb_idx++;
    }
    return result;
}

CrystalDimers Crystal::unit_cell_dimers(double radius) const {
    using occ::chem::Dimer;
    CrystalDimers result;
    auto &dimers = result.unique_dimers;
    auto &mol_nbs = result.molecule_neighbors;

    occ::Vec3 frac_radius = radius * 2 / m_unit_cell.lengths().array();
    auto hklmax = frac_radius.array().ceil().cast<int>();
    HKL upper{hklmax(0), hklmax(1), hklmax(2)};
    HKL lower{-hklmax(0), -hklmax(1), -hklmax(2)};

    const auto &uc_mols = unit_cell_molecules();
    mol_nbs.resize(uc_mols.size());
    result.unique_dimer_idx.resize(uc_mols.size());

    for (int h = lower.h; h <= upper.h; h++) {
        for (int k = lower.k; k <= upper.k; k++) {
            for (int l = lower.l; l <= upper.l; l++) {
                occ::Vec3 cart_shift = to_cartesian(
                    occ::Vec3{static_cast<double>(h), static_cast<double>(k),
                              static_cast<double>(l)});
                int uc_idx_a = 0;
                for (const auto &uc_mol1 : uc_mols) {
                    for (const auto &uc_mol2 : uc_mols) {
                        auto mol_translated = uc_mol2.translated(cart_shift);
                        mol_translated.set_cell_shift({h, k, l});
                        double distance =
                            std::get<2>(uc_mol1.nearest_atom(mol_translated));
                        if ((distance < radius) && (distance > 1e-1)) {
                            Dimer d(uc_mol1, mol_translated);
                            mol_nbs[uc_idx_a].push_back(d);
                            if (std::any_of(
                                    dimers.begin(), dimers.end(),
                                    [&d](const Dimer &d2) { return d == d2; }))
                                continue;
                            dimers.push_back(d);
                        }
                    }
                    uc_idx_a++;
                }
            }
        }
    }

    auto sort_func = [](const Dimer &a, const Dimer &b) {
        return a.nearest_distance() < b.nearest_distance();
    };

    std::sort(dimers.begin(), dimers.end(), sort_func);
    size_t nb_idx = 0;
    for (auto &vec : mol_nbs) {
        std::sort(vec.begin(), vec.end(), sort_func);
        for (const auto &d : vec) {
            size_t idx = std::distance(
                dimers.begin(), std::find(dimers.begin(), dimers.end(), d));
            result.unique_dimer_idx[nb_idx].push_back(idx);
        }
        nb_idx++;
    }
    return result;
}

Crystal Crystal::create_primitive_supercell(const Crystal& c, HKL hkl) {
    const auto& uc = c.unit_cell();
    auto supercell = UnitCell(
        uc.a() * hkl.h, uc.b() * hkl.k, uc.c() * hkl.l,
        uc.alpha(), uc.beta(), uc.gamma()
    );
    const auto& uc_mols = c.unit_cell_molecules();
    size_t natoms = std::accumulate(uc_mols.begin(), uc_mols.end(), 0, 
            [](size_t a, const auto& mol) {
                return a + mol.size();
            }) * hkl.h * hkl.k * hkl.l;
    Mat3N positions(3, natoms);
    IVec numbers(natoms);
    Vec3 t;
    size_t offset{0};
    for (int h = 0; h < hkl.h; h++) {
        for (int k = 0; k < hkl.k; k++) {
            for (int l = 0; l < hkl.l; l++) {
                for(const auto& uc_mol: uc_mols) {
                    t = Vec3(h, k, l);
                    size_t n = uc_mol.size();
                    positions.block(0, offset, 3, n) = c.to_fractional(uc_mol.positions());
                    positions.block(0, offset, 3, n).colwise() += t;
                    numbers.block(offset, 0, n, 1) = uc_mol.atomic_numbers();
                    offset += n;
                }
            }
        }
    }
    return Crystal(AsymmetricUnit(positions, numbers), SpaceGroup(1), supercell);
}

std::string Crystal::dimer_symmetry_string(const occ::chem::Dimer &dimer) const {
    const auto &a = dimer.a();
    const auto &b = dimer.b();
    if (a.asymmetric_molecule_idx() != b.asymmetric_molecule_idx())
        return "-";

    int sa_int = a.asymmetric_unit_symop()(0);
    int sb_int = b.asymmetric_unit_symop()(0);

    SymmetryOperation symop_a(sa_int);
    SymmetryOperation symop_b(sb_int);

    auto symop_ab = symop_b * symop_a.inverted();
    occ::Vec3 c_a = symop_ab(to_fractional(a.positions())).rowwise().mean();
    occ::Vec3 v_ab = to_fractional(b.centroid()) - c_a;

    symop_ab = symop_ab.translated(v_ab);
    return symop_ab.to_string();
}

double Crystal::volume() const {
    return m_unit_cell.volume();
}

} // namespace occ::crystal
