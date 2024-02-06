#include <algorithm>
#include <fmt/os.h>
#include <numeric>
#include <occ/core/eem.h>
#include <occ/core/log.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/surface.h>
#include <vector>

namespace occ::crystal {

template <typename T>
void loop_over_miller_indices(
    T &func, const Crystal &c,
    const CrystalSurfaceGenerationParameters &params) {
    const auto &uc = c.unit_cell();
    HKL limits = uc.hkl_limits(params.d_min);
    const auto rasu = c.space_group().reciprocal_asu();
    const Mat3 &lattice = uc.reciprocal();
    HKL m;
    for (m.h = -limits.h; m.h <= limits.h; m.h++)
        for (m.k = -limits.k; m.k <= limits.k; m.k++)
            for (m.l = -limits.l; m.l <= limits.l; m.l++) {
                if (!params.unique || rasu.is_in(m)) {
                    double d = m.d(lattice);
                    if (d > params.d_min && d < params.d_max)
                        func(m);
                }
            }
}

std::vector<size_t> argsort(const Vec &vec) {
    std::vector<size_t> idx(vec.rows());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(), [&vec](size_t i1, size_t i2) {
        return std::abs(vec(i1)) < std::abs(vec(i2));
    });

    return idx;
}

Surface::Surface(const HKL &miller, const Crystal &crystal)
    : m_hkl{miller}, m_crystal_unit_cell(crystal.unit_cell()) {
    int cd = std::gcd(std::gcd(m_hkl.h, m_hkl.k), std::gcd(m_hkl.k, m_hkl.l));
    m_depth = cd / d();
    Vec3 unit_normal = normal_vector();

    std::vector<Vec3> vector_candidates;
    {
        int c = std::gcd(m_hkl.h, m_hkl.k);
        double cd = (c != 0) ? c : 1.0;
        Vec3 a = m_hkl.k / cd * m_crystal_unit_cell.a_vector() -
                 m_hkl.h / cd * m_crystal_unit_cell.b_vector();
        if (a.squaredNorm() > 1e-3)
            vector_candidates.push_back(a);
    }
    {
        int c = std::gcd(m_hkl.h, m_hkl.l);
        double cd = (c != 0) ? c : 1.0;
        Vec3 a = m_hkl.l / cd * m_crystal_unit_cell.a_vector() -
                 m_hkl.h / cd * m_crystal_unit_cell.c_vector();
        if (a.squaredNorm() > 1e-3)
            vector_candidates.push_back(a);
    }
    {
        int c = std::gcd(m_hkl.k, m_hkl.l);
        double cd = (c != 0) ? c : 1.0;
        Vec3 a = m_hkl.l / cd * m_crystal_unit_cell.b_vector() -
                 m_hkl.k / cd * m_crystal_unit_cell.c_vector();
        if (a.squaredNorm() > 1e-3)
            vector_candidates.push_back(a);
    }
    std::vector<Vec3> temp_vectors;
    // add linear combinations
    for (int i = 0; i < vector_candidates.size() - 1; i++) {
        const auto &v_a = vector_candidates[i];
        for (int j = i + 1; j < vector_candidates.size(); j++) {
            const auto &v_b = vector_candidates[j];
            Vec3 a = v_a + v_b;
            if (a.squaredNorm() > 1e-3) {
                temp_vectors.push_back(a);
            }
            a = v_a - v_b;
            if (a.squaredNorm() > 1e-3) {
                temp_vectors.push_back(a);
            }
        }
    }
    vector_candidates.insert(vector_candidates.end(), temp_vectors.begin(),
                             temp_vectors.end());

    occ::log::trace("Candidate surface vectors:");
    for (const auto &x : vector_candidates) {
        occ::log::trace("v = {:.5f} {:.5f} {:.5f}", x(0), x(1), x(2));
    }

    std::sort(vector_candidates.begin(), vector_candidates.end(),
              [](const Vec3 &a, const Vec3 &b) {
                  return a.squaredNorm() < b.squaredNorm();
              });
    m_a_vector = vector_candidates[0];
    bool found = false;
    for (int i = 1; i < vector_candidates.size(); i++) {
        Vec3 a = m_a_vector.cross(vector_candidates[i]);
        if (a.squaredNorm() > 1e-3) {
            found = true;
            m_b_vector = vector_candidates[i];
            break;
        }
    }
    if (!found)
        occ::log::error("No valid second vector for surface was found!");
    occ::log::trace("Found vectors:");
    occ::log::trace("A = {:.5f} {:.5f} {:.5f}", m_a_vector(0), m_a_vector(1),
                    m_a_vector(2));
    occ::log::trace("B = {:.5f} {:.5f} {:.5f}", m_b_vector(0), m_b_vector(1),
                    m_b_vector(2));

    m_angle = std::acos(m_a_vector.normalized().dot(m_b_vector.normalized()));
    m_depth_vector = m_depth * unit_normal;
}

Vec3 Surface::normal_vector() const {
    std::vector<Vec3> vecs;
    if (m_hkl.h == 0)
        vecs.push_back(m_crystal_unit_cell.a_vector());
    if (m_hkl.k == 0)
        vecs.push_back(m_crystal_unit_cell.b_vector());
    if (m_hkl.l == 0)
        vecs.push_back(m_crystal_unit_cell.c_vector());

    if (vecs.size() < 2) {
        std::vector<Vec3> points;
        if (m_hkl.h != 0)
            points.push_back(
                m_crystal_unit_cell.to_cartesian(Vec3(1.0 / m_hkl.h, 0, 0)));
        if (m_hkl.k != 0)
            points.push_back(
                m_crystal_unit_cell.to_cartesian(Vec3(0, 1.0 / m_hkl.k, 0)));
        if (m_hkl.l != 0)
            points.push_back(
                m_crystal_unit_cell.to_cartesian(Vec3(0, 0, 1.0 / m_hkl.l)));
        if (points.size() == 2) {
            vecs.push_back(points[1] - points[0]);
        } else {
            vecs.push_back(points[1] - points[0]);
            vecs.push_back(points[2] - points[0]);
        }
    }
    Vec3 v = vecs[0].cross(vecs[1]);
    v.normalize();
    return v;
}

Mat3 Surface::basis_matrix(double depth_scale) const {
    Mat3 basis;
    basis.col(0) = m_a_vector;
    basis.col(1) = m_b_vector;
    basis.col(2) = m_depth_vector * depth_scale;
    return basis;
}

double Surface::depth() const { return m_depth; }

double Surface::d() const { return m_hkl.d(m_crystal_unit_cell.reciprocal()); }

Vec3 Surface::dipole() const { return m_dipole; }

void Surface::print() const {
    occ::log::info("({} {} {}) spacing = {:.3f} area = {:.3f}", m_hkl.h,
                   m_hkl.k, m_hkl.l, d(), area());
    occ::log::debug(
        "A vector    : [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})",
        m_a_vector(0), m_a_vector(1), m_a_vector(2), m_a_vector.norm());
    occ::log::debug(
        "B vector    : [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})",
        m_b_vector(0), m_b_vector(1), m_b_vector(2), m_b_vector.norm());
    occ::log::debug(
        "Depth vector: [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})\n",
        m_depth_vector(0), m_depth_vector(1), m_depth_vector(2), depth());
}

std::vector<Surface>
generate_surfaces(const Crystal &c,
                  const CrystalSurfaceGenerationParameters &params) {
    std::vector<Surface> result;
    auto f = [&](const HKL &m) {
        if (params.systematic_absences_allowed || !Surface::check_systematic_absence(c, m))
            result.emplace_back(Surface(m, c));
    };
    loop_over_miller_indices(f, c, params);
    std::sort(result.begin(), result.end(),
              [](const Surface &a, const Surface &b) { return a.d() < b.d(); });
    return result;
}

bool Surface::check_systematic_absence(const Crystal &crystal, const HKL &hkl) {
    Vec3 f(hkl.h, hkl.k, hkl.l);
    constexpr double position_tolerance = 1e-6;
    for (const auto &symop : crystal.space_group().symmetry_operations()) {
        if (symop.is_identity())
            continue;

        Vec3 df = f - symop.rotation() * f;
        if (df.squaredNorm() < position_tolerance) {
            Vec3 offset = symop.translation();
            offset.array() *= f.array();
            double integer_part;
            double frac_part = std::modf(offset.sum(), &integer_part);
            if (std::abs(frac_part) > position_tolerance)
                return true;
        }
    }
    return false;
}

bool Surface::faces_are_equivalent(const Crystal &crystal, const HKL &hkl1,
                                   const HKL &hkl2) {
    IVec3 a(hkl1.h, hkl1.k, hkl1.l);
    IVec3 b(hkl2.h, hkl2.k, hkl2.l);
    constexpr double position_tolerance = 1e-6;
    for (const auto &symop : crystal.space_group().symmetry_operations()) {
        Eigen::Matrix3i rot = symop.rotation().cast<int>();
        IVec3 a_rotated = rot * a;
        if (b == a_rotated)
            return true;
    }
    return false;
}

std::vector<Molecule> Surface::find_molecule_cell_translations(
    const std::vector<Molecule> &mols, double depth, double cut_offset) {
    const double epsilon = 1e-3;
    const int num_mols = mols.size();
    std::vector<Molecule> result;
    Vec3 unit_normal = normal_vector();

    Mat3 basis = basis_matrix(depth);
    Mat3 basis_inverse = basis.inverse();

    // cb = crystal basis, sb = surface basis
    Mat3 cb_in_sb = basis_inverse * m_crystal_unit_cell.direct();
    Mat3 sb_in_cb = m_crystal_unit_cell.to_fractional(basis);

    occ::log::debug("Crystal basis in Surface coordinates");
    occ::log::debug("[{:9.3f}, {:9.3f}, {:9.3f}]", cb_in_sb(0, 0),
                    cb_in_sb(0, 1), cb_in_sb(0, 2));
    occ::log::debug("[{:9.3f}, {:9.3f}, {:9.3f}]", cb_in_sb(1, 0),
                    cb_in_sb(1, 1), cb_in_sb(1, 2));
    occ::log::debug("[{:9.3f}, {:9.3f}, {:9.3f}]", cb_in_sb(2, 0),
                    cb_in_sb(2, 1), cb_in_sb(2, 2));

    occ::log::debug("Surface basis in Crystal coordinates");
    occ::log::debug("[{:9.3f}, {:9.3f}, {:9.3f}]", sb_in_cb(0, 0),
                    sb_in_cb(0, 1), sb_in_cb(0, 2));
    occ::log::debug("[{:9.3f}, {:9.3f}, {:9.3f}]", sb_in_cb(1, 0),
                    sb_in_cb(1, 1), sb_in_cb(1, 2));
    occ::log::debug("[{:9.3f}, {:9.3f}, {:9.3f}]", sb_in_cb(2, 0),
                    sb_in_cb(2, 1), sb_in_cb(2, 2));

    Vec3 origin_cart = m_depth_vector * cut_offset;
    Vec3 origin_crystal = m_crystal_unit_cell.to_fractional(origin_cart);
    Vec3 origin_surf = basis_inverse * origin_cart;

    // get box corners in surface coordinates
    Vec3 zeros = Vec3(0.0, 0.0, 0.0);
    Vec3 upper_vec_surface = origin_surf.array() + epsilon + 1.0;
    Vec3 ones = Vec3(1.0, 1.0, 1.0);
    Vec3 lower_vec_surface = origin_surf.array() + epsilon;

    occ::log::debug("Origin (crystal) = [{:9.3f}, {:9.3f} {:9.3f}]",
                    origin_crystal(0), origin_crystal(1), origin_crystal(2));
    occ::log::debug("Origin (cart)    = [{:9.3f}, {:9.3f} {:9.3f}]",
                    origin_cart(0), origin_cart(1), origin_cart(2));
    occ::log::debug("Origin (surf)    = [{:9.3f}, {:9.3f} {:9.3f}]",
                    origin_surf(0), origin_surf(1), origin_surf(2));

    // get corners of the surface box in crystal coordinates
    Mat3N corners(3, 8);
    corners.col(0) = origin_crystal;
    corners.col(1) = origin_crystal + sb_in_cb.col(0);
    corners.col(2) = origin_crystal + sb_in_cb.col(1);
    corners.col(3) = origin_crystal + sb_in_cb.col(2);
    corners.col(4) = origin_crystal + sb_in_cb.col(0) + sb_in_cb.col(1);
    corners.col(5) = origin_crystal + sb_in_cb.col(0) + sb_in_cb.col(2);
    corners.col(6) = origin_crystal + sb_in_cb.col(1) + sb_in_cb.col(2);
    corners.col(7) =
        origin_crystal + sb_in_cb.col(0) + sb_in_cb.col(1) + sb_in_cb.col(2);
    Vec3 upper_vec_crystal = corners.rowwise().maxCoeff();
    Vec3 lower_vec_crystal = corners.rowwise().minCoeff();

    occ::log::debug("Lower corner crystal: [{:9.3f}, {:9.3f}, {:9.3f}]",
                    lower_vec_crystal(0), lower_vec_crystal(1),
                    lower_vec_crystal(2));
    occ::log::debug("Upper corner crystal: [{:9.3f}, {:9.3f}, {:9.3f}]",
                    upper_vec_crystal(0), upper_vec_crystal(1),
                    upper_vec_crystal(2));
    occ::log::debug("Lower corner surface: [{:9.3f}, {:9.3f}, {:9.3f}]",
                    lower_vec_surface(0), lower_vec_surface(1),
                    lower_vec_surface(2));
    occ::log::debug("Upper corner surface: [{:9.3f}, {:9.3f}, {:9.3f}]",
                    upper_vec_surface(0), upper_vec_surface(1),
                    upper_vec_surface(2));

    Mat3N centroids(3, num_mols);
    for (int i = 0; i < num_mols; i++)
        centroids.col(i) = mols[i].centroid();

    Mat3N frac_centroids_crystal = m_crystal_unit_cell.to_fractional(centroids);

    // buffer these with an extra 1 cell - may not be necessary
    HKL upper = HKL::ceil(upper_vec_crystal.array() + 1.0);
    HKL lower = HKL::floor(lower_vec_crystal.array() - 1.0);

    for (int h = lower.h; h <= upper.h; h++) {
        for (int k = lower.k; k <= upper.k; k++) {
            for (int l = lower.l; l <= upper.l; l++) {
                Vec3 frac_t_crystal = HKL{h, k, l}.vector();
                // cartesian translation of HKL in crystal basis + cut offset
                Vec3 cart_shift =
                    m_crystal_unit_cell.to_cartesian(frac_t_crystal);
                Mat3N tmp = basis_inverse * (centroids.colwise() + cart_shift);
                for (int i = 0; i < num_mols; i++) {

                    if ((tmp.col(i).array() > lower_vec_surface.array())
                            .all() &&
                        (tmp.col(i).array() < upper_vec_surface.array())
                            .all()) {
                        auto cell_shift = mols[i].cell_shift();
                        auto mol_t = mols[i].translated(cart_shift);
                        mol_t.set_cell_shift({h, k, l});
                        mol_t.set_unit_cell_molecule_idx(i);
                        const auto &tpos = mol_t.positions();
                        result.push_back(mol_t);
                        Vec3 centroid = mol_t.centroid();
                        Vec3 centroid_frac = basis_inverse * centroid;
                        occ::log::debug(
                            "Molecule {} added with fractional (surface) "
                            "centroid: [{:9.3f}, {:9.3f}, {:9.3f}]",
                            i, tmp(0, i), tmp(1, i), tmp(2, i));
                    }
                }
            }
        }
    }

    std::vector<Vec> partial_charges;
    for (const auto &mol : mols) {
        const auto &nums = mol.atomic_numbers();
        const auto &pos = mol.positions();
        partial_charges.push_back(
            occ::core::charges::eem_partial_charges(nums, pos));
    }

    // sort them by their depth;
    std::stable_sort(result.begin(), result.end(),
                     [&basis_inverse](const Molecule &l, const Molecule &r) {
                         Vec3 fracl = basis_inverse * l.centroid();
                         Vec3 fracr = basis_inverse * r.centroid();
                         return fracl(2) < fracr(2);
                     });

    m_dipole = Vec3::Zero();
    for (const auto &mol : result) {
        const auto &q = partial_charges[mol.unit_cell_molecule_idx()];
        const auto &pos = mol.positions();
        for (int i = 0; i < pos.cols(); i++) {
            m_dipole.array() += pos.col(i).array() * q(i);
        }
    }

    occ::log::debug("Dipole (EEM charges): {:.3f} {:.3f} {:.3f}", m_dipole(0),
                    m_dipole(1), m_dipole(2));

    if (m_dipole.norm() > 1e-2)
        bool valid = false;

    return result;
}

std::vector<std::vector<size_t>>
unique_counts_from_dimers(const SurfaceCutResult::DimerCounts &counts,
                          const CrystalDimers &dimers) {

    size_t max_unique_interaction = 0;
    int max_asym_idx = 0;
    for (const auto &mol_neighbors : dimers.molecule_neighbors) {
        for (const auto &pair : mol_neighbors) {
            max_unique_interaction =
                std::max(max_unique_interaction, pair.dimer.interaction_id());
            max_asym_idx = std::max(max_asym_idx,
                                    pair.dimer.a().asymmetric_molecule_idx());
        }
    }

    std::vector<std::vector<size_t>> result;
    for (size_t i = 0; i <= max_asym_idx; i++) {
        result.push_back(std::vector<size_t>(max_unique_interaction + 1, 0));
    }

    for (int i = 0; i < counts.size(); i++) {
        const auto &neighbor_counts = counts[i];
        const auto &neighbors = dimers.molecule_neighbors[i];
        for (int j = 0; j < neighbor_counts.size(); j++) {
            if (neighbor_counts[j] > 0) {
                const auto &d = neighbors[j].dimer;
                size_t asym_idx = d.a().asymmetric_molecule_idx();
                result[asym_idx][d.interaction_id()] += neighbor_counts[j];
            }
        }
    }
    return result;
}

double
energy_from_counts_and_dimers(const SurfaceCutResult::DimerCounts &counts,
                              const CrystalDimers &dimers) {
    double energy_total = 0.0;
    for (int i = 0; i < counts.size(); i++) {
        const auto &neighbor_counts = counts[i];
        const auto &neighbors = dimers.molecule_neighbors[i];
        for (int j = 0; j < neighbor_counts.size(); j++) {
            if (neighbor_counts[j] > 0) {
                double e_int = neighbors[j].dimer.interaction_energy();
                energy_total += neighbor_counts[j] * e_int;
            }
        }
    }
    return energy_total;
}

std::vector<std::vector<size_t>>
SurfaceCutResult::unique_counts_above(const CrystalDimers &dimers) const {
    return unique_counts_from_dimers(above, dimers);
}

double SurfaceCutResult::total_above(const CrystalDimers &dimers) const {
    return energy_from_counts_and_dimers(above, dimers);
}

double SurfaceCutResult::total_below(const CrystalDimers &dimers) const {
    return energy_from_counts_and_dimers(below, dimers);
}

double SurfaceCutResult::total_slab(const CrystalDimers &dimers) const {
    return energy_from_counts_and_dimers(slab, dimers);
}

double SurfaceCutResult::total_bulk(const CrystalDimers &dimers) const {
    return energy_from_counts_and_dimers(bulk, dimers);
}

SurfaceCutResult::SurfaceCutResult(const CrystalDimers &dimers) {

    for (const auto &neighbors : dimers.molecule_neighbors) {
        if (neighbors.size() < 1) {
            above.push_back({});
            below.push_back({});
            slab.push_back({});
            bulk.push_back({});
            continue;
        }
        above.push_back(std::vector<int>(neighbors.size(), 0));
        below.push_back(std::vector<int>(neighbors.size(), 0));
        slab.push_back(std::vector<int>(neighbors.size(), 0));
        bulk.push_back(std::vector<int>(neighbors.size(), 0));
    }
}

SurfaceCutResult Surface::count_crystal_dimers_cut_by_surface(
    const CrystalDimers &crystal_dimers, double cut_offset) {
    const double epsilon = 1e-3;
    SurfaceCutResult result(crystal_dimers);
    result.cut_offset = cut_offset;

    double required_depth = 0.0;
    Vec3 unit_normal = normal_vector();

    std::vector<Molecule> molecules;
    for (const auto &neighbors : crystal_dimers.molecule_neighbors) {
        molecules.push_back(neighbors[0].dimer.a());
        for (const auto &[dimer, unique_idx] : neighbors) {
            Vec3 v_ab = dimer.b().centroid() - dimer.a().centroid();
            required_depth =
                std::max(required_depth, std::abs(v_ab.dot(unit_normal)));
        }
    }
    double depth_scale = std::ceil(required_depth / m_depth_vector.norm());
    occ::log::debug("Required depth = {} x depth ({:12.6f} angs)", depth_scale,
                    required_depth);

    result.molecules =
        find_molecule_cell_translations(molecules, depth_scale, cut_offset);

    Mat3 basis = basis_matrix(depth_scale);
    occ::log::debug("Basis");
    occ::log::debug(
        "A vector    : [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})",
        basis(0, 0), basis(1, 0), basis(2, 0), basis.col(0).norm());
    occ::log::debug(
        "B vector    : [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})",
        basis(0, 1), basis(1, 1), basis(2, 1), basis.col(1).norm());
    occ::log::debug(
        "Depth vector: [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})",
        basis(0, 2), basis(1, 2), basis(2, 2), basis.col(2).norm());

    Vec3 origin = m_depth_vector * cut_offset;
    occ::log::debug("Origin: [{:9.3f}, {:9.3f}, {:9.3f}]", origin(0), origin(1),
                    origin(2));

    const double upper_bound = (1.0 + cut_offset / depth_scale) + epsilon;
    const double lower_bound = (cut_offset / depth_scale) - epsilon;
    Mat3 basis_inverse = basis.inverse();
    result.basis = basis;

    auto cart2surf = [&basis_inverse](const Vec3 &v) {
        return basis_inverse * v;
    };

    struct CutCount {
        int above{0};
        double energy_above{0.0};
        int below{0};
        double energy_below{0.0};
        int slab{0};
        double energy_slab{0.0};
        int neighbors{0};
        double energy_neighbors{0.0};
    };
    CutCount total_cut_count;
    int t_index = 0;
    auto filename = fmt::format("crystal_surface_{}_{}_{}_cut{:.3f}.txt",
                                m_hkl.h, m_hkl.k, m_hkl.l, cut_offset);
    auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                 fmt::file::CREATE);

    std::vector<std::string> lines;
    for (const auto &mol : result.molecules) {
        int index = mol.unit_cell_molecule_idx();
        if (index > result.above.size()) {
            throw std::runtime_error("Must pass unit cell dimers to "
                                     "count_crystal_dimers_cut_by_surface");
        }

        Mat3N mol_pos_cart = mol.positions();

        const auto &elements = mol.elements();
        for (int atom_idx = 0; atom_idx < elements.size(); atom_idx++) {
            lines.push_back(fmt::format(
                "{} {:12.6f} {:12.6f} {:12.6f} {} {}",
                elements[atom_idx].symbol(), mol_pos_cart(0, atom_idx),
                mol_pos_cart(1, atom_idx), mol_pos_cart(2, atom_idx), t_index,
                index));
        }

        const auto &uc_shift = mol.cell_shift();
        Vec3 shift(uc_shift[0], uc_shift[1], uc_shift[2]);
        Vec3 translation = m_crystal_unit_cell.to_cartesian(shift);
        Vec3 frac_c = cart2surf(mol.centroid());
        occ::log::debug("mol in surface index = {}, unit cell index = {}",
                        t_index, index);
        occ::log::debug("Fractional center: {:12.6f} {:12.6f} {:12.6f}",
                        frac_c(0), frac_c(1), frac_c(2));
        occ::log::debug("Shift (hkl): {} {} {}", uc_shift[0], uc_shift[1],
                        uc_shift[2]);
        occ::log::debug("Shift (cart): {:12.6f} {:12.6f} {:12.6f}",
                        translation(0), translation(1), translation(2));

        const auto &neighbors = crystal_dimers.molecule_neighbors[index];
        auto &neighbor_counts_above = result.above[index];
        auto &neighbor_counts_below = result.below[index];
        auto &neighbor_counts_slab = result.slab[index];
        auto &neighbor_counts_bulk = result.bulk[index];

        CutCount cut_count;
        for (int neighbor_index = 0; neighbor_index < neighbors.size();
             neighbor_index++) {
            const auto &dimer = neighbors[neighbor_index].dimer;
            double e = dimer.interaction_energy();
            double r = dimer.centroid_distance();
            Vec3 c_a = dimer.a().centroid() + translation;
            Vec3 c_b = dimer.b().centroid() + translation;
            cut_count.neighbors++;
            cut_count.energy_neighbors += e;
            neighbor_counts_bulk[neighbor_index]++;
            if ((c_a - mol.centroid()).norm() > 1e-3) {
                occ::log::error(
                    "check for mol_a translated positions matching failed");
            }
            if ((std::abs(r - (c_a - c_b).norm()) > 1e-3)) {
                occ::log::error(
                    "check for translated distance in Surface cut failed");
            }
            Molecule mol2 = dimer.b();
            mol2.translate(translation);
            int region = 0;

            Vec3 frac_coords = cart2surf(c_b);
            if (frac_coords(2) > upper_bound) {
                region = 1;
                cut_count.above++;
                cut_count.energy_above += e;
                neighbor_counts_above[neighbor_index]++;
            } else if (frac_coords(2) <= lower_bound) {
                region = 2;
                cut_count.below++;
                cut_count.energy_below += e;
                neighbor_counts_below[neighbor_index]++;
            } else {
                region = 3;
                cut_count.slab++;
                cut_count.energy_slab += e;
                neighbor_counts_slab[neighbor_index]++;
            }
            Vec3 pos = mol2.centroid();
        }
        total_cut_count.energy_slab += cut_count.energy_slab;
        total_cut_count.energy_above += cut_count.energy_above;
        total_cut_count.energy_below += cut_count.energy_below;
        total_cut_count.energy_neighbors += cut_count.energy_neighbors;
        occ::log::debug("Cut interactions for mol in surface slab {}", t_index);
        occ::log::debug("Above:     {:3d}  e = {:12.6f}", cut_count.above,
                        cut_count.energy_above);
        occ::log::debug("Below:     {:3d}  e = {:12.6f}", cut_count.below,
                        cut_count.energy_below);
        occ::log::debug("In slab:   {:3d}  e = {:12.6f}", cut_count.slab,
                        cut_count.energy_slab);
        occ::log::debug("Neighbors: {:3d}  e = {:12.6f}", cut_count.neighbors,
                        cut_count.energy_neighbors);
        t_index++;
    }

    output.print("{}\n", lines.size());
    output.print(
        R"""(Lattice="{:3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}" Properties={} Origin="{:.3f} {:.3f} {:.3f}"{})""",
        basis(0, 0), basis(1, 0), basis(2, 0), basis(0, 1), basis(1, 1),
        basis(2, 1), basis(0, 2), basis(1, 2), basis(2, 2),
        "species:S:1:pos:R:3:mol_idx:I:1:uc_idx:I:1", 0.0, 0.0, 0.0, "\n");

    for (const auto &line : lines) {
        output.print("{}\n", line);
    }

    occ::log::debug("slab energy  (S) = {}", total_cut_count.energy_slab);
    occ::log::debug("above energy (A) = {}", total_cut_count.energy_above);
    occ::log::debug("below energy (B) = {}", total_cut_count.energy_below);
    occ::log::debug("total energy (T) = {}", total_cut_count.energy_neighbors);
    occ::log::debug("T - A - B - S = {}", total_cut_count.energy_neighbors -
                                              total_cut_count.energy_slab -
                                              total_cut_count.energy_above -
                                              total_cut_count.energy_below);
    return result;
}
std::vector<double> Surface::possible_cuts(Eigen::Ref<const Mat3N> unique_positions, double epsilon) const {
    std::vector<double> result;
    result.reserve(unique_positions.cols() - 1); // Adjust for between positions

    Mat3 basis = basis_matrix(1.0);
    Mat3 basis_inverse = basis.inverse();

    Mat3N pos_frac = basis_inverse * unique_positions;

    // Get sorted distances along surface normal
    std::vector<double> zpos(pos_frac.cols());
    for (int i = 0; i < pos_frac.cols(); ++i) {
	zpos[i] = pos_frac(2, i);
    }
    std::sort(zpos.begin(), zpos.end());

    // Calculate midpoints between unique distances along normal
    // ensuring we're between 0 and 1
    for (size_t i = 0; i < zpos.size() - 1; ++i) {
	double mid = 0.5 * (zpos[i] + zpos[i + 1]);
	result.push_back(std::fmod(mid + 7.0, 1.0));
    }

    // Add a cut between the last and first (periodic) 
    double mid = 0.5 * (zpos.back() + (zpos.front() + 1.0));
    result.push_back(std::fmod(mid + 7.0, 1.0)); // Ensure within [0,1)

    // Remove duplicates
    std::sort(result.begin(), result.end());
    auto last = std::unique(result.begin(), result.end(),
			    [epsilon](double a, double b) { return std::abs(a - b) < epsilon; });
    result.erase(last, result.end());

    // Round to mitigate floating-point arithmetic issues
    std::transform(result.begin(), result.end(), result.begin(),
		   [](double value) { return std::round(value * 1000000.0) / 1000000.0; });

    return result;
}

} // namespace occ::crystal
