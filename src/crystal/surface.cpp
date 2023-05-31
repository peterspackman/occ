#include <algorithm>
#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <numeric>
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
    m_depth = 1.0 / d();
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

double Surface::depth() const { return m_depth; }

double Surface::d() const { return m_hkl.d(m_crystal_unit_cell.reciprocal()); }

Vec3 Surface::dipole() const { return Vec3(0, 0, 0); }

void Surface::print() const {
    fmt::print("Surface ({:d}, {:d}, {:d})\n", m_hkl.h, m_hkl.k, m_hkl.l);
    fmt::print("Surface area {:.3f}\n", area());
    fmt::print("A vector    : [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})\n",
               m_a_vector(0), m_a_vector(1), m_a_vector(2), m_a_vector.norm());
    fmt::print("B vector    : [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})\n",
               m_b_vector(0), m_b_vector(1), m_b_vector(2), m_b_vector.norm());
    fmt::print("Depth vector: [{:9.3f}, {:9.3f}, {:9.3f}] (length = {:9.3f})\n",
               m_depth_vector(0), m_depth_vector(1), m_depth_vector(2),
               depth());
}

bool Surface::cuts_line_segment(const Vec3 &surface_origin, const Vec3 &point1,
                                const Vec3 &point2, Vec3 &intersection,
                                bool infinite) const {
    constexpr double epsilon = 1e-6;
    Vec3 line_direction = (point2 - point1).normalized();
    Vec3 u_n = normal_vector();

    // line is parallel to the plane
    if (u_n.dot(line_direction) <= epsilon) {
        return false;
    }

    // find the intersection of the line segment and the plane
    double d = (surface_origin - point1).dot(u_n) / u_n.dot(line_direction);

    intersection = point1 + line_direction * d;
    Vec3 v_i1 = point1 - intersection;
    Vec3 v_i2 = point2 - intersection;

    // check if both points lie on the same side of the plane
    if (v_i1.dot(u_n) * v_i2.dot(u_n) > -epsilon) {
        return false;
    }

    // if our plane is infinite it therefore must cut
    if (infinite)
        return true;

    // Basis matrix of surface
    Mat3 basis;
    basis.col(0) = m_a_vector;
    basis.col(1) = m_b_vector;
    basis.col(2) = m_depth_vector;

    Vec3 frac_coords = basis.inverse() * (intersection - surface_origin);

    // check if the intersection occurs inside the plane
    return (-epsilon < frac_coords(0) && frac_coords(0) < (1 + epsilon) &&
            -epsilon < frac_coords(1) && frac_coords(1) < (1 + epsilon));
}

std::vector<Surface>
generate_surfaces(const Crystal &c,
                  const CrystalSurfaceGenerationParameters &params) {
    std::vector<Surface> result;
    auto f = [&](const HKL &m) {
        if (!Surface::check_systematic_absence(c, m))
            result.emplace_back(Surface(m, c));
    };
    loop_over_miller_indices(f, c, params);
    std::sort(result.begin(), result.end(),
              [](const Surface &a, const Surface &b) {
                  return a.depth() > b.depth();
              });
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

std::vector<Molecule>
Surface::find_molecule_cell_translations(const std::vector<Molecule> &mols,
                                         double depth) const {
    const int num_mols = mols.size();
    std::vector<Molecule> result;
    Vec3 unit_normal = normal_vector();

    Mat3 basis;
    basis.col(0) = m_a_vector;
    basis.col(1) = m_b_vector;
    basis.col(2) = m_depth_vector * depth;
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

    // get box corners in surface coordinates
    Vec3 upper_vec_surface = Vec3(1.0, 1.0, 1.0);
    Vec3 lower_vec_surface = Vec3(0.0, 0.0, 0.0);

    // get corners of the surface box in crystal coordinates
    Mat3N corners(3, 8);
    corners.col(0).setZero();
    corners.col(1) = sb_in_cb.col(0);
    corners.col(2) = sb_in_cb.col(1);
    corners.col(3) = sb_in_cb.col(2);
    corners.col(4) = sb_in_cb.col(0) + sb_in_cb.col(1);
    corners.col(5) = sb_in_cb.col(0) + sb_in_cb.col(2);
    corners.col(6) = sb_in_cb.col(1) + sb_in_cb.col(2);
    corners.col(7) = sb_in_cb.col(0) + sb_in_cb.col(1) + sb_in_cb.col(2);
    Vec3 upper_vec_crystal = corners.rowwise().maxCoeff();
    Vec3 lower_vec_crystal = corners.rowwise().minCoeff();

    occ::log::debug("Lower corner crystal: [{:9.3f}, {:9.3f}, {:9.3f}]",
                    lower_vec_crystal(0), lower_vec_crystal(1),
                    lower_vec_crystal(2));
    occ::log::debug("Upper corner crystal: [{:9.3f}, {:9.3f}, {:9.3f}]",
                    upper_vec_crystal(0), upper_vec_crystal(1),
                    upper_vec_crystal(2));

    Mat3N centroids(3, num_mols);
    for (int i = 0; i < num_mols; i++)
        centroids.col(i) = mols[i].center_of_mass();

    Mat3N frac_centroids_crystal = m_crystal_unit_cell.to_fractional(centroids);

    // buffer these with an extra 1 cell - may not be necessary
    HKL upper = HKL::ceil(upper_vec_crystal.array() + 0.0);
    HKL lower = HKL::floor(lower_vec_crystal.array() - 0.0);

    auto filename =
        fmt::format("surface_{}_{}_{}_mols.txt", m_hkl.h, m_hkl.k, m_hkl.l);
    auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                 fmt::file::CREATE);

    for (int i = 0; i <= 10; i++) {
        for (int j = 0; j <= 10; j++) {
            for (int k = 0; k <= 20; k++) {
                output.print("Xe {}\n",
                             (0.1 * i * basis.col(0) + 0.1 * j * basis.col(1) +
                              0.05 * k * basis.col(2))
                                 .transpose());
            }
        }
    }

    for (int h = lower.h; h <= upper.h; h++) {
        for (int k = lower.k; k <= upper.k; k++) {
            for (int l = lower.l; l <= upper.l; l++) {
                Vec3 frac_t_crystal = HKL{h, k, l}.vector();
                Vec3 cart_shift =
                    m_crystal_unit_cell.to_cartesian(frac_t_crystal);
                output.print("Ru {}\n", cart_shift.transpose());
                Mat3N tmp = cb_in_sb *
                            (frac_centroids_crystal.colwise() + frac_t_crystal);

                for (int i = 0; i < num_mols; i++) {

                    if ((tmp.col(i).array() > lower_vec_surface.array())
                            .all() &&
                        (tmp.col(i).array() <= upper_vec_surface.array())
                            .all()) {
                        auto mol_t = mols[i].translated(cart_shift);
                        mol_t.set_cell_shift({h, k, l});
                        mol_t.set_unit_cell_molecule_idx(i);
                        const auto &tpos = mol_t.positions();
                        for (int j = 0; j < mols[i].size(); j++) {
                            output.print("{} {} {} {}\n",
                                         mols[i].elements()[j].symbol(),
                                         tpos(0, j), tpos(1, j), tpos(2, j));
                        }
                        result.push_back(mol_t);
                    }
                }
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
    const CrystalDimers &crystal_dimers) const {
    SurfaceCutResult result(crystal_dimers);
    result.origin = Vec3(0.0, 0.0, 0.0);

    double required_depth = 0.0;
    Vec3 unit_normal = normal_vector();

    std::vector<Molecule> molecules;
    for (const auto &neighbors : crystal_dimers.molecule_neighbors) {
        molecules.push_back(neighbors[0].dimer.a());
        for (const auto &[dimer, unique_idx] : neighbors) {
            Vec3 v_ab = dimer.b().center_of_mass() - dimer.a().center_of_mass();
            required_depth =
                std::max(required_depth, std::abs(v_ab.dot(unit_normal)));
        }
    }
    double depth_scale = std::ceil(required_depth / m_depth_vector.norm());
    fmt::print("Required depth = {} x depth ({:12.6f} angs)\n", depth_scale,
               required_depth);

    result.molecules = find_molecule_cell_translations(molecules, depth_scale);

    Mat3 basis;
    basis.col(0) = m_a_vector;
    basis.col(1) = m_b_vector;
    basis.col(2) = m_depth_vector * depth_scale;
    Mat3 basis_inverse = basis.inverse();
    result.basis = basis;

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
    for (const auto &mol : result.molecules) {
        auto filename = fmt::format("crystal_surface_{}_{}_{}_t{}.txt", m_hkl.h,
                                    m_hkl.k, m_hkl.l, t_index);
        auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                     fmt::file::CREATE);

        int index = mol.unit_cell_molecule_idx();
        for (int i = 0; i < mol.size(); i++) {
            const auto &nums = mol.elements();
            const auto &pos = mol.positions();
            output.print("{} {:12.6f} {:12.6f} {:12.6f} {} {} {} 0.0 0.0\n",
                         nums[i].symbol(), pos(0, i), pos(1, i), pos(2, i),
                         t_index, index, 0);
        }
        if (index > result.above.size()) {
            throw std::runtime_error("Must pass unit cell dimers to "
                                     "count_crystal_dimers_cut_by_surface");
        }
        const auto &uc_shift = mol.cell_shift();
        Vec3 shift(uc_shift[0], uc_shift[1], uc_shift[2]);
        Vec3 translation = m_crystal_unit_cell.to_cartesian(shift);
        Vec3 frac_c = basis_inverse * (mol.center_of_mass());
        occ::log::debug("mol in surface index = {}, unit cell index = {}",
                        t_index, index);
        occ::log::debug("Fractional center: {:12.6f} {:12.6f} {:12.6f}",
                        frac_c(0), frac_c(1), frac_c(2));

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
            double r = dimer.center_of_mass_distance();
            Vec3 c_a = dimer.a().center_of_mass() + translation;
            Vec3 c_b = dimer.b().center_of_mass() + translation;
            cut_count.neighbors++;
            cut_count.energy_neighbors += e;
            neighbor_counts_bulk[neighbor_index]++;
            if ((c_a - mol.center_of_mass()).norm() > 1e-3) {
                fmt::print("Error in difference, mol_a doesn't match "
                           "translated mol\n");
            }
            if ((std::abs(r - (c_a - c_b).norm()) > 1e-3)) {
                fmt::print("Error in difference, distance doesn't match"
                           "translated mol\n");
            }
            Molecule mol2 = dimer.b();
            mol2.translate(translation);
            int region = 0;

            Vec3 frac_coords = basis.inverse() * c_b;
            if (frac_coords(2) > (1 + 1e-6)) {
                region = 1;
                cut_count.above++;
                cut_count.energy_above += e;
                neighbor_counts_above[neighbor_index]++;
            } else if (frac_coords(2) < -1e-6) {
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
            for (int ii = 0; ii < mol2.size(); ii++) {
                const auto &nums = mol2.elements();
                const auto &pos = mol2.positions();
                output.print("{} {:12.6f} {:12.6f} {:12.6f} {} {} {} {} {}\n",
                             nums[ii].symbol(), pos(0, ii), pos(1, ii),
                             pos(2, ii), t_index, index, region, e, r);
            }
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

} // namespace occ::crystal
