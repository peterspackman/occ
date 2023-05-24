#include <algorithm>
#include <fmt/core.h>
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
                                const Vec3 &point2) const {
    Vec3 line_direction = (point2 - point1).normalized();
    Vec3 u_n = normal_vector();

    // line is parallel to the plane
    if (u_n.dot(line_direction) <= 1e-6)
        return false;

    // find the intersection of the line segment and the plane
    double d = (surface_origin - point1).dot(u_n) / u_n.dot(line_direction);

    Vec3 intersection = point1 + line_direction * d;

    // check if intersecting point lies on the line segment
    if ((intersection - point1).dot(point2 - intersection) < 0) {
        return false;
    }

    // Basis matrix of surface
    Mat3 basis;
    basis.col(0) = m_a_vector;
    basis.col(1) = m_b_vector;
    basis.col(2) = m_depth_vector;

    Vec3 frac_coords = basis.inverse() * (intersection - surface_origin);

    // check if it's inside the cell
    return (0 <= frac_coords(0) && frac_coords(0) <= 1 && 0 <= frac_coords(1) &&
            frac_coords(1) <= 1);
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
                                   const HKL &hkl2) {}

} // namespace occ::crystal
