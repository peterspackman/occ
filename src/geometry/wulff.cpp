#include <numeric>
#include <occ/core/log.h>
#include <occ/geometry/quickhull.h>
#include <occ/geometry/wulff.h>

namespace occ::geometry {

Mat3N project_to_plane(const Mat3N &points, const Vec3 &plane_normal) {
    Mat3N projected_points =
        points.array() -
        (plane_normal * (plane_normal.transpose() * points)).array();

    Vec3 a_vector = projected_points.col(1) - projected_points.col(0);
    Vec3 b_vector = plane_normal.cross(a_vector);

    Vec u = projected_points.transpose() * a_vector;
    Vec v = projected_points.transpose() * b_vector;

    Mat3N result = Mat3N::Zero(3, points.cols());
    result.row(0) = u.transpose();
    result.row(1) = v.transpose();

    return result;
}

void Facet::reorder(const Mat3N &points) {
    if (point_index.empty())
        return;

    Mat3N points_2d = project_to_plane(points, normal);

    Vec3 centroid = points_2d.rowwise().mean();
    Mat3N directions = points_2d.colwise() - centroid;
    for (int i = 0; i < directions.cols(); ++i) {
        directions.col(i).normalize();
    }

    std::sort(
        point_index.begin(), point_index.end(), [&directions](int i1, int i2) {
            double angle1 = std::atan2(directions(1, i1), directions(0, i1));
            double angle2 = std::atan2(directions(1, i2), directions(0, i2));
            return angle1 < angle2;
        });
}

void Facet::reorder_and_triangulate(const Mat3N &points) {
    if (point_index.empty())
        return;

    // assumes we have at least 3 points
    const size_t N = point_index.size();
    reorder(points);

    this->triangles = IMat3N(3, N - 2);

    this->triangles.row(0).array() = point_index[0];
    this->triangles.row(1) =
        Eigen::Map<const IVec>(point_index.data() + 1, N - 2);
    this->triangles.row(2) =
        Eigen::Map<const IVec>(point_index.data() + 2, N - 2);
}

WulffConstruction::WulffConstruction(
    const Mat3N &facet_normals, const Vec &facet_energies,
    const std::vector<std::string> &facet_labels) {

    const size_t N = facet_energies.rows();

    Mat3N dual_points(3, N);
    for (int i = 0; i < N; i++) {
        double energy = facet_energies(i);
        // dual = p / (|p|^2), since we haven't scaled p just divide by energy
        Vec3 dual = facet_normals.col(i).array() / energy;
        m_facets.push_back(Facet{energy, facet_normals.col(i),
                                 (facet_labels.size() > i)
                                     ? facet_labels[i]
                                     : fmt::format("facet_{}", i),
                                 dual});
        dual_points.col(i) = dual;
    }
    quickhull::QuickHull<double> hull_builder;

    auto hull = hull_builder.getConvexHull(dual_points.data(),
                                           dual_points.cols(), true);

    occ::log::debug("Convex hull has {} faces, {} vertices",
                    hull.triangles().cols(), hull.vertices().cols());
    IMat3N triangles = hull.triangles().cast<int>();
    extract_wulff_from_dual_hull_simplices(triangles);
}

void WulffConstruction::extract_wulff_from_dual_hull_simplices(
    const IMat3N &simplices) {

    m_wulff_vertices = Mat3N(3, simplices.cols());
    for (int i = 0; i < m_wulff_vertices.cols(); i++) {
        auto &facet_a = m_facets[simplices(0, i)];
        auto &facet_b = m_facets[simplices(1, i)];
        auto &facet_c = m_facets[simplices(2, i)];
        m_wulff_vertices.col(i) =
            (facet_b.dual - facet_a.dual).cross(facet_c.dual - facet_a.dual);
        double scale_factor =
            facet_a.energy / m_wulff_vertices.col(i).dot(facet_a.normal);
        m_wulff_vertices.col(i).array() *= scale_factor;

        // push_back facet_indices
        facet_a.point_index.push_back(i);
        facet_b.point_index.push_back(i);
        facet_c.point_index.push_back(i);
    }

    size_t N = 0;
    for (auto &facet : m_facets) {
        facet.reorder_and_triangulate(m_wulff_vertices);
        if (facet.point_index.size() > 0) {
            N += facet.triangles.cols();
        }
    }

    m_wulff_triangles = IMat3N(3, N);
    m_wulff_triangle_indices = IVec(N);
    N = 0;
    for (int f = 0; f < m_facets.size(); f++) {
        const auto &facet = m_facets[f];
        if (facet.point_index.size() <= 0)
            continue;
        int size = facet.triangles.cols();
        m_wulff_triangles.block(0, N, 3, size) = facet.triangles;
        m_wulff_triangle_indices.block(N, 0, size, 1).array() = f;
        N += size;
    }
}

const Mat3N &WulffConstruction::vertices() const { return m_wulff_vertices; }

const IMat3N &WulffConstruction::triangles() const { return m_wulff_triangles; }

} // namespace occ::geometry
