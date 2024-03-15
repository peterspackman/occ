#include <chrono>
#include <filesystem>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/kdtree.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/numpy.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/geometry/linear_hashed_marching_cubes.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/io/xyz.h>
#include <occ/io/obj.h>
#include <occ/io/ply.h>
#include <occ/main/isosurface.h>
#include <occ/main/occ_isosurface.h>

namespace fs = std::filesystem;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Element;
using occ::core::Interpolator1D;
using occ::core::Molecule;
using occ::main::PromoleculeDensityFunctor;
using occ::main::StockholderWeightFunctor;
using occ::io::IsosurfaceMesh;
using occ::io::VertexProperties;


template <typename F>
IsosurfaceMesh as_mesh(const F &b, const std::vector<float> &vertices,
                       const std::vector<uint32_t> &indices,
                       const std::vector<float> &normals) {

    IsosurfaceMesh result(vertices.size() / 3, indices.size() / 3);

    float length = b.side_length();
    const auto &origin = b.origin();
    for (size_t i = 0; i < vertices.size(); i += 3) {
        result.vertices(0, i / 3) = vertices[i] * length + origin(0);
        result.vertices(1, i / 3) = vertices[i + 1] * length + origin(1);
        result.vertices(2, i / 3) = vertices[i + 2] * length + origin(2);

        Eigen::Vector3f normal(normals[i], normals[i + 1], normals[i + 2]);
        result.normals.col(i / 3) = normal.normalized();
    }

    result.vertices.array() *= occ::units::BOHR_TO_ANGSTROM;

    for (size_t i = 0; i < indices.size(); i += 3) {
        result.faces(0, i / 3) = indices[i];
        result.faces(1, i / 3) = indices[i + 2];
        result.faces(2, i / 3) = indices[i + 1];
    }

    return result;
}

template <typename F> IsosurfaceMesh extract_surface(F &func) {
    occ::timing::StopWatch sw;
    size_t max_depth = func.subdivisions();
    auto mc = occ::geometry::mc::MarchingCubes(std::pow(2, max_depth));
    occ::log::debug("target separation: {}",
                    (func.side_length() / std::pow(2, max_depth)) *
                        occ::units::BOHR_TO_ANGSTROM);
    occ::log::debug("naive voxel count: {}",
                    std::pow(std::pow(2, max_depth) + 1, 3));
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<uint32_t> faces;
    sw.start();
    mc.extract_with_normals(func, vertices, faces, normals);
    sw.stop();
    double max_calls = std::pow(2, 3 * max_depth);
    occ::log::debug("{} calls ({} % of conventional)", func.num_calls(),
                    (func.num_calls() / max_calls) * 100);
    occ::log::info("Surface extraction took {:.5f} s", sw.read());

    occ::log::info("Surface has {} vertices, {} faces", vertices.size() / 3,
                   faces.size() / 3);
    return as_mesh(func, vertices, faces, normals);
}

template <typename F> IsosurfaceMesh extract_surface_hashed(F &func) {
    size_t min_depth = 4;
    occ::timing::StopWatch sw;
    size_t max_depth = func.subdivisions();
    occ::log::debug("minimum subdivisions = {}, maximum subdivisions = {}",
                    min_depth, max_depth);
    auto mc = occ::geometry::mc::LinearHashedMarchingCubes(max_depth);
    mc.min_depth = min_depth;
    occ::log::debug("target separation: {}",
                    (func.side_length() / std::pow(2, max_depth)) *
                        occ::units::BOHR_TO_ANGSTROM);
    occ::log::debug("naive voxel count: {}",
                    std::pow(std::pow(2, max_depth) + 1, 3));
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<uint32_t> faces;
    sw.start();
    mc.extract_with_normals(func, vertices, faces, normals);
    sw.stop();
    double max_calls = std::pow(2, 3 * max_depth);
    occ::log::debug("{} calls ({} % of conventional)", func.num_calls(),
                    (func.num_calls() / max_calls) * 100);
    occ::log::info("Surface extraction took {:.5f} s", sw.read());

    occ::log::info("Surface has {} vertices, {} faces", vertices.size() / 3,
                   faces.size() / 3);
    return as_mesh(func, vertices, faces, normals);
}

VertexProperties compute_surface_properties(const Molecule &m1,
                                            const Molecule &m2,
                                            const Eigen::Matrix3Xf &vertices) {
    Eigen::Matrix3Xf inside = m1.positions().cast<float>();
    Eigen::Matrix3Xf outside = m2.positions().cast<float>();
    Eigen::VectorXf vdw_inside = m1.vdw_radii().cast<float>();
    Eigen::VectorXf vdw_outside = m2.vdw_radii().cast<float>();

    VertexProperties properties(vertices.cols());
    int nthreads = occ::parallel::get_num_threads();

    occ::core::KDTree<float> interior_tree(inside.rows(), inside,
                                           occ::core::max_leaf);
    interior_tree.index->buildIndex();

    occ::core::KDTree<float> exterior_tree(outside.rows(), outside,
                                           occ::core::max_leaf);
    exterior_tree.index->buildIndex();

    occ::log::info("Indexes built");
    constexpr size_t num_results = 6;

    auto fill_properties = [&](int thread_id) {
        std::vector<size_t> indices(num_results);
        std::vector<float> dist_sq(num_results);
        std::vector<float> dist_norm(num_results);

        for (int i = 0; i < vertices.cols(); i++) {
            if (i % nthreads != thread_id)
                continue;

            Eigen::Vector3f v = vertices.col(i);
            {

                float dist_inside_norm = std::numeric_limits<float>::max();
                nanoflann::KNNResultSet<float> results(num_results);
                results.init(&indices[0], &dist_sq[0]);
                bool populated = interior_tree.index->findNeighbors(
                    results, v.data(), nanoflann::SearchParams());
                properties.di(i) = std::sqrt(dist_sq[0]);
                properties.di_idx(i) = indices[0];

                size_t inside_idx = 0;
                for (int idx = 0; idx < results.size(); idx++) {

                    float vdw = vdw_inside(indices[idx]);
                    float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

                    if (dnorm < dist_inside_norm) {
                        inside_idx = indices[idx];
                        dist_inside_norm = dnorm;
                    }
                }
                properties.di_norm(i) = dist_inside_norm;
                properties.di_norm_idx(i) = inside_idx;
            }

            {
                float dist_outside_norm = std::numeric_limits<float>::max();
                nanoflann::KNNResultSet<float> results(num_results);
                results.init(&indices[0], &dist_sq[0]);
                bool populated = exterior_tree.index->findNeighbors(
                    results, v.data(), nanoflann::SearchParams());
                properties.de(i) = std::sqrt(dist_sq[0]);
                properties.de_idx(i) = indices[0];

                size_t outside_idx = 0;
                for (int idx = 0; idx < results.size(); idx++) {

                    float vdw = vdw_outside(indices[idx]);
                    float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

                    if (dnorm < dist_outside_norm) {
                        outside_idx = indices[idx];
                        dist_outside_norm = dnorm;
                    }
                }
                properties.de_norm(i) = dist_outside_norm;
                properties.de_norm_idx(i) = outside_idx;
            }

            properties.dnorm(i) = properties.de_norm(i) + properties.di_norm(i);
        }
    };

    occ::timing::start(occ::timing::category::isosurface_properties);
    occ::parallel::parallel_do(fill_properties);
    occ::timing::stop(occ::timing::category::isosurface_properties);

    return properties;
}

namespace occ::main {

CLI::App *add_isosurface_subcommand(CLI::App &app) {
    CLI::App *iso =
        app.add_subcommand("isosurface", "compute molecular isosurfaces");
    auto config = std::make_shared<IsosurfaceConfig>();

    iso->add_option("geometry", config->geometry_filename,
                    "input geometry file (xyz)")
        ->required();

    iso->add_option("environment", config->environment_filename,
                    "environment geometry file (xyz)");

    iso->add_option("--kind", config->kind,
                    "surface kind");

    iso->add_option("--max-depth", config->max_depth, "Maximum voxel depth");
    iso->add_option("--separation", config->separation,
                    "targt voxel separation");
    iso->add_option("--isovalue", config->isovalue, "target isovalue");
    iso->add_flag("--hashed", config->use_hashed_mc, "use linear hashed octree");
    iso->add_option("--background-density", config->background_density,
                    "add background density to close surface");

    iso->add_option("--output,-o", config->output_filename,
                    "destination to write file");

    iso->fallthrough();
    iso->callback([config]() { run_isosurface_subcommand(*config); });
    return iso;
}

void run_isosurface_subcommand(IsosurfaceConfig const &config) {
    IsosurfaceMesh mesh;
    VertexProperties properties;

    if(occ::qm::Wavefunction::is_likely_wavefunction_filename(config.geometry_filename)) {
	if(config.kind == "esp") {
	    auto wfn = occ::qm::Wavefunction::load(config.geometry_filename);
	    auto func = ElectricPotentialFunctor(wfn, config.separation * occ::units::ANGSTROM_TO_BOHR);
	    func.set_isovalue(config.isovalue);
	    mesh = extract_surface(func);
	}
	else {
	    auto wfn = occ::qm::Wavefunction::load(config.geometry_filename);
	    auto func = ElectronDensityFunctor(wfn, config.separation * occ::units::ANGSTROM_TO_BOHR);
	    func.set_isovalue(config.isovalue);
	    mesh = extract_surface(func);
	}
    }
    else if (!config.environment_filename.empty()) {
        Molecule m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);
        Molecule m2 =
            occ::io::molecule_from_xyz_file(config.environment_filename);

        occ::log::info("Interior has {} atoms", m1.size());
        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}",
                        config.geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m1.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        occ::log::info("Environment has {} atoms", m2.size());
        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}",
                        config.geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m2.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        auto func = StockholderWeightFunctor(
            m1, m2, config.separation * occ::units::ANGSTROM_TO_BOHR);
        func.set_background_density(config.background_density);
	if(config.use_hashed_mc) {
	    mesh = extract_surface_hashed(func);
	}
	else {
	    mesh = extract_surface(func);
	}
        properties = compute_surface_properties(m1, m2, mesh.vertices);
    } else {
        Molecule m = occ::io::molecule_from_xyz_file(config.geometry_filename);
        occ::log::info("Molecule has {} atoms", m.size());

        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}",
                        config.geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        auto func = PromoleculeDensityFunctor(
            m, config.separation * occ::units::ANGSTROM_TO_BOHR);
        func.set_isovalue(config.isovalue);
	if(config.use_hashed_mc) {
	    mesh = extract_surface_hashed(func);
	}
	else {
	    mesh = extract_surface(func);
	}
    }

    Eigen::Vector3f lower_left = mesh.vertices.rowwise().minCoeff();
    Eigen::Vector3f upper_right = mesh.vertices.rowwise().maxCoeff();
    occ::log::info("Lower corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   lower_left(0), lower_left(1), lower_left(2));
    occ::log::info("Upper corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   upper_right(0), upper_right(1), upper_right(2));

    occ::log::info("Writing surface to {}", config.output_filename);
    write_obj_file(config.output_filename, mesh, properties);
}

} // namespace occ::main
