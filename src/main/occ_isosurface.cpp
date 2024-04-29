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
#include <occ/geometry/marching_cubes.h>
#include <occ/io/cifparser.h>
#include <occ/io/xyz.h>
#include <occ/io/obj.h>
#include <occ/io/ply.h>
#include <occ/io/tinyply.h>
#include <occ/isosurface/isosurface.h>
#include <occ/isosurface/curvature.h>
#include <occ/main/occ_isosurface.h>
#include <occ/crystal/crystal.h>

namespace fs = std::filesystem;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::FVec;
using occ::core::Element;
using occ::core::Interpolator1D;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::io::IsosurfaceMesh;
using occ::io::VertexProperties;
using occ::qm::Wavefunction;
namespace iso = occ::isosurface;

namespace occ::main {

std::string to_string(IsosurfaceConfig::Property prop) {
    switch(prop) {
	case IsosurfaceConfig::Property::Dnorm: return "d_norm";
	case IsosurfaceConfig::Property::Dint_norm: return "di_norm";
	case IsosurfaceConfig::Property::Dint: return "di";
	case IsosurfaceConfig::Property::Dext_norm: return "de_norm";
	case IsosurfaceConfig::Property::Dext: return "de";
	case IsosurfaceConfig::Property::FragmentPatch: return "fragment_patch";
	case IsosurfaceConfig::Property::ShapeIndex: return "shape_index";
	case IsosurfaceConfig::Property::Curvedness: return "curvedness";
	case IsosurfaceConfig::Property::PromoleculeDensity: return "promolecule_density";
	case IsosurfaceConfig::Property::EEQ_ESP: return "eeq_esp";
	case IsosurfaceConfig::Property::ESP: return "esp";
	case IsosurfaceConfig::Property::ElectronDensity: return "electron_density";
	case IsosurfaceConfig::Property::DeformationDensity: return "deformation_density";
	case IsosurfaceConfig::Property::Orbital: return "orbital_density";
	case IsosurfaceConfig::Property::SpinDensity: return "spin_density";
    }
}

std::string to_string(IsosurfaceConfig::Surface surface) {
    switch(surface) {
	case IsosurfaceConfig::Surface::PromoleculeDensity: return "promolecule_density";
	case IsosurfaceConfig::Surface::Hirshfeld: return "hirshfeld";
	case IsosurfaceConfig::Surface::EEQ_ESP: return "eeq_esp";
	case IsosurfaceConfig::Surface::ESP: return "esp";
	case IsosurfaceConfig::Surface::ElectronDensity: return "electron_density";
	case IsosurfaceConfig::Surface::DeformationDensity: return "deformation_density";
	case IsosurfaceConfig::Surface::Orbital: return "orbital_density";
	case IsosurfaceConfig::Surface::SpinDensity: return "spin_density";
	case IsosurfaceConfig::Surface::CrystalVoid: return "void";
    }
}
}

template <typename F>
IsosurfaceMesh as_mesh(const F &func, const std::vector<float> &vertices,
                       const std::vector<uint32_t> &indices,
                       const std::vector<float> &normals,
		       const std::vector<float> &curvature) {

    IsosurfaceMesh result;

    func.remap_vertices(vertices, result.vertices);
    result.normals.resize(vertices.size());
    result.faces.resize(indices.size());
    result.gaussian_curvature.reserve(curvature.size() / 2);
    result.mean_curvature.reserve(curvature.size() / 2);

    for (size_t i = 0; i < normals.size(); i += 3) {
        Eigen::Vector3f normal = Eigen::Vector3f(normals[i], normals[i + 1], normals[i + 2]);
	result.normals[i] = normal(0);
	result.normals[i + 1] = normal(1);
	result.normals[i + 2] = normal(2);
    }

    for (size_t i = 0; i < curvature.size(); i += 2) {
	result.mean_curvature.push_back(curvature[i]);
	result.gaussian_curvature.push_back(curvature[i + 1]);
    }

    // winding is backward for some reason out of the marching cubes.
    for(size_t i = 0; i < indices.size(); i += 3) {
	result.faces[i] = indices[i];
	result.faces[i + 1] = indices[i + 1];
	result.faces[i + 2] = indices[i + 2];
    }

    return result;
}

template <typename F>
IsosurfaceMesh extract_surface(F &func, float isovalue) {
    occ::timing::StopWatch sw;
    auto cubes = func.cubes_per_side();
    occ::log::info("Marching cubes voxels: {}x{}x{}", cubes(0), cubes(1), cubes(2));
    auto mc = occ::geometry::mc::MarchingCubes(cubes(0), cubes(1), cubes(2));
    mc.set_origin_and_side_lengths(func.origin(), func.side_length());
    mc.isovalue = isovalue;

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> curvature;
    std::vector<uint32_t> faces;
    sw.start();
    mc.extract_with_curvature(func, vertices, faces, normals, curvature);
    sw.stop();
    occ::log::debug("Required {} function calls ", func.num_calls());
    occ::log::info("Surface extraction took {:.5f} s", sw.read());

    occ::log::info("Surface has {} vertices, {} faces", vertices.size() / 3,
                   faces.size() / 3);
    if(vertices.size() < 3) {
	throw std::runtime_error("Invalid isosurface encountered, not enough vertices?");
    }
    return as_mesh(func, vertices, faces, normals, curvature);
}

VertexProperties compute_atom_surface_properties(const Molecule &m1, const Molecule &m2, 
						 Eigen::Ref<const Eigen::Matrix3Xf> vertices) {
    const size_t N = vertices.cols();
    constexpr size_t num_results = 6;
    FVec di_norm(N), dnorm(N);
    VertexProperties properties;
    int nthreads = occ::parallel::get_num_threads();

    if(m1.size() > 0) {
	Eigen::Matrix3Xf inside = m1.positions().cast<float>();
	Eigen::VectorXf vdw_inside = m1.vdw_radii().cast<float>();

	occ::core::KDTree<float> interior_tree(inside.rows(), inside,
					       occ::core::max_leaf);
	interior_tree.index->buildIndex();

	FVec di(N);
	IVec di_idx(N), di_norm_idx(N);

	auto fill_interior_properties = [&](int thread_id) {
	    std::vector<size_t> indices(num_results);
	    std::vector<float> dist_sq(num_results);
	    std::vector<float> dist_norm(num_results);

	    for (int i = 0; i < vertices.cols(); i++) {
		if (i % nthreads != thread_id)
		    continue;

		Eigen::Vector3f v = vertices.col(i);
		float dist_inside_norm = std::numeric_limits<float>::max();
		nanoflann::KNNResultSet<float> results(num_results);
		results.init(&indices[0], &dist_sq[0]);
		bool populated = interior_tree.index->findNeighbors(
		    results, v.data(), nanoflann::SearchParams());
		di(i) = std::sqrt(dist_sq[0]);
		di_idx(i) = indices[0];

		size_t inside_idx = 0;
		for (int idx = 0; idx < results.size(); idx++) {

		    float vdw = vdw_inside(indices[idx]);
		    float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

		    if (dnorm < dist_inside_norm) {
			inside_idx = indices[idx];
			dist_inside_norm = dnorm;
		    }
		}
		di_norm(i) = dist_inside_norm;
		di_norm_idx(i) = inside_idx;
	    }
	};


	occ::timing::start(occ::timing::category::isosurface_properties);
	occ::parallel::parallel_do(fill_interior_properties);
	occ::timing::stop(occ::timing::category::isosurface_properties);


	properties.add_property("di", di);
	properties.add_property("di_idx", di_idx);
	properties.add_property("di_norm", di_norm);
	properties.add_property("di_norm_idx", di_norm_idx);
    }

    if(m2.size() > 0) {
	FVec de(N), de_norm(N);
	IVec de_idx(N), de_norm_idx(N);
	Eigen::Matrix3Xf outside = m2.positions().cast<float>();
	Eigen::VectorXf vdw_outside = m2.vdw_radii().cast<float>();
	occ::core::KDTree<float> exterior_tree(outside.rows(), outside,
					       occ::core::max_leaf);
	exterior_tree.index->buildIndex();
	auto fill_exterior_properties = [&](int thread_id) {
	    std::vector<size_t> indices(num_results);
	    std::vector<float> dist_sq(num_results);
	    std::vector<float> dist_norm(num_results);

	    for (int i = 0; i < vertices.cols(); i++) {
		if (i % nthreads != thread_id)
		    continue;

		Eigen::Vector3f v = vertices.col(i);
		float dist_outside_norm = std::numeric_limits<float>::max();
		nanoflann::KNNResultSet<float> results(num_results);
		results.init(&indices[0], &dist_sq[0]);
		bool populated = exterior_tree.index->findNeighbors(
		    results, v.data(), nanoflann::SearchParams());
		de(i) = std::sqrt(dist_sq[0]);
		de_idx(i) = indices[0];

		size_t outside_idx = 0;
		for (int idx = 0; idx < results.size(); idx++) {

		    float vdw = vdw_outside(indices[idx]);
		    float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

		    if (dnorm < dist_outside_norm) {
			outside_idx = indices[idx];
			dist_outside_norm = dnorm;
		    }
		}
		de_norm(i) = dist_outside_norm;
		de_norm_idx(i) = outside_idx;

		if(m1.size() > 0) {
		    dnorm(i) = de_norm(i) + di_norm(i);
		}
	    }
	};
	occ::timing::start(occ::timing::category::isosurface_properties);
	occ::parallel::parallel_do(fill_exterior_properties);
	occ::timing::stop(occ::timing::category::isosurface_properties);

	properties.add_property("de_idx", de_idx);
	properties.add_property("de", de);
	properties.add_property("de_norm", de_norm);
	properties.add_property("de_norm_idx", de_norm_idx);
	if(m1.size() > 0) {
	    properties.add_property("dnorm", dnorm);
	}

    }
    
    return properties;
}

namespace occ::main {

IsosurfaceConfig::Surface IsosurfaceConfig::surface_type() const {
    std::vector<IsosurfaceConfig::Surface> surfaces{
	IsosurfaceConfig::Surface::PromoleculeDensity,
	IsosurfaceConfig::Surface::Hirshfeld,
	IsosurfaceConfig::Surface::EEQ_ESP,
	IsosurfaceConfig::Surface::ElectronDensity,
	IsosurfaceConfig::Surface::ESP,
	IsosurfaceConfig::Surface::SpinDensity,
	IsosurfaceConfig::Surface::DeformationDensity,
	IsosurfaceConfig::Surface::Orbital,
	IsosurfaceConfig::Surface::CrystalVoid
    };

    ankerl::unordered_dense::map<std::string, IsosurfaceConfig::Surface> name2surface{
	// ESP
	{"electric_potential", IsosurfaceConfig::Surface::ESP},
	{"electric potential", IsosurfaceConfig::Surface::ESP},
	// Electron density
	{"electron_density", IsosurfaceConfig::Surface::ElectronDensity},
	{"electron density", IsosurfaceConfig::Surface::ElectronDensity},
	{"rho", IsosurfaceConfig::Surface::ElectronDensity},
	{"density", IsosurfaceConfig::Surface::ElectronDensity},
	// Promolecule
	{"promol", IsosurfaceConfig::Surface::PromoleculeDensity},
	{"pro", IsosurfaceConfig::Surface::PromoleculeDensity},
	// Hirshfeld
	{"stockholder weight", IsosurfaceConfig::Surface::Hirshfeld},
	{"hs", IsosurfaceConfig::Surface::Hirshfeld},
	{"stockholder_weight", IsosurfaceConfig::Surface::Hirshfeld},
	// void
	{"crystal_void", IsosurfaceConfig::Surface::CrystalVoid},
    };
    for(const auto &s: surfaces) {
	name2surface.insert({to_string(s), s});
    }

    auto s = occ::util::to_lower_copy(kind);
    auto loc = name2surface.find(s);
    if(loc != name2surface.end()) {
	return loc->second;
    }
    throw std::runtime_error(fmt::format("Unknown surface type: {}", kind));
}

std::vector<IsosurfaceConfig::Property> IsosurfaceConfig::surface_properties() const {
    std::vector<IsosurfaceConfig::Property> properties{
	IsosurfaceConfig::Property::Dnorm,
	IsosurfaceConfig::Property::Dint_norm,
	IsosurfaceConfig::Property::Dext_norm,
	IsosurfaceConfig::Property::Dint,
	IsosurfaceConfig::Property::Dext,
	IsosurfaceConfig::Property::FragmentPatch,
	IsosurfaceConfig::Property::ShapeIndex,
	IsosurfaceConfig::Property::Curvedness,
	IsosurfaceConfig::Property::EEQ_ESP,
	IsosurfaceConfig::Property::PromoleculeDensity,
	IsosurfaceConfig::Property::ESP,
	IsosurfaceConfig::Property::ElectronDensity,
	IsosurfaceConfig::Property::SpinDensity,
	IsosurfaceConfig::Property::DeformationDensity,
	IsosurfaceConfig::Property::Orbital
    };

    ankerl::unordered_dense::set<IsosurfaceConfig::Property> result{
	IsosurfaceConfig::Property::Dint,
	IsosurfaceConfig::Property::Dint_norm,
	IsosurfaceConfig::Property::ShapeIndex,
	IsosurfaceConfig::Property::Curvedness,
    };

    if(have_environment_file()) {
	result.insert(IsosurfaceConfig::Property::Dext);
	result.insert(IsosurfaceConfig::Property::Dext_norm);
	result.insert(IsosurfaceConfig::Property::Dnorm);
	result.insert(IsosurfaceConfig::Property::FragmentPatch);
    }

    ankerl::unordered_dense::map<std::string, IsosurfaceConfig::Property> name2prop{
	// ESP
	{"electric_potential", IsosurfaceConfig::Property::ESP},
	{"electric potential", IsosurfaceConfig::Property::ESP},
	// Electron density
	{"electron density", IsosurfaceConfig::Property::ElectronDensity},
	{"rho", IsosurfaceConfig::Property::ElectronDensity},
	{"density", IsosurfaceConfig::Property::ElectronDensity},
    };

    for(const auto &p: properties) {
	name2prop.insert({to_string(p), p});
    }

    for(const auto &p: additional_properties) {
	auto s = occ::util::to_lower_copy(p);
	auto loc = name2prop.find(s);
	if(loc != name2prop.end()) {
	    result.insert(loc->second);
	}
	else {
	    occ::log::warn("Unknown property: {}, ignoring", p);
	}
    }
    return std::vector<IsosurfaceConfig::Property>(result.begin(), result.end());
}

bool IsosurfaceConfig::have_environment_file() const {
    return !environment_filename.empty();
}

bool IsosurfaceConfig::requires_crystal() const {
    return surface_type() == IsosurfaceConfig::Surface::CrystalVoid;
}

bool IsosurfaceConfig::requires_environment() const {
    auto s = surface_type();
    switch(s) {
	case IsosurfaceConfig::Surface::Hirshfeld: return true;
	default: break;
    }

    for(const auto &prop: surface_properties()) {
	switch(prop) {
	    case IsosurfaceConfig::Property::Dext: return true;
	    case IsosurfaceConfig::Property::Dext_norm: return true;
	    case IsosurfaceConfig::Property::Dnorm: return true;
	    case IsosurfaceConfig::Property::FragmentPatch: return true;
	    default: break;
	}
    }
    return false;
}

bool IsosurfaceConfig::requires_wavefunction() const {
    auto s = surface_type();
    switch(s) {
	case IsosurfaceConfig::Surface::ESP: return true;
	case IsosurfaceConfig::Surface::ElectronDensity: return true;
	case IsosurfaceConfig::Surface::DeformationDensity: return true;
	case IsosurfaceConfig::Surface::Orbital: return true;
	case IsosurfaceConfig::Surface::SpinDensity: return true;
	default: break;
    }

    for(const auto &prop: surface_properties()) {
	switch(prop) {
	    case IsosurfaceConfig::Property::ESP: return true;
	    case IsosurfaceConfig::Property::ElectronDensity: return true;
	    case IsosurfaceConfig::Property::DeformationDensity: return true;
	    case IsosurfaceConfig::Property::Orbital: return true;
	    case IsosurfaceConfig::Property::SpinDensity: return true;
	    default: break;
	}
    }
    return false;
}

void ensure_isosurface_configuration_valid(const IsosurfaceConfig &config, bool have_wavefunction, bool have_crystal) {
    if(config.requires_wavefunction() && !have_wavefunction) {
	throw std::runtime_error("Surface, or surface properties require a wavefunction");
    }
    if(config.requires_environment() && !config.have_environment_file()) {
	throw std::runtime_error("Surface, or surface properties require an environment");
    }
    if(config.requires_crystal() && !have_crystal) {
	throw std::runtime_error("Surface, or surface properties requires a crystal structure");
    }
}

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

    iso->add_flag("--binary,!--ascii", config->binary_output, "Write binary/ascii file format (default binary)");

    iso->add_option("--wavefunction,-w", config->wavefunction_filename,
		    "Wavefunction filename for geometry");
    iso->add_option("--wfn-rotation,--wfn_rotation", config->wfn_rotation,
                     "Rotation for supplied wavefunction (row major order)")
        ->expected(9);

    iso->add_option("--wfn-translation,--wfn_translation", config->wfn_translation,
                     "Translation for monomer A")
        ->expected(3);

    iso->add_option("--properties,--additional_properties",
	            config->additional_properties,
                    "Additional properties to compute");

    iso->add_option("--max-depth", config->max_depth, "Maximum voxel depth");
    iso->add_option("--separation", config->separation,
                    "targt voxel separation (Bohr)");
    iso->add_option("--isovalue", config->isovalue, "target isovalue");
    iso->add_option("--background-density", config->background_density,
                    "add background density to close surface");

    iso->add_option("--output,-o", config->output_filename,
                    "destination to write file");

    iso->fallthrough();
    iso->callback([config]() { run_isosurface_subcommand(*config); });
    return iso;
}

Wavefunction load_wfn(const IsosurfaceConfig &config) {
    Wavefunction wfn;
    if(Wavefunction::is_likely_wavefunction_filename(
		config.wavefunction_filename)) {
	occ::log::info("Loading wavefunction data from '{}'", config.wavefunction_filename);
	wfn = Wavefunction::load(config.wavefunction_filename);
    }
    else if(Wavefunction::is_likely_wavefunction_filename(
		config.geometry_filename)) {
	occ::log::info("Loading wavefunction data from geometry file '{}'", config.geometry_filename);
	wfn = Wavefunction::load(config.geometry_filename);
    }
    if(wfn.atoms.size() > 0) {
	occ::log::info("Loaded wavefunction, applying transformation:");
	occ::Mat3 rotation = Eigen::Map<const Mat3RM>(config.wfn_rotation.data());
	occ::log::info("Rotation\n{}", rotation);
	occ::Vec3 translation = Eigen::Map<const Vec3>(config.wfn_translation.data());
	occ::log::info("Translation\n{}", translation.transpose());
	wfn.apply_transformation(rotation, translation);
    }
    return wfn;
}

void run_isosurface_subcommand(IsosurfaceConfig const &config) {
    IsosurfaceMesh mesh;
    VertexProperties properties;

    const auto properties_to_compute = config.surface_properties();
    if(properties_to_compute.size() > 0) {
	occ::log::info("Properties to compute:");
	for(const auto &prop: properties_to_compute) {
	    occ::log::info("{}", to_string(prop));
	}
    }

    Wavefunction wfn = load_wfn(config);
    bool have_wfn = wfn.atoms.size() > 0;

    Molecule m1, m2;

    bool have_crystal = occ::io::CifParser::is_likely_cif_filename(config.geometry_filename);
    ensure_isosurface_configuration_valid(config, have_wfn, have_crystal);

    if(!config.environment_filename.empty()) {
	m2 = occ::io::molecule_from_xyz_file(config.environment_filename);
    }

    const auto surface_type = config.surface_type();
    occ::log::info("Isosurface kind: {}", to_string(surface_type));
    switch(config.surface_type()) {
	case IsosurfaceConfig::Surface::ESP: {
	    auto func = iso::ElectricPotentialFunctor(wfn, config.separation);
	    mesh = extract_surface(func, config.isovalue);
	    break;
	}
	case IsosurfaceConfig::Surface::ElectronDensity: {
	    auto func = iso::ElectronDensityFunctor(wfn, config.separation);
	    mesh = extract_surface(func, config.isovalue);
	    break;
	}
	case IsosurfaceConfig::Surface::CrystalVoid: {
	    occ::io::CifParser parser;
	    auto crystal = parser.parse_crystal(config.geometry_filename).value();
	    auto func = iso::VoidSurfaceFunctor(crystal, config.separation);
	    if(m2.size() == 0) {
		m2 = func.molecule();
	    }
	    mesh = extract_surface(func, -config.isovalue);
	    break;
	}
	case IsosurfaceConfig::Surface::Hirshfeld: {
	    m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);

	    occ::log::info("Interior region has {} atoms", m1.size());
	    occ::log::info("Exterior region has {} atoms", m2.size());

	    auto func = iso::StockholderWeightFunctor(m1, m2, config.separation);
	    func.set_background_density(config.background_density);
	    mesh = extract_surface(func, 0.5f);
	    break;
	}
	case IsosurfaceConfig::Surface::PromoleculeDensity: {
	    m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);
	    occ::log::info("Interior region has {} atoms", m1.size());
	    auto func = iso::PromoleculeDensityFunctor(m1, config.separation);
	    func.set_isovalue(config.isovalue);
	    mesh = extract_surface(func, config.isovalue);
	    break;
	}
	default: {
	    throw std::runtime_error("Not implemented");
	    break;
	}
    }

    Eigen::Map<const FMat3N> verts(mesh.vertices.data(), 3, mesh.vertices.size() / 3);
    Eigen::Map<const FMat3N> normals(mesh.normals.data(), 3, mesh.normals.size() / 3);
    Eigen::Map<const Eigen::Matrix<uint32_t, 3, Eigen::Dynamic>> faces(mesh.faces.data(), 3, mesh.faces.size() / 3);

    occ::log::info("Computing surface curvature properties");
    auto c = occ::isosurface::calculate_curvature(mesh.mean_curvature, mesh.gaussian_curvature); 

    occ::log::info("Computing atom internal/external neighbor properties");
    properties = compute_atom_surface_properties(m1, m2, verts);

    properties.add_property("shape_index", c.shape_index);
    properties.add_property("curvedness", c.curvedness);
    properties.add_property("gaussian_curvature", c.gaussian);
    properties.add_property("mean_curvature", c.mean);
    properties.add_property("k1", c.k1);
    properties.add_property("k2", c.k2);

    for(const auto &prop: properties_to_compute) {
	const auto s = to_string(prop);
	if(properties.fprops.contains(s)) continue;
	if(properties.iprops.contains(s)) continue;
    }

    Eigen::Vector3f lower_left = verts.rowwise().minCoeff();
    Eigen::Vector3f upper_right = verts.rowwise().maxCoeff();
    occ::log::info("Lower corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   lower_left(0), lower_left(1), lower_left(2));
    occ::log::info("Upper corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   upper_right(0), upper_right(1), upper_right(2));

    occ::log::info("Writing surface to {}", config.output_filename);
    occ::io::write_ply_mesh(config.output_filename, mesh, properties, config.binary_output);
}

} // namespace occ::main
