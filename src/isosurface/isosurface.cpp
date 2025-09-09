#include <occ/core/eeq.h>
#include <occ/core/kdtree.h>
#include <occ/core/parallel.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/isosurface/curvature.h>
#include <occ/isosurface/isosurface.h>

namespace occ::isosurface {

namespace {

template <typename F>
Isosurface convert_to_isosurface(const F &func,
                                 const std::vector<float> &vertices,
                                 const std::vector<uint32_t> &indices,
                                 const std::vector<float> &normals,
                                 const std::vector<float> &curvature) {

  // Calculate sizes
  const size_t num_vertices = vertices.size() / 3;
  const size_t num_faces = indices.size() / 3;
  const size_t num_curvature_points = curvature.size() / 2;

  std::vector<float> remapped(vertices.size());
  func.remap_vertices(vertices, remapped);

  Isosurface result;
  result.vertices = Eigen::Map<const FMat3N>(remapped.data(), 3, num_vertices);
  result.normals = Eigen::Map<const FMat3N>(normals.data(), 3, num_vertices);

  result.faces = Eigen::Map<const Eigen::Matrix<uint32_t, 3, Eigen::Dynamic>>(
                     indices.data(), 3, num_faces)
                     .cast<int>();

  // Map curvature data (stored as alternating mean/gaussian values)
  Eigen::Map<const Eigen::Matrix<float, 2, Eigen::Dynamic>> curvature_map(
      curvature.data(), 2, num_curvature_points);
  result.mean_curvature = curvature_map.row(0).transpose();
  result.gaussian_curvature = curvature_map.row(1).transpose();

  return result;
}

template <typename F>
Isosurface extract_surface(F &func, float isovalue, bool flip = false) {
  occ::timing::StopWatch sw;
  auto cubes = func.cubes_per_side();
  occ::log::info("Begin marching cubes with dimensions: {}x{}x{}", cubes(0),
                 cubes(1), cubes(2));
  auto mc = occ::geometry::mc::MarchingCubes(cubes(0), cubes(1), cubes(2));
  mc.set_origin_and_side_lengths(func.origin(), func.side_length());
  mc.isovalue = isovalue;
  mc.flip_normals = (isovalue < 0.0) || flip;
  if (mc.flip_normals)
    occ::log::debug("Negative isovalue provided, will flip normals");

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
  if (vertices.size() < 3) {
    throw std::runtime_error(
        "Invalid isosurface encountered, not enough vertices?");
  }
  return convert_to_isosurface(func, vertices, faces, normals, curvature);
}
} // namespace

float Isosurface::volume() const {
  if (vertices.cols() == 0 || faces.cols() == 0) {
    return 0.0f;
  }

  float total_volume = 0.0f;
  for (int i = 0; i < faces.cols(); ++i) {
    const auto v1 = vertices.col(faces(0, i));
    const auto v2 = vertices.col(faces(1, i));
    const auto v3 = vertices.col(faces(2, i));
    total_volume += v1.dot((v2.cross(v3))) / 6.0f;
  }
  return std::abs(total_volume);
}

float Isosurface::surface_area() const {
  if (vertices.cols() == 0 || faces.cols() == 0) {
    return 0.0f;
  }

  float total_area = 0.0f;
  for (int i = 0; i < faces.cols(); ++i) {
    const FVec3 v1 = vertices.col(faces(0, i));
    const FVec3 v2 = vertices.col(faces(1, i));
    const FVec3 v3 = vertices.col(faces(2, i));

    const FVec3 edge1 = v2 - v1;
    const FVec3 edge2 = v3 - v1;

    total_area += 0.5f * edge1.cross(edge2).norm();
  }

  return total_area;
}

void IsosurfaceCalculator::set_molecule(const occ::core::Molecule &mol) {
  m_molecule = mol;
}
void IsosurfaceCalculator::set_environment(const occ::core::Molecule &env) {
  m_environment = env;
}
void IsosurfaceCalculator::set_wavefunction(const occ::qm::Wavefunction &wfn) {
  m_wavefunction = wfn;
}
void IsosurfaceCalculator::set_crystal(const occ::crystal::Crystal &crystal) {
  m_crystal = crystal;
}
void IsosurfaceCalculator::set_parameters(
    const IsosurfaceGenerationParameters &params) {
  m_params = params;
}

void IsosurfaceCalculator::compute_default_atom_surface_properties() {
  const auto &vertices = m_isosurface.vertices;
  const size_t N = vertices.cols();
  constexpr size_t num_results = 6;
  FVec di_norm = FVec::Constant(N, std::numeric_limits<float>::max());
  FVec dnorm = FVec::Constant(N, std::numeric_limits<float>::max());

  if (m_molecule.size() > 0) {
    Eigen::Matrix3Xf inside = m_molecule.positions().cast<float>();
    Eigen::VectorXf vdw_inside = m_molecule.vdw_radii().cast<float>();

    occ::core::KDTree<float> interior_tree(inside.rows(), inside,
                                           occ::core::max_leaf);
    interior_tree.index->buildIndex();

    FVec di = FVec::Constant(N, std::numeric_limits<float>::max());
    IVec di_idx = IVec::Constant(N, -1);
    IVec di_norm_idx = IVec::Constant(N, -1);

    // Use TBB-based thread-local storage for temporary arrays
    occ::parallel::thread_local_storage<std::vector<size_t>> indices_local(
      [num_results]() { return std::vector<size_t>(num_results); }
    );
    occ::parallel::thread_local_storage<std::vector<float>> dist_sq_local(
      [num_results]() { return std::vector<float>(num_results); }
    );

    occ::timing::start(occ::timing::category::isosurface_properties);
    occ::parallel::parallel_for(size_t(0), size_t(N), [&](size_t i) {
      auto &indices = indices_local.local();
      auto &dist_sq = dist_sq_local.local();

      Eigen::Vector3f v = vertices.col(i);
      float dist_inside_norm = std::numeric_limits<float>::max();
      nanoflann::KNNResultSet<float> results(num_results);
      results.init(&indices[0], &dist_sq[0]);
      bool populated = interior_tree.index->findNeighbors(
          results, v.data(), nanoflann::SearchParams());
      if (!populated)
        return;
      
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
    });
    occ::timing::stop(occ::timing::category::isosurface_properties);

    m_isosurface.properties.add("di", di);
    m_isosurface.properties.add("di_idx", di_idx);
    m_isosurface.properties.add("di_norm", di_norm);
    m_isosurface.properties.add("di_norm_idx", di_norm_idx);
  }

  if (m_environment.size() > 0) {
    FVec de = FVec::Constant(N, std::numeric_limits<float>::max());
    FVec de_norm = FVec::Constant(N, std::numeric_limits<float>::max());
    IVec de_idx = IVec::Constant(N, -1);
    IVec de_norm_idx = IVec::Constant(N, -1);
    Eigen::Matrix3Xf outside = m_environment.positions().cast<float>();
    Eigen::VectorXf vdw_outside = m_environment.vdw_radii().cast<float>();
    occ::core::KDTree<float> exterior_tree(outside.rows(), outside,
                                           occ::core::max_leaf);
    exterior_tree.index->buildIndex();
    // Use TBB-based thread-local storage for temporary arrays
    occ::parallel::thread_local_storage<std::vector<size_t>> ext_indices_local(
      [num_results]() { return std::vector<size_t>(num_results); }
    );
    occ::parallel::thread_local_storage<std::vector<float>> ext_dist_sq_local(
      [num_results]() { return std::vector<float>(num_results); }
    );

    occ::timing::start(occ::timing::category::isosurface_properties);
    occ::parallel::parallel_for(size_t(0), size_t(N), [&](size_t i) {
      auto &indices = ext_indices_local.local();
      auto &dist_sq = ext_dist_sq_local.local();

      Eigen::Vector3f v = vertices.col(i);
      float dist_outside_norm = std::numeric_limits<float>::max();
      nanoflann::KNNResultSet<float> results(num_results);
      results.init(&indices[0], &dist_sq[0]);
      bool populated = exterior_tree.index->findNeighbors(
          results, v.data(), nanoflann::SearchParams());
      if (!populated)
        return;
        
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

      if (m_molecule.size() > 0) {
        dnorm(i) = de_norm(i) + di_norm(i);
      }
    });
    occ::timing::stop(occ::timing::category::isosurface_properties);

    m_isosurface.properties.add("de_idx", de_idx);
    m_isosurface.properties.add("de", de);
    m_isosurface.properties.add("de_norm", de_norm);
    m_isosurface.properties.add("de_norm_idx", de_norm_idx);
    if (m_molecule.size() > 0) {
      m_isosurface.properties.add("dnorm", dnorm);
    }
  }
}

FVec IsosurfaceCalculator::compute_surface_property(PropertyKind prop) const {

  const auto &vertices = m_isosurface.vertices;
  FVec result = FVec::Zero(vertices.cols());

  switch (prop) {
  case PropertyKind::PromoleculeDensity: {
    auto func = BatchFunctor<slater::PromoleculeDensity>(m_molecule);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::debug("Min {} Max {} Mean {}", result.minCoeff(),
                    result.maxCoeff(), result.mean());
    occ::log::debug("Computed Promoecule Density for {} vertices",
                    func.num_calls());
    break;
  }
  case PropertyKind::ElectronDensity: {
    auto func = ElectronDensityFunctor(m_wavefunction);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::debug("Min {} Max {} Mean {}", result.minCoeff(),
                    result.maxCoeff(), result.mean());
    occ::log::debug("Computed Electron Density for {} vertices",
                    func.num_calls());
    break;
  }
  case PropertyKind::Orbital: {
    // Handle orbital property
    if (m_params.property_orbital_indices.empty()) {
      throw std::runtime_error(
          "No orbital indices specified for orbital property");
    }

    auto func = ElectronDensityFunctor(m_wavefunction, -1);
    int prev_calls = 0;
    for (const auto &orbital_index : m_params.property_orbital_indices) {
      func.set_orbital_index(orbital_index.resolve(m_wavefunction.mo.n_alpha,
                                                   m_wavefunction.mo.n_beta));
      func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
      occ::log::debug("Computed Orbital {} Density for {} vertices",
                      orbital_index.format(), func.num_calls() - prev_calls);
      occ::log::debug("Min {} Max {} Mean {}", result.minCoeff(),
                      result.maxCoeff(), result.mean());
      prev_calls = func.num_calls();
    }
    break;
  }
  case PropertyKind::DeformationDensity: {
    auto func = DeformationDensityFunctor(m_molecule, m_wavefunction);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::debug("Min {} Max {} Mean {}", result.minCoeff(),
                    result.maxCoeff(), result.mean());
    occ::log::debug("Computed Deformation Density for {} vertices",
                    func.num_calls());
    break;
  }

  case PropertyKind::ESP: {
    auto func = ElectricPotentialFunctor(m_wavefunction);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::debug("Min {} Max {} Mean {}", result.minCoeff(),
                    result.maxCoeff(), result.mean());
    occ::log::debug("Computed ESP (QM) for {} vertices", func.num_calls());
    break;
  }
  case PropertyKind::EEQ_ESP: {
    // copy the molecule
    auto m = m_molecule;
    auto q = occ::core::charges::eeq_partial_charges(m.atomic_numbers(),
                                                     m.positions(), m.charge());
    occ::log::debug("Molecule partial charges (EEQ)");
    for (int i = 0; i < q.rows(); i++) {
      occ::log::debug("Atom {}: {:12.5f}", i, q(i));
    }
    m.set_partial_charges(q);
    auto func = ElectricPotentialFunctorPC(m);
    func.batch(vertices, result);
    occ::log::debug("Min {} Max {} Mean {}", result.minCoeff(),
                    result.maxCoeff(), result.mean());
    occ::log::debug("Computed EEQ ESP for {} vertices", func.num_calls());
    break;
  }
  default:
    break;
  }

  return result;
}

void IsosurfaceCalculator::compute_isosurface() {
  double isovalue = m_params.isovalue;
  const double separation = m_params.separation;
  switch (m_params.surface_kind) {
  case SurfaceKind::ESP: {
    auto func = MCElectricPotentialFunctor(m_wavefunction, separation);
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "ESP";
    break;
  }
  case SurfaceKind::ElectronDensity: {
    auto func = MCElectronDensityFunctor(m_wavefunction, separation);
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "Electron Density";
    break;
  }
  case SurfaceKind::CrystalVoid: {
    if (!m_crystal)
      throw std::runtime_error("Void surface requires crystal");
    auto func = VoidSurfaceFunctor(*m_crystal, separation);
    if (m_environment.size() == 0) {
      m_environment = func.molecule();
    }
    m_isosurface = extract_surface(func, isovalue, true);
    m_isosurface.description = "Void";
    break;
  }
  case SurfaceKind::Hirshfeld: {
    auto func = StockholderWeightFunctor(m_molecule, m_environment, separation);
    func.set_background_density(m_params.background_density);
    isovalue = 0.5f;
    m_isosurface = extract_surface(func, isovalue); // Hirshfeld always uses 0.5
    m_isosurface.description = "Hirshfeld";
    break;
  }
  case SurfaceKind::VDWLogSumExp: {
    auto metric = RadiusMetric(RadiusMetric::RadiusKind::VDW);
    auto func = LogSumExpFunctor(metric, m_molecule, m_environment, separation);
    isovalue = 0.0f;
    m_isosurface = extract_surface(func, isovalue); // LSE always uses 0.0
    m_isosurface.description = "VDWLogSumExp";
    break;
  }
  case SurfaceKind::SoftVoronoi: {
    auto metric = RadiusMetric(RadiusMetric::RadiusKind::Unit);
    auto func = LogSumExpFunctor(metric, m_molecule, m_environment, separation);
    isovalue = 0.0f;
    m_isosurface = extract_surface(func, isovalue); // LSE always uses 0.0
    m_isosurface.description = "SoftVoronoi";
    break;
  }
  case SurfaceKind::HSRinv: {
    auto wfunc = RInvFunc{m_params.power};
    auto func = GenericStockholderWeightFunctor(wfunc, m_molecule,
                                                m_environment, separation);
    isovalue = 0.5f;
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "HS-Rinv";
    break;
  }
  case SurfaceKind::HSExp: {
    auto wfunc = ExpFunc{m_params.power};
    auto func = GenericStockholderWeightFunctor(wfunc, m_molecule,
                                                m_environment, separation);
    isovalue = 0.5f;
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "HS-Exp";
    break;
  }
  case SurfaceKind::PromoleculeDensity: {
    auto func = MCPromoleculeDensityFunctor(m_molecule, separation);
    func.set_isovalue(isovalue);
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "Promolecule Density";
    break;
  }
  case SurfaceKind::DeformationDensity: {
    auto func =
        MCDeformationDensityFunctor(m_molecule, m_wavefunction, separation);
    func.set_isovalue(isovalue);
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "Deformation Density";
    break;
  }
  case SurfaceKind::Orbital: {
    int orbital_index =
        (m_params.surface_orbital_index)
            .resolve(m_wavefunction.mo.n_alpha, m_wavefunction.mo.n_beta);
    occ::log::info("Surface orbital index = {}", orbital_index);
    auto func =
        MCElectronDensityFunctor(m_wavefunction, separation, orbital_index);
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description =
        fmt::format("Orbital {}", m_params.surface_orbital_index.format());
    break;
  }
  case SurfaceKind::VolumeGrid: {
    auto func = VolumeGridFunctor(m_grid, separation);
    m_isosurface = extract_surface(func, isovalue);
    m_isosurface.description = "Volume Grid";
    break;
  }

  default: {
    throw std::runtime_error("Not implemented");
    break;
  }
  }

  m_isosurface.kind = surface_to_string(m_params.surface_kind);
  m_isosurface.isovalue = isovalue;
  m_isosurface.separation = separation;

  auto curvature = occ::isosurface::calculate_curvature(
      m_isosurface.mean_curvature, m_isosurface.gaussian_curvature);

  occ::log::debug("Computing atom internal/external neighbor properties");
  compute_default_atom_surface_properties();

  m_isosurface.properties.add("shape_index", curvature.shape_index);
  m_isosurface.properties.add("curvedness", curvature.curvedness);
  m_isosurface.properties.add("gaussian_curvature", curvature.gaussian);
  m_isosurface.properties.add("mean_curvature", curvature.mean);
  m_isosurface.properties.add("k1", curvature.k1);
  m_isosurface.properties.add("k2", curvature.k2);

  for (const auto &prop : m_params.properties) {
    const auto s = isosurface::property_to_string(prop);
    if (m_isosurface.properties.has_property(s))
      continue;
    occ::log::debug("Need to compute: {}", s);
    m_isosurface.properties.add(s, compute_surface_property(prop));
  }
}

void IsosurfaceCalculator::compute() {
  if (!validate()) {
    throw std::runtime_error("Invalid parameters for isosurface calculation: " +
                             m_error_message);
  }

  compute_isosurface();
}

bool IsosurfaceCalculator::requires_grid() const {
  return m_params.surface_kind == SurfaceKind::VolumeGrid;
}

bool IsosurfaceCalculator::requires_crystal() const {
  return m_params.surface_kind == SurfaceKind::CrystalVoid;
}

bool IsosurfaceCalculator::requires_wavefunction() const {
  if (isosurface::surface_requires_wavefunction(m_params.surface_kind))
    return true;

  for (const auto &prop : m_params.properties) {
    if (isosurface::property_requires_wavefunction(prop))
      return true;
  }
  return false;
}

bool IsosurfaceCalculator::requires_environment() const {
  if (isosurface::surface_requires_environment(m_params.surface_kind))
    return true;

  for (const auto &prop : m_params.properties) {
    if (isosurface::property_requires_environment(prop))
      return true;
  }
  return false;
}

bool IsosurfaceCalculator::validate() {
  if (requires_wavefunction() && !have_wavefunction()) {
    m_error_message = "Surface or property requires a wavefunction";
    return false;
  }

  if (requires_crystal() && !have_crystal()) {
    m_error_message = "Surface or property requires a crystal";
    return false;
  }

  if (requires_environment() && !have_environment()) {
    m_error_message =
        "Surface or property requires a surrounding atom environment";
    return false;
  }

  if (requires_grid() && !have_grid()) {
    m_error_message = "Surfaces requires a volume grid";
    return false;
  }
  m_error_message = "";
  return true;
}

} // namespace occ::isosurface
