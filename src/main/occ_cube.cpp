#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/io/cifparser.h>
#include <occ/io/load_geometry.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_cube.h>
#include <occ/isosurface/volume_calculator.h>
#include <occ/isosurface/point_functors.h>
#include <scn/scan.h>

namespace fs = std::filesystem;
using occ::IVec;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;
using occ::core::Element;
using occ::core::Molecule;
using occ::io::Cube;
using occ::qm::Wavefunction;
using occ::isosurface::ElectronDensityFunctor;
using occ::isosurface::EspFunctor;
using occ::isosurface::EEQEspFunctor;
using occ::isosurface::PromolDensityFunctor;
using occ::isosurface::DeformationDensityFunctor;
using occ::isosurface::XCDensityFunctor;
using occ::isosurface::SpinConstraint;

namespace occ::main {

void require_wfn(const CubeConfig &config, bool have_wfn) {
  if (have_wfn)
    return;
  throw std::runtime_error("Property requires a wavefunction: " +
                           config.property);
}

// Convert CubeConfig to VolumeGenerationParameters
isosurface::VolumeGenerationParameters config_to_volume_params(const CubeConfig &config) {
    isosurface::VolumeGenerationParameters params;
    
    // Convert property name
    params.property = isosurface::VolumeCalculator::property_from_string(config.property);
    
    // Convert spin constraint
    params.spin = isosurface::VolumeCalculator::spin_from_string(config.spin);
    
    // Set other parameters
    params.functional = config.functional;
    params.mo_number = config.mo_number;
    
    // Grid parameters (preserve exact vector structure)
    params.steps = config.steps;
    params.da = config.da;
    params.db = config.db;
    params.dc = config.dc;
    params.origin = config.origin;
    
    // Adaptive bounds
    params.adaptive_bounds = config.adaptive_bounds;
    params.value_threshold = config.value_threshold;
    params.buffer_distance = config.buffer_distance;
    
    // Output format
    params.format = isosurface::VolumeCalculator::format_from_string(config.format);
    
    // Crystal parameters
    params.crystal_filename = config.crystal_filename;
    params.unit_cell_only = config.unit_cell_only;
    
    // Custom points
    params.points_filename = config.points_filename;
    
    return params;
}

CLI::App *add_cube_subcommand(CLI::App &app) {

  CLI::App *cube =
      app.add_subcommand("cube", "compute molecule/qm properties on points");
  auto config = std::make_shared<CubeConfig>();

  cube->add_option("input", config->input_filename,
                   "input geometry file (xyz, wavefunction ...)")
      ->required();

  cube->add_option("property", config->property,
                   "property to evaluate (default=density)");

  cube->add_option(
      "spin", config->spin,
      "spin (for e.g. electron density) [alpha,beta,default=both]");

  cube->add_option("--orbital", config->orbitals_input,
                   "orbital specification (e.g., 'homo', 'lumo', 'homo-1', 'lumo+2', '5', 'all') [default=all]");
  cube->add_option("--functional", config->functional,
                   "DFT functional for XC density [default=blyp]");

  cube->add_option("-n,--steps", config->steps,
                   "Number of steps in each direction")
      ->expected(0, 3)
      ->default_val("{}");

  cube->add_option(
      "--points", config->points_filename,
      "list of points/mesh file requesting points to evaluate the property");
  cube->add_option("--output,-o", config->output_filename,
                   "destination to write file");

  cube->add_option("--origin", config->origin, "origin/starting point for grid")
      ->expected(0, 3)
      ->default_val("{}");

  cube->add_option("--direction-a,--direction_a,--da,-a", config->da,
                   "Translation in direction A")
      ->expected(0, 3)
      ->default_val("{}");
  cube->add_option("--direction-b,--direction_b,--db,-b", config->db,
                   "Translation in direction B")
      ->expected(0, 3)
      ->default_val("{}");
  cube->add_option("--direction-c,--direction_c,--dc,-c", config->dc,
                   "Translation in direction C")
      ->expected(0, 3)
      ->default_val("{}");

  cube->add_flag("--adaptive", config->adaptive_bounds,
                 "Use adaptive bounds calculation");
  cube->add_option("--threshold", config->value_threshold,
                   "Value threshold for adaptive bounds (default=1e-8)");
  cube->add_option("--buffer", config->buffer_distance,
                   "Buffer distance for adaptive bounds in Angstrom (default=2.0)");
  cube->add_option("--format", config->format,
                   "Output format: cube, ggrid, pgrid (default=cube)");
  
  cube->add_option("--crystal", config->crystal_filename,
                   "CIF file for crystal structure (enables symmetry-aware pgrid generation)");
  cube->add_flag("--unit-cell", config->unit_cell_only,
                 "Generate grid for unit cell only (requires --crystal)");

  cube->add_flag("--list-properties", config->list_properties,
                 "List all supported properties and exit");

  cube->fallthrough();
  cube->callback([config]() { run_cube_subcommand(*config); });
  return cube;
}

inline void fail_with_error(const std::string &msg, int line) {
  throw std::runtime_error(fmt::format(
      "Unable to parse points file, error: {}, line {}", msg, line));
}

Mat3N read_points_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    fail_with_error(fmt::format("Failed to open points file '{}'", filename),
                    0);
  }

  std::string line;
  std::getline(file, line);

  auto result = scn::scan<int>(line, "{}");
  if (!result) {
    fail_with_error(result.error().msg(), 0);
  }
  int n = result->value();

  Mat3N points(3, n);

  for (int i = 0; i < n; ++i) {
    std::getline(file, line);
    auto scan_result = scn::scan<double, double, double>(line, "{} {} {}");
    if (!scan_result) {
      fail_with_error(scan_result.error().msg(), i + 1);
    }
    auto [px, py, pz] = scan_result->values();
    points(0, i) = px;
    points(1, i) = py;
    points(2, i) = pz;
  }

  return points;
}

Mat3N load_points_for_evaluation(const std::string &points_filename) {
  return read_points_file(points_filename);
}

void write_results(CubeConfig const &config, const Mat3N &points,
                   const Vec &data) {
  auto output = fmt::output_file(
      config.output_filename, fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
  output.print("{}\n", data.rows());
  for (int i = 0; i < data.rows(); i++) {
    output.print("{:20.12f} {:20.12f} {:20.12f} {:20.12f}\n", points(0, i),
                 points(1, i), points(2, i), data(i));
  }
}

void evaluate_custom_points(const Wavefunction &wfn, CubeConfig const &config,
                            bool have_wfn) {

  auto points = load_points_for_evaluation(config.points_filename);
  Vec data = Vec::Zero(points.cols());

  if (config.mo_number > -1) {
    occ::log::info("Specified MO number:    {}", config.mo_number);
  }
  occ::timing::start(occ::timing::category::cube_evaluation);
  if (config.property == "eeqesp") {
    EEQEspFunctor func(wfn.atoms);
    func(points, data);
  } else if (config.property == "promolecule") {
    PromolDensityFunctor func(wfn.atoms);
    func(points, data);
  } else if (config.property == "deformation_density") {
    require_wfn(config, have_wfn);
    DeformationDensityFunctor func(wfn);
    func(points, data);
  } else if (config.property == "rho" ||
             config.property == "electron_density" ||
             config.property == "density") {
    require_wfn(config, have_wfn);
    ElectronDensityFunctor func(wfn);
    func.mo_index = config.mo_number;
    func(points, data);
  } else if (config.property == "rho_alpha") {
    require_wfn(config, have_wfn);
    ElectronDensityFunctor func(wfn);
    func.spin = SpinConstraint::Alpha;
    func.mo_index = config.mo_number;
    func(points, data);
  } else if (config.property == "rho_beta") {
    require_wfn(config, have_wfn);
    ElectronDensityFunctor func(wfn);
    func.spin = SpinConstraint::Beta;
    func.mo_index = config.mo_number;
    func(points, data);
  } else if (config.property == "esp") {
    require_wfn(config, have_wfn);
    EspFunctor func(wfn);
    func(points, data);
  } else if (config.property == "xc") {
    require_wfn(config, have_wfn);
    XCDensityFunctor func(wfn, config.functional);
    func(points, data);
  } else
    throw std::runtime_error(
        fmt::format("Unknown property to evaluate: {}", config.property));
  occ::timing::stop(occ::timing::category::cube_evaluation);

  write_results(config, points, data);
}

void run_cube_subcommand(CubeConfig const &config_in) {
  // Handle list-properties flag first
  if (config_in.list_properties) {
    isosurface::VolumeCalculator::list_supported_properties();
    return;
  }
  
  // Make a mutable copy so we can modify for convenience features
  CubeConfig config = config_in;
  Wavefunction wfn;
  bool have_wfn{false};

  // Load input file (wavefunction, XYZ, or CIF)
  bool is_cif_input = occ::io::CifParser::is_likely_cif_filename(config.input_filename);
  if (is_cif_input && config.crystal_filename.empty()) {
    // If input is a CIF and no separate crystal file specified, use input as crystal
    config.crystal_filename = config.input_filename;
  }

  if (Wavefunction::is_likely_wavefunction_filename(config.input_filename)) {
    wfn = Wavefunction::load(config.input_filename);
    occ::log::info("Loaded wavefunction from {}", config.input_filename);
    occ::log::info("Spinorbital kind: {}",
                   occ::qm::spinorbital_kind_to_string(wfn.mo.kind));
    occ::log::info("Num alpha:        {}", wfn.mo.n_alpha);
    occ::log::info("Num beta:         {}", wfn.mo.n_beta);
    occ::log::info("Num AOs:          {}", wfn.mo.n_ao);
    have_wfn = true;
    
    // Parse orbital specification if property uses wavefunction
    auto property_kind = isosurface::VolumeCalculator::property_from_string(config.property);
    if (isosurface::VolumeCalculator::requires_wavefunction(property_kind)) {
      if (config.orbitals_input == "all") {
        // "all" means use all orbitals (mo_number = -1, which is already the default)
        occ::log::info("Using all orbitals for density calculation");
      } else {
        try {
          auto orbital_indices = isosurface::parse_orbital_descriptions(config.orbitals_input);
          if (orbital_indices.size() != 1) {
            throw std::runtime_error("Cube generation supports only one orbital at a time");
          }
          config.mo_number = orbital_indices[0].resolve(wfn.mo.n_alpha, wfn.mo.n_beta);
          occ::log::info("Orbital specification '{}' resolved to MO index {}", 
                        config.orbitals_input, config.mo_number);
        } catch (const std::exception& e) {
          throw std::runtime_error(fmt::format("Invalid orbital specification '{}': {}", 
                                              config.orbitals_input, e.what()));
        }
      }
    }
  } else if (!is_cif_input) {
    Molecule m = occ::io::molecule_from_xyz_file(config.input_filename);
    wfn.atoms = m.atoms();
  }

  // Handle void as an alias for promolecule when using crystals
  if (config.property == "void") {
    if (config.crystal_filename.empty()) {
      throw std::runtime_error("Void calculation requires a crystal structure (use --crystal)");
    }
    // Keep property as "void" so VolumeCalculator uses CrystalVoid with crystal atoms
    config.unit_cell_only = true;  // Voids are typically calculated for unit cell
    if (config.format == "cube") {
      config.format = "pgrid";  // Default to pgrid for crystal voids
    }
    // Also update output filename if it ends with .cube
    if (config.output_filename.ends_with(".cube")) {
      config.output_filename = config.output_filename.substr(0, config.output_filename.length() - 5) + ".pgrid";
    }
    occ::log::info("Void calculation: using crystal structure for unit cell");
  }

  // CRITICAL: Check for custom points first (restore accidentally broken feature)
  if (!config.points_filename.empty()) {
    occ::log::info("Evaluating properties at custom points from {}", config.points_filename);
    evaluate_custom_points(wfn, config, have_wfn);
    return; // Early return - skip cube generation
  }

  // Convert config to VolumeCalculator parameters
  auto volume_params = config_to_volume_params(config);
  
  // Create and configure VolumeCalculator
  isosurface::VolumeCalculator calc;
  
  // Set wavefunction if available
  if (have_wfn) {
    calc.set_wavefunction(wfn);
  } else if (!wfn.atoms.empty()) {
    // Create molecule from atoms for non-wavefunction properties
    occ::core::Molecule mol(wfn.atoms);
    calc.set_molecule(mol);
  }
  
  // Load and set crystal if specified
  if (!config.crystal_filename.empty()) {
    occ::log::info("Loading crystal structure from {}", config.crystal_filename);
    auto crystal = occ::io::load_crystal(config.crystal_filename);
    calc.set_crystal(crystal);
    
    const auto &cell = crystal.unit_cell();
    occ::log::info("Unit cell parameters: a={:.3f} b={:.3f} c={:.3f} alpha={:.1f} beta={:.1f} gamma={:.1f}",
                   cell.a(), cell.b(), cell.c(), cell.alpha(), cell.beta(), cell.gamma());
  }
  
  // Compute volume
  occ::log::info("Computing volume for property: {}", config.property);
  if (config.mo_number > -1) {
    occ::log::info("Specified MO number:    {}", config.mo_number);
  }
  isosurface::VolumeData volume = calc.compute_volume(volume_params);
  
  // Save results in the requested format
  occ::timing::start(occ::timing::category::io);
  
  if (config.format == "cube") {
    calc.save_cube(volume, config.output_filename);
  } else if (config.format == "ggrid") {
    calc.save_ggrid(volume, config.output_filename);
    occ::log::info("Saved ggrid file: {}", config.output_filename);
  } else if (config.format == "pgrid") {
    calc.save_pgrid(volume, config.output_filename);
    occ::log::info("Saved pgrid file: {}", config.output_filename);
  } else {
    throw std::runtime_error("Unknown output format: " + config.format);
  }
  
  occ::timing::stop(occ::timing::category::io);
}

} // namespace occ::main
