#include <occ/isosurface/volume_calculator.h>
#include <occ/isosurface/point_functors.h>
#include <occ/io/adaptive_grid.h>
#include <occ/core/units.h>
#include <occ/core/log.h>
#include <occ/crystal/hkl.h>
#include <fmt/core.h>
#include <algorithm>

namespace occ::isosurface {

// Helper function to compute adaptive bounds for any functor and apply user constraints
template<typename Functor>
void apply_adaptive_bounds(Functor& func, const VolumeGenerationParameters& params,
                          const std::vector<core::Atom>& atoms, VolumeData& volume) {
    typename io::AdaptiveGridBounds<Functor>::Parameters adapt_params;
    adapt_params.value_threshold = params.value_threshold;
    adapt_params.extra_buffer = params.buffer_distance * occ::units::ANGSTROM_TO_BOHR;
    
    auto bounds_calc = io::make_adaptive_bounds(func, adapt_params);
    core::Molecule mol(atoms);
    auto raw_bounds = bounds_calc.compute(mol);
    
    // Check what user specified to determine what to preserve
    bool user_specified_steps = !params.steps.empty();
    bool user_specified_spacing = !params.da.empty() || !params.db.empty() || !params.dc.empty();
    
    if (user_specified_steps && !user_specified_spacing) {
        // User specified steps - keep those, adjust spacing to fit adaptive bounds
        occ::log::info("Preserving user-specified steps: {} x {} x {}", 
                      volume.steps(0), volume.steps(1), volume.steps(2));
        
        // Calculate new basis vectors to fit the adaptive extent with user's steps
        Vec3 extent = raw_bounds.max_corner() - raw_bounds.origin;
        volume.origin = raw_bounds.origin;
        volume.basis = Mat3::Zero();
        volume.basis(0,0) = extent(0) / volume.steps(0);
        volume.basis(1,1) = extent(1) / volume.steps(1);
        volume.basis(2,2) = extent(2) / volume.steps(2);
        
        occ::log::info("Adjusted spacing to fit adaptive bounds: [{:.3f}, {:.3f}, {:.3f}] Bohr/step",
                      volume.basis(0,0), volume.basis(1,1), volume.basis(2,2));
        
    } else if (user_specified_spacing && !user_specified_steps) {
        // User specified spacing - keep that, adjust steps to fit adaptive bounds
        occ::log::info("Preserving user-specified spacing");
        
        // Keep the user's basis vectors, calculate steps to fit adaptive bounds
        Vec3 extent = raw_bounds.max_corner() - raw_bounds.origin;
        volume.origin = raw_bounds.origin;
        // volume.basis already set by user in setup_grid_parameters
        
        // Calculate steps needed for this extent with user's spacing
        volume.steps(0) = std::max(1, static_cast<int>(std::ceil(extent(0) / std::abs(volume.basis(0,0)))));
        volume.steps(1) = std::max(1, static_cast<int>(std::ceil(extent(1) / std::abs(volume.basis(1,1)))));
        volume.steps(2) = std::max(1, static_cast<int>(std::ceil(extent(2) / std::abs(volume.basis(2,2)))));
        
        occ::log::info("Adjusted steps to fit adaptive bounds: {} x {} x {}",
                      volume.steps(0), volume.steps(1), volume.steps(2));
        
    } else {
        // Neither or both specified - use adaptive bounds as-is (original behavior)
        volume.origin = raw_bounds.origin;
        volume.basis = raw_bounds.basis;
        volume.steps = raw_bounds.steps;
        
        occ::log::info("Using adaptive bounds as-is: {} x {} x {} points",
                      volume.steps(0), volume.steps(1), volume.steps(2));
    }
    
    // Log final grid information
    occ::log::info("Grid origin: [{:.3f}, {:.3f}, {:.3f}] Bohr", 
                  volume.origin(0), volume.origin(1), volume.origin(2));
    
    Vec3 corner = volume.origin + Vec3(volume.steps(0) * volume.basis(0,0),
                                      volume.steps(1) * volume.basis(1,1), 
                                      volume.steps(2) * volume.basis(2,2));
    occ::log::info("Grid extent: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}] Bohr",
                  volume.origin(0), volume.origin(1), volume.origin(2),
                  corner(0), corner(1), corner(2));
    
    Vec3 spacing = volume.basis.diagonal();
    occ::log::info("Final grid spacing: [{:.3f}, {:.3f}, {:.3f}] Bohr/step", 
                  spacing(0), spacing(1), spacing(2));
    
    double grid_volume = std::abs(volume.basis.determinant()) * volume.steps.prod();
    occ::log::info("Final grid volume: {:.2f} Bohr³", grid_volume);
}

// Setup methods
void VolumeCalculator::set_wavefunction(const occ::qm::Wavefunction& wfn) {
    m_wavefunction = wfn;
    // If no molecule is set, create one from the wavefunction atoms
    if (!m_molecule.has_value()) {
        m_molecule = core::Molecule(wfn.atoms);
    }
}

// Static conversion methods for backward compatibility
VolumePropertyKind VolumeCalculator::property_from_string(const std::string& name) {
    if (name == "electron_density" || name == "density" || name == "rho") {
        return VolumePropertyKind::ElectronDensity;
    } else if (name == "rho_alpha") {
        return VolumePropertyKind::ElectronDensityAlpha;
    } else if (name == "rho_beta") {
        return VolumePropertyKind::ElectronDensityBeta;
    } else if (name == "esp") {
        return VolumePropertyKind::ElectricPotential;
    } else if (name == "eeqesp") {
        return VolumePropertyKind::EEQ_ESP;
    } else if (name == "promolecule") {
        return VolumePropertyKind::PromoleculeDensity;
    } else if (name == "deformation_density") {
        return VolumePropertyKind::DeformationDensity;
    } else if (name == "xc") {
        return VolumePropertyKind::XCDensity;
    } else if (name == "void") {
        return VolumePropertyKind::CrystalVoid;
    }
    throw std::runtime_error("Unknown property: " + name);
}

std::string VolumeCalculator::property_to_string(VolumePropertyKind prop) {
    switch (prop) {
        case VolumePropertyKind::ElectronDensity: return "electron_density";
        case VolumePropertyKind::ElectronDensityAlpha: return "rho_alpha";
        case VolumePropertyKind::ElectronDensityBeta: return "rho_beta";
        case VolumePropertyKind::ElectricPotential: return "esp";
        case VolumePropertyKind::EEQ_ESP: return "eeqesp";
        case VolumePropertyKind::PromoleculeDensity: return "promolecule";
        case VolumePropertyKind::DeformationDensity: return "deformation_density";
        case VolumePropertyKind::XCDensity: return "xc";
        case VolumePropertyKind::CrystalVoid: return "void";
    }
    throw std::runtime_error("Unknown property kind");
}

SpinConstraint VolumeCalculator::spin_from_string(const std::string& name) {
    if (name == "both" || name == "total") {
        return SpinConstraint::Total;
    } else if (name == "alpha") {
        return SpinConstraint::Alpha;
    } else if (name == "beta") {
        return SpinConstraint::Beta;
    }
    throw std::runtime_error("Unknown spin constraint: " + name);
}

OutputFormat VolumeCalculator::format_from_string(const std::string& name) {
    if (name == "cube") {
        return OutputFormat::Cube;
    } else if (name == "ggrid") {
        return OutputFormat::GGrid;
    } else if (name == "pgrid") {
        return OutputFormat::PGrid;
    }
    throw std::runtime_error("Unknown output format: " + name);
}

void VolumeCalculator::list_supported_properties() {
    occ::log::info("Supported properties for cube generation:");
    occ::log::info("");
    occ::log::info("Electron density properties:");
    occ::log::info("  electron_density, density, rho  - Total electron density (requires wavefunction)");
    occ::log::info("  rho_alpha                       - Alpha spin density (requires wavefunction)");  
    occ::log::info("  rho_beta                        - Beta spin density (requires wavefunction)");
    occ::log::info("");
    occ::log::info("Electrostatic properties:");
    occ::log::info("  esp                             - Electrostatic potential (requires wavefunction)");
    occ::log::info("  eeqesp                          - EEQ electrostatic potential (atomic charges only)");
    occ::log::info("");
    occ::log::info("Molecular properties:");
    occ::log::info("  promolecule                     - Promolecule density (atomic densities)");
    occ::log::info("  deformation_density             - Deformation density (requires wavefunction)");
    occ::log::info("  xc                              - Exchange-correlation density (requires wavefunction)");
    occ::log::info("");
    occ::log::info("Crystal properties:");
    occ::log::info("  void                            - Crystal void space (requires --crystal)");
    occ::log::info("");
    occ::log::info("Notes:");
    occ::log::info("  - Properties marked 'requires wavefunction' need wavefunction files (owf.json, .fchk, .molden, etc.)");
    occ::log::info("  - Properties marked 'requires --crystal' need --crystal with .cif file");
    occ::log::info("  - Use --orbital to specify which orbital for electron density properties");
    occ::log::info("  - Use --spin to specify alpha/beta spin components");
}

// Requirements checking (static)
bool VolumeCalculator::requires_wavefunction(VolumePropertyKind property) {
    switch (property) {
        case VolumePropertyKind::ElectronDensity:
        case VolumePropertyKind::ElectronDensityAlpha:
        case VolumePropertyKind::ElectronDensityBeta:
        case VolumePropertyKind::ElectricPotential:
        case VolumePropertyKind::DeformationDensity:
        case VolumePropertyKind::XCDensity:
            return true;
        case VolumePropertyKind::EEQ_ESP:
        case VolumePropertyKind::PromoleculeDensity:
        case VolumePropertyKind::CrystalVoid:
            return false;
    }
    return false;
}

bool VolumeCalculator::requires_crystal(VolumePropertyKind property) {
    return property == VolumePropertyKind::CrystalVoid;
}

// Main computation method
VolumeData VolumeCalculator::compute_volume(const VolumeGenerationParameters& params) {
    validate_parameters(params);
    
    if (!have_required_inputs(params)) {
        throw std::runtime_error("Missing required inputs for property: " + property_to_string(params.property));
    }
    
    VolumeData volume;
    volume.property = params.property;
    volume.name = fmt::format("Generated by OCC VolumeCalculator - {}", property_to_string(params.property));
    
    // Get atoms for this property
    volume.atoms = get_atoms_for_property(params);
    
    // Set up grid parameters
    setup_grid_parameters(volume, params, volume.atoms);
    
    // Initialize the tensor with the correct dimensions
    volume.data = Eigen::Tensor<double, 3>(volume.nx(), volume.ny(), volume.nz());
    volume.data.setZero();
    
    // Fill the volume data
    fill_volume_data(volume, params);
    
    return volume;
}

// Helper methods
std::vector<core::Atom> VolumeCalculator::get_atoms_for_property(const VolumeGenerationParameters& params) {
    if (requires_wavefunction(params.property) && m_wavefunction.has_value()) {
        return m_wavefunction->atoms;
    } else if (m_molecule.has_value()) {
        // Convert molecule atoms to vector
        const auto& mol = m_molecule.value();
        std::vector<core::Atom> atoms;
        const auto& mol_atoms = mol.atoms();
        for (size_t i = 0; i < mol.size(); i++) {
            atoms.push_back(mol_atoms[i]);
        }
        return atoms;
    } else if (requires_crystal(params.property) && m_crystal.has_value()) {
        // For crystal void, we need expanded atoms (cluster around unit cell)
        const auto& crystal = m_crystal.value();
        const auto& uc_atoms = crystal.unit_cell_atoms();
        
        // Use same approach as VoidSurfaceFunctor - expand by buffer radius
        double buffer_radius = params.crystal_buffer; // Use parameter or default 6.0 Angstrom
        crystal::HKL upper = crystal::HKL::minimum();
        crystal::HKL lower = crystal::HKL::maximum();
        occ::Vec3 frac_radius = buffer_radius * 2 / crystal.unit_cell().lengths().array();

        for (size_t i = 0; i < uc_atoms.frac_pos.cols(); i++) {
            const auto& pos = uc_atoms.frac_pos.col(i);
            upper.h = std::max(upper.h, static_cast<int>(ceil(pos(0) + frac_radius(0))));
            upper.k = std::max(upper.k, static_cast<int>(ceil(pos(1) + frac_radius(1))));
            upper.l = std::max(upper.l, static_cast<int>(ceil(pos(2) + frac_radius(2))));

            lower.h = std::min(lower.h, static_cast<int>(floor(pos(0) - frac_radius(0))));
            lower.k = std::min(lower.k, static_cast<int>(floor(pos(1) - frac_radius(1))));
            lower.l = std::min(lower.l, static_cast<int>(floor(pos(2) - frac_radius(2))));
        }
        
        auto slab = crystal.slab(lower, upper);
        
        std::vector<core::Atom> atoms;
        for (size_t i = 0; i < slab.atomic_numbers.size(); i++) {
            atoms.push_back(core::Atom{
                slab.atomic_numbers(i),
                slab.cart_pos(0, i) * occ::units::ANGSTROM_TO_BOHR,
                slab.cart_pos(1, i) * occ::units::ANGSTROM_TO_BOHR,
                slab.cart_pos(2, i) * occ::units::ANGSTROM_TO_BOHR
            });
        }
        occ::log::info("Crystal void calculation using {} atoms in expanded cluster (buffer: {:.1f} Å)", 
                      atoms.size(), buffer_radius);
        return atoms;
    }
    throw std::runtime_error("No suitable atoms found for property: " + property_to_string(params.property));
}

void VolumeCalculator::setup_grid_parameters(VolumeData& volume, const VolumeGenerationParameters& params, 
                                             const std::vector<core::Atom>& atoms) {
    // Set default grid parameters - ALL IN BOHR UNITS
    volume.origin = Vec3::Zero();
    volume.basis = Mat3::Zero();
    
    // Set default steps
    volume.steps = IVec3::Constant(11);    // Default 11x11x11
    
    // Apply parameter overrides for steps
    if (!params.steps.empty()) {
        if (params.steps.size() == 1) {
            volume.steps.setConstant(params.steps[0]);
        } else {
            for (int i = 0; i < std::min(3, static_cast<int>(params.steps.size())); i++) {
                volume.steps(i) = params.steps[i];
            }
        }
    }
    
    // Handle adaptive bounds if requested (only for molecular calculations, not crystals)
    if (params.adaptive_bounds && !requires_crystal(params.property)) {
        occ::log::info("Computing adaptive bounds for property: {}", property_to_string(params.property));
        occ::log::info("Adaptive parameters: threshold={:.2e}, buffer={:.1f} Å", 
                      params.value_threshold, params.buffer_distance);
        
        // Create adaptive grid based on the property
        switch (params.property) {
            case VolumePropertyKind::ElectronDensity:
            case VolumePropertyKind::ElectronDensityAlpha: 
            case VolumePropertyKind::ElectronDensityBeta: {
                if (m_wavefunction.has_value()) {
                    Point_ElectronDensityFunctor func(m_wavefunction.value());
                    func.mo_index = params.mo_number;
                    if (params.property == VolumePropertyKind::ElectronDensityAlpha) {
                        func.spin = SpinConstraint::Alpha;
                    } else if (params.property == VolumePropertyKind::ElectronDensityBeta) {
                        func.spin = SpinConstraint::Beta;
                    }
                    apply_adaptive_bounds(func, params, atoms, volume);
                }
                break;
            }
            case VolumePropertyKind::ElectricPotential: {
                if (m_wavefunction.has_value()) {
                    EspFunctor func(m_wavefunction.value());
                    apply_adaptive_bounds(func, params, atoms, volume);
                }
                break;
            }
            case VolumePropertyKind::EEQ_ESP: {
                EEQEspFunctor func(atoms);
                apply_adaptive_bounds(func, params, atoms, volume);
                break;
            }
            case VolumePropertyKind::PromoleculeDensity: {
                PromolDensityFunctor func(atoms);
                apply_adaptive_bounds(func, params, atoms, volume);
                break;
            }
            case VolumePropertyKind::DeformationDensity: {
                if (m_wavefunction.has_value()) {
                    Point_DeformationDensityFunctor func(m_wavefunction.value());
                    apply_adaptive_bounds(func, params, atoms, volume);
                }
                break;
            }
            case VolumePropertyKind::XCDensity: {
                if (m_wavefunction.has_value()) {
                    XCDensityFunctor func(m_wavefunction.value(), params.functional);
                    apply_adaptive_bounds(func, params, atoms, volume);
                }
                break;
            }
            default:
                occ::log::warn("Adaptive bounds not supported for property: {}, using regular grid", 
                              property_to_string(params.property));
                break;
        }
    }
    
    // If adaptive bounds were not used, set basis vectors based on final step counts
    if (!params.adaptive_bounds) {
        if (requires_crystal(params.property) && m_crystal.has_value()) {
            const auto& crystal = m_crystal.value();
            const auto& cell = crystal.unit_cell();
            
            // Grid basis = unit cell vectors (in Bohr) / number of steps
            volume.basis.col(0) = cell.direct().col(0) * occ::units::ANGSTROM_TO_BOHR / volume.steps(0);
            volume.basis.col(1) = cell.direct().col(1) * occ::units::ANGSTROM_TO_BOHR / volume.steps(1); 
            volume.basis.col(2) = cell.direct().col(2) * occ::units::ANGSTROM_TO_BOHR / volume.steps(2);
            
            occ::log::info("Using crystal unit cell grid: {} x {} x {} steps", 
                          volume.steps(0), volume.steps(1), volume.steps(2));
            occ::log::info("Unit cell: a={:.3f} b={:.3f} c={:.3f} Å", 
                          cell.a(), cell.b(), cell.c());
        } else {
            // Regular molecular calculation - use diagonal basis
            volume.basis.diagonal().array() = 0.2; // Default 0.2 Bohr spacing
        }
    }
    
    // Apply origin if specified (assumed to be in Bohr units)
    if (!params.origin.empty()) {
        if (params.origin.size() == 1) {
            volume.origin.setConstant(params.origin[0]);
        } else {
            for (int i = 0; i < std::min(3, static_cast<int>(params.origin.size())); i++) {
                volume.origin(i) = params.origin[i];
            }
        }
    } else {
        if (requires_crystal(params.property) && m_crystal.has_value()) {
            // For crystal calculations, origin is at unit cell origin (0,0,0 fractional)
            volume.origin = Vec3::Zero();
            occ::log::info("Grid origin set to unit cell origin");
        } else {
            // Center on atoms if no origin specified (molecular calculations)
            if (!atoms.empty()) {
                Vec3 center = Vec3::Zero();
                for (const auto& atom : atoms) {
                    center += Vec3(atom.x, atom.y, atom.z);
                }
                center /= atoms.size();
                
                // Set origin to center minus half the grid extent
                Vec3 extent = Vec3(volume.steps(0) * volume.basis(0,0), 
                                  volume.steps(1) * volume.basis(1,1), 
                                  volume.steps(2) * volume.basis(2,2));
                volume.origin = center - 0.5 * extent;
            }
        }
    }
}

void VolumeCalculator::fill_volume_data(VolumeData& volume, const VolumeGenerationParameters& params) {
    occ::log::info("Computing volume data for property: {}", property_to_string(params.property));
    occ::log::info("Grid: {} x {} x {} points", volume.nx(), volume.ny(), volume.nz());
    
    // Create points matrix for evaluation - ALL COORDINATES IN BOHR
    Mat3N points(3, volume.total_points());
    size_t point_idx = 0;
    
    for (int i = 0; i < volume.nx(); i++) {
        for (int j = 0; j < volume.ny(); j++) {
            for (int k = 0; k < volume.nz(); k++) {
                Vec3 pos = volume.grid_to_coords(i, j, k); // Returns Bohr coordinates
                points.col(point_idx) = pos;
                point_idx++;
            }
        }
    }
    
    // Evaluate property at all points
    Vec values = Vec::Zero(volume.total_points());
    
    switch (params.property) {
        case VolumePropertyKind::ElectronDensity: {
            Point_ElectronDensityFunctor func(m_wavefunction.value());
            func.mo_index = params.mo_number;
            func(points, values);
            break;
        }
        case VolumePropertyKind::ElectronDensityAlpha: {
            Point_ElectronDensityFunctor func(m_wavefunction.value());
            func.spin = SpinConstraint::Alpha;
            func.mo_index = params.mo_number;
            func(points, values);
            break;
        }
        case VolumePropertyKind::ElectronDensityBeta: {
            Point_ElectronDensityFunctor func(m_wavefunction.value());
            func.spin = SpinConstraint::Beta;
            func.mo_index = params.mo_number;
            func(points, values);
            break;
        }
        case VolumePropertyKind::ElectricPotential: {
            EspFunctor func(m_wavefunction.value());
            func(points, values);
            break;
        }
        case VolumePropertyKind::EEQ_ESP: {
            EEQEspFunctor func(volume.atoms);
            func(points, values);
            break;
        }
        case VolumePropertyKind::PromoleculeDensity:
        case VolumePropertyKind::CrystalVoid: {
            PromolDensityFunctor func(volume.atoms);
            func(points, values);
            break;
        }
        case VolumePropertyKind::DeformationDensity: {
            Point_DeformationDensityFunctor func(m_wavefunction.value());
            func(points, values);
            break;
        }
        case VolumePropertyKind::XCDensity: {
            XCDensityFunctor func(m_wavefunction.value(), params.functional);
            func(points, values);
            break;
        }
    }
    
    // Copy values back to tensor
    point_idx = 0;
    for (int i = 0; i < volume.nx(); i++) {
        for (int j = 0; j < volume.ny(); j++) {
            for (int k = 0; k < volume.nz(); k++) {
                volume.data(i, j, k) = values(point_idx);
                point_idx++;
            }
        }
    }
}

// Validation
void VolumeCalculator::validate_parameters(const VolumeGenerationParameters& params) {
    if (requires_wavefunction(params.property) && !m_wavefunction.has_value()) {
        throw std::runtime_error("Property requires a wavefunction: " + property_to_string(params.property));
    }
    
    if (requires_crystal(params.property) && !m_crystal.has_value()) {
        throw std::runtime_error("Property requires a crystal structure: " + property_to_string(params.property));
    }
    
    if (params.mo_number >= 0 && !requires_wavefunction(params.property)) {
        throw std::runtime_error("MO index specified but property does not use wavefunction");
    }
    
    if (params.mo_number >= 0 && m_wavefunction.has_value()) {
        int max_mo = static_cast<int>(m_wavefunction->mo.n_ao);
        if (params.mo_number >= max_mo) {
            throw std::runtime_error(fmt::format(
                "Invalid MO index: {} (have {} MOs)", params.mo_number, max_mo));
        }
    }
}

bool VolumeCalculator::have_required_inputs(const VolumeGenerationParameters& params) {
    if (requires_wavefunction(params.property)) {
        return m_wavefunction.has_value();
    }
    if (requires_crystal(params.property)) {
        return m_crystal.has_value();
    }
    return m_molecule.has_value() || m_wavefunction.has_value() || m_crystal.has_value();
}

// Custom points evaluation
Vec VolumeCalculator::evaluate_at_points(const Mat3N& points, const VolumeGenerationParameters& params) {
    validate_parameters(params);
    
    if (!have_required_inputs(params)) {
        throw std::runtime_error("Missing required inputs for property: " + property_to_string(params.property));
    }
    
    Vec result = Vec::Zero(points.cols());
    std::vector<core::Atom> atoms = get_atoms_for_property(params);
    
    switch (params.property) {
        case VolumePropertyKind::ElectronDensity: {
            Point_ElectronDensityFunctor func(m_wavefunction.value());
            func.mo_index = params.mo_number;
            func(points, result);
            break;
        }
        case VolumePropertyKind::ElectronDensityAlpha: {
            Point_ElectronDensityFunctor func(m_wavefunction.value());
            func.spin = SpinConstraint::Alpha;
            func.mo_index = params.mo_number;
            func(points, result);
            break;
        }
        case VolumePropertyKind::ElectronDensityBeta: {
            Point_ElectronDensityFunctor func(m_wavefunction.value());
            func.spin = SpinConstraint::Beta;
            func.mo_index = params.mo_number;
            func(points, result);
            break;
        }
        case VolumePropertyKind::ElectricPotential: {
            EspFunctor func(m_wavefunction.value());
            func(points, result);
            break;
        }
        case VolumePropertyKind::EEQ_ESP: {
            EEQEspFunctor func(atoms);
            func(points, result);
            break;
        }
        case VolumePropertyKind::PromoleculeDensity:
        case VolumePropertyKind::CrystalVoid: {
            PromolDensityFunctor func(atoms);
            func(points, result);
            break;
        }
        case VolumePropertyKind::DeformationDensity: {
            Point_DeformationDensityFunctor func(m_wavefunction.value());
            func(points, result);
            break;
        }
        case VolumePropertyKind::XCDensity: {
            XCDensityFunctor func(m_wavefunction.value(), params.functional);
            func(points, result);
            break;
        }
    }
    
    return result;
}

// I/O methods - convert VolumeData to different formats
void VolumeCalculator::save_cube(const VolumeData& volume, const std::string& filename) {
    auto cube = to_cube(volume);
    cube.save(filename);
}

void VolumeCalculator::save_ggrid(const VolumeData& volume, const std::string& filename) {
    auto cube = to_cube(volume);
    auto grid = io::PeriodicGrid::from_cube(cube, io::GridFormat::GeneralGrid);
    grid.save(filename);
}

void VolumeCalculator::save_pgrid(const VolumeData& volume, const std::string& filename) {
    auto cube = to_cube(volume);
    auto grid = io::PeriodicGrid::from_cube(cube, io::GridFormat::PeriodicGrid);
    grid.save(filename);
}

std::string VolumeCalculator::volume_as_cube_string(const VolumeData& volume) {
    auto cube = to_cube(volume);
    std::ostringstream oss;
    cube.save(oss);
    return oss.str();
}

// Conversion to legacy io::Cube for backward compatibility
io::Cube VolumeCalculator::to_cube(const VolumeData& volume) {
    io::Cube cube;
    cube.name = volume.name;
    cube.description = fmt::format("Scalar values for property '{}'", property_to_string(volume.property));
    cube.atoms = volume.atoms;
    cube.origin = volume.origin;
    cube.basis = volume.basis;
    cube.steps = volume.steps;
    
    // Create volume grid with proper dimensions
    // This will create the internal storage
    // Note: VolumeGrid constructor takes size_t arguments in (nx, ny, nz) order
    cube.grid() = geometry::VolumeGrid(volume.nx(), volume.ny(), volume.nz());
    float* data = cube.data();
    
    // Convert to cube file format: [x][y][z] order (x outer loop, z inner loop)
    size_t idx = 0;
    for (int i = 0; i < volume.nx(); i++) {        // x outer loop
        for (int j = 0; j < volume.ny(); j++) {    // y middle loop  
            for (int k = 0; k < volume.nz(); k++) { // z inner loop
                data[idx] = static_cast<float>(volume.data(i, j, k));
                idx++;
            }
        }
    }
    
    return cube;
}

// Convenience static methods
VolumeData VolumeCalculator::compute_mo_volume(const occ::qm::Wavefunction& wfn, int mo_index, 
                                              const VolumeGenerationParameters& params) {
    VolumeCalculator calc;
    calc.set_wavefunction(wfn);
    
    VolumeGenerationParameters mo_params = params;
    mo_params.property = VolumePropertyKind::ElectronDensity;
    mo_params.mo_number = mo_index;
    
    return calc.compute_volume(mo_params);
}

VolumeData VolumeCalculator::compute_density_volume(const occ::qm::Wavefunction& wfn,
                                                   const VolumeGenerationParameters& params) {
    VolumeCalculator calc;
    calc.set_wavefunction(wfn);
    
    VolumeGenerationParameters density_params = params;
    density_params.property = VolumePropertyKind::ElectronDensity;
    
    return calc.compute_volume(density_params);
}

VolumeData VolumeCalculator::compute_esp_volume(const occ::qm::Wavefunction& wfn,
                                               const VolumeGenerationParameters& params) {
    VolumeCalculator calc;
    calc.set_wavefunction(wfn);
    
    VolumeGenerationParameters esp_params = params;
    esp_params.property = VolumePropertyKind::ElectricPotential;
    
    return calc.compute_volume(esp_params);
}

} // namespace occ::isosurface