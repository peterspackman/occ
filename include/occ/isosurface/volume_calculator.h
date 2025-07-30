#pragma once
#include <occ/isosurface/volume_data.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>
#include <occ/io/cube.h>
#include <occ/io/periodic_grid.h>
#include <optional>

namespace occ::isosurface {

// Output format options
enum class OutputFormat {
    Cube,     // "cube" (default)
    GGrid,    // "ggrid" 
    PGrid     // "pgrid"
};

struct VolumeGenerationParameters {
    // Property selection
    VolumePropertyKind property{VolumePropertyKind::ElectronDensity};
    SpinConstraint spin{SpinConstraint::Total};
    std::string functional{"blyp"};
    int mo_number{-1};  // -1 means all MOs
    
    // Grid parameters
    std::vector<int> steps;         // 0-3 values, default 11x11x11 if empty
    std::vector<double> da;         // direction A (0-3 values)
    std::vector<double> db;         // direction B (0-3 values)
    std::vector<double> dc;         // direction C (0-3 values)
    std::vector<double> origin;     // origin (0-3 values)
    
    // Adaptive bounds
    bool adaptive_bounds{false};
    double value_threshold{1e-6};
    double buffer_distance{2.0};   // Angstrom
    
    // Crystal parameters
    std::string crystal_filename{""};
    bool unit_cell_only{false};
    double crystal_buffer{6.0};     // Angstrom
    
    // Output format
    OutputFormat format{OutputFormat::Cube};
    
    // Custom points
    std::string points_filename{""};
};

/**
 * @brief Calculator for generating volume data from quantum mechanical properties
 * 
 * Clean, functional interface similar to IsosurfaceCalculator. Set up the required
 * inputs (wavefunction, molecule, or crystal), then call compute_volume() with
 * parameters to get a VolumeData result.
 */
class VolumeCalculator {
public:
    VolumeCalculator() = default;

    // Setup methods (IsosurfaceCalculator pattern)
    void set_wavefunction(const occ::qm::Wavefunction& wfn);
    void set_molecule(const occ::core::Molecule& mol) { m_molecule = mol; }
    void set_crystal(const occ::crystal::Crystal& crystal) { m_crystal = crystal; }
    
    // Main computation method - dispatches based on property requirements
    VolumeData compute_volume(const VolumeGenerationParameters& params);
    
    // Custom points evaluation
    Vec evaluate_at_points(const Mat3N& points, const VolumeGenerationParameters& params);
    
    // I/O methods - convert VolumeData to different formats
    void save_cube(const VolumeData& volume, const std::string& filename);
    void save_ggrid(const VolumeData& volume, const std::string& filename);
    void save_pgrid(const VolumeData& volume, const std::string& filename);
    std::string volume_as_cube_string(const VolumeData& volume);
    
    // Conversion to legacy io::Cube for backward compatibility 
    io::Cube to_cube(const VolumeData& volume);
    
    // String property name conversion (for backward compatibility)
    static VolumePropertyKind property_from_string(const std::string& name);
    static std::string property_to_string(VolumePropertyKind prop);
    static SpinConstraint spin_from_string(const std::string& name);
    static OutputFormat format_from_string(const std::string& name);
    
    // List all supported properties
    static void list_supported_properties();
    
    // Requirements checking (static - no state needed)
    static bool requires_wavefunction(VolumePropertyKind property);
    static bool requires_crystal(VolumePropertyKind property);
    
    // Convenience static methods for common usage patterns
    static VolumeData compute_mo_volume(const occ::qm::Wavefunction& wfn, int mo_index, 
                                       const VolumeGenerationParameters& params = {});
    static VolumeData compute_density_volume(const occ::qm::Wavefunction& wfn,
                                            const VolumeGenerationParameters& params = {});
    static VolumeData compute_esp_volume(const occ::qm::Wavefunction& wfn,
                                        const VolumeGenerationParameters& params = {});

private:
    // Simple state - just hold the inputs
    std::optional<occ::core::Molecule> m_molecule;
    std::optional<occ::qm::Wavefunction> m_wavefunction;
    std::optional<occ::crystal::Crystal> m_crystal;
    
    // Helper methods - clean functional approach
    std::vector<core::Atom> get_atoms_for_property(const VolumeGenerationParameters& params);
    void setup_grid_parameters(VolumeData& volume, const VolumeGenerationParameters& params, 
                              const std::vector<core::Atom>& atoms);
    void fill_volume_data(VolumeData& volume, const VolumeGenerationParameters& params);
    
    // Validation
    void validate_parameters(const VolumeGenerationParameters& params);
    bool have_required_inputs(const VolumeGenerationParameters& params);
};

} // namespace occ::isosurface