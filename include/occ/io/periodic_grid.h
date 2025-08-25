#pragma once
#include <array>
#include <fstream>
#include <memory>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/geometry/volume_grid.h>
#include <vector>

namespace occ::io {

enum class GridFormat { 
    GeneralGrid,  // *.ggrid
    PeriodicGrid  // *.pgrid
};

enum class GridDataType {
    Raw,          // fType = 0: raw data
    Indexed       // fType = 1: paired record of index and value
};

struct GridFileHeader {
    char title[80];
    std::array<int, 4> version{3, 0, 0, 0};
    int grid_type{0};     // 0: general grid, 1: periodic grid
    int data_type{0};     // 0: raw data, 1: indexed
    int num_values{1};    // Number of data values per voxel (1 or 2)
    int dimension{3};     // Only 3 is supported
    std::array<int, 3> num_voxels{0, 0, 0};
    int num_asymmetric{0};  // Total voxels for indexed format
    std::array<float, 6> cell_params{0, 0, 0, 90, 90, 90}; // a, b, c, alpha, beta, gamma
};

struct GridSymmetryInfo {
    int num_operations{0};
    int centrosymmetric{0};  // 0: non-centrosym or all ops recorded, 1: centrosym with half ops
    int num_sublattices{1};  // Number of lattice points (1 for P, >1 for I, F, etc)
    std::vector<Mat4> symmetry_operations;  // 4x4 matrices in column-major format
    std::vector<IVec3> sublattice_positions;
};

class PeriodicGrid {
public:
    PeriodicGrid();
    PeriodicGrid(const PeriodicGrid &other);
    PeriodicGrid &operator=(const PeriodicGrid &other);
    
    // File I/O
    static PeriodicGrid load_pgrid(const std::string &filename);
    static PeriodicGrid load_ggrid(const std::string &filename);
    static PeriodicGrid load(const std::string &filename); // Auto-detects format
    
    void save_pgrid(const std::string &filename) const;
    void save_ggrid(const std::string &filename) const;
    void save(const std::string &filename) const; // Auto-detects format
    
    // Convert between formats
    static PeriodicGrid from_cube(const class Cube &cube, GridFormat format = GridFormat::GeneralGrid);
    Cube to_cube() const;
    
    // Data access
    const float *data() const { return m_grid.data(); }
    float *data() { return m_grid.data(); }
    const geometry::VolumeGrid &grid() const { return m_grid; }
    geometry::VolumeGrid &grid() { return m_grid; }
    
    // Metadata
    std::string title{"Grid file from OCC"};
    GridFormat format{GridFormat::GeneralGrid};
    GridDataType data_type{GridDataType::Raw};
    std::array<float, 6> cell_parameters{1.0, 1.0, 1.0, 90.0, 90.0, 90.0};
    Vec3 origin{0, 0, 0};
    Mat3 basis;
    
    // Atoms (optional)
    std::vector<core::Atom> atoms;
    Vec charges;
    
    // Symmetry information (for indexed format)
    GridSymmetryInfo symmetry;
    
    // Helper functions
    bool is_periodic() const { return format == GridFormat::PeriodicGrid; }
    bool is_indexed() const { return data_type == GridDataType::Indexed; }
    IVec3 dimensions() const;
    
    // Fill data from function
    template <typename F> 
    void fill_data_from_function(F &func, const IVec3 &steps);
    
private:
    geometry::VolumeGrid m_grid;
    
    // Internal I/O helpers
    void write_header(std::ofstream &file) const;
    void write_raw_data(std::ofstream &file) const;
    void write_indexed_data(std::ofstream &file) const;
    
    static GridFileHeader read_header(std::ifstream &file);
    static GridSymmetryInfo read_symmetry_info(std::ifstream &file, const GridFileHeader &header);
    void read_raw_data(std::ifstream &file, const GridFileHeader &header);
    void read_indexed_data(std::ifstream &file, const GridFileHeader &header, const GridSymmetryInfo &sym);
    
    // Conversion helpers
    void expand_periodic_to_general();
    void reduce_general_to_periodic();
};

// Template implementation
template <typename F>
void PeriodicGrid::fill_data_from_function(F &func, const IVec3 &steps) {
    m_grid = geometry::VolumeGrid(steps(0), steps(1), steps(2));
    
    // For periodic grids, we don't include the redundant points
    IVec3 actual_steps = steps;
    if (is_periodic()) {
        actual_steps = steps.array() - 1;
    }
    
    // Use TBB thread-local storage for efficient memory reuse
    occ::parallel::thread_local_storage<Mat3N> tl_points;
    occ::parallel::thread_local_storage<Vec> tl_temp;
    
    // Let TBB handle work distribution with automatic load balancing
    // Process z-slices individually for optimal granularity and cache locality
    occ::parallel::parallel_for(0, actual_steps(2), [&](int z) {
        const size_t slice_size = actual_steps(0) * actual_steps(1);
        
        // Get thread-local storage, resize if needed
        auto& points = tl_points.local();
        auto& temp = tl_temp.local();
        
        if (points.cols() < slice_size) {
            points.resize(3, slice_size);
        }
        if (temp.size() < slice_size) {
            temp.resize(slice_size);
        }
        
        // Generate points for this z-slice
        size_t local_idx = 0;
        for (int y = 0; y < actual_steps(1); y++) {
            for (int x = 0; x < actual_steps(0); x++, local_idx++) {
                points.col(local_idx) = basis * Vec3(x, y, z) + origin;
            }
        }
        
        // Process slice (only use the needed portion)
        auto points_view = points.leftCols(slice_size);
        auto temp_view = temp.head(slice_size);
        temp_view.setZero();
        func(points_view, temp_view);
        
        // Copy results back to grid
        size_t grid_offset = z * actual_steps(0) * actual_steps(1);
        std::copy(temp_view.data(), temp_view.data() + slice_size, m_grid.data() + grid_offset);
    });
    
    if (is_periodic()) {
        // Need to expand the data to include redundant points
        expand_periodic_to_general();
    }
}

} // namespace occ::io