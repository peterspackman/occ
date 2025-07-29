#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/log.h>
#include <occ/io/cube.h>
#include <occ/io/periodic_grid.h>

namespace occ::io {

PeriodicGrid::PeriodicGrid() : basis(Mat3::Identity()), m_grid(11, 11, 11) {}

PeriodicGrid::PeriodicGrid(const PeriodicGrid &other) 
    : m_grid(other.dimensions()(0), other.dimensions()(1), other.dimensions()(2)),
      title(other.title),
      format(other.format),
      data_type(other.data_type),
      cell_parameters(other.cell_parameters),
      origin(other.origin),
      basis(other.basis),
      atoms(other.atoms),
      charges(other.charges),
      symmetry(other.symmetry) {
    std::copy(other.data(), other.data() + other.m_grid.size(), m_grid.data());
}

PeriodicGrid &PeriodicGrid::operator=(const PeriodicGrid &other) {
    if (this != &other) {
        auto dims = other.dimensions();
        m_grid = geometry::VolumeGrid(dims(0), dims(1), dims(2));
        std::copy(other.data(), other.data() + other.m_grid.size(), m_grid.data());
        
        title = other.title;
        format = other.format;
        data_type = other.data_type;
        cell_parameters = other.cell_parameters;
        origin = other.origin;
        basis = other.basis;
        atoms = other.atoms;
        charges = other.charges;
        symmetry = other.symmetry;
    }
    return *this;
}

IVec3 PeriodicGrid::dimensions() const {
    auto dims = m_grid.dimensions();
    return IVec3(dims[0], dims[1], dims[2]);
}

PeriodicGrid PeriodicGrid::load_pgrid(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open pgrid file: " + filename);
    }
    
    PeriodicGrid grid;
    grid.format = GridFormat::PeriodicGrid;
    
    auto header = read_header(file);
    grid.title = std::string(header.title);
    grid.data_type = header.data_type == 0 ? GridDataType::Raw : GridDataType::Indexed;
    grid.cell_parameters = header.cell_params;
    
    if (header.data_type == 1) {
        grid.symmetry = read_symmetry_info(file, header);
        grid.read_indexed_data(file, header, grid.symmetry);
    } else {
        grid.read_raw_data(file, header);
    }
    
    return grid;
}

PeriodicGrid PeriodicGrid::load_ggrid(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open ggrid file: " + filename);
    }
    
    PeriodicGrid grid;
    grid.format = GridFormat::GeneralGrid;
    
    auto header = read_header(file);
    grid.title = std::string(header.title);
    grid.data_type = header.data_type == 0 ? GridDataType::Raw : GridDataType::Indexed;
    grid.cell_parameters = header.cell_params;
    
    if (header.data_type == 1) {
        grid.symmetry = read_symmetry_info(file, header);
        grid.read_indexed_data(file, header, grid.symmetry);
    } else {
        grid.read_raw_data(file, header);
    }
    
    return grid;
}

PeriodicGrid PeriodicGrid::load(const std::string &filename) {
    // Auto-detect format based on extension
    if (filename.ends_with(".pgrid")) {
        return load_pgrid(filename);
    } else if (filename.ends_with(".ggrid")) {
        return load_ggrid(filename);
    } else {
        throw std::runtime_error("Unknown grid file format: " + filename);
    }
}

void PeriodicGrid::save_pgrid(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Ensure we're saving as periodic
    if (format != GridFormat::PeriodicGrid) {
        PeriodicGrid periodic_copy = *this;
        periodic_copy.format = GridFormat::PeriodicGrid;
        periodic_copy.reduce_general_to_periodic();
        periodic_copy.save_pgrid(filename);
        return;
    }
    
    write_header(file);
    
    if (data_type == GridDataType::Indexed) {
        // Write symmetry info
        int npos = symmetry.num_operations;
        int ncen = symmetry.centrosymmetric;
        int nsub = symmetry.num_sublattices;
        
        file.write(reinterpret_cast<const char*>(&npos), sizeof(int));
        file.write(reinterpret_cast<const char*>(&ncen), sizeof(int));
        file.write(reinterpret_cast<const char*>(&nsub), sizeof(int));
        
        // Write symmetry operations
        for (const auto &op : symmetry.symmetry_operations) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int val = static_cast<int>(op(k, j));
                    file.write(reinterpret_cast<const char*>(&val), sizeof(int));
                }
            }
            for (int j = 0; j < 3; j++) {
                int val = static_cast<int>(op(j, 3) * dimensions()(j));
                file.write(reinterpret_cast<const char*>(&val), sizeof(int));
            }
        }
        
        // Write sublattice positions
        for (const auto &pos : symmetry.sublattice_positions) {
            file.write(reinterpret_cast<const char*>(pos.data()), 3 * sizeof(int));
        }
        
        write_indexed_data(file);
    } else {
        write_raw_data(file);
    }
}

void PeriodicGrid::save_ggrid(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Ensure we're saving as general
    if (format != GridFormat::GeneralGrid) {
        PeriodicGrid general_copy = *this;
        general_copy.format = GridFormat::GeneralGrid;
        general_copy.expand_periodic_to_general();
        general_copy.save_ggrid(filename);
        return;
    }
    
    write_header(file);
    
    if (data_type == GridDataType::Indexed) {
        // Similar to pgrid but for general grid
        write_indexed_data(file);
    } else {
        write_raw_data(file);
    }
}

void PeriodicGrid::save(const std::string &filename) const {
    if (filename.ends_with(".pgrid")) {
        save_pgrid(filename);
    } else if (filename.ends_with(".ggrid")) {
        save_ggrid(filename);
    } else {
        throw std::runtime_error("Unknown grid file format: " + filename);
    }
}

PeriodicGrid PeriodicGrid::from_cube(const Cube &cube, GridFormat format) {
    PeriodicGrid grid;
    grid.format = format;
    grid.title = cube.name;
    grid.origin = cube.origin;
    grid.basis = cube.basis;
    grid.atoms = cube.atoms;
    grid.charges = cube.charges;
    
    // Copy grid data
    auto dims = cube.grid().dimensions();
    grid.m_grid = geometry::VolumeGrid(dims[0], dims[1], dims[2]);
    std::copy(cube.data(), cube.data() + cube.grid().size(), grid.m_grid.data());
    
    // Set cell parameters from basis
    grid.cell_parameters[0] = cube.basis.col(0).norm() * cube.steps(0);
    grid.cell_parameters[1] = cube.basis.col(1).norm() * cube.steps(1);
    grid.cell_parameters[2] = cube.basis.col(2).norm() * cube.steps(2);
    
    // Calculate angles
    Vec3 a = cube.basis.col(0).normalized();
    Vec3 b = cube.basis.col(1).normalized();
    Vec3 c = cube.basis.col(2).normalized();
    
    grid.cell_parameters[3] = std::acos(b.dot(c)) * 180.0 / M_PI;  // alpha
    grid.cell_parameters[4] = std::acos(a.dot(c)) * 180.0 / M_PI;  // beta
    grid.cell_parameters[5] = std::acos(a.dot(b)) * 180.0 / M_PI;  // gamma
    
    if (format == GridFormat::PeriodicGrid) {
        grid.reduce_general_to_periodic();
    }
    
    return grid;
}

Cube PeriodicGrid::to_cube() const {
    Cube cube;
    cube.name = title;
    cube.description = "Converted from grid file";
    cube.origin = origin;
    cube.basis = basis;
    cube.atoms = atoms;
    cube.charges = charges;
    
    // Set steps from grid dimensions
    auto dims = dimensions();
    cube.steps = dims;
    
    // Copy grid data
    cube.grid() = geometry::VolumeGrid(dims(0), dims(1), dims(2));
    
    if (format == GridFormat::PeriodicGrid) {
        // Need to expand periodic to general
        PeriodicGrid general_copy = *this;
        general_copy.expand_periodic_to_general();
        std::copy(general_copy.data(), general_copy.data() + general_copy.grid().size(), 
                  cube.data());
    } else {
        std::copy(data(), data() + m_grid.size(), cube.data());
    }
    
    return cube;
}

void PeriodicGrid::write_header(std::ofstream &file) const {
    GridFileHeader header;
    
    // Copy title
    std::strncpy(header.title, title.c_str(), 79);
    header.title[79] = '\0';
    
    header.grid_type = format == GridFormat::PeriodicGrid ? 1 : 0;
    header.data_type = data_type == GridDataType::Indexed ? 1 : 0;
    header.num_values = 1;
    header.dimension = 3;
    
    auto dims = dimensions();
    header.num_voxels = {dims(0), dims(1), dims(2)};
    
    if (data_type == GridDataType::Raw) {
        header.num_asymmetric = dims(0) * dims(1) * dims(2);
    } else {
        // For indexed format, this would be the number of unique voxels
        // This needs to be calculated based on symmetry
        header.num_asymmetric = dims(0) * dims(1) * dims(2); // Simplified for now
    }
    
    header.cell_params = cell_parameters;
    
    // Write header
    file.write(reinterpret_cast<const char*>(&header.version), 4 * sizeof(int));
    file.write(header.title, 80);
    file.write(reinterpret_cast<const char*>(&header.grid_type), sizeof(int));
    file.write(reinterpret_cast<const char*>(&header.data_type), sizeof(int));
    file.write(reinterpret_cast<const char*>(&header.num_values), sizeof(int));
    file.write(reinterpret_cast<const char*>(&header.dimension), sizeof(int));
    file.write(reinterpret_cast<const char*>(&header.num_voxels), 3 * sizeof(int));
    file.write(reinterpret_cast<const char*>(&header.num_asymmetric), sizeof(int));
    file.write(reinterpret_cast<const char*>(&header.cell_params), 6 * sizeof(float));
}

void PeriodicGrid::write_raw_data(std::ofstream &file) const {
    auto dims = dimensions();
    
    for (int k = 0; k < dims(2); k++) {
        for (int j = 0; j < dims(1); j++) {
            for (int i = 0; i < dims(0); i++) {
                float val = m_grid(i, j, k);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
    }
}

void PeriodicGrid::write_indexed_data(std::ofstream &file) const {
    // Simplified version - writes all voxels as if they were unique
    auto dims = dimensions();
    
    for (int k = 0; k < dims(2); k++) {
        for (int j = 0; j < dims(1); j++) {
            for (int i = 0; i < dims(0); i++) {
                int index = k * dims(0) * dims(1) + j * dims(0) + i;
                float val = m_grid(i, j, k);
                file.write(reinterpret_cast<const char*>(&index), sizeof(int));
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
    }
}

GridFileHeader PeriodicGrid::read_header(std::ifstream &file) {
    GridFileHeader header;
    
    file.read(reinterpret_cast<char*>(&header.version), 4 * sizeof(int));
    file.read(header.title, 80);
    file.read(reinterpret_cast<char*>(&header.grid_type), sizeof(int));
    file.read(reinterpret_cast<char*>(&header.data_type), sizeof(int));
    file.read(reinterpret_cast<char*>(&header.num_values), sizeof(int));
    file.read(reinterpret_cast<char*>(&header.dimension), sizeof(int));
    file.read(reinterpret_cast<char*>(&header.num_voxels), 3 * sizeof(int));
    file.read(reinterpret_cast<char*>(&header.num_asymmetric), sizeof(int));
    file.read(reinterpret_cast<char*>(&header.cell_params), 6 * sizeof(float));
    
    return header;
}

GridSymmetryInfo PeriodicGrid::read_symmetry_info(std::ifstream &file, const GridFileHeader &header) {
    GridSymmetryInfo sym;
    
    if (header.data_type != 1) {
        return sym; // No symmetry info for raw data
    }
    
    file.read(reinterpret_cast<char*>(&sym.num_operations), sizeof(int));
    file.read(reinterpret_cast<char*>(&sym.centrosymmetric), sizeof(int));
    file.read(reinterpret_cast<char*>(&sym.num_sublattices), sizeof(int));
    
    // Read symmetry operations
    sym.symmetry_operations.resize(sym.num_operations);
    for (int i = 0; i < sym.num_operations; i++) {
        Mat4 op = Mat4::Identity();
        
        // Read rotation part
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int val;
                file.read(reinterpret_cast<char*>(&val), sizeof(int));
                op(k, j) = static_cast<double>(val);
            }
        }
        
        // Read translation part
        for (int j = 0; j < 3; j++) {
            int val;
            file.read(reinterpret_cast<char*>(&val), sizeof(int));
            op(j, 3) = static_cast<double>(val) / header.num_voxels[j];
        }
        
        sym.symmetry_operations[i] = op;
    }
    
    // Read sublattice positions
    sym.sublattice_positions.resize(sym.num_sublattices);
    for (int i = 0; i < sym.num_sublattices; i++) {
        IVec3 pos;
        file.read(reinterpret_cast<char*>(pos.data()), 3 * sizeof(int));
        sym.sublattice_positions[i] = pos;
    }
    
    return sym;
}

void PeriodicGrid::read_raw_data(std::ifstream &file, const GridFileHeader &header) {
    m_grid = geometry::VolumeGrid(header.num_voxels[0], header.num_voxels[1], 
                                  header.num_voxels[2]);
    
    file.read(reinterpret_cast<char*>(m_grid.data()), 
              m_grid.size() * sizeof(float));
}

void PeriodicGrid::read_indexed_data(std::ifstream &file, const GridFileHeader &header, 
                            const GridSymmetryInfo &sym) {
    m_grid = geometry::VolumeGrid(header.num_voxels[0], header.num_voxels[1], 
                                  header.num_voxels[2]);
    
    // Initialize grid to zero
    m_grid.set_zero();
    
    // Read indexed data
    for (int i = 0; i < header.num_asymmetric; i++) {
        int index;
        float value;
        file.read(reinterpret_cast<char*>(&index), sizeof(int));
        file.read(reinterpret_cast<char*>(&value), sizeof(float));
        
        // Convert linear index to 3D indices
        int k = index / (header.num_voxels[0] * header.num_voxels[1]);
        int j = (index % (header.num_voxels[0] * header.num_voxels[1])) / header.num_voxels[0];
        int ii = index % header.num_voxels[0];
        
        m_grid(ii, j, k) = value;
        
        // Apply symmetry operations to fill the rest of the grid
        // This is a simplified version - full implementation would need proper symmetry handling
    }
}

void PeriodicGrid::expand_periodic_to_general() {
    if (format != GridFormat::PeriodicGrid) return;
    
    auto dims = dimensions();
    
    // Create new grid with one extra point in each dimension
    geometry::VolumeGrid new_grid(dims(0) + 1, dims(1) + 1, dims(2) + 1);
    
    // Copy existing data
    for (int k = 0; k < dims(2); k++) {
        for (int j = 0; j < dims(1); j++) {
            for (int i = 0; i < dims(0); i++) {
                new_grid(i, j, k) = m_grid(i, j, k);
            }
        }
    }
    
    // Fill boundary points with periodic values
    for (int j = 0; j <= dims(1); j++) {
        for (int i = 0; i <= dims(0); i++) {
            new_grid(i, j, dims(2)) = new_grid(i % dims(0), j % dims(1), 0);
        }
    }
    
    for (int k = 0; k <= dims(2); k++) {
        for (int i = 0; i <= dims(0); i++) {
            new_grid(i, dims(1), k) = new_grid(i % dims(0), 0, k % dims(2));
        }
    }
    
    for (int k = 0; k <= dims(2); k++) {
        for (int j = 0; j <= dims(1); j++) {
            new_grid(dims(0), j, k) = new_grid(0, j % dims(1), k % dims(2));
        }
    }
    
    m_grid = std::move(new_grid);
    format = GridFormat::GeneralGrid;
}

void PeriodicGrid::reduce_general_to_periodic() {
    if (format != GridFormat::PeriodicGrid) return;
    
    auto dims = dimensions();
    
    // Create new grid with one less point in each dimension
    geometry::VolumeGrid new_grid(dims(0) - 1, dims(1) - 1, dims(2) - 1);
    
    // Copy data excluding the redundant boundary points
    for (int k = 0; k < dims(2) - 1; k++) {
        for (int j = 0; j < dims(1) - 1; j++) {
            for (int i = 0; i < dims(0) - 1; i++) {
                new_grid(i, j, k) = m_grid(i, j, k);
            }
        }
    }
    
    m_grid = std::move(new_grid);
    format = GridFormat::PeriodicGrid;
}


} // namespace occ::io