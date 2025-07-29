#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/io/adaptive_grid.h>
#include <occ/io/cube.h>
#include <occ/io/periodic_grid.h>
#include <occ/io/xyz.h>
#include <occ/core/molecule.h>
#include <filesystem>

using namespace occ;
using namespace occ::io;

TEST_CASE("PeriodicGrid basic functionality", "[io][periodic_grid]") {
    SECTION("Create and save ggrid file") {
        PeriodicGrid grid;
        grid.title = "Test general grid";
        grid.format = GridFormat::GeneralGrid;
        grid.cell_parameters = {10.0, 10.0, 10.0, 90.0, 90.0, 90.0};
        
        // Create a simple 5x5x5 grid
        grid.grid() = geometry::VolumeGrid(5, 5, 5);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    grid.grid()(i, j, k) = i + j + k;
                }
            }
        }
        
        // Save and reload
        std::string filename = "test.ggrid";
        grid.save_ggrid(filename);
        
        auto loaded = PeriodicGrid::load_ggrid(filename);
        
        REQUIRE(loaded.title == grid.title);
        REQUIRE(loaded.dimensions() == grid.dimensions());
        REQUIRE(loaded.cell_parameters == grid.cell_parameters);
        
        // Check data
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    REQUIRE(loaded.grid()(i, j, k) == Catch::Approx(i + j + k));
                }
            }
        }
        
        std::filesystem::remove(filename);
    }
    
    SECTION("Convert between cube and grid formats") {
        // Create a cube
        Cube cube;
        cube.name = "Test cube";
        cube.origin = Vec3(0, 0, 0);
        cube.basis = Mat3::Identity() * 0.2;
        cube.steps = IVec3(10, 10, 10);
        
        // Fill with test data
        auto func = [](Eigen::Ref<const Mat3N> pos, Eigen::Ref<Vec> values) {
            for (int i = 0; i < pos.cols(); i++) {
                values(i) = pos.col(i).norm();
            }
        };
        cube.fill_data_from_function(func);
        
        // Convert to ggrid
        auto ggrid = PeriodicGrid::from_cube(cube, GridFormat::GeneralGrid);
        REQUIRE(ggrid.dimensions() == cube.steps);
        
        // Convert to pgrid
        auto pgrid = PeriodicGrid::from_cube(cube, GridFormat::PeriodicGrid);
        // Periodic grid should have one less point in each dimension
        REQUIRE(pgrid.dimensions() == IVec3(9, 9, 9));
        
        // Convert back to cube
        auto cube_from_ggrid = ggrid.to_cube();
        REQUIRE(cube_from_ggrid.steps == cube.steps);
        
        // Check some data points
        for (int i = 0; i < 5; i++) {
            REQUIRE(cube.grid()(i, i, i) == Catch::Approx(cube_from_ggrid.grid()(i, i, i)));
        }
    }
}

TEST_CASE("Adaptive grid bounds calculation", "[io][adaptive_grid]") {
    SECTION("Simple exponential decay function") {
        // Create a test function that decays exponentially from origin
        auto func = [](Eigen::Ref<const Mat3N> pos, Eigen::Ref<Vec> values) {
            for (int i = 0; i < pos.cols(); i++) {
                double r = pos.col(i).norm();
                values(i) = std::exp(-r);
            }
        };
        
        AdaptiveGridBounds<decltype(func)>::Parameters params;
        params.value_threshold = 1e-4;
        params.extra_buffer = 1.0;
        
        auto bounds_calc = make_adaptive_bounds(func, params);
        
        // Test with center at origin
        auto bounds = bounds_calc.compute(Vec3(0, 0, 0));
        
        // For exp(-r) = 1e-4, r ≈ 9.21
        // So we expect bounds roughly from -10.21 to 10.21 (with buffer)
        double expected_size = 2 * (9.21 + 1.0);
        Vec3 actual_size = bounds.max_corner() - bounds.min_corner();
        
        REQUIRE(actual_size.x() == Catch::Approx(expected_size).margin(1.0));
        REQUIRE(actual_size.y() == Catch::Approx(expected_size).margin(1.0));
        REQUIRE(actual_size.z() == Catch::Approx(expected_size).margin(1.0));
    }
    
    SECTION("Molecule-based bounds") {
        // Create a simple water molecule
        core::Molecule mol;
        mol = core::Molecule(std::vector<core::Atom>{
            core::Atom{8, 0.0, 0.0, 0.0},
            core::Atom{1, 0.0, 0.757, 0.587},
            core::Atom{1, 0.0, -0.757, 0.587}
        });
        
        // Mock electron density function
        auto func = [&mol](Eigen::Ref<const Mat3N> pos, Eigen::Ref<Vec> values) {
            values.setZero();
            for (int i = 0; i < pos.cols(); i++) {
                for (int j = 0; j < mol.size(); j++) {
                    Vec3 atom_pos = mol.positions().col(j);
                    double r = (pos.col(i) - atom_pos).norm();
                    values(i) += std::exp(-2.0 * r);
                }
            }
        };
        
        AdaptiveGridBounds<decltype(func)>::Parameters params;
        params.value_threshold = 1e-6;
        params.extra_buffer = 2.0;
        
        auto bounds_calc = make_adaptive_bounds(func, params);
        auto bounds = bounds_calc.compute(mol);
        
        // Check that bounds include all atoms with margin
        Vec3 min_bound = bounds.min_corner();
        Vec3 max_bound = bounds.max_corner();
        
        for (int i = 0; i < mol.size(); i++) {
            Vec3 atom = mol.positions().col(i);
            REQUIRE(atom.x() > min_bound.x());
            REQUIRE(atom.y() > min_bound.y());
            REQUIRE(atom.z() > min_bound.z());
            REQUIRE(atom.x() < max_bound.x());
            REQUIRE(atom.y() < max_bound.y());
            REQUIRE(atom.z() < max_bound.z());
        }
    }
}

TEST_CASE("PeriodicGrid file format details", "[io][periodic_grid]") {
    SECTION("Header writing and reading") {
        PeriodicGrid grid;
        grid.title = "Test grid with special characters: αβγ";
        grid.format = GridFormat::GeneralGrid;
        grid.cell_parameters = {5.5, 6.6, 7.7, 89.0, 91.0, 92.0};
        grid.grid() = geometry::VolumeGrid(3, 4, 5);
        
        // Fill with test pattern
        float value = 0.0;
        for (int k = 0; k < 5; k++) {
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 3; i++) {
                    grid.grid()(i, j, k) = value++;
                }
            }
        }
        
        std::string filename = "test_header.ggrid";
        grid.save(filename);
        
        auto loaded = PeriodicGrid::load(filename);
        
        // Title might be truncated to 79 chars
        REQUIRE(loaded.title.substr(0, 20) == "Test grid with speci");
        REQUIRE(loaded.cell_parameters[0] == Catch::Approx(5.5));
        REQUIRE(loaded.cell_parameters[1] == Catch::Approx(6.6));
        REQUIRE(loaded.cell_parameters[2] == Catch::Approx(7.7));
        REQUIRE(loaded.cell_parameters[3] == Catch::Approx(89.0));
        REQUIRE(loaded.cell_parameters[4] == Catch::Approx(91.0));
        REQUIRE(loaded.cell_parameters[5] == Catch::Approx(92.0));
        
        std::filesystem::remove(filename);
    }
}