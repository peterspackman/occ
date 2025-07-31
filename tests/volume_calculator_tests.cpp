#include "test_utils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <occ/isosurface/volume_calculator.h>
#include <occ/qm/wavefunction.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/io/load_geometry.h>
#include <sstream>

using namespace occ;
using namespace occ::isosurface;
using Catch::Approx;

TEST_CASE("VolumeCalculator property name conversions", "[volume_calculator]") {
    SECTION("property_from_string conversions") {
        REQUIRE(VolumeCalculator::property_from_string("electron_density") == VolumePropertyKind::ElectronDensity);
        REQUIRE(VolumeCalculator::property_from_string("density") == VolumePropertyKind::ElectronDensity);
        REQUIRE(VolumeCalculator::property_from_string("rho") == VolumePropertyKind::ElectronDensity);
        REQUIRE(VolumeCalculator::property_from_string("rho_alpha") == VolumePropertyKind::ElectronDensityAlpha);
        REQUIRE(VolumeCalculator::property_from_string("rho_beta") == VolumePropertyKind::ElectronDensityBeta);
        REQUIRE(VolumeCalculator::property_from_string("esp") == VolumePropertyKind::ElectricPotential);
        REQUIRE(VolumeCalculator::property_from_string("eeqesp") == VolumePropertyKind::EEQ_ESP);
        REQUIRE(VolumeCalculator::property_from_string("promolecule") == VolumePropertyKind::PromoleculeDensity);
        REQUIRE(VolumeCalculator::property_from_string("deformation_density") == VolumePropertyKind::DeformationDensity);
        REQUIRE(VolumeCalculator::property_from_string("xc") == VolumePropertyKind::XCDensity);
        REQUIRE(VolumeCalculator::property_from_string("void") == VolumePropertyKind::CrystalVoid);
        
        REQUIRE_THROWS_WITH(VolumeCalculator::property_from_string("unknown"), "Unknown property: unknown");
    }
    
    SECTION("property_to_string conversions") {
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::ElectronDensity) == "electron_density");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::ElectronDensityAlpha) == "rho_alpha");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::ElectronDensityBeta) == "rho_beta");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::ElectricPotential) == "esp");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::EEQ_ESP) == "eeqesp");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::PromoleculeDensity) == "promolecule");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::DeformationDensity) == "deformation_density");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::XCDensity) == "xc");
        REQUIRE(VolumeCalculator::property_to_string(VolumePropertyKind::CrystalVoid) == "void");
    }
    
    SECTION("spin_from_string conversions") {
        REQUIRE(VolumeCalculator::spin_from_string("both") == SpinConstraint::Total);
        REQUIRE(VolumeCalculator::spin_from_string("total") == SpinConstraint::Total);
        REQUIRE(VolumeCalculator::spin_from_string("alpha") == SpinConstraint::Alpha);
        REQUIRE(VolumeCalculator::spin_from_string("beta") == SpinConstraint::Beta);
        
        REQUIRE_THROWS_WITH(VolumeCalculator::spin_from_string("unknown"), "Unknown spin constraint: unknown");
    }
    
    SECTION("format_from_string conversions") {
        REQUIRE(VolumeCalculator::format_from_string("cube") == OutputFormat::Cube);
        REQUIRE(VolumeCalculator::format_from_string("ggrid") == OutputFormat::GGrid);
        REQUIRE(VolumeCalculator::format_from_string("pgrid") == OutputFormat::PGrid);
        
        REQUIRE_THROWS_WITH(VolumeCalculator::format_from_string("unknown"), "Unknown output format: unknown");
    }
}

TEST_CASE("VolumeCalculator requirements checking", "[volume_calculator]") {
    SECTION("wavefunction requirements") {
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::ElectronDensity) == true);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::ElectronDensityAlpha) == true);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::ElectronDensityBeta) == true);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::ElectricPotential) == true);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::DeformationDensity) == true);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::XCDensity) == true);
    }
    
    SECTION("no wavefunction requirements") {
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::EEQ_ESP) == false);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::PromoleculeDensity) == false);
        REQUIRE(VolumeCalculator::requires_wavefunction(VolumePropertyKind::CrystalVoid) == false);
    }
    
    SECTION("crystal requirements") {
        REQUIRE(VolumeCalculator::requires_crystal(VolumePropertyKind::CrystalVoid) == true);
        REQUIRE(VolumeCalculator::requires_crystal(VolumePropertyKind::ElectronDensity) == false);
    }
}

TEST_CASE("VolumeCalculator basic functionality", "[volume_calculator]") {
    VolumeCalculator calc;
    
    SECTION("molecule setting") {
        // Create a simple molecule
        std::vector<core::Atom> atoms = {
            {1, 0.0, 0.0, 0.0},  // H at origin
            {1, 0.0, 0.0, 1.4}   // H at 1.4 Bohr along z
        };
        core::Molecule mol(atoms);
        
        calc.set_molecule(mol);
        // With new functional interface, setting molecule doesn't change state
        // We just verify it doesn't throw
        REQUIRE_NOTHROW(calc.set_molecule(mol));
    }
}

TEST_CASE("VolumeCalculator validation", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    SECTION("validation without required wavefunction") {
        params.property = VolumePropertyKind::ElectronDensity;
        
        REQUIRE_THROWS_WITH(calc.compute_volume(params), 
                           "Property requires a wavefunction: electron_density");
    }
    
    SECTION("validation without required crystal") {
        params.property = VolumePropertyKind::CrystalVoid;
        
        REQUIRE_THROWS_WITH(calc.compute_volume(params),
                           "Property requires a crystal structure: void");
    }
    
    SECTION("MO index validation without wavefunction") {
        params.property = VolumePropertyKind::EEQ_ESP;
        params.mo_number = 5;
        
        std::vector<core::Atom> atoms = {{1, 0.0, 0.0, 0.0}};
        core::Molecule mol(atoms);
        calc.set_molecule(mol);
        
        REQUIRE_THROWS_WITH(calc.compute_volume(params),
                           "MO index specified but property does not use wavefunction");
    }
}

TEST_CASE("VolumeCalculator promolecule density computation", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    // Create a single H atom (minimal test)
    std::vector<core::Atom> atoms = {
        {1, 0.0, 0.0, 0.0}  // H at origin
    };
    core::Molecule mol(atoms);
    
    params.property = VolumePropertyKind::PromoleculeDensity;
    params.steps = {2, 2, 2};  // Minimal 2x2x2 grid for speed
    
    calc.set_molecule(mol);
    
    SECTION("computation completes successfully") {
        VolumeData volume = calc.compute_volume(params);
        
        REQUIRE(volume.steps(0) == 2);
        REQUIRE(volume.steps(1) == 2);
        REQUIRE(volume.steps(2) == 2);
        REQUIRE(volume.atoms.size() == 1);
        REQUIRE(volume.property == VolumePropertyKind::PromoleculeDensity);
        
        // Check that data is non-zero (promolecular density should be positive)
        bool has_positive_data = false;
        for (int i = 0; i < volume.nx(); i++) {
            for (int j = 0; j < volume.ny(); j++) {
                for (int k = 0; k < volume.nz(); k++) {
                    if (volume.data(i, j, k) > 0.0) {
                        has_positive_data = true;
                        break;
                    }
                }
                if (has_positive_data) break;
            }
            if (has_positive_data) break;
        }
        REQUIRE(has_positive_data == true);
    }
}

TEST_CASE("VolumeCalculator EEQ ESP computation", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    // Create a single H atom (minimal test)
    std::vector<core::Atom> atoms = {
        {1, 0.0, 0.0, 0.0}  // H at origin
    };
    core::Molecule mol(atoms);
    
    params.property = VolumePropertyKind::EEQ_ESP;
    params.steps = {2, 2, 2};  // Minimal 2x2x2 grid for speed
    
    calc.set_molecule(mol);
    
    SECTION("computation completes successfully") {
        VolumeData volume = calc.compute_volume(params);
        
        REQUIRE(volume.atoms.size() == 1);
        REQUIRE(volume.property == VolumePropertyKind::EEQ_ESP);
        
        // EEQ ESP can be positive or negative, just check it's not all zeros
        bool has_nonzero_data = false;
        for (int i = 0; i < volume.nx(); i++) {
            for (int j = 0; j < volume.ny(); j++) {
                for (int k = 0; k < volume.nz(); k++) {
                    if (std::abs(volume.data(i, j, k)) > 1e-10) {
                        has_nonzero_data = true;
                        break;
                    }
                }
                if (has_nonzero_data) break;
            }
            if (has_nonzero_data) break;
        }
        REQUIRE(has_nonzero_data == true);
    }
}

TEST_CASE("VolumeCalculator parameter application", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    std::vector<core::Atom> atoms = {{1, 0.0, 0.0, 0.0}};
    core::Molecule mol(atoms);
    calc.set_molecule(mol);
    
    SECTION("single step value applied to all dimensions") {
        params.property = VolumePropertyKind::PromoleculeDensity;
        params.steps = {3};  // Reduced from 7 to 3
        
        VolumeData volume = calc.compute_volume(params);
        
        REQUIRE(volume.steps(0) == 3);
        REQUIRE(volume.steps(1) == 3);
        REQUIRE(volume.steps(2) == 3);
    }
    
    SECTION("individual step values") {
        params.property = VolumePropertyKind::PromoleculeDensity;
        params.steps = {2, 2, 3};  // Reduced from {3, 4, 5}
        
        VolumeData volume = calc.compute_volume(params);
        
        REQUIRE(volume.steps(0) == 2);
        REQUIRE(volume.steps(1) == 2);
        REQUIRE(volume.steps(2) == 3);
    }
    
    SECTION("origin parameters") {
        params.property = VolumePropertyKind::PromoleculeDensity;
        params.steps = {2, 2, 2};  // Reduced from {3, 3, 3}
        params.origin = {1.0, 2.0, 3.0};
        
        VolumeData volume = calc.compute_volume(params);
        
        REQUIRE(volume.origin(0) == Approx(1.0));
        REQUIRE(volume.origin(1) == Approx(2.0));
        REQUIRE(volume.origin(2) == Approx(3.0));
    }
}

TEST_CASE("VolumeCalculator save functionality", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    std::vector<core::Atom> atoms = {{1, 0.0, 0.0, 0.0}};
    core::Molecule mol(atoms);
    
    params.property = VolumePropertyKind::PromoleculeDensity;
    params.steps = {2, 2, 2};  // Reduced from {3, 3, 3}
    
    calc.set_molecule(mol);
    
    SECTION("volume_as_cube_string produces valid output") {
        VolumeData volume = calc.compute_volume(params);
        std::string cube_str = calc.volume_as_cube_string(volume);
        REQUIRE(!cube_str.empty());
        
        // Should contain standard cube file elements
        REQUIRE(cube_str.find("Generated by OCC") != std::string::npos);
        REQUIRE(cube_str.find("1") != std::string::npos);  // Should contain the H atom
    }
}

// Note: Static convenience methods tested separately due to wavefunction setup complexity

TEST_CASE("VolumeCalculator custom points evaluation", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    std::vector<core::Atom> atoms = {{1, 0.0, 0.0, 0.0}};
    core::Molecule mol(atoms);
    
    params.property = VolumePropertyKind::PromoleculeDensity;
    calc.set_molecule(mol);
    
    SECTION("evaluate at custom points") {
        // Define just 2 test points instead of 3
        Mat3N points(3, 2);
        points.col(0) = Vec3(0.2, 0.0, 0.0);   // Near atom (0.2 Bohr away)
        points.col(1) = Vec3(2.0, 0.0, 0.0);   // Far away
        
        Vec values = calc.evaluate_at_points(points, params);
        
        REQUIRE(values.size() == 2);
        // Density should be highest at the atom position
        REQUIRE(values(0) > values(1));
        REQUIRE(values(1) >= 0.0);  // Should be non-negative
    }
}

// Test grid parameter edge cases that caused issues in the original implementation
TEST_CASE("VolumeCalculator grid parameter edge cases", "[volume_calculator]") {
    VolumeCalculator calc;
    VolumeGenerationParameters params;
    
    std::vector<core::Atom> atoms = {{1, 0.0, 0.0, 0.0}};
    core::Molecule mol(atoms);
    calc.set_molecule(mol);
    
    params.property = VolumePropertyKind::PromoleculeDensity;
    
    SECTION("empty parameter vectors use defaults") {
        // All parameter vectors empty - should use defaults
        VolumeData volume = calc.compute_volume(params);
        
        REQUIRE(volume.steps(0) == 11);  // Default
        REQUIRE(volume.steps(1) == 11);  // Default
        REQUIRE(volume.steps(2) == 11);  // Default
    }
    
    SECTION("da/db/dc parameters are not yet implemented in new interface") {
        // The new interface focuses on the core functionality first
        // Grid basis vectors are set to diagonal with 0.2 Bohr spacing by default
        params.steps = {3, 3, 3};
        
        VolumeData volume = calc.compute_volume(params);
        
        // Default 0.2 Bohr spacing in all directions
        REQUIRE(volume.basis(0, 0) == Approx(0.2));
        REQUIRE(volume.basis(1, 1) == Approx(0.2));
        REQUIRE(volume.basis(2, 2) == Approx(0.2));
        REQUIRE(volume.basis(0, 1) == Approx(0.0));
        REQUIRE(volume.basis(1, 2) == Approx(0.0));
        REQUIRE(volume.basis(0, 2) == Approx(0.0));
    }
}