#include "volume_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/isosurface/volume_calculator.h>
#include <occ/isosurface/volume_data.h>
#include <occ/core/molecule.h>
#include <occ/qm/wavefunction.h>

using namespace emscripten;
using namespace occ;

void register_volume_bindings() {
    // Register VolumePropertyKind enum
    enum_<isosurface::VolumePropertyKind>("VolumePropertyKind")
        .value("ElectronDensity", isosurface::VolumePropertyKind::ElectronDensity)
        .value("ElectronDensityAlpha", isosurface::VolumePropertyKind::ElectronDensityAlpha)
        .value("ElectronDensityBeta", isosurface::VolumePropertyKind::ElectronDensityBeta)
        .value("ElectricPotential", isosurface::VolumePropertyKind::ElectricPotential)
        .value("EEQ_ESP", isosurface::VolumePropertyKind::EEQ_ESP)
        .value("PromoleculeDensity", isosurface::VolumePropertyKind::PromoleculeDensity)
        .value("DeformationDensity", isosurface::VolumePropertyKind::DeformationDensity)
        .value("XCDensity", isosurface::VolumePropertyKind::XCDensity)
        .value("CrystalVoid", isosurface::VolumePropertyKind::CrystalVoid);

    // Register SpinConstraint enum
    enum_<isosurface::SpinConstraint>("SpinConstraint")
        .value("Total", isosurface::SpinConstraint::Total)
        .value("Alpha", isosurface::SpinConstraint::Alpha)
        .value("Beta", isosurface::SpinConstraint::Beta);

    // Register VolumeGenerationParameters
    class_<isosurface::VolumeGenerationParameters>("VolumeGenerationParameters")
        .constructor<>()
        .property("property", &isosurface::VolumeGenerationParameters::property)
        .property("spin", &isosurface::VolumeGenerationParameters::spin)
        .property("functional", &isosurface::VolumeGenerationParameters::functional)
        .property("mo_number", &isosurface::VolumeGenerationParameters::mo_number)
        .function("setSteps", optional_override([](isosurface::VolumeGenerationParameters &params, 
                                                   int nx, int ny, int nz) {
            params.steps = {nx, ny, nz};
        }))
        .function("setOrigin", optional_override([](isosurface::VolumeGenerationParameters &params,
                                                    double x, double y, double z) {
            params.origin = {x, y, z};
        }));

    // Register VolumeData
    class_<isosurface::VolumeData>("VolumeData")
        .property("name", &isosurface::VolumeData::name)
        .property("property", &isosurface::VolumeData::property)
        .function("nx", &isosurface::VolumeData::nx)
        .function("ny", &isosurface::VolumeData::ny)
        .function("nz", &isosurface::VolumeData::nz)
        .function("totalPoints", &isosurface::VolumeData::total_points)
        .function("getOrigin", optional_override([](const isosurface::VolumeData &volume) {
            val result = val::array();
            result.set(0, volume.origin(0));
            result.set(1, volume.origin(1));
            result.set(2, volume.origin(2));
            return result;
        }))
        .function("getBasis", optional_override([](const isosurface::VolumeData &volume) {
            val result = val::global("Float64Array").new_(9);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    result.set(i * 3 + j, volume.basis(i, j));
                }
            }
            return result;
        }))
        .function("getSteps", optional_override([](const isosurface::VolumeData &volume) {
            val result = val::array();
            result.set(0, volume.steps(0));
            result.set(1, volume.steps(1));
            result.set(2, volume.steps(2));
            return result;
        }))
        .function("getData", optional_override([](const isosurface::VolumeData &volume) {
            // Return data as Float32Array for efficiency
            size_t totalSize = volume.total_points();
            val result = val::global("Float32Array").new_(totalSize);
            
            // Copy data in cube file order: [x][y][z]
            size_t idx = 0;
            for (int i = 0; i < volume.nx(); i++) {
                for (int j = 0; j < volume.ny(); j++) {
                    for (int k = 0; k < volume.nz(); k++) {
                        result.set(idx++, static_cast<float>(volume.data(i, j, k)));
                    }
                }
            }
            return result;
        }));

    // Register VolumeCalculator
    class_<isosurface::VolumeCalculator>("VolumeCalculator")
        .constructor<>()
        .function("setWavefunction", &isosurface::VolumeCalculator::set_wavefunction)
        .function("setMolecule", &isosurface::VolumeCalculator::set_molecule)
        .function("computeVolume", optional_override([](isosurface::VolumeCalculator &calc,
                                                       const isosurface::VolumeGenerationParameters &params) {
            return calc.compute_volume(params);
        }))
        .function("volumeAsCubeString", &isosurface::VolumeCalculator::volume_as_cube_string)
        .class_function("computeDensityVolume", 
            optional_override([](const qm::Wavefunction &wfn,
                               const isosurface::VolumeGenerationParameters &params) {
                return isosurface::VolumeCalculator::compute_density_volume(wfn, params);
            }))
        .class_function("computeMOVolume",
            optional_override([](const qm::Wavefunction &wfn, int mo_index,
                               const isosurface::VolumeGenerationParameters &params) {
                return isosurface::VolumeCalculator::compute_mo_volume(wfn, mo_index, params);
            }));

    // Convenience functions for electron density and MO cube generation
    function("generateElectronDensityCube", 
        optional_override([](const qm::Wavefunction &wfn, int nx, int ny, int nz) {
            isosurface::VolumeCalculator calc;
            calc.set_wavefunction(wfn);
            
            isosurface::VolumeGenerationParameters params;
            params.property = isosurface::VolumePropertyKind::ElectronDensity;
            params.steps = {nx, ny, nz};
            
            auto volume = calc.compute_volume(params);
            return calc.volume_as_cube_string(volume);
        }));

    function("generateMOCube",
        optional_override([](const qm::Wavefunction &wfn, int mo_index, int nx, int ny, int nz) {
            isosurface::VolumeCalculator calc;
            calc.set_wavefunction(wfn);
            
            isosurface::VolumeGenerationParameters params;
            params.property = isosurface::VolumePropertyKind::ElectronDensity;
            params.mo_number = mo_index;
            params.steps = {nx, ny, nz};
            
            auto volume = calc.compute_volume(params);
            return calc.volume_as_cube_string(volume);
        }));

    // Function to generate MO cube with spin constraint
    function("generateMOCubeWithSpin",
        optional_override([](const qm::Wavefunction &wfn, int mo_index, 
                           isosurface::SpinConstraint spin, int nx, int ny, int nz) {
            isosurface::VolumeCalculator calc;
            calc.set_wavefunction(wfn);
            
            isosurface::VolumeGenerationParameters params;
            if (spin == isosurface::SpinConstraint::Alpha) {
                params.property = isosurface::VolumePropertyKind::ElectronDensityAlpha;
            } else if (spin == isosurface::SpinConstraint::Beta) {
                params.property = isosurface::VolumePropertyKind::ElectronDensityBeta;
            } else {
                params.property = isosurface::VolumePropertyKind::ElectronDensity;
            }
            params.mo_number = mo_index;
            params.spin = spin;
            params.steps = {nx, ny, nz};
            
            auto volume = calc.compute_volume(params);
            return calc.volume_as_cube_string(volume);
        }));
}