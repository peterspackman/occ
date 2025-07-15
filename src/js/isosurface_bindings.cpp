#include "isosurface_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/io/cube.h>
#include <occ/io/isosurface_json.h>
#include <occ/isosurface/isosurface.h>
#include <occ/core/molecule.h>
#include <occ/qm/wavefunction.h>
#include <sstream>

using namespace emscripten;
using namespace occ;

void register_isosurface_bindings() {
    // Helper function to generate an isosurface mesh with promolecule density
    function("generatePromoleculeDensityIsosurface", optional_override([](
        const core::Molecule &mol,
        double isovalue,
        double separation
    ) {
        isosurface::IsosurfaceCalculator calc;
        calc.set_molecule(mol);
        
        isosurface::IsosurfaceGenerationParameters params;
        params.isovalue = isovalue;
        params.separation = separation;
        params.surface_kind = isosurface::SurfaceKind::PromoleculeDensity;
        
        calc.set_parameters(params);
        
        if (!calc.validate()) {
            throw std::runtime_error("Failed to validate isosurface calculation: " + calc.error_message());
        }
        
        calc.compute();
        const auto &surf = calc.isosurface();
        
        // Return mesh data as a JavaScript object
        val result = val::object();
        
        // Get vertices
        size_t numVertices = surf.vertices.cols();
        val vertices = val::global("Float32Array").new_(3 * numVertices);
        for (size_t i = 0; i < numVertices; ++i) {
            vertices.set(i * 3 + 0, surf.vertices(0, i));
            vertices.set(i * 3 + 1, surf.vertices(1, i));
            vertices.set(i * 3 + 2, surf.vertices(2, i));
        }
        
        // Get faces
        size_t numFaces = surf.faces.cols();
        val faces = val::global("Uint32Array").new_(3 * numFaces);
        for (size_t i = 0; i < numFaces; ++i) {
            faces.set(i * 3 + 0, surf.faces(0, i));
            faces.set(i * 3 + 1, surf.faces(1, i));
            faces.set(i * 3 + 2, surf.faces(2, i));
        }
        
        // Get normals
        val normals = val::global("Float32Array").new_(3 * numVertices);
        for (size_t i = 0; i < numVertices; ++i) {
            normals.set(i * 3 + 0, surf.normals(0, i));
            normals.set(i * 3 + 1, surf.normals(1, i));
            normals.set(i * 3 + 2, surf.normals(2, i));
        }
        
        result.set("vertices", vertices);
        result.set("faces", faces);
        result.set("normals", normals);
        result.set("numVertices", numVertices);
        result.set("numFaces", numFaces);
        result.set("volume", surf.volume());
        result.set("surfaceArea", surf.surface_area());
        
        return result;
    }));
    
    // Helper function to generate an isosurface mesh with electron density
    function("generateElectronDensityIsosurface", optional_override([](
        const qm::Wavefunction &wfn,
        double isovalue,
        double separation
    ) {
        isosurface::IsosurfaceCalculator calc;
        calc.set_wavefunction(wfn);
        
        // Extract molecule from wavefunction atoms
        std::vector<int> atomic_numbers;
        Mat3N positions(3, wfn.atoms.size());
        for (size_t i = 0; i < wfn.atoms.size(); ++i) {
            atomic_numbers.push_back(wfn.atoms[i].atomic_number);
            positions.col(i) = wfn.atoms[i].position();
        }
        core::Molecule mol(IVec::Map(atomic_numbers.data(), atomic_numbers.size()), positions);
        calc.set_molecule(mol);
        
        isosurface::IsosurfaceGenerationParameters params;
        params.isovalue = isovalue;
        params.separation = separation;
        params.surface_kind = isosurface::SurfaceKind::ElectronDensity;
        
        calc.set_parameters(params);
        
        if (!calc.validate()) {
            throw std::runtime_error("Failed to validate isosurface calculation: " + calc.error_message());
        }
        
        calc.compute();
        const auto &surf = calc.isosurface();
        
        // Return mesh data as a JavaScript object
        val result = val::object();
        
        // Get vertices
        size_t numVertices = surf.vertices.cols();
        val vertices = val::global("Float32Array").new_(3 * numVertices);
        for (size_t i = 0; i < numVertices; ++i) {
            vertices.set(i * 3 + 0, surf.vertices(0, i));
            vertices.set(i * 3 + 1, surf.vertices(1, i));
            vertices.set(i * 3 + 2, surf.vertices(2, i));
        }
        
        // Get faces
        size_t numFaces = surf.faces.cols();
        val faces = val::global("Uint32Array").new_(3 * numFaces);
        for (size_t i = 0; i < numFaces; ++i) {
            faces.set(i * 3 + 0, surf.faces(0, i));
            faces.set(i * 3 + 1, surf.faces(1, i));
            faces.set(i * 3 + 2, surf.faces(2, i));
        }
        
        // Get normals
        val normals = val::global("Float32Array").new_(3 * numVertices);
        for (size_t i = 0; i < numVertices; ++i) {
            normals.set(i * 3 + 0, surf.normals(0, i));
            normals.set(i * 3 + 1, surf.normals(1, i));
            normals.set(i * 3 + 2, surf.normals(2, i));
        }
        
        result.set("vertices", vertices);
        result.set("faces", faces);
        result.set("normals", normals);
        result.set("numVertices", numVertices);
        result.set("numFaces", numFaces);
        result.set("volume", surf.volume());
        result.set("surfaceArea", surf.surface_area());
        
        return result;
    }));
    
    // Simple function to export isosurface as JSON string
    function("isosurfaceToJSON", optional_override([](const isosurface::Isosurface &surf) {
        return io::isosurface_to_json_string(surf);
    }));

}