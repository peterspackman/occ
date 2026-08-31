#include "cg_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/cg/morphology_types.h>
#include <occ/cg/result_types.h>
#include <occ/driver/cg_runner.h>
#include <occ/mults/dma_cg.h>

using namespace emscripten;
using occ::cg::CrystalGrowthResult;
using occ::driver::CGConfig;
using occ::driver::DMAReferenceLevel;

namespace {

val hkl_to_array(const occ::crystal::HKL &h) {
    val a = val::array();
    a.set(0, h.h);
    a.set(1, h.k);
    a.set(2, h.l);
    return a;
}

val morphology_to_val(const occ::cg::MorphologyResult &m) {
    val o = val::object();
    o.set("shape", m.shape);
    o.set("muBulk", m.mu_bulk);
    o.set("molecularVolume", m.molecular_volume);

    val facets = val::array();
    for (size_t i = 0; i < m.facets.size(); ++i) {
        const auto &f = m.facets[i];
        val fo = val::object();
        fo.set("hkl", hkl_to_array(f.hkl));
        fo.set("gamma", f.gamma);
        fo.set("area", f.area);
        facets.set(i, fo);
    }
    o.set("facets", facets);

    val edges = val::array();
    for (size_t i = 0; i < m.edges.size(); ++i) {
        const auto &e = m.edges[i];
        val eo = val::object();
        eo.set("hklA", hkl_to_array(e.hkl_a));
        eo.set("hklB", hkl_to_array(e.hkl_b));
        eo.set("length", e.length);
        eo.set("lineTension", e.lambda);
        edges.set(i, eo);
    }
    o.set("edges", edges);

    val corners = val::array();
    for (size_t i = 0; i < m.corners.size(); ++i) {
        const auto &c = m.corners[i];
        val co = val::object();
        val hkls = val::array();
        for (size_t k = 0; k < c.hkls.size(); ++k)
            hkls.set(k, hkl_to_array(c.hkls[k]));
        co.set("hkls", hkls);
        co.set("count", c.count);
        co.set("epsilon", c.epsilon);
        corners.set(i, co);
    }
    o.set("corners", corners);

    val samples = val::array();
    for (size_t i = 0; i < m.samples.size(); ++i) {
        const auto &s = m.samples[i];
        val so = val::object();
        so.set("sizeScale", s.size_scale);
        so.set("nMolecules", s.n_molecules);
        so.set("eExcess", s.e_excess);
        so.set("eSurface", s.e_surface);
        so.set("eEdge", s.e_edge);
        so.set("eCorner", s.e_corner);
        so.set("eSurfaceAnalytic", s.e_surface_analytic);
        so.set("area", s.area);
        so.set("edgeLength", s.edge_length);
        so.set("nCorners", s.n_corners);
        samples.set(i, so);
    }
    o.set("samples", samples);
    return o;
}

val result_to_val(const CrystalGrowthResult &result) {
    val o = val::object();
    val molecules = val::array();
    for (size_t i = 0; i < result.molecule_results.size(); ++i) {
        const auto &mr = result.molecule_results[i];
        val mo = val::object();
        mo.set("totalEnergy", mr.total_energy());
        mo.set("crystalEnergy", mr.total.crystal_energy);
        mo.set("interactionEnergy", mr.total.interaction_energy);
        mo.set("solutionTerm", mr.total.solution_term);
        molecules.set(i, mo);
    }
    o.set("moleculeResults", molecules);
    if (!result.morphology.empty())
        o.set("morphology", morphology_to_val(result.morphology));
    return o;
}

val calculate_crystal_growth_energies(const CGConfig &config) {
    return result_to_val(occ::mults::run_crystal_growth(config));
}

std::string get_crystal_filename(const CGConfig &c) {
    return c.lattice_settings.crystal_filename;
}
void set_crystal_filename(CGConfig &c, std::string v) {
    c.lattice_settings.crystal_filename = std::move(v);
}
std::string get_model_name(const CGConfig &c) {
    return c.lattice_settings.model_name;
}
void set_model_name(CGConfig &c, std::string v) {
    c.lattice_settings.model_name = std::move(v);
}
double get_max_radius(const CGConfig &c) {
    return c.lattice_settings.max_radius;
}
void set_max_radius(CGConfig &c, double v) {
    c.lattice_settings.max_radius = v;
}

} // namespace

void register_cg_bindings() {

    class_<DMAReferenceLevel>("DMAReferenceLevel")
        .constructor<>()
        .property("model", &DMAReferenceLevel::model)
        .property("method", &DMAReferenceLevel::method)
        .property("basis", &DMAReferenceLevel::basis);

    class_<CGConfig>("CrystalGrowthConfig")
        .constructor<>()
        .property("crystalFilename", &get_crystal_filename,
                  &set_crystal_filename)
        .property("modelName", &get_model_name, &set_model_name)
        .property("maxRadius", &get_max_radius, &set_max_radius)
        .property("solvent", &CGConfig::solvent)
        .property("solvationModel", &CGConfig::solvation_model)
        .property("temperature", &CGConfig::temperature)
        .property("chargeString", &CGConfig::charge_string)
        .property("dmaReference", &CGConfig::dma_reference)
        .property("cgRadius", &CGConfig::cg_radius)
        .property("computeMorphology", &CGConfig::compute_morphology)
        .property("numSurfaceEnergies", &CGConfig::max_facets);

    // Returns a plain JS object: {moleculeResults: [...], morphology?: {...}}
    function("calculateCrystalGrowthEnergies",
             &calculate_crystal_growth_energies);
}
