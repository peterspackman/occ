#include "dma_bindings.h"
#include <occ/driver/dma_driver.h>
#include <occ/dma/dma.h>
#include <occ/dma/mult.h>
#include <occ/core/element.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;
using namespace occ::driver;

void register_dma_bindings() {
    
    // Vector binding for std::vector<Mult>
    register_vector<occ::dma::Mult>("VectorMult");
    
    // Mult class (from occ::dma namespace)
    class_<occ::dma::Mult>("Mult")
        .constructor<>()
        .constructor<int>()
        .property("max_rank", &occ::dma::Mult::max_rank)
        .property("q", &occ::dma::Mult::q)
        .function("numComponents", &occ::dma::Mult::num_components)
        .function("Q00", select_overload<double&()>(&occ::dma::Mult::Q00))
        .function("charge", select_overload<double&()>(&occ::dma::Mult::charge))
        .function("Q10", select_overload<double&()>(&occ::dma::Mult::Q10))
        .function("Q11c", select_overload<double&()>(&occ::dma::Mult::Q11c))
        .function("Q11s", select_overload<double&()>(&occ::dma::Mult::Q11s))
        .function("Q20", select_overload<double&()>(&occ::dma::Mult::Q20))
        .function("Q21c", select_overload<double&()>(&occ::dma::Mult::Q21c))
        .function("Q21s", select_overload<double&()>(&occ::dma::Mult::Q21s))
        .function("Q22c", select_overload<double&()>(&occ::dma::Mult::Q22c))
        .function("Q22s", select_overload<double&()>(&occ::dma::Mult::Q22s))
        .function("Q30", select_overload<double&()>(&occ::dma::Mult::Q30))
        .function("Q31c", select_overload<double&()>(&occ::dma::Mult::Q31c))
        .function("Q31s", select_overload<double&()>(&occ::dma::Mult::Q31s))
        .function("Q32c", select_overload<double&()>(&occ::dma::Mult::Q32c))
        .function("Q32s", select_overload<double&()>(&occ::dma::Mult::Q32s))
        .function("Q33c", select_overload<double&()>(&occ::dma::Mult::Q33c))
        .function("Q33s", select_overload<double&()>(&occ::dma::Mult::Q33s))
        .function("Q40", select_overload<double&()>(&occ::dma::Mult::Q40))
        .function("Q41c", select_overload<double&()>(&occ::dma::Mult::Q41c))
        .function("Q41s", select_overload<double&()>(&occ::dma::Mult::Q41s))
        .function("Q42c", select_overload<double&()>(&occ::dma::Mult::Q42c))
        .function("Q42s", select_overload<double&()>(&occ::dma::Mult::Q42s))
        .function("Q43c", select_overload<double&()>(&occ::dma::Mult::Q43c))
        .function("Q43s", select_overload<double&()>(&occ::dma::Mult::Q43s))
        .function("Q44c", select_overload<double&()>(&occ::dma::Mult::Q44c))
        .function("Q44s", select_overload<double&()>(&occ::dma::Mult::Q44s))
        .function("getMultipole", select_overload<double(int, int) const>(&occ::dma::Mult::get_multipole))
        .function("getComponent", select_overload<double(const std::string&) const>(&occ::dma::Mult::get_component));

    // DMAResult class (from occ::dma namespace)
    class_<occ::dma::DMAResult>("DMAResult")
        .constructor<>()
        .property("max_rank", &occ::dma::DMAResult::max_rank)
        .property("multipoles", &occ::dma::DMAResult::multipoles);

    // DMASites class (from occ::dma namespace)
    class_<occ::dma::DMASites>("DMASites")
        .constructor<>()
        .property("atoms", &occ::dma::DMASites::atoms)
        .property("name", &occ::dma::DMASites::name)
        .property("positions", &occ::dma::DMASites::positions)
        .property("atom_indices", &occ::dma::DMASites::atom_indices)
        .property("radii", &occ::dma::DMASites::radii)
        .property("limits", &occ::dma::DMASites::limits)
        .function("size", &occ::dma::DMASites::size)
        .function("numAtoms", &occ::dma::DMASites::num_atoms);
    
    // DMASettings class (from occ::dma namespace)
    class_<occ::dma::DMASettings>("DMASettings")
        .constructor<>()
        .property("max_rank", &occ::dma::DMASettings::max_rank)
        .property("big_exponent", &occ::dma::DMASettings::big_exponent)
        .property("include_nuclei", &occ::dma::DMASettings::include_nuclei);
    
    // DMAConfig class
    class_<DMAConfig>("DMAConfig")
        .constructor<>()
        .property("wavefunction_filename", &DMAConfig::wavefunction_filename)
        .property("punch_filename", &DMAConfig::punch_filename)
        .property("settings", &DMAConfig::settings)
        .property("write_punch", &DMAConfig::write_punch)
        .function("setAtomRadius", optional_override([](DMAConfig& config, const std::string& element, double radius) {
            config.atom_radii.insert_or_assign(element, radius);
        }))
        .function("setAtomLimit", optional_override([](DMAConfig& config, const std::string& element, int limit) {
            config.atom_limits.insert_or_assign(element, limit);
        }))
        .function("setMaxRank", optional_override([](DMAConfig& config, int max_rank) {
            config.settings.max_rank = max_rank;
        }))
        .function("setBigExponent", optional_override([](DMAConfig& config, double big_exponent) {
            config.settings.big_exponent = big_exponent;
        }))
        .function("setIncludeNuclei", optional_override([](DMAConfig& config, bool include_nuclei) {
            config.settings.include_nuclei = include_nuclei;
        }));

    // DMADriver::DMAOutput class  
    class_<DMADriver::DMAOutput>("DMAOutput")
        .constructor<>()
        .property("result", &DMADriver::DMAOutput::result)
        .property("sites", &DMADriver::DMAOutput::sites);

    // DMADriver class
    class_<DMADriver>("DMADriver")
        .constructor<>()
        .constructor<const DMAConfig&>()
        .function("set_config", &DMADriver::set_config)
        .function("config", &DMADriver::config)
        .function("run", static_cast<DMADriver::DMAOutput(DMADriver::*)()>(&DMADriver::run))
        .function("runWithWavefunction", static_cast<DMADriver::DMAOutput(DMADriver::*)(const occ::qm::Wavefunction&)>(&DMADriver::run));
        
    // Static helper functions
    function("generate_punch_file", &DMADriver::generate_punch_file);
}