#include "js/core_bindings.h"
#include "js/qm_bindings.h"
#include "js/dft_bindings.h"
#include "js/opt_bindings.h"
#include "js/isosurface_bindings.h"
#include "js/cube_bindings.h"
#include "js/crystal_bindings.h"
#include "js/dma_bindings.h"
#include "js/volume_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(occ) {
    // Register all binding modules
    register_core_bindings();
    register_qm_bindings();
    register_dft_bindings();
    register_opt_bindings();
    register_isosurface_bindings();
    register_cube_bindings();
    register_crystal_bindings();
    register_dma_bindings();
    register_volume_bindings();

    // Global utility functions
    // Note: LogLevel enum and logging functions are now registered in core_bindings.cpp
    function("setLogFile", &occ::log::set_log_file);

    // Version information
    constant("version", std::string("0.7.6"));
}
