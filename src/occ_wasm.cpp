#include "core_bindings.h"
#include <emscripten/bind.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/main/occ_cg.h>
#include <occ/qm/shell.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(occ_module) {
  // Register all submodules
  occ::wasm::register_core_bindings();
}
