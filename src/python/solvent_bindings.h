#pragma once
#include <nanobind/nanobind.h>

namespace nb = nanobind;

nb::module_ register_solvent_bindings(nb::module_ &parent);
