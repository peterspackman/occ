#pragma once
#include <nanobind/nanobind.h>

namespace nb = nanobind;

nb::module_ register_cg_bindings(nb::module_ &parent);
