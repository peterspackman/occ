#pragma once
#include <nanobind/nanobind.h>

namespace nb = nanobind;

nb::module_ register_core_bindings(nb::module_ &parent);
