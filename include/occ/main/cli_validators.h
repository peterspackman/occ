#pragma once
#include <CLI/Validators.hpp>

namespace occ::main {

struct MultiplicityValidator : public CLI::Validator {
  MultiplicityValidator();
};

struct SpinorbitalKindValidator : public CLI::Validator {
  SpinorbitalKindValidator();
};

namespace validator {
const static MultiplicityValidator Multiplicity;
const static SpinorbitalKindValidator SpinorbitalKind;
} // namespace validator

} // namespace occ::main
