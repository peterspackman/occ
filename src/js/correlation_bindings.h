#pragma once

// Register post-HF correlation methods (runCorrelation, MP2, CCSD/(T),
// UCCSD) with embind. Mirrors src/python/correlation_bindings.cpp.
void register_correlation_bindings();
