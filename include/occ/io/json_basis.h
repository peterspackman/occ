#pragma once
// Forwarding header for backward compatibility
// JsonBasisReader has moved to occ/gto/io/json_basis.h
#include <occ/gto/io/json_basis.h>

namespace occ::io {
// Re-export types for backward compatibility
using occ::gto::io::ElectronShell;
using occ::gto::io::ECPShell;
using occ::gto::io::ReferenceData;
using occ::gto::io::ElementBasis;
using occ::gto::io::ElementMap;
using occ::gto::io::JsonBasis;
using occ::gto::io::JsonBasisReader;
} // namespace occ::io
