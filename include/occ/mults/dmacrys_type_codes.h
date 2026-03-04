#pragma once
#include <map>
#include <string>

namespace occ::mults {

struct DmacrysInput;  // Forward declaration

/// Bidirectional mapping between DMACRYS type strings and integer type codes.
struct DmacrysTypeCodeTables {
    std::map<std::string, int> type_to_code;
    std::map<int, std::string> code_to_type;
};

/// Normalize a type key: strip whitespace, uppercase.
std::string normalize_type_key(std::string s);

/// Extract atom type from site label using two-underscore parsing.
/// e.g. "C_W4_1" -> "C_W4", "H1" -> "H1".
std::string infer_site_atom_type_from_label(const std::string& label);

/// Infer atomic number from a type string like "C_W4" -> 6, "BR01" -> 35.
int infer_atomic_number_from_type(const std::string& atom_type);

/// Build canonical type code tables from DMACRYS input.
/// Known Williams codes (H_W1=501, C_W3=512, etc.) are used when possible;
/// custom types get auto-generated codes: 10000 + 100*Z + index.
DmacrysTypeCodeTables build_dmacrys_type_code_tables(const DmacrysInput& input);

} // namespace occ::mults
