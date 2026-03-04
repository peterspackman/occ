#include <occ/mults/dmacrys_type_codes.h>
#include <occ/mults/dmacrys_input.h>
#include <occ/core/element.h>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <vector>

namespace occ::mults {

std::string normalize_type_key(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(),
                           [](unsigned char c) { return std::isspace(c) != 0; }),
            s.end());
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string infer_site_atom_type_from_label(const std::string& label) {
    auto stripped = label;
    while (!stripped.empty() && stripped.back() == '_') {
        stripped.pop_back();
    }
    const auto first = stripped.find('_');
    if (first == std::string::npos) {
        return stripped;
    }
    const auto second = stripped.find('_', first + 1);
    if (second == std::string::npos) {
        return stripped.substr(0, first);
    }
    return stripped.substr(0, second);
}

int infer_atomic_number_from_type(const std::string& atom_type) {
    auto token = atom_type;
    const auto us = token.find('_');
    if (us != std::string::npos) {
        token = token.substr(0, us);
    }
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) { return !std::isalpha(c); }),
                token.end());
    if (token.empty()) {
        return 0;
    }

    if (token.size() >= 2) {
        std::string sym2;
        sym2 += static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
        sym2 += static_cast<char>(std::tolower(static_cast<unsigned char>(token[1])));
        occ::core::Element el2(sym2);
        if (el2.atomic_number() > 0) {
            return el2.atomic_number();
        }
    }

    std::string sym1(1, static_cast<char>(
        std::toupper(static_cast<unsigned char>(token[0]))));
    occ::core::Element el1(sym1);
    if (el1.atomic_number() > 0) {
        return el1.atomic_number();
    }
    return 0;
}

DmacrysTypeCodeTables build_dmacrys_type_code_tables(const DmacrysInput& input) {

    static const std::map<std::string, int> kKnownTypeCodes = {
        {"H_W1", 501}, {"H_W2", 502}, {"H_W3", 503}, {"H_W4", 504},
        {"H_WA", 505}, {"C_W4", 511}, {"C_W3", 512}, {"C_W2", 513},
        {"N_W1", 521}, {"N_W2", 522}, {"N_W3", 523}, {"N_W4", 524},
        {"O_W1", 531}, {"O_W2", 532}, {"O_WA", 533}, {"F_01", 540},
        {"F01", 540}, {"CL01", 541}, {"CL_01", 541}, {"S_01", 542},
        {"S01", 542}, {"K_01", 543}, {"K01", 543}, {"BR01", 544},
        {"BR_01", 544}, {"I_01", 545}, {"I01", 545},
    };

    std::unordered_map<std::string, int> type_to_z;
    std::vector<std::string> ordered_types;

    auto note_type = [&](const std::string& raw_type, int z_hint) {
        if (raw_type.empty()) return;
        const std::string key = normalize_type_key(raw_type);
        if (key.empty()) return;
        if (std::find(ordered_types.begin(), ordered_types.end(), key) ==
            ordered_types.end()) {
            ordered_types.push_back(key);
        }
        if (z_hint > 0) {
            type_to_z[key] = z_hint;
        } else if (!type_to_z.count(key)) {
            type_to_z[key] = infer_atomic_number_from_type(raw_type);
        }
    };

    for (const auto& s : input.molecule.sites) {
        const std::string at = !s.atom_type.empty()
                                   ? s.atom_type
                                   : infer_site_atom_type_from_label(s.label);
        note_type(at, s.atomic_number);
    }

    for (const auto& p : input.potentials) {
        if (!p.type1.empty()) {
            note_type(p.type1, occ::core::Element(p.el1).atomic_number());
        }
        if (!p.type2.empty()) {
            note_type(p.type2, occ::core::Element(p.el2).atomic_number());
        }
    }

    DmacrysTypeCodeTables tables;
    std::unordered_map<int, int> custom_counts_by_z;
    for (const auto& key : ordered_types) {
        auto it_known = kKnownTypeCodes.find(key);
        int code = 0;
        if (it_known != kKnownTypeCodes.end()) {
            code = it_known->second;
        } else {
            int z = 0;
            auto itz = type_to_z.find(key);
            if (itz != type_to_z.end()) {
                z = itz->second;
            }
            if (z <= 0) {
                z = 99;
            }
            const int idx = ++custom_counts_by_z[z];
            code = 10000 + 100 * z + idx;
        }
        tables.type_to_code[key] = code;
        tables.code_to_type[code] = key;
    }
    return tables;
}

} // namespace occ::mults
