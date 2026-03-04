#include <occ/mults/force_field_params.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <algorithm>
#include <cmath>

namespace occ::mults {

namespace {

constexpr double kBondToleranceAngstrom = 0.4;

std::pair<int, int> canonical_pair(int a, int b) {
    return (a <= b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct TypedSelfBuckingham {
    int code = 0;
    double A = 0.0;
    double rho = 0.0;
    double C = 0.0;
};

std::vector<TypedSelfBuckingham> williams_typed_self_params() {
    std::vector<TypedSelfBuckingham> self{
        {513, 1069.960000, 0.277778, 14.874827}, // C_W2
        {512, 2802.120000, 0.277778, 17.638572}, // C_W3
        {511, 1363.640000, 0.277778, 10.140782}, // C_W4
        {501,  131.420000, 0.280899,  2.885328}, // H_W1
        {502,    3.740000, 0.280899,  0.000000}, // H_W2
        {503,    1.200000, 0.280899,  0.000000}, // H_W3
        {504,    7.930000, 0.280899,  0.000000}, // H_W4
        {521,  998.590000, 0.287356, 14.589580}, // N_W1
        {522, 1060.980000, 0.287356, 14.491940}, // N_W2
        {523, 1989.270000, 0.287356, 24.633137}, // N_W3
        {524, 4201.060000, 0.287356, 58.353550}, // N_W4
        {531, 2498.220000, 0.252525, 13.067571}, // O_W1
        {532, 2949.910000, 0.252525, 13.328149}, // O_W2
        {540, 3761.006673, 0.240385,  7.144500}, // F_01
        {541, 5903.747391, 0.299155, 86.716330}, // Cl01
        {544,12272.878680, 0.303030,168.478200}, // Br01
        {545,13072.690000, 0.318249,172.380900}, // I_01
    };
    self.push_back({505, 3.740000, 0.280899, 0.000000}); // H_Wa -> H_W2
    self.push_back({533, 2949.910000, 0.252525, 13.328149}); // O_Wa -> O_W2
    return self;
}

} // namespace

// ============================================================================
// Element-based Buckingham
// ============================================================================

void ForceFieldParams::set_buckingham(int Z1, int Z2, const BuckinghamParams& p) {
    m_buckingham_params[{Z1, Z2}] = p;
    m_buckingham_params[{Z2, Z1}] = p;
}

BuckinghamParams ForceFieldParams::get_buckingham(int Z1, int Z2) const {
    auto it = m_buckingham_params.find({Z1, Z2});
    if (it != m_buckingham_params.end()) {
        return it->second;
    }
    const auto key = std::make_pair(std::min(Z1, Z2), std::max(Z1, Z2));
    if (m_missing_buckingham_warned.insert(key).second) {
        occ::log::warn(
            "Missing Buckingham parameters for Z{}-Z{}; using fallback A=1000, B=3.5, C=10",
            key.first, key.second);
    }
    return {1000.0, 3.5, 10.0};
}

bool ForceFieldParams::has_buckingham(int Z1, int Z2) const {
    return m_buckingham_params.find({Z1, Z2}) != m_buckingham_params.end();
}

// ============================================================================
// Type-code-based Buckingham
// ============================================================================

void ForceFieldParams::set_typed_buckingham(int type1, int type2, const BuckinghamParams& p) {
    m_typed_buckingham_params[{type1, type2}] = p;
    m_typed_buckingham_params[{type2, type1}] = p;
    m_use_short_range_typing = true;
}

void ForceFieldParams::set_typed_buckingham(
    const std::map<std::pair<int,int>, BuckinghamParams>& params) {
    m_typed_buckingham_params = params;
    if (!m_typed_buckingham_params.empty()) {
        m_use_short_range_typing = true;
    }
}

void ForceFieldParams::clear_typed_buckingham() {
    m_typed_buckingham_params.clear();
    m_missing_typed_buckingham_warned.clear();
    m_use_short_range_typing = m_use_williams_atom_typing;
}

bool ForceFieldParams::has_typed_buckingham(int type1, int type2) const {
    return m_typed_buckingham_params.find({type1, type2}) !=
           m_typed_buckingham_params.end();
}

BuckinghamParams ForceFieldParams::get_typed_buckingham(int type1, int type2) const {
    auto it = m_typed_buckingham_params.find({type1, type2});
    if (it != m_typed_buckingham_params.end()) {
        return it->second;
    }
    return {0.0, 0.0, 0.0};
}

BuckinghamParams ForceFieldParams::get_buckingham_for_types(int type1, int type2) const {
    auto it = m_typed_buckingham_params.find({type1, type2});
    if (it != m_typed_buckingham_params.end()) {
        return it->second;
    }

    const auto key = canonical_pair(type1, type2);
    if (m_use_short_range_typing &&
        type1 > 0 && type2 > 0 &&
        m_missing_typed_buckingham_warned.insert(key).second) {
        occ::log::warn(
            "Missing typed Buckingham parameters for {}-{}; falling back to element pair",
            type_name(type1), type_name(type2));
    }

    const int z1 = short_range_type_atomic_number(type1);
    const int z2 = short_range_type_atomic_number(type2);
    if (z1 > 0 && z2 > 0) {
        return get_buckingham(z1, z2);
    }
    return {1000.0, 3.5, 10.0};
}

// ============================================================================
// Anisotropic repulsion
// ============================================================================

void ForceFieldParams::set_typed_aniso(
    const std::map<std::pair<int,int>, AnisotropicRepulsionParams>& params) {
    m_typed_aniso_params = params;
}

bool ForceFieldParams::has_aniso(int type1, int type2) const {
    return m_typed_aniso_params.count({type1, type2}) > 0;
}

AnisotropicRepulsionParams ForceFieldParams::get_aniso(int type1, int type2) const {
    auto it = m_typed_aniso_params.find({type1, type2});
    if (it != m_typed_aniso_params.end()) {
        return it->second;
    }
    return {};
}

// ============================================================================
// Type labels
// ============================================================================

void ForceFieldParams::set_type_labels(const std::map<int, std::string>& labels) {
    m_short_range_type_labels = labels;
    if (!m_short_range_type_labels.empty()) {
        m_use_short_range_typing = true;
    }
}

std::string ForceFieldParams::type_name(int type_code) const {
    auto it = m_short_range_type_labels.find(type_code);
    if (it != m_short_range_type_labels.end() && !it->second.empty()) {
        return it->second;
    }
    const char* label = short_range_type_label(type_code);
    if (label && std::string(label) != "UNKN") {
        return label;
    }
    return std::string("type") + std::to_string(type_code);
}

// ============================================================================
// Williams DE built-in tables (static)
// ============================================================================

std::map<std::pair<int,int>, BuckinghamParams> ForceFieldParams::williams_de_params() {
    std::map<std::pair<int,int>, BuckinghamParams> params;
    params[{1, 1}] = {2650.8, 3.74, 27.3};
    params[{6, 6}] = {369742.2, 3.60, 2439.8};
    params[{7, 7}] = {254501.2, 3.78, 1378.4};
    params[{8, 8}] = {230064.3, 3.96, 1123.6};
    params[{1, 6}] = {31368.8, 3.67, 258.0};
    params[{6, 1}] = params[{1, 6}];
    params[{1, 7}] = {25988.3, 3.76, 194.0};
    params[{7, 1}] = params[{1, 7}];
    params[{1, 8}] = {24716.7, 3.85, 175.2};
    params[{8, 1}] = params[{1, 8}];
    params[{6, 7}] = {306739.8, 3.69, 1834.1};
    params[{7, 6}] = params[{6, 7}];
    params[{6, 8}] = {291770.4, 3.78, 1655.4};
    params[{8, 6}] = params[{6, 8}];
    params[{7, 8}] = {242022.9, 3.87, 1244.5};
    params[{8, 7}] = params[{7, 8}];
    return params;
}

std::map<std::pair<int,int>, BuckinghamParams> ForceFieldParams::williams_typed_params() {
    std::map<std::pair<int, int>, BuckinghamParams> params;
    const auto self = williams_typed_self_params();
    const double eV_to_kJ = occ::units::EV_TO_KJ_PER_MOL;

    for (size_t i = 0; i < self.size(); ++i) {
        for (size_t j = i; j < self.size(); ++j) {
            const auto& a = self[i];
            const auto& b = self[j];
            if (a.code <= 0 || b.code <= 0) continue;
            if (a.rho <= 0.0 || b.rho <= 0.0) continue;

            BuckinghamParams p;
            p.A = std::sqrt(a.A * b.A) * eV_to_kJ;
            p.B = 0.5 * ((1.0 / a.rho) + (1.0 / b.rho));
            p.C = std::sqrt(std::max(0.0, a.C) * std::max(0.0, b.C)) * eV_to_kJ;
            params[{a.code, b.code}] = p;
            params[{b.code, a.code}] = p;
        }
    }
    return params;
}

const char* ForceFieldParams::short_range_type_label(int type_code) {
    switch (type_code) {
    case 501: return "H_W1";
    case 502: return "H_W2";
    case 503: return "H_W3";
    case 504: return "H_W4";
    case 505: return "H_Wa";
    case 511: return "C_W4";
    case 512: return "C_W3";
    case 513: return "C_W2";
    case 521: return "N_W1";
    case 522: return "N_W2";
    case 523: return "N_W3";
    case 524: return "N_W4";
    case 531: return "O_W1";
    case 532: return "O_W2";
    case 533: return "O_Wa";
    case 540: return "F_01";
    case 541: return "Cl01";
    case 542: return "S_01";
    case 543: return "K_01";
    case 544: return "Br01";
    case 545: return "I_01";
    default:  return "UNKN";
    }
}

int ForceFieldParams::short_range_type_atomic_number(int type_code) {
    switch (type_code) {
    case 501: case 502: case 503: case 504: case 505: return 1;
    case 511: case 512: case 513: return 6;
    case 521: case 522: case 523: case 524: return 7;
    case 531: case 532: case 533: return 8;
    case 540: return 9;
    case 541: return 17;
    case 542: return 16;
    case 543: return 19;
    case 544: return 35;
    case 545: return 53;
    default:
        if (type_code >= 10000) {
            const int z = (type_code - 10000) / 100;
            if (z > 0 && z <= 118) {
                return z;
            }
        }
        return 0;
    }
}

// ============================================================================
// Williams atom type classification
// ============================================================================

std::vector<std::vector<int>> ForceFieldParams::bonded_neighbors(
    const std::vector<int>& atomic_numbers,
    const std::vector<Vec3>& positions) {

    const int n = static_cast<int>(atomic_numbers.size());
    std::vector<std::vector<int>> neighbors(n);

    for (int i = 0; i < n; ++i) {
        const occ::core::Element ei(atomic_numbers[i]);
        const double ri = ei.covalent_radius();
        if (ri <= 0.0) continue;

        for (int j = i + 1; j < n; ++j) {
            const occ::core::Element ej(atomic_numbers[j]);
            const double rj = ej.covalent_radius();
            if (rj <= 0.0) continue;

            const double cutoff = ri + rj + kBondToleranceAngstrom;
            const double dist = (positions[j] - positions[i]).norm();
            if (dist >= 0.1 && dist <= cutoff) {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
            }
        }
    }
    return neighbors;
}

int ForceFieldParams::classify_williams_type(
    int idx,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<int>& atomic_numbers) {

    const int z = atomic_numbers[idx];
    const int nnb = static_cast<int>(neighbors[idx].size());

    // Hydrogen
    if (z == 1) {
        if (nnb != 1) return 0;
        const int n1 = neighbors[idx][0];
        const int z1 = atomic_numbers[n1];
        if (z1 == 6) return 501; // H_W1
        if (z1 == 7) return 504; // H_W4
        if (z1 == 8) {
            int code = 502; // H_W2 default for O-H
            const auto& o_neigh = neighbors[n1];
            if (static_cast<int>(o_neigh.size()) == 2) {
                bool all_h = true;
                for (int k : o_neigh) {
                    if (atomic_numbers[k] != 1) {
                        all_h = false;
                        break;
                    }
                }
                if (all_h) return 505; // H_Wa
            }

            for (int c : o_neigh) {
                if (c == idx || atomic_numbers[c] != 6) continue;
                for (int o2 : neighbors[c]) {
                    if (o2 == n1) continue;
                    if (atomic_numbers[o2] == 8 &&
                        static_cast<int>(neighbors[o2].size()) == 1) {
                        code = 503; // H_W3, carboxylic OH
                        break;
                    }
                }
                if (code == 503) break;
            }
            return code;
        }
        return 0;
    }

    // Carbon
    if (z == 6) {
        if (nnb == 4) return 511; // C_W4
        if (nnb == 3) return 512; // C_W3
        if (nnb == 2) return 513; // C_W2
        return 0;
    }

    // Nitrogen
    if (z == 7) {
        if (nnb == 1) return 521; // N_W1
        int h_count = 0;
        for (int n : neighbors[idx]) {
            if (atomic_numbers[n] == 1) ++h_count;
        }
        if (h_count == 0) return 522; // N_W2
        if (h_count == 1) return 523; // N_W3
        return 524;                   // N_W4
    }

    // Oxygen
    if (z == 8) {
        if (nnb == 1) return 531; // O_W1
        if (nnb == 2) {
            int h_count = 0;
            for (int n : neighbors[idx]) {
                if (atomic_numbers[n] == 1) ++h_count;
            }
            if (h_count == 2) return 533; // O_Wa
            return 532;                   // O_W2
        }
        return 0;
    }

    if (z == 9) return 540;   // F_01
    if (z == 17) return 541;  // Cl01
    if (z == 16) return 542;  // S_01
    if (z == 19) return 543;  // K_01
    if (z == 35) return 544;  // Br01
    if (z == 53) return 545;  // I_01

    return 0;
}

} // namespace occ::mults
