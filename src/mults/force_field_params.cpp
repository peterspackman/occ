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

// FIT atom types, in the order the pair table below indexes them. FIT is
// coarser than NEIGHCRYS: every polar hydrogen (502-505) shares H_F2.
struct FitType {
    const char *label;
    std::vector<int> codes;
};

const std::vector<FitType>& fit_types() {
    static const std::vector<FitType> types{
        {"H_F1", {501}},                 // H on carbon
        {"H_F2", {502, 503, 504, 505}},  // polar H (on N/O)
        {"C_F1", {511, 512, 513}},
        {"N_F1", {521, 522, 523, 524}},
        {"O_F1", {531, 532, 533}},
        {"F_F1", {540}},
        {"ClF1", {541}},
        {"S_F1", {542}},
    };
    return types;
}

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

// FIT (Williams/Cox) potential: explicit pairs (no combining rule) over eight
// atom types, H split into H_F1 (on C) / H_F2 (polar, on N/O). A in eV,
// rho = 1/B in Angstrom, C in eV*Angstrom^6. Source: mol-cspy fit.pots.
std::map<std::pair<int,int>, BuckinghamParams> ForceFieldParams::fit_typed_params() {
    // label indices: 0 H_F1, 1 H_F2, 2 C_F1, 3 N_F1, 4 O_F1, 5 F_F1, 6 ClF1, 7 S_F1
    struct Pair { int i, j; double A_eV, rho, C_eV; };
    static const Pair pairs[] = {
        {2, 2, 3832.147000, 0.277778, 25.286950}, // C_F1 C_F1
        {2, 3, 3179.514586, 0.271003, 19.006711}, // C_F1 N_F1
        {2, 4, 3022.850285, 0.264550, 17.160239}, // C_F1 O_F1
        {2, 0,  689.536726, 0.272480,  5.978972}, // C_F1 H_F1
        {2, 1,  446.949968, 0.242131,  2.373772}, // C_F1 H_F2
        {2, 5, 3800.832789, 0.257732, 14.872725}, // C_F1 F_F1
        {2, 6, 6060.195609, 0.281294, 45.040509}, // C_F1 ClF1
        {2, 7, 3991.008388, 0.289824, 38.956795}, // C_F1 S_F1
        {3, 3, 2638.028500, 0.264550, 14.286225}, // N_F1 N_F1
        {3, 4, 2508.044856, 0.258398, 12.898341}, // N_F1 O_F1
        {3, 0,  572.105423, 0.265957,  4.494041}, // N_F1 H_F1
        {3, 1,  370.832315, 0.236967,  1.784224}, // N_F1 H_F2
        {3, 5, 3153.533331, 0.251889, 11.178951}, // N_F1 F_F1
        {3, 6, 5028.116179, 0.274348, 33.854298}, // N_F1 ClF1
        {3, 7, 3311.321142, 0.282456, 29.281528}, // N_F1 S_F1
        {4, 4, 2384.465900, 0.252525, 11.645288}, // O_F1 O_F1
        {4, 0,  543.916058, 0.259740,  4.057452}, // O_F1 H_F1
        {4, 1,  352.560285, 0.232018,  1.610890}, // O_F1 H_F2
        {4, 5, 2998.149205, 0.246306, 10.092934}, // O_F1 F_F1
        {4, 6, 4780.365686, 0.267737, 30.565407}, // O_F1 ClF1
        {4, 7, 3148.162333, 0.275454, 26.436874}, // O_F1 S_F1
        {0, 0,  124.071675, 0.267380,  1.413698}, // H_F1 H_F1
        {0, 1,   80.421867, 0.238095,  0.561266}, // H_F1 H_F2
        {0, 5,  683.902209, 0.253165,  3.516581}, // H_F1 F_F1
        {0, 6, 1090.440278, 0.275862, 10.649602}, // H_F1 ClF1
        {0, 7,  718.121424, 0.284062,  9.211139}, // H_F1 S_F1
        {1, 1,   52.128552, 0.214592,  0.222834}, // H_F2 H_F2
        {1, 5,  443.297737, 0.226757,  1.396153}, // H_F2 F_F1
        {1, 6,  706.811152, 0.244798,  4.228105}, // H_F2 ClF1
        {1, 7,  465.478248, 0.251233,  3.657006}, // H_F2 S_F1
        {5, 5, 3769.774461, 0.240385,  8.747514}, // F_F1 F_F1
        {5, 6, 6010.675003, 0.260756, 26.490941}, // F_F1 ClF1
        {5, 7, 3958.396048, 0.268070, 22.912755}, // F_F1 S_F1
        {6, 6, 9583.653972, 0.284900, 80.225075}, // ClF1 ClF1
        {6, 7, 6311.420597, 0.293654, 69.388909}, // ClF1 S_F1
        {7, 7, 4156.455363, 0.302963, 60.016406}, // S_F1 S_F1
    };
    const double eV = occ::units::EV_TO_KJ_PER_MOL;
    BuckinghamParams by_label[8][8];
    for (const auto& p : pairs) {
        BuckinghamParams bp{p.A_eV * eV, 1.0 / p.rho, p.C_eV * eV};
        by_label[p.i][p.j] = bp;
        by_label[p.j][p.i] = bp;
    }

    // Expand label pairs over the NEIGHCRYS codes each label covers, so the
    // table can be looked up by the codes classify_williams_type() produces.
    const auto& types = fit_types();
    std::map<std::pair<int, int>, BuckinghamParams> params;
    for (size_t li = 0; li < types.size(); ++li)
        for (size_t lj = 0; lj < types.size(); ++lj)
            for (int ci : types[li].codes)
                for (int cj : types[lj].codes)
                    params[{ci, cj}] = by_label[li][lj];
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

const char* ForceFieldParams::fit_type_label(int type_code) {
    for (const auto& type : fit_types()) {
        if (std::find(type.codes.begin(), type.codes.end(), type_code) !=
            type.codes.end())
            return type.label;
    }
    return nullptr;
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

std::vector<int> ForceFieldParams::classify_atom_types(
    const std::vector<int>& atomic_numbers,
    const std::vector<Vec3>& positions) {

    const auto neighbors = bonded_neighbors(atomic_numbers, positions);
    std::vector<int> codes(atomic_numbers.size(), 0);
    for (size_t i = 0; i < atomic_numbers.size(); ++i)
        codes[i] = classify_williams_type(static_cast<int>(i), neighbors,
                                          atomic_numbers);
    return codes;
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
