#include <occ/core/element.h>
#include <occ/core/units.h>
#include <occ/solvent/smd.h>

namespace occ::solvent::smd {

namespace detail {

inline constexpr double smd_sigma_gamma{0.35};
inline constexpr double smd_sigma_phi2{-4.19};
inline constexpr double smd_sigma_psi2{-6.68};
inline constexpr double smd_sigma_beta2{0.0};

enum ElementPair {
    H_C,
    C_C,
    H_O,
    O_C,
    O_O,
    C_N,
    N_C,
    N_C3,
    O_N,
    O_P,
    C_F,
    C_P,
    C_S,
    C_Cl,
    C_Br,
    C_I,
    C_H,
    C_O,
    Other
};

constexpr int H = 1, C = 6, N = 7, O = 8, F = 9, Si = 14, P = 15, S = 16,
              Cl = 17, Br = 35, I = 53;

;

constexpr double sigma_aqueous(int z) {
    switch (z) {
    case H:
        return 48.69;
    case C:
        return 129.74;
    case F:
        return 38.18;
    case S:
        return -9.10;
    case Cl:
        return 9.82;
    case Br:
        return -8.72;
    default:
        return 0.0;
    }
}

constexpr double sigma_aqueous(ElementPair p) {
    switch (p) {
    case H_C:
        return -60.77;
    case C_C:
        return -72.95;
    case O_C:
        return 68.69;
    case N_C:
        return -48.22;
    case N_C3:
        return 84.10;
    case O_N:
        return 121.98;
    case O_P:
        return 68.85;
    default:
        return 0;
    }
}

constexpr double sigma_n(int z) {
    switch (z) {
    case C:
        return 58.10;
    case N:
        return 32.62;
    case O:
        return -17.56;
    case Si:
        return -18.04;
    case Cl:
        return -24.31;
    case S:
        return -33.17;
    case Br:
        return -35.42;
    default:
        return 0;
    }
}

constexpr double sigma_n(ElementPair p) {
    switch (p) {
    case H_C:
        return -36.37;
    case C_C:
        return -62.05;
    case H_O:
        return -19.39;
    case O_C:
        return -15.70;
    case C_N:
        return -99.76;
    default:
        return 0;
    }
}

constexpr double sigma_alpha(int z) {
    switch (z) {
    case C:
        return 48.10;
    case O:
        return 193.06;
    default:
        return 0;
    }
}

constexpr double sigma_alpha(ElementPair p) {
    switch (p) {
    case O_C:
        return 95.99;
    case C_N:
        return 152.20;
    case N_C:
        return -41.00;
    default:
        return 0.0;
    }
}

constexpr double sigma_beta(int z) {
    switch (z) {
    case C:
        return 32.87;
    case O:
        return -43.79;
    default:
        return 0;
    }
}

constexpr double sigma_beta(ElementPair p) {
    switch (p) {
    case O_O:
        return -128.16;
    case O_N:
        return 79.13;
    default:
        return 0;
    }
}

constexpr double rzz(ElementPair p) {
    switch (p) {
    case H_C:
    case H_O:
    case C_H:
        return 1.55;
    case C_C:
    case C_N:
    case C_O:
    case C_F:
    case N_C:
        return 1.84;
    case C_P:
    case C_S:
        return 2.2;
    case C_Cl:
    case O_P:
        return 2.1;
    case C_Br:
        return 2.3;
    case C_I:
        return 2.6;
    case N_C3:
        return 1.225;
    case O_C:
        return 1.3;
    case O_N:
        return 1.5;
    case O_O:
        return 1.8;
    default:
        return 0;
    }
}

constexpr double delta_rzz(ElementPair p) {
    switch (p) {
    case H_C:
    case H_O:
    case C_H:
    case C_C:
    case C_N:
    case C_O:
    case C_F:
    case C_P:
    case C_S:
    case C_Cl:
    case C_Br:
    case C_I:
    case N_C:
    case O_N:
    case O_O:
    case O_P:
        return 0.3;
    case N_C3:
        return 0.065;
    case O_C:
        return 0.1;
    default:
        return 0;
    }
}

constexpr const char *pair_string(ElementPair p) {
    switch (p) {
    case H_C:
        return "H,C";
    case H_O:
        return "H,O";
    case O_C:
        return "O,C";
    case O_O:
        return "O,O";
    case O_N:
        return "O,N";
    case O_P:
        return "O,P";
    case C_C:
        return "C,C";
    case C_N:
        return "C,N";
    case N_C3:
        return "N,C(3)";
    case N_C:
        return "N,C";
    default:
        return "X,X";
    }
}

double T_switching_function(ElementPair p, double r) {
    const double rz = rzz(p);
    const double delta_rz = delta_rzz(p);
    if (r < (rz + delta_rz))
        return std::exp(delta_rz / (r - rz - delta_rz));
    else
        return 0.0;
}

double element_sigma(const SMDSolventParameters &params, int z) {
    if (params.is_water) {
        return sigma_aqueous(z);
    } else {
        return sigma_n(z) * params.refractive_index_293K +
               sigma_alpha(z) * params.acidity +
               sigma_beta(z) * params.basicity;
    }
}

double element_pair_prefactor(const SMDSolventParameters &params,
                              ElementPair p) {
    if (params.is_water) {
        return sigma_aqueous(p);
    } else {
        return sigma_n(p) * params.refractive_index_293K +
               sigma_alpha(p) * params.acidity +
               sigma_beta(p) * params.basicity;
    }
}

int get_second_element(ElementPair p) {
    switch (p) {
    case C_Br:
        return Br;
    case C_C:
        return C;
    case C_Cl:
        return Cl;
    case C_F:
        return F;
    case C_H:
        return H;
    case C_I:
        return I;
    case C_N:
        return N;
    case C_O:
        return O;
    case C_P:
        return P;
    case C_S:
        return S;
    case H_C:
        return C;
    case H_O:
        return O;
    case N_C:
        return C;
    case N_C3:
        return C;
    case O_C:
        return C;
    case O_N:
        return N;
    case O_O:
        return O;
    case O_P:
        return P;
    default:
        return -1;
    }
}

ElementPair get_element_pair(int z1, int z2) {
    if (z1 == H) {
        switch (z2) {
        case C:
            return H_C;
        case O:
            return H_O;
        default:
            return Other;
        }
    } else if (z1 == C) {
        switch (z2) {
        case H:
            return C_H;
        case C:
            return C_C;
        case N:
            return C_N;
        case O:
            return C_O;
        case F:
            return C_F;
        case P:
            return C_P;
        case S:
            return C_S;
        case Cl:
            return C_Cl;
        case Br:
            return C_Br;
        case I:
            return C_I;
        default:
            return Other;
        }
    } else if (z1 == N) {
        switch (z2) {
        case C:
            return N_C;
        default:
            return Other;
        }
    } else if (z1 == O) {
        switch (z2) {
        case C:
            return O_C;
        case N:
            return O_N;
        case O:
            return O_O;
        case P:
            return O_P;
        default:
            return Other;
        }
    }
    return Other;
}

Mat cot_matrix(const SMDSolventParameters &params, const IVec &nums,
               const Mat3N &positions) {
    Mat result = Mat::Zero(nums.rows(), nums.rows());
    size_t N = nums.rows();
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double r = (positions.col(i) - positions.col(j)).norm();
            ElementPair p1 = get_element_pair(nums(i), nums(j));
            ElementPair p2 = get_element_pair(nums(j), nums(i));
            result(i, j) = T_switching_function(p1, r);
            result(j, i) = T_switching_function(p2, r);
        }
    }
    return result;
}

Mat sigma_matrix(const SMDSolventParameters &params, const IVec &nums) {
    int max_el = nums.maxCoeff() + 1;
    Mat result = Mat::Zero(max_el, max_el);
    for (int i = 1; i < max_el; i++) {
        for (int j = 1; j < max_el; j++) {
            ElementPair p = get_element_pair(i, j);
            result(i, j) = element_pair_prefactor(params, p);
        }
    }
    return result;
}

Vec sigma_vector(const SMDSolventParameters &params, const IVec &nums) {
    Vec result = Vec::Zero(nums.rows());
    for (int i = 0; i < nums.rows(); i++) {
        result(i) = element_sigma(params, nums(i));
    }
    return result;
}

double pair_term(int index, int n, const IVec &nums, const Mat &cot,
                 int power = 1) {
    double result = 0.0;
    for (int j = 0; j < nums.rows(); j++) {
        if (j == index)
            continue;
        if (nums(j) != n)
            continue;
        result += cot(index, j);
    }
    return std::pow(result, power);
}

double nc_term(int k, const IVec &nums, const Mat &cot) {
    double result{0.0};
    size_t N = nums.rows();
    for (int kp = 0; kp < N; kp++) {
        if (nums(kp) != 6)
            continue;

        double csum{0.0};
        for (int kpp = 0; kpp < N; kpp++) {
            if ((kpp == k) || (kpp == kp))
                continue;
            csum += cot(kp, kpp);
        }
        result += csum * csum * cot(k, kp);
    }
    return std::pow(result, 1.3);
}

double nc3_term(const SMDSolventParameters &params, int index, const IVec &nums,
                const Mat3N &positions) {
    size_t N = nums.rows();
    double result = 0.0;
    const ElementPair p = ElementPair::N_C3;
    for (int j = 0; j < N; j++) {
        if (j == index)
            continue;
        if (nums(j) != 6)
            continue;
        double r = (positions.col(j) - positions.col(index)).norm();
        result += T_switching_function(p, r);
    }
    return result;
}

} // namespace detail

Vec atomic_surface_tension(const SMDSolventParameters &params, const IVec &nums,
                           const Mat3N &positions, const Vec &areas) {
    int N = nums.rows();
    Vec result = Vec::Zero(nums.rows());
    Vec per_element_sigma =
        Vec::Zero(static_cast<size_t>(detail::ElementPair::Other));
    Vec per_element_pair = Vec::Zero(nums.maxCoeff() + 1);
    Vec per_element_pair2 = Vec::Zero(nums.maxCoeff() + 1);
    Mat cot = detail::cot_matrix(params, nums, positions);
    Mat sigma_mat = detail::sigma_matrix(params, nums);
    Vec sigma_vec = detail::sigma_vector(params, nums);
    int max_number = nums.maxCoeff();

    for (int i = 0; i < N; i++) {
        int ni = nums(i);
        result(i) += sigma_vec(i);
        switch (ni) {
        case 1: {
            if (6 > max_number)
                continue;
            double hc_sigma = sigma_mat(1, 6);
            if (hc_sigma != 0.0) {
                double hc = detail::pair_term(i, 6, nums, cot, 1);
                result(i) += hc * hc_sigma;
            }
            if (8 > max_number)
                continue;
            double ho_sigma = sigma_mat(1, 8);
            if (ho_sigma != 0.0) {
                double ho = detail::pair_term(i, 8, nums, cot, 1);
                result(i) += ho * ho_sigma;
            }
            break;
        }
        case 6: {
            double cc_sigma = sigma_mat(6, 6);
            if (cc_sigma != 0.0) {
                double cc = detail::pair_term(i, 6, nums, cot, 1);
                result(i) += cc * cc_sigma;
            }
            if (7 > max_number)
                continue;
            double cn_sigma = sigma_mat(6, 7);
            if (cn_sigma != 0.0) {
                double cn = detail::pair_term(i, 7, nums, cot, 2);
                result(i) += cn * cn_sigma;
            }
            break;
        }
        case 7: {
            // definitely the problem
            double nc_sigma = sigma_mat(7, 6);
            if (nc_sigma != 0.0) {
                double nc = detail::nc_term(i, nums, cot);
                result(i) += nc * nc_sigma;
            }
            double nc3_sigma = detail::element_pair_prefactor(
                params, detail::ElementPair::N_C3);
            if (nc3_sigma != 0.0) {
                double nc3 = detail::nc3_term(params, i, nums, positions);
                result(i) += nc3 * nc3_sigma;
            }
            break;
        }
        case 8: {
            // O,C; O,N; O,O
            for (int j = 6; j < 9; j++) {
                if (j > max_number)
                    continue;
                double ox_sigma = sigma_mat(8, j);
                if (ox_sigma != 0.0) {
                    double ox = detail::pair_term(i, j, nums, cot, 1);
                    result(i) += ox * ox_sigma;
                }
            }

            if (15 > max_number)
                continue;
            double op_sigma = sigma_mat(8, 15);
            if (op_sigma != 0.0) {
                double op = detail::pair_term(i, 15, nums, cot, 1);
                result(i) += op * op_sigma;
            }
            break;
        }
        default:
            break;
        }
    }

    return result;
}

Vec atomic_surface_tension(const SMDSolventParameters &params, const IVec &nums,
                           const Mat3N &positions) {
    int N = nums.rows();
    Vec result = Vec::Zero(nums.rows());
    Mat cot = detail::cot_matrix(params, nums, positions);
    Mat sigma_mat = detail::sigma_matrix(params, nums);
    Vec sigma_vec = detail::sigma_vector(params, nums);
    int max_number = nums.maxCoeff();

    for (int i = 0; i < N; i++) {
        int ni = nums(i);
        result(i) += sigma_vec(i);
        switch (ni) {
        case 1: {
            if (6 > max_number)
                continue;
            double hc_sigma = sigma_mat(1, 6);
            if (hc_sigma != 0.0) {
                double hc = detail::pair_term(i, 6, nums, cot, 1);
                result(i) += hc * hc_sigma;
            }
            if (8 > max_number)
                continue;
            double ho_sigma = sigma_mat(1, 8);
            if (ho_sigma != 0.0) {
                double ho = detail::pair_term(i, 8, nums, cot, 1);
                result(i) += ho * ho_sigma;
            }
            break;
        }
        case 6: {
            double cc_sigma = sigma_mat(6, 6);
            if (cc_sigma != 0.0) {
                double cc = detail::pair_term(i, 6, nums, cot, 1);
                result(i) += cc * cc_sigma;
            }
            if (7 > max_number)
                continue;
            double cn_sigma = sigma_mat(6, 7);
            if (cn_sigma != 0.0) {
                double cn = detail::pair_term(i, 7, nums, cot, 2);
                result(i) += cn * cn_sigma;
            }
            break;
        }
        case 7: {
            double nc_sigma = sigma_mat(7, 6);
            if (nc_sigma != 0.0) {
                double nc = detail::nc_term(i, nums, cot);
                result(i) += nc * nc_sigma;
            }
            double nc3_sigma = detail::element_pair_prefactor(
                params, detail::ElementPair::N_C3);
            if (nc3_sigma != 0.0) {
                double nc3 = detail::nc3_term(params, i, nums, positions);
                result(i) += nc3 * nc3_sigma;
            }
            break;
        }
        case 8: {
            // O,C; O,N; O,O
            for (int j = 6; j < 9; j++) {
                if (j > max_number)
                    continue;
                double ox_sigma = sigma_mat(8, j);
                if (ox_sigma != 0.0) {
                    double ox = detail::pair_term(i, j, nums, cot, 1);
                    result(i) += ox * ox_sigma;
                }
            }

            if (15 > max_number)
                continue;
            double op_sigma = sigma_mat(8, 15);
            if (op_sigma != 0.0) {
                double op = detail::pair_term(i, 15, nums, cot, 1);
                result(i) += op * op_sigma;
            }
            break;
        }
        default:
            break;
        }
    }

    return result;
}

double molecular_surface_tension(const SMDSolventParameters &params) {
    if (params.is_water)
        return 0.0;
    return detail::smd_sigma_gamma * params.gamma +
           detail::smd_sigma_phi2 * params.aromaticity * params.aromaticity +
           detail::smd_sigma_psi2 * params.electronegative_halogenicity *
               params.electronegative_halogenicity +
           detail::smd_sigma_beta2 * params.basicity * params.basicity;
}

Vec intrinsic_coulomb_radii(const IVec &nums,
                            const SMDSolventParameters &params) {

    // see Table 3 of https://pubs.acs.org/doi/10.1021/jp810292n
    occ::Vec result(nums.rows());
    for (int i = 0; i < nums.rows(); i++) {
        int n = nums(i);
        double r = 1.2;
        switch (n) {
        case 1:
            r = 1.2;
            break;
        case 6:
            r = 1.85;
            break;
        case 7:
            r = 1.89;
            break;
        case 8:
            r = (params.acidity >= 0.43 || params.is_water)
                    ? 1.52
                    : 1.52 + 1.8 * (0.43 - params.acidity);
            break;
        case 9:
            r = 1.73;
            break;
        case 14:
            r = 2.47;
            break;
        case 15:
            r = 2.12;
            break;
        case 16:
            r = 2.49;
            break;
        case 17:
            r = 2.38;
            break;
        case 35:
            r = 3.06;
            break;
        default:
            r = occ::core::Element(n).van_der_waals_radius();
            break;
        }
        result(i) = r;
    }
    return result * occ::units::ANGSTROM_TO_BOHR;
}

Vec cds_radii(const IVec &nums, const SMDSolventParameters &params) {
    occ::Vec result(nums.rows());
    for (int i = 0; i < nums.rows(); i++) {
        int n = nums(i);
        result(i) = occ::core::Element(n).van_der_waals_radius() + 0.4;
    }
    return result * occ::units::ANGSTROM_TO_BOHR;
}


Vec intrinsic_coulomb_radii(const std::vector<core::Atom> &atoms, const SMDSolventParameters &params) {
    IVec nums(atoms.size());
    for(int i = 0; i < atoms.size(); i++) nums(i) = atoms[i].atomic_number;
    return intrinsic_coulomb_radii(nums, params);
}

Vec cds_radii(const std::vector<core::Atom> &atoms, const SMDSolventParameters &params) {
    IVec nums(atoms.size());
    for(int i = 0; i < atoms.size(); i++) nums(i) = atoms[i].atomic_number;
    return cds_radii(nums, params);
}

} // namespace occ::solvent::smd
