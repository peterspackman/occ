#include <occ/solvent/smd.h>

namespace occ::solvent::smd{

namespace detail{

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


constexpr int H = 1, C = 6, N = 7, O = 8,
    F = 9, Si = 14, P = 15, S = 16, Cl = 17, Br = 35, I = 53;

;

constexpr double sigma_aqueous(int z)
{
    switch(z)
    {
        case H: return 48.69;
        case C: return 129.74;
        case F: return 38.18;
        case S: return -9.10;
        case Cl: return 9.82;
        case Br: return -8.72;
        default:
            return 0.0;
    }
}

constexpr double sigma_aqueous(ElementPair p)
{
    switch(p)
    {
        case H_C: return -60.77;
        case C_C: return -72.95;
        case O_C: return 68.69;
        case N_C: return -48.22;
        case N_C3: return 84.10;
        case O_N: return 121.98;
        case O_P: return 68.85;
        default:
            return 0;
    }
}

constexpr double sigma_n(int z)
{
    switch(z)
    {
        case C: return 58.10;
        case N: return 32.62;
        case O: return -17.56;
        case Si: return -18.04;
        case Cl: return -24.31;
        case S: return -33.17;
        case Br: return -35.42;
        default: return 0;
    }
}

constexpr double sigma_n(ElementPair p)
{
    switch(p)
    {
        case H_C: return -36.37;
        case C_C: return -62.05;
        case H_O: return -19.39;
        case O_C: return -15.70;
        case C_N: return -99.76;
        default:
            return 0;
    }
}

constexpr double sigma_alpha(int z)
{
    switch(z)
    {
        case C: return 48.10;
        case O: return 193.06;
        default:
            return 0;
    }
}

constexpr double sigma_alpha(ElementPair p)
{
    switch(p)
    {
        case O_C: return 95.99;
        case C_N: return 152.20;
        case N_C: return -41.00;
        default:
            return 0.0;
    }
}

constexpr double sigma_beta(int z)
{
    switch(z)
    {
        case C: return 32.87;
        case O: return -43.79;
        default:
            return 0;
    }
}

constexpr double sigma_beta(ElementPair p)
{
    switch(p)
    {
        case O_O: return -128.16;
        case O_N: return 79.13;
        default:
            return 0;
    }
}

constexpr double rzz(ElementPair p)
{
    switch(p)
    {
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

constexpr double delta_rzz(ElementPair p)
{
    switch(p)
    {
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

constexpr double T_switching_function(ElementPair p, double r)
{
    double rz = rzz(p);
    double delta_rz = delta_rzz(p);
    if(r < (rz + delta_rz)) return std::exp(delta_rz / (r - delta_rz - rz));
    return 0.0;
}

double element_sigma(const SMDSolventParameters &params, int z)
{
    if(params.is_water)
    {
        return sigma_aqueous(z);
    }
    else
    {
        return sigma_n(z) * params.refractive_index_293K + sigma_alpha(z) * params.acidity + sigma_beta(z) * params.basicity;
    }
}

double element_pair_prefactor(const SMDSolventParameters &params, ElementPair p)
{
    if(params.is_water)
    {
        return sigma_aqueous(p);
    }
    else
    {
        return sigma_n(p) * params.refractive_index_293K +
            sigma_alpha(p) * params.acidity +
            sigma_beta(p) * params.basicity;
    }
}

int get_second_element(ElementPair p)
{
    switch(p)
    {
        case C_Br: return Br;
        case C_C: return C;
        case C_Cl: return Cl;
        case C_F: return F;
        case C_H: return H;
        case C_I: return I;
        case C_N: return N;
        case C_O: return O;
        case C_P: return P;
        case C_S: return S;
        case H_C: return C;
        case H_O: return O;
        case N_C: return C;
        case N_C3: return C;
        case O_C: return C;
        case O_N: return N;
        case O_O: return O;
        case O_P: return P;
    }
    return -1;
}

double element_pair_sum(const SMDSolventParameters &params, int index, ElementPair p, const IVec &nums, const Mat3N &positions, double power)
{
    double result{0.0};
    int z2 = get_second_element(p);
    double prefactor = element_pair_prefactor(params, p);
    if(prefactor == 0.0) return prefactor;
    for(int i = 0; i < nums.rows(); i++)
    {
        if(i == index) continue;
        if(nums(i) != z2) continue;
        double r = (positions.col(index) - positions.col(i)).norm();
        result += T_switching_function(p, r);
    }
    return std::pow(result, power) * prefactor;
}

ElementPair get_element_pair(int z1, int z2)
{
    if(z1 == H)
    {
        switch(z2)
        {
            case C: return H_C;
            case O: return H_O;
            default: return Other;
        }
    }
    else if(z1 == C)
    {
        switch(z2)
        {
            case C: return C_C;
            case N: return C_N;
            case O: return C_O;
            case F: return C_F;
            case P: return C_P;
            case S: return C_S;
            case Cl: return C_Cl;
            case Br: return C_Br;
            case I: return C_I;
            default: return Other;
        }
    }
    else if(z1 == N)
    {
        switch(z2)
        {
            case C: return N_C;
            default: return Other;
        }
    }
    else if (z1 == O)
    {
        switch(z2)
        {
            case C: return O_C;
            case N: return O_N;
            case O: return O_O;
            case P: return O_P;
            default: return Other;
        }

    }
    return Other;
}

double nitrogen_N_C_term(const SMDSolventParameters &params, int index, const IVec &nums, const Mat3N &positions)
{
    double result{0.0};
    ElementPair p = N_C;
    int z2 = get_second_element(p);
    double nc_prefactor = element_pair_prefactor(params, p);
    if(nc_prefactor == 0.0) return nc_prefactor;
    for(int i = 0; i < nums.rows(); i++)
    {
        if(i == index) continue;
        if(nums(i) != z2) continue;
        double r = (positions.col(index) - positions.col(i)).norm();
        double tmp = T_switching_function(p, r);
        double csum{0.0};
        for(int j = 0; j < nums.rows(); j++)
        {
            if(j == index || j == i) continue;
            double r = (positions.col(j) - positions.col(i)).norm();
            ElementPair p2 = get_element_pair(nums(i), nums(j));
            csum += T_switching_function(p2, r);
        }
        result += csum * csum * tmp;
    }
    return std::pow(result, 1.3) * nc_prefactor;
}



}


Vec atomic_surface_tension(const SMDSolventParameters &params, const IVec &nums, const Mat3N &positions)
{
    Vec result = Vec::Zero(nums.rows());
    for(int i = 0; i < nums.rows(); i++)
    {
        int n = nums(i);
        result(i) = detail::element_sigma(params, n);
        switch(n)
        {
            case 1:
            {
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::H_C, nums, positions, 1);
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::H_O, nums, positions, 1);
                break;
            }
            case 6:
            {
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::C_C, nums, positions, 1);
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::C_N, nums, positions, 2);
                break;
            }
            case 7:
            {
                result(i) += detail::nitrogen_N_C_term(params, i, nums, positions);
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::N_C3, nums, positions, 1);
                break;
            }
            case 8:
            {
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::O_C, nums, positions, 1);
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::O_N, nums, positions, 1);
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::O_O, nums, positions, 1);
                result(i) += detail::element_pair_sum(params, i, detail::ElementPair::O_P, nums, positions, 1);
            }
            default:
                continue;
        }
    }
    return result;
}

double molecular_surface_tension(const SMDSolventParameters &params)
{
    return detail::smd_sigma_gamma * params.gamma +
        detail::smd_sigma_phi2 * params.aromaticity * params.aromaticity +
        detail::smd_sigma_psi2 * params.electronegative_halogenicity * params.electronegative_halogenicity + 
        detail::smd_sigma_beta2 * params.basicity * params.basicity;
}

}
