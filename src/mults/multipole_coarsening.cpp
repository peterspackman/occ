#include <occ/mults/multipole_coarsening.h>
#include <occ/ints/rints.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::mults {

namespace {

/// Compile-time binomial coefficient table C(n, k) for n, k <= 8.
/// MaxL=4 means max index t+u+v=4 and the shift can produce up to t=4,
/// so we need C(n,k) for n up to 4.
constexpr int binom_table[9][9] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 1, 0, 0, 0, 0, 0, 0},
    {1, 3, 3, 1, 0, 0, 0, 0, 0},
    {1, 4, 6, 4, 1, 0, 0, 0, 0},
    {1, 5, 10, 10, 5, 1, 0, 0, 0},
    {1, 6, 15, 20, 15, 6, 1, 0, 0},
    {1, 7, 21, 35, 35, 21, 7, 1, 0},
    {1, 8, 28, 56, 70, 56, 28, 8, 1},
};

constexpr int binom(int n, int k) {
    return binom_table[n][k];
}

/// Integer power for small exponents (matches pattern in cartesian_rotation.h).
inline double ipow(double x, int n) {
    switch (n) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return x * x;
        case 3: return x * x * x;
        case 4: return x * x * x * x;
        default: return std::pow(x, n);
    }
}

} // anonymous namespace

void shift_multipole_to_origin(const CartesianMultipole<4> &input,
                               int input_rank,
                               const Vec3 &displacement,
                               CartesianMultipole<4> &output) {
    using occ::ints::hermite_index;

    const double dx = displacement[0];
    const double dy = displacement[1];
    const double dz = displacement[2];

    constexpr int MaxL = 4;

    // For each output index (t, u, v) with t+u+v <= MaxL,
    // accumulate contributions from source indices (tp, up, vp)
    // with tp <= t, up <= u, vp <= v, and tp+up+vp <= input_rank.
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u <= MaxL - t; ++u) {
            for (int v = 0; v <= MaxL - t - u; ++v) {
                double sum = 0.0;

                for (int tp = 0; tp <= t; ++tp) {
                    double cx = binom(t, tp) * ipow(dx, t - tp);
                    for (int up = 0; up <= u; ++up) {
                        double cy = binom(u, up) * ipow(dy, u - up);
                        for (int vp = 0; vp <= v; ++vp) {
                            if (tp + up + vp > input_rank) continue;
                            double cz = binom(v, vp) * ipow(dz, v - vp);
                            sum += cx * cy * cz *
                                   input.data[hermite_index(tp, up, vp)];
                        }
                    }
                }

                output.data[hermite_index(t, u, v)] += sum;
            }
        }
    }
}

CartesianMolecule merge_to_single_site(const CartesianMolecule &mol) {
    if (mol.sites.empty()) return mol;

    // Compute total charge and charge-weighted centroid.
    // For near-neutral molecules, charge-weighted centroid diverges,
    // so fall back to geometric centroid.
    double total_charge = 0.0;
    double sum_abs_charge = 0.0;
    Vec3 centroid = Vec3::Zero();

    for (const auto &site : mol.sites) {
        double q = site.cart(0, 0, 0); // charge = M_000
        total_charge += q;
        sum_abs_charge += std::abs(q);
        centroid += q * site.position;
    }

    // Use charge-weighted centroid only if net charge is a significant
    // fraction of the total absolute charge (i.e., clearly ionic)
    if (sum_abs_charge > 1e-15 &&
        std::abs(total_charge) > 0.1 * sum_abs_charge) {
        centroid /= total_charge;
    } else {
        centroid = Vec3::Zero();
        for (const auto &site : mol.sites) {
            centroid += site.position;
        }
        centroid /= static_cast<double>(mol.sites.size());
    }

    return merge_to_single_site(mol, centroid);
}

CartesianMolecule merge_to_single_site(const CartesianMolecule &mol,
                                       const Vec3 &origin) {
    CartesianMultipole<4> merged;

    // Positions are in Angstrom but multipole moments are in atomic units
    // (e * bohr^l), so displacement must be in Bohr for consistent shift.
    constexpr double ang_to_bohr = 1.0 / occ::units::BOHR_TO_ANGSTROM;

    for (const auto &site : mol.sites) {
        if (site.rank < 0) continue;
        Vec3 displacement = (site.position - origin) * ang_to_bohr;
        shift_multipole_to_origin(site.cart, site.rank, displacement, merged);
    }

    CartesianMolecule result;
    CartesianSite merged_site;
    merged_site.cart = merged;
    merged_site.position = origin;
    merged_site.rank = merged.effective_rank();
    result.sites.push_back(std::move(merged_site));
    return result;
}

} // namespace occ::mults
