#include <occ/mults/ewald_sum.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::mults {

std::vector<EwaldSite> gather_ewald_sites(
    const std::vector<CartesianMolecule>& cart_mols,
    bool include_dipole) {

    std::vector<EwaldSite> sites;
    for (int m = 0; m < static_cast<int>(cart_mols.size()); ++m) {
        for (const auto& site : cart_mols[m].sites) {
            if (site.rank < 0) continue;

            EwaldSite es;
            es.position = site.position;
            es.charge = site.cart.data[0];  // Q00
            if (include_dipole && site.rank >= 1) {
                es.dipole = Vec3(site.cart.data[1], site.cart.data[2], site.cart.data[3]);
            } else {
                es.dipole = Vec3::Zero();
            }
            es.mol_index = m;
            sites.push_back(es);
        }
    }
    return sites;
}

std::vector<std::vector<size_t>> build_mol_site_indices(
    const std::vector<CartesianMolecule>& cart_mols) {

    std::vector<std::vector<size_t>> indices(cart_mols.size());
    size_t idx = 0;
    for (int m = 0; m < static_cast<int>(cart_mols.size()); ++m) {
        for (const auto& site : cart_mols[m].sites) {
            if (site.rank < 0) continue; // skip without incrementing
            indices[m].push_back(idx);
            idx++;
        }
    }
    return indices;
}

EwaldResult compute_ewald_correction(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params) {

    const int N = static_cast<int>(mol_site_indices.size());
    EwaldResult result;
    result.site_forces.resize(sites.size(), Vec3::Zero());

    if (sites.empty()) return result;

    double alpha = params.alpha;
    int kmax = params.kmax;

    double alpha_bohr = alpha * occ::units::BOHR_TO_ANGSTROM;
    double two_alpha_over_sqrt_pi = 2.0 * alpha_bohr / std::sqrt(M_PI);

    occ::log::debug("Ewald correction: alpha = {:.4f} /Ang ({:.6f} /Bohr), kmax = {}",
                    alpha, alpha_bohr, kmax);

    // Convert site positions to Bohr for internal computation
    struct InternalSite {
        Vec3 pos_bohr;
        double charge;
        Vec3 dipole;
        int mol_index;
    };
    std::vector<InternalSite> isites(sites.size());
    for (size_t i = 0; i < sites.size(); ++i) {
        isites[i].pos_bohr = sites[i].position * occ::units::ANGSTROM_TO_BOHR;
        isites[i].charge = sites[i].charge;
        isites[i].dipole = sites[i].dipole;
        isites[i].mol_index = sites[i].mol_index;
    }

    // Per-site force accumulator (Hartree/Bohr)
    std::vector<Vec3> forces_ha_bohr(sites.size(), Vec3::Zero());
    double energy_ha = 0.0;

    // Helper: compute erf correction for a site pair (qq + qmu + mumu).
    // R_vec = r_b - r_a (Bohr). Returns energy correction in Hartree.
    auto erf_pair_correction = [&](const InternalSite& sa, const InternalSite& sb,
                                   const Vec3& R_vec, double w,
                                   Vec3& f_a, Vec3& f_b) {
        double r = R_vec.norm();
        if (r < 1e-10) return 0.0;

        double ar = alpha_bohr * r;
        double r2 = r * r;
        double r3 = r2 * r;
        double erf_val = std::erf(ar);
        double exp_val = std::exp(-ar * ar);
        double qa = sa.charge, qb = sb.charge;
        const Vec3& da = sa.dipole;
        const Vec3& db = sb.dipole;

        // Charge-charge erf correction: -q_a * q_b * erf(alpha*r) / r
        double e_qq = -w * qa * qb * erf_val / r;

        // Charge-charge force correction
        double f_over_r = qa * qb *
            (-erf_val / r2 + two_alpha_over_sqrt_pi * exp_val / r);
        Vec3 f_corr = w * f_over_r * (R_vec / r);
        f_a -= f_corr;
        f_b += f_corr;

        double e_qmu = 0.0, e_mumu = 0.0;

        if (params.include_dipole) {
            // Charge-dipole erf correction
            double f1 = two_alpha_over_sqrt_pi * exp_val / r2 - erf_val / r3;
            double muA_dot_R = da.dot(R_vec);
            double muB_dot_R = db.dot(R_vec);
            e_qmu = -w * (qa * muB_dot_R - qb * muA_dot_R) * f1;

            // Dipole-dipole erf correction
            double r4 = r2 * r2;
            double r5 = r4 * r;
            double Ap_over_r = -two_alpha_over_sqrt_pi * exp_val *
                (2.0 * alpha_bohr * alpha_bohr / r2 + 3.0 / r4) +
                3.0 * erf_val / r5;
            e_mumu = w * (da.dot(db) * f1 + muA_dot_R * muB_dot_R * Ap_over_r);
        }

        return e_qq + e_qmu + e_mumu;
    };

    // Real-space erf correction over neighbor pairs (inter-molecular)
    const bool use_erf_site_cutoff = (elec_site_cutoff > 0.0);
    const double erf_site_cutoff_bohr = elec_site_cutoff * occ::units::ANGSTROM_TO_BOHR;
    for (const auto& pair : neighbors) {
        if (use_com_gate && pair.com_distance > cutoff_radius)
            continue;

        int mi = pair.mol_i;
        int mj = pair.mol_j;

        Vec3 cell_trans_bohr = unit_cell.to_cartesian(
            pair.cell_shift.cast<double>()) * occ::units::ANGSTROM_TO_BOHR;

        for (size_t ai : mol_site_indices[mi]) {
            for (size_t bj : mol_site_indices[mj]) {
                Vec3 R_vec = isites[bj].pos_bohr + cell_trans_bohr
                           - isites[ai].pos_bohr;
                if (use_erf_site_cutoff && R_vec.norm() > erf_site_cutoff_bohr)
                    continue;
                energy_ha += erf_pair_correction(
                    isites[ai], isites[bj], R_vec, pair.weight,
                    forces_ha_bohr[ai], forces_ha_bohr[bj]);
            }
        }
    }

    // Intra-molecular erf correction
    for (int m = 0; m < N; ++m) {
        const auto& indices = mol_site_indices[m];
        for (size_t ii = 0; ii < indices.size(); ++ii) {
            size_t ai = indices[ii];
            for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
                size_t bi = indices[jj];
                Vec3 R_vec = isites[bi].pos_bohr - isites[ai].pos_bohr;
                energy_ha += erf_pair_correction(
                    isites[ai], isites[bi], R_vec, 1.0,
                    forces_ha_bohr[ai], forces_ha_bohr[bi]);
            }
        }
    }

    // Reciprocal-space sum
    Mat3 A_bohr = unit_cell.direct() * occ::units::ANGSTROM_TO_BOHR;
    double volume_bohr = unit_cell.volume() *
        std::pow(occ::units::ANGSTROM_TO_BOHR, 3);
    Mat3 B_bohr = 2.0 * M_PI * A_bohr.inverse().transpose();

    double four_pi_over_vol = 4.0 * M_PI / volume_bohr;
    double inv_4alpha2 = 1.0 / (4.0 * alpha_bohr * alpha_bohr);

    for (int hx = -kmax; hx <= kmax; ++hx) {
        for (int hy = -kmax; hy <= kmax; ++hy) {
            for (int hz = -kmax; hz <= kmax; ++hz) {
                if (hx == 0 && hy == 0 && hz == 0) continue;

                Vec3 G = B_bohr * Vec3(hx, hy, hz);
                double G2 = G.squaredNorm();
                double coeff = std::exp(-G2 * inv_4alpha2) / G2;

                // Structure factors
                double Sq_re = 0.0, Sq_im = 0.0;
                double Smu_re = 0.0, Smu_im = 0.0;
                for (const auto& s : isites) {
                    double phase = G.dot(s.pos_bohr);
                    double cos_p = std::cos(phase);
                    double sin_p = std::sin(phase);
                    Sq_re += s.charge * cos_p;
                    Sq_im += s.charge * sin_p;
                    if (params.include_dipole) {
                        double mu_dot_G = s.dipole.dot(G);
                        Smu_re += mu_dot_G * cos_p;
                        Smu_im += mu_dot_G * sin_p;
                    }
                }

                // Energy: (2pi/V) * coeff * |S_q + i*S_mu|^2
                double qq_recip = Sq_re * Sq_re + Sq_im * Sq_im;
                double qmu_cross = params.include_dipole ?
                    -2.0 * (Sq_re * Smu_im - Sq_im * Smu_re) : 0.0;
                double mumu_recip = params.include_dipole ?
                    Smu_re * Smu_re + Smu_im * Smu_im : 0.0;
                double prefactor = 0.5 * four_pi_over_vol * coeff;
                energy_ha += prefactor * (qq_recip + qmu_cross + mumu_recip);

                // Forces
                double P = Sq_re - Smu_im;
                double Q = Sq_im + Smu_re;
                for (size_t k = 0; k < isites.size(); ++k) {
                    double q_k = isites[k].charge;
                    double mk = params.include_dipole ?
                        isites[k].dipole.dot(G) : 0.0;

                    double phase_k = G.dot(isites[k].pos_bohr);
                    double sin_k = std::sin(phase_k);
                    double cos_k = std::cos(phase_k);

                    double force_factor = q_k * (P * sin_k - Q * cos_k)
                                        + mk * (P * cos_k + Q * sin_k);

                    forces_ha_bohr[k] += four_pi_over_vol * coeff * force_factor * G;
                }
            }
        }
    }

    // Self correction
    double alpha3 = alpha_bohr * alpha_bohr * alpha_bohr;
    for (const auto& s : isites) {
        // Charge-charge self: -(alpha/sqrt(pi)) * q^2
        energy_ha -= (alpha_bohr / std::sqrt(M_PI)) * s.charge * s.charge;
        // Dipole-dipole self: -(2*alpha^3/(3*sqrt(pi))) * |mu|^2
        if (params.include_dipole) {
            energy_ha -= (2.0 * alpha3 / (3.0 * std::sqrt(M_PI))) *
                s.dipole.squaredNorm();
        }
    }

    // Convert to kJ/mol and Angstrom
    result.energy = energy_ha * occ::units::AU_TO_KJ_PER_MOL;

    double force_conv = occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;
    for (size_t k = 0; k < sites.size(); ++k) {
        result.site_forces[k] = forces_ha_bohr[k] * force_conv;
    }

    occ::log::debug("Ewald correction: {:.4f} kJ/mol ({} sites, {} neighbor pairs)",
                    result.energy, sites.size(), neighbors.size());

    return result;
}

} // namespace occ::mults
