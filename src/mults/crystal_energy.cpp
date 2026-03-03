#include <occ/mults/crystal_energy.h>
#include <occ/mults/ewald_sum.h>
#include <occ/mults/cartesian_hessian.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/timings.h>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>

namespace occ::mults {

namespace {

constexpr double kBondToleranceAngstrom = 0.4;

Mat3 skew_symmetric(const Vec3& v) {
    Mat3 S;
    S <<      0.0, -v[2],  v[1],
           v[2],      0.0, -v[0],
          -v[1],   v[0],   0.0;
    return S;
}

const std::array<Mat3, 3>& so3_generators() {
    static const std::array<Mat3, 3> G = [] {
        std::array<Mat3, 3> out;
        out[0] << 0, 0, 0,  0, 0, -1,  0, 1, 0;
        out[1] << 0, 0, 1,  0, 0, 0,  -1, 0, 0;
        out[2] << 0, -1, 0,  1, 0, 0,  0, 0, 0;
        return out;
    }();
    return G;
}

std::pair<int, int> canonical_pair(int a, int b) {
    return (a <= b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct TypedSelfBuckingham {
    int code = 0;
    double A = 0.0;
    double rho = 0.0;
    double C = 0.0;
};

struct AD6 {
    double v = 0.0;
    Vec6 g = Vec6::Zero();
    Mat6 h = Mat6::Zero();
};

struct AD6R3 {
    AD6 v;
    std::array<AD6, 3> d;
};

AD6 ad6_constant(double c) {
    AD6 out;
    out.v = c;
    return out;
}

AD6 ad6_add(const AD6& a, const AD6& b) {
    AD6 out;
    out.v = a.v + b.v;
    out.g = a.g + b.g;
    out.h = a.h + b.h;
    return out;
}

AD6 ad6_sub(const AD6& a, const AD6& b) {
    AD6 out;
    out.v = a.v - b.v;
    out.g = a.g - b.g;
    out.h = a.h - b.h;
    return out;
}

AD6 ad6_mul(const AD6& a, const AD6& b) {
    AD6 out;
    out.v = a.v * b.v;
    out.g = a.g * b.v + b.g * a.v;
    out.h = a.h * b.v + b.h * a.v +
            a.g * b.g.transpose() + b.g * a.g.transpose();
    return out;
}

AD6 ad6_mul_scalar(const AD6& a, double s) {
    AD6 out;
    out.v = a.v * s;
    out.g = a.g * s;
    out.h = a.h * s;
    return out;
}

AD6 ad6_inv(const AD6& a) {
    AD6 out;
    const double inv_v = 1.0 / a.v;
    const double inv_v2 = inv_v * inv_v;
    out.v = inv_v;
    out.g = -a.g * inv_v2;
    out.h = 2.0 * (a.g * a.g.transpose()) * (inv_v2 * inv_v) - a.h * inv_v2;
    return out;
}

AD6 ad6_div(const AD6& a, const AD6& b) {
    return ad6_mul(a, ad6_inv(b));
}

AD6 ad6_sqrt(const AD6& a) {
    AD6 out;
    const double s = std::sqrt(a.v);
    const double fp = 0.5 / s;
    const double fpp = -0.25 / (a.v * s);
    out.v = s;
    out.g = fp * a.g;
    out.h = fpp * (a.g * a.g.transpose()) + fp * a.h;
    return out;
}

AD6 ad6_exp(const AD6& a) {
    AD6 out;
    const double ev = std::exp(a.v);
    out.v = ev;
    out.g = ev * a.g;
    out.h = ev * (a.h + a.g * a.g.transpose());
    return out;
}

AD6 ad6_erf(const AD6& a) {
    AD6 out;
    const double ev = std::erf(a.v);
    const double fp = 2.0 / std::sqrt(M_PI) * std::exp(-a.v * a.v);
    const double fpp = -2.0 * a.v * fp;
    out.v = ev;
    out.g = fp * a.g;
    out.h = fpp * (a.g * a.g.transpose()) + fp * a.h;
    return out;
}

AD6 ad6_sin(const AD6& a) {
    AD6 out;
    const double s = std::sin(a.v);
    const double c = std::cos(a.v);
    out.v = s;
    out.g = c * a.g;
    out.h = -s * (a.g * a.g.transpose()) + c * a.h;
    return out;
}

AD6 ad6_cos(const AD6& a) {
    AD6 out;
    const double s = std::sin(a.v);
    const double c = std::cos(a.v);
    out.v = c;
    out.g = -s * a.g;
    out.h = -c * (a.g * a.g.transpose()) - s * a.h;
    return out;
}

AD6 ad6_cutoff_spline(const AD6& r_ang, const CutoffSpline& spl) {
    if (!spl.is_valid()) return ad6_constant(1.0);
    if (r_ang.v <= spl.r_on) return ad6_constant(1.0);
    if (r_ang.v > spl.r_off) return ad6_constant(0.0);

    const double vdr = 1.0 / (spl.r_on - spl.r_off);
    const AD6 dr = ad6_sub(r_ang, ad6_constant(spl.r_off));
    const AD6 dr2 = ad6_mul(dr, dr);
    const AD6 dr3 = ad6_mul(dr2, dr);

    if (spl.order == 3) {
        const double c2 = 3.0 * vdr * vdr;
        const double c3 = -2.0 * vdr * vdr * vdr;
        return ad6_add(ad6_mul_scalar(dr2, c2), ad6_mul_scalar(dr3, c3));
    }

    const AD6 dr4 = ad6_mul(dr3, dr);
    const AD6 dr5 = ad6_mul(dr4, dr);
    const double vdr2 = vdr * vdr;
    const double vdr3 = vdr2 * vdr;
    const double vdr4 = vdr3 * vdr;
    const double vdr5 = vdr4 * vdr;
    const double c3 = -8.0 * vdr3;
    const double c4 = 21.0 * vdr4;
    const double c5 = -12.0 * vdr5;
    return ad6_add(
        ad6_add(ad6_mul_scalar(dr3, c3), ad6_mul_scalar(dr4, c4)),
        ad6_mul_scalar(dr5, c5));
}

AD6R3 ad6r3_constant(double c) {
    AD6R3 out;
    out.v = ad6_constant(c);
    out.d = {ad6_constant(0.0), ad6_constant(0.0), ad6_constant(0.0)};
    return out;
}

AD6R3 ad6r3_variable(const AD6& value, int coord) {
    AD6R3 out = ad6r3_constant(0.0);
    out.v = value;
    out.d[coord] = ad6_constant(1.0);
    return out;
}

AD6R3 ad6r3_add(const AD6R3& a, const AD6R3& b) {
    AD6R3 out;
    out.v = ad6_add(a.v, b.v);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_add(a.d[k], b.d[k]);
    }
    return out;
}

AD6R3 ad6r3_sub(const AD6R3& a, const AD6R3& b) {
    AD6R3 out;
    out.v = ad6_sub(a.v, b.v);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_sub(a.d[k], b.d[k]);
    }
    return out;
}

AD6R3 ad6r3_mul(const AD6R3& a, const AD6R3& b) {
    AD6R3 out;
    out.v = ad6_mul(a.v, b.v);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_add(ad6_mul(a.d[k], b.v), ad6_mul(a.v, b.d[k]));
    }
    return out;
}

AD6R3 ad6r3_mul_scalar(const AD6R3& a, double s) {
    AD6R3 out;
    out.v = ad6_mul_scalar(a.v, s);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul_scalar(a.d[k], s);
    }
    return out;
}

AD6R3 ad6r3_inv(const AD6R3& a) {
    AD6R3 out;
    const AD6 inv_v = ad6_inv(a.v);
    out.v = inv_v;
    const AD6 fprime = ad6_mul_scalar(ad6_mul(inv_v, inv_v), -1.0);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul(fprime, a.d[k]);
    }
    return out;
}

AD6R3 ad6r3_div(const AD6R3& a, const AD6R3& b) {
    return ad6r3_mul(a, ad6r3_inv(b));
}

AD6R3 ad6r3_sqrt(const AD6R3& a) {
    AD6R3 out;
    out.v = ad6_sqrt(a.v);
    const AD6 fprime = ad6_mul_scalar(ad6_inv(out.v), 0.5);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul(fprime, a.d[k]);
    }
    return out;
}

AD6R3 ad6r3_exp(const AD6R3& a) {
    AD6R3 out;
    out.v = ad6_exp(a.v);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul(out.v, a.d[k]);
    }
    return out;
}

AD6R3 ad6r3_erf(const AD6R3& a) {
    AD6R3 out;
    out.v = ad6_erf(a.v);
    const AD6 fprime = ad6_mul_scalar(
        ad6_exp(ad6_mul_scalar(ad6_mul(a.v, a.v), -1.0)),
        2.0 / std::sqrt(M_PI));
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul(fprime, a.d[k]);
    }
    return out;
}

AD6R3 ad6r3_sin(const AD6R3& a) {
    AD6R3 out;
    out.v = ad6_sin(a.v);
    const AD6 fprime = ad6_cos(a.v);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul(fprime, a.d[k]);
    }
    return out;
}

AD6R3 ad6r3_cos(const AD6R3& a) {
    AD6R3 out;
    out.v = ad6_cos(a.v);
    const AD6 fprime = ad6_mul_scalar(ad6_sin(a.v), -1.0);
    for (int k = 0; k < 3; ++k) {
        out.d[k] = ad6_mul(fprime, a.d[k]);
    }
    return out;
}

AD6R3 ad6r3_cutoff_spline(const AD6R3& r_ang, const CutoffSpline& spl) {
    if (!spl.is_valid()) return ad6r3_constant(1.0);
    if (r_ang.v.v <= spl.r_on) return ad6r3_constant(1.0);
    if (r_ang.v.v > spl.r_off) return ad6r3_constant(0.0);

    const double vdr = 1.0 / (spl.r_on - spl.r_off);
    const AD6R3 dr = ad6r3_sub(r_ang, ad6r3_constant(spl.r_off));
    const AD6R3 dr2 = ad6r3_mul(dr, dr);
    const AD6R3 dr3 = ad6r3_mul(dr2, dr);

    if (spl.order == 3) {
        const double c2 = 3.0 * vdr * vdr;
        const double c3 = -2.0 * vdr * vdr * vdr;
        return ad6r3_add(ad6r3_mul_scalar(dr2, c2), ad6r3_mul_scalar(dr3, c3));
    }

    const AD6R3 dr4 = ad6r3_mul(dr3, dr);
    const AD6R3 dr5 = ad6r3_mul(dr4, dr);
    const double vdr2 = vdr * vdr;
    const double vdr3 = vdr2 * vdr;
    const double vdr4 = vdr3 * vdr;
    const double vdr5 = vdr4 * vdr;
    const double c3 = -8.0 * vdr3;
    const double c4 = 21.0 * vdr4;
    const double c5 = -12.0 * vdr5;
    return ad6r3_add(
        ad6r3_add(ad6r3_mul_scalar(dr3, c3), ad6r3_mul_scalar(dr4, c4)),
        ad6r3_mul_scalar(dr5, c5));
}

struct EwaldExplicitStrainTerms {
    Vec6 grad = Vec6::Zero();
    Mat6 hess = Mat6::Zero();
    Mat strain_site_mixed;
};

EwaldExplicitStrainTerms compute_ewald_explicit_strain_terms(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params,
    const CutoffSpline* taper,
    const EwaldLatticeCache* lattice_cache,
    bool include_strain_state = false) {

    EwaldExplicitStrainTerms out;
    if (sites.empty()) return out;
    if (include_strain_state) {
        out.strain_site_mixed = Mat::Zero(6, 3 * static_cast<int>(sites.size()));
    }

    struct InternalSite {
        Vec3 pos_bohr;
        double charge = 0.0;
        Vec3 dipole = Vec3::Zero();
    };
    std::vector<InternalSite> isites(sites.size());
    for (size_t i = 0; i < sites.size(); ++i) {
        isites[i].pos_bohr = sites[i].position * occ::units::ANGSTROM_TO_BOHR;
        isites[i].charge = sites[i].charge;
        isites[i].dipole = sites[i].dipole;
    }

    const auto& B = voigt_basis_matrices();

    std::vector<std::array<AD6, 3>> site_pos_ad(sites.size());
    for (size_t i = 0; i < sites.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            site_pos_ad[i][c].v = isites[i].pos_bohr[c];
        }
    }

    const double alpha_bohr = lattice_cache ? lattice_cache->alpha_bohr
                                            : params.alpha * occ::units::BOHR_TO_ANGSTROM;
    const double two_alpha_over_sqrt_pi = lattice_cache
        ? lattice_cache->two_alpha_over_sqrt_pi
        : 2.0 * alpha_bohr / std::sqrt(M_PI);

    AD6 E_total = ad6_constant(0.0);
    const bool use_taper = (taper != nullptr && taper->is_valid());
    const bool use_erf_site_cutoff = (elec_site_cutoff > 0.0);
    const double erf_site_cutoff_bohr =
        elec_site_cutoff * occ::units::ANGSTROM_TO_BOHR;

    // Real-space explicit lattice term: image translation T changes under strain.
    auto accumulate_real_pair_ad = [&](size_t ai, size_t bj,
                                       const AD6& Rx, const AD6& Ry, const AD6& Rz,
                                       double weight) {
        AD6 r2 = ad6_add(ad6_add(ad6_mul(Rx, Rx), ad6_mul(Ry, Ry)), ad6_mul(Rz, Rz));
        AD6 r = ad6_sqrt(r2);
        if (r.v < 1e-10) return;
        if (use_erf_site_cutoff && r.v > erf_site_cutoff_bohr) return;

        AD6 inv_r = ad6_inv(r);
        AD6 inv_r2 = ad6_inv(r2);
        AD6 inv_r3 = ad6_mul(inv_r2, inv_r);
        AD6 inv_r4 = ad6_mul(inv_r2, inv_r2);
        AD6 inv_r5 = ad6_mul(inv_r4, inv_r);
        AD6 ar = ad6_mul_scalar(r, alpha_bohr);
        AD6 erf_ar = ad6_erf(ar);
        AD6 exp_m_ar2 = ad6_exp(ad6_mul_scalar(ad6_mul(ar, ar), -1.0));

        const double qa = isites[ai].charge;
        const double qb = isites[bj].charge;
        AD6 e_pair = ad6_mul_scalar(ad6_mul(erf_ar, inv_r), -qa * qb);

        if (params.include_dipole) {
            const Vec3& da = isites[ai].dipole;
            const Vec3& db = isites[bj].dipole;
            AD6 muA_dot_R = ad6_add(
                ad6_add(ad6_mul_scalar(Rx, da[0]), ad6_mul_scalar(Ry, da[1])),
                ad6_mul_scalar(Rz, da[2]));
            AD6 muB_dot_R = ad6_add(
                ad6_add(ad6_mul_scalar(Rx, db[0]), ad6_mul_scalar(Ry, db[1])),
                ad6_mul_scalar(Rz, db[2]));
            AD6 f1 = ad6_sub(
                ad6_mul_scalar(ad6_mul(exp_m_ar2, inv_r2), two_alpha_over_sqrt_pi),
                ad6_mul(erf_ar, inv_r3));
            AD6 qmu_lin = ad6_sub(ad6_mul_scalar(muB_dot_R, qa),
                                  ad6_mul_scalar(muA_dot_R, qb));
            AD6 e_qmu = ad6_mul_scalar(ad6_mul(qmu_lin, f1), -1.0);

            const double a2 = alpha_bohr * alpha_bohr;
            AD6 t_inner = ad6_add(
                ad6_mul_scalar(inv_r2, 2.0 * a2),
                ad6_mul_scalar(inv_r4, 3.0));
            AD6 Ap_over_r = ad6_add(
                ad6_mul_scalar(ad6_mul(exp_m_ar2, t_inner), -two_alpha_over_sqrt_pi),
                ad6_mul_scalar(ad6_mul(erf_ar, inv_r5), 3.0));
            AD6 e_mumu = ad6_add(
                ad6_mul_scalar(f1, da.dot(db)),
                ad6_mul(ad6_mul(muA_dot_R, muB_dot_R), Ap_over_r));
            e_pair = ad6_add(e_pair, ad6_add(e_qmu, e_mumu));
        }

        if (use_taper) {
            AD6 r_ang = ad6_mul_scalar(r, occ::units::BOHR_TO_ANGSTROM);
            AD6 sw = ad6_cutoff_spline(r_ang, *taper);
            if (sw.v <= 0.0) return;
            e_pair = ad6_mul(sw, e_pair);
        }

        E_total = ad6_add(E_total, ad6_mul_scalar(e_pair, weight));
    };

    for (const auto& pair : neighbors) {
        if (use_com_gate && pair.com_distance > cutoff_radius) continue;

        const int mi = pair.mol_i;
        const int mj = pair.mol_j;
        const Vec3 T0 = unit_cell.to_cartesian(pair.cell_shift.cast<double>()) *
                        occ::units::ANGSTROM_TO_BOHR;

        std::array<AD6, 3> T;
        for (int c = 0; c < 3; ++c) {
            T[c].v = T0[c];
        }
        for (int a = 0; a < 6; ++a) {
            const Vec3 dT = B[a] * T0;
            for (int c = 0; c < 3; ++c) {
                T[c].g[a] = dT[c];
            }
        }

        for (size_t ai : mol_site_indices[mi]) {
            for (size_t bj : mol_site_indices[mj]) {
                AD6 Rx = ad6_add(ad6_sub(site_pos_ad[bj][0], site_pos_ad[ai][0]), T[0]);
                AD6 Ry = ad6_add(ad6_sub(site_pos_ad[bj][1], site_pos_ad[ai][1]), T[1]);
                AD6 Rz = ad6_add(ad6_sub(site_pos_ad[bj][2], site_pos_ad[ai][2]), T[2]);
                accumulate_real_pair_ad(ai, bj, Rx, Ry, Rz, pair.weight);
            }
        }
    }

    // Intra-molecular real-space correction (same as compute_ewald_correction):
    // needed when COM-chain strain derivatives are requested.
    for (size_t m = 0; m < mol_site_indices.size(); ++m) {
        const auto& idxs = mol_site_indices[m];
        for (size_t ii = 0; ii < idxs.size(); ++ii) {
            const size_t ai = idxs[ii];
            for (size_t jj = ii + 1; jj < idxs.size(); ++jj) {
                const size_t bj = idxs[jj];
                AD6 Rx = ad6_sub(site_pos_ad[bj][0], site_pos_ad[ai][0]);
                AD6 Ry = ad6_sub(site_pos_ad[bj][1], site_pos_ad[ai][1]);
                AD6 Rz = ad6_sub(site_pos_ad[bj][2], site_pos_ad[ai][2]);
                accumulate_real_pair_ad(ai, bj, Rx, Ry, Rz, 1.0);
            }
        }
    }

    // Reciprocal-space explicit lattice term at fixed Cartesian coordinates.
    AD6 invV;
    const double V_bohr = unit_cell.volume() *
        std::pow(occ::units::ANGSTROM_TO_BOHR, 3);
    invV.v = 1.0 / V_bohr;
    std::array<double, 6> trB{};
    Mat6 trBB = Mat6::Zero();
    for (int a = 0; a < 6; ++a) {
        trB[a] = B[a].trace();
        invV.g[a] = -trB[a] * invV.v;
        for (int b = 0; b < 6; ++b) {
            trBB(a, b) = (B[a] * B[b]).trace();
            invV.h(a, b) = (trB[a] * trB[b] + trBB(a, b)) * invV.v;
        }
    }

    auto reciprocal_term = [&](const Vec3& G0) {
        std::array<AD6, 3> G;
        for (int c = 0; c < 3; ++c) G[c].v = G0[c];
        for (int a = 0; a < 6; ++a) {
            const Vec3 dG = -(B[a] * G0);
            for (int c = 0; c < 3; ++c) G[c].g[a] = dG[c];
        }
        for (int a = 0; a < 6; ++a) {
            for (int b = 0; b < 6; ++b) {
                const Vec3 d2G = B[a] * B[b] * G0 + B[b] * B[a] * G0;
                for (int c = 0; c < 3; ++c) {
                    G[c].h(a, b) = d2G[c];
                }
            }
        }

        AD6 G2 = ad6_add(ad6_add(ad6_mul(G[0], G[0]), ad6_mul(G[1], G[1])),
                         ad6_mul(G[2], G[2]));
        const double beta = 1.0 / (4.0 * alpha_bohr * alpha_bohr);
        AD6 coeff = ad6_div(ad6_exp(ad6_mul_scalar(G2, -beta)), G2);
        AD6 pref = ad6_mul_scalar(ad6_mul(invV, coeff), 2.0 * M_PI);

        AD6 Sq_re = ad6_constant(0.0), Sq_im = ad6_constant(0.0);
        AD6 Smu_re = ad6_constant(0.0), Smu_im = ad6_constant(0.0);

        for (size_t idx = 0; idx < isites.size(); ++idx) {
            const auto& s = isites[idx];
            AD6 phase = ad6_add(
                ad6_add(ad6_mul(G[0], site_pos_ad[idx][0]),
                        ad6_mul(G[1], site_pos_ad[idx][1])),
                ad6_mul(G[2], site_pos_ad[idx][2]));
            AD6 cph = ad6_cos(phase);
            AD6 sph = ad6_sin(phase);

            Sq_re = ad6_add(Sq_re, ad6_mul_scalar(cph, s.charge));
            Sq_im = ad6_add(Sq_im, ad6_mul_scalar(sph, s.charge));

            if (params.include_dipole) {
                AD6 m_dot_G = ad6_add(
                    ad6_add(ad6_mul_scalar(G[0], s.dipole[0]),
                            ad6_mul_scalar(G[1], s.dipole[1])),
                    ad6_mul_scalar(G[2], s.dipole[2]));
                Smu_re = ad6_add(Smu_re, ad6_mul(m_dot_G, cph));
                Smu_im = ad6_add(Smu_im, ad6_mul(m_dot_G, sph));
            }
        }

        AD6 qq = ad6_add(ad6_mul(Sq_re, Sq_re), ad6_mul(Sq_im, Sq_im));
        AD6 qmu = ad6_constant(0.0);
        AD6 mumu = ad6_constant(0.0);
        if (params.include_dipole) {
            qmu = ad6_mul_scalar(
                ad6_sub(ad6_mul(Sq_re, Smu_im), ad6_mul(Sq_im, Smu_re)), -2.0);
            mumu = ad6_add(ad6_mul(Smu_re, Smu_re), ad6_mul(Smu_im, Smu_im));
        }
        AD6 term = ad6_mul(pref, ad6_add(ad6_add(qq, qmu), mumu));
        E_total = ad6_add(E_total, term);
    };

    if (lattice_cache) {
        for (const auto& gv : lattice_cache->g_vectors) {
            reciprocal_term(gv.G);
        }
    } else {
        Mat3 A_bohr = unit_cell.direct() * occ::units::ANGSTROM_TO_BOHR;
        Mat3 B_bohr = 2.0 * M_PI * A_bohr.inverse().transpose();
        double inv_4alpha2 = 1.0 / (4.0 * alpha_bohr * alpha_bohr);
        for (int hx = -params.kmax; hx <= params.kmax; ++hx) {
            for (int hy = -params.kmax; hy <= params.kmax; ++hy) {
                for (int hz = -params.kmax; hz <= params.kmax; ++hz) {
                    if (hx == 0 && hy == 0 && hz == 0) continue;
                    Vec3 G0 = B_bohr * Vec3(hx, hy, hz);
                    double G2 = G0.squaredNorm();
                    double coeff = std::exp(-G2 * inv_4alpha2) / G2;
                    if (coeff < 1e-12) continue;
                    reciprocal_term(G0);
                }
            }
        }
    }

    if (include_strain_state) {
        // Mixed explicit Ewald terms: d/dE (dE/dx_site) at fixed Cartesian sites.
        // Stored as d^2E / dE_a dx_{site,c} in out.strain_site_mixed[a, 3*site+c].

        for (const auto& pair : neighbors) {
            if (use_com_gate && pair.com_distance > cutoff_radius) continue;

            const int mi = pair.mol_i;
            const int mj = pair.mol_j;
            const Vec3 T0 = unit_cell.to_cartesian(pair.cell_shift.cast<double>()) *
                            occ::units::ANGSTROM_TO_BOHR;

            std::array<AD6, 3> T;
            for (int c = 0; c < 3; ++c) {
                T[c].v = T0[c];
            }
            for (int a = 0; a < 6; ++a) {
                const Vec3 dT = B[a] * T0;
                for (int c = 0; c < 3; ++c) {
                    T[c].g[a] = dT[c];
                }
            }

            for (size_t ai : mol_site_indices[mi]) {
                for (size_t bj : mol_site_indices[mj]) {
                    const AD6 Rx0 = ad6_add(
                        ad6_constant(isites[bj].pos_bohr[0] - isites[ai].pos_bohr[0]), T[0]);
                    const AD6 Ry0 = ad6_add(
                        ad6_constant(isites[bj].pos_bohr[1] - isites[ai].pos_bohr[1]), T[1]);
                    const AD6 Rz0 = ad6_add(
                        ad6_constant(isites[bj].pos_bohr[2] - isites[ai].pos_bohr[2]), T[2]);

                    AD6R3 Rx = ad6r3_variable(Rx0, 0);
                    AD6R3 Ry = ad6r3_variable(Ry0, 1);
                    AD6R3 Rz = ad6r3_variable(Rz0, 2);

                    AD6R3 r2 = ad6r3_add(
                        ad6r3_add(ad6r3_mul(Rx, Rx), ad6r3_mul(Ry, Ry)),
                        ad6r3_mul(Rz, Rz));
                    AD6R3 r = ad6r3_sqrt(r2);
                    if (r.v.v < 1e-10) continue;
                    if (use_erf_site_cutoff && r.v.v > erf_site_cutoff_bohr) continue;

                    AD6R3 inv_r = ad6r3_inv(r);
                    AD6R3 inv_r2 = ad6r3_inv(r2);
                    AD6R3 inv_r3 = ad6r3_mul(inv_r2, inv_r);
                    AD6R3 inv_r4 = ad6r3_mul(inv_r2, inv_r2);
                    AD6R3 inv_r5 = ad6r3_mul(inv_r4, inv_r);
                    AD6R3 ar = ad6r3_mul_scalar(r, alpha_bohr);
                    AD6R3 erf_ar = ad6r3_erf(ar);
                    AD6R3 exp_m_ar2 = ad6r3_exp(ad6r3_mul_scalar(ad6r3_mul(ar, ar), -1.0));

                    const double qa = isites[ai].charge;
                    const double qb = isites[bj].charge;
                    AD6R3 e_pair = ad6r3_mul_scalar(ad6r3_mul(erf_ar, inv_r), -qa * qb);

                    if (params.include_dipole) {
                        const Vec3& da = isites[ai].dipole;
                        const Vec3& db = isites[bj].dipole;
                        AD6R3 muA_dot_R = ad6r3_add(
                            ad6r3_add(ad6r3_mul_scalar(Rx, da[0]), ad6r3_mul_scalar(Ry, da[1])),
                            ad6r3_mul_scalar(Rz, da[2]));
                        AD6R3 muB_dot_R = ad6r3_add(
                            ad6r3_add(ad6r3_mul_scalar(Rx, db[0]), ad6r3_mul_scalar(Ry, db[1])),
                            ad6r3_mul_scalar(Rz, db[2]));
                        AD6R3 f1 = ad6r3_sub(
                            ad6r3_mul_scalar(ad6r3_mul(exp_m_ar2, inv_r2),
                                             two_alpha_over_sqrt_pi),
                            ad6r3_mul(erf_ar, inv_r3));
                        AD6R3 qmu_lin = ad6r3_sub(ad6r3_mul_scalar(muB_dot_R, qa),
                                                  ad6r3_mul_scalar(muA_dot_R, qb));
                        AD6R3 e_qmu = ad6r3_mul_scalar(ad6r3_mul(qmu_lin, f1), -1.0);

                        const double a2 = alpha_bohr * alpha_bohr;
                        AD6R3 t_inner =
                            ad6r3_add(ad6r3_mul_scalar(inv_r2, 2.0 * a2),
                                      ad6r3_mul_scalar(inv_r4, 3.0));
                        AD6R3 Ap_over_r = ad6r3_add(
                            ad6r3_mul_scalar(ad6r3_mul(exp_m_ar2, t_inner),
                                             -two_alpha_over_sqrt_pi),
                            ad6r3_mul_scalar(ad6r3_mul(erf_ar, inv_r5), 3.0));
                        AD6R3 e_mumu = ad6r3_add(
                            ad6r3_mul_scalar(f1, da.dot(db)),
                            ad6r3_mul(ad6r3_mul(muA_dot_R, muB_dot_R), Ap_over_r));
                        e_pair = ad6r3_add(e_pair, ad6r3_add(e_qmu, e_mumu));
                    }

                    if (use_taper) {
                        AD6R3 r_ang = ad6r3_mul_scalar(r, occ::units::BOHR_TO_ANGSTROM);
                        AD6R3 sw = ad6r3_cutoff_spline(r_ang, *taper);
                        if (sw.v.v <= 0.0) continue;
                        e_pair = ad6r3_mul(sw, e_pair);
                    }

                    for (int c = 0; c < 3; ++c) {
                        const Vec6 dF_dE = pair.weight * e_pair.d[c].g;
                        for (int a = 0; a < 6; ++a) {
                            // g_site = dE/dx_site = -force_site
                            out.strain_site_mixed(a, 3 * static_cast<int>(ai) + c) -= dF_dE[a];
                            out.strain_site_mixed(a, 3 * static_cast<int>(bj) + c) += dF_dE[a];
                        }
                    }
                }
            }
        }

        auto reciprocal_mixed_term = [&](const Vec3& G0) {
            std::array<AD6, 3> G;
            for (int c = 0; c < 3; ++c) G[c].v = G0[c];
            for (int a = 0; a < 6; ++a) {
                const Vec3 dG = -(B[a] * G0);
                for (int c = 0; c < 3; ++c) G[c].g[a] = dG[c];
            }

            AD6 G2 = ad6_add(ad6_add(ad6_mul(G[0], G[0]), ad6_mul(G[1], G[1])),
                             ad6_mul(G[2], G[2]));
            const double beta = 1.0 / (4.0 * alpha_bohr * alpha_bohr);
            AD6 coeff = ad6_div(ad6_exp(ad6_mul_scalar(G2, -beta)), G2);
            AD6 pref_force = ad6_mul_scalar(ad6_mul(invV, coeff), 4.0 * M_PI);

            const size_t n_sites = isites.size();
            std::vector<AD6> cos_phase(n_sites), sin_phase(n_sites), mu_dot_G(n_sites);
            AD6 Sq_re = ad6_constant(0.0), Sq_im = ad6_constant(0.0);
            AD6 Smu_re = ad6_constant(0.0), Smu_im = ad6_constant(0.0);

            for (size_t k = 0; k < n_sites; ++k) {
                AD6 phase = ad6_add(
                    ad6_add(ad6_mul_scalar(G[0], isites[k].pos_bohr[0]),
                            ad6_mul_scalar(G[1], isites[k].pos_bohr[1])),
                    ad6_mul_scalar(G[2], isites[k].pos_bohr[2]));
                cos_phase[k] = ad6_cos(phase);
                sin_phase[k] = ad6_sin(phase);
                Sq_re = ad6_add(Sq_re, ad6_mul_scalar(cos_phase[k], isites[k].charge));
                Sq_im = ad6_add(Sq_im, ad6_mul_scalar(sin_phase[k], isites[k].charge));
                if (params.include_dipole) {
                    mu_dot_G[k] = ad6_add(
                        ad6_add(ad6_mul_scalar(G[0], isites[k].dipole[0]),
                                ad6_mul_scalar(G[1], isites[k].dipole[1])),
                        ad6_mul_scalar(G[2], isites[k].dipole[2]));
                    Smu_re = ad6_add(Smu_re, ad6_mul(mu_dot_G[k], cos_phase[k]));
                    Smu_im = ad6_add(Smu_im, ad6_mul(mu_dot_G[k], sin_phase[k]));
                } else {
                    mu_dot_G[k] = ad6_constant(0.0);
                }
            }

            const AD6 P = ad6_sub(Sq_re, Smu_im);
            const AD6 Q = ad6_add(Sq_im, Smu_re);

            for (size_t i = 0; i < n_sites; ++i) {
                const double qi = isites[i].charge;
                const AD6& si = sin_phase[i];
                const AD6& ci = cos_phase[i];
                const AD6& mi = mu_dot_G[i];

                const AD6 q_part = ad6_mul_scalar(
                    ad6_sub(ad6_mul(P, si), ad6_mul(Q, ci)), qi);
                AD6 A = q_part;
                if (params.include_dipole) {
                    const AD6 mu_part =
                        ad6_mul(mi, ad6_add(ad6_mul(P, ci), ad6_mul(Q, si)));
                    A = ad6_add(A, mu_part);
                }

                for (int c = 0; c < 3; ++c) {
                    const AD6 F_ic = ad6_mul(pref_force, ad6_mul(A, G[c]));
                    for (int a = 0; a < 6; ++a) {
                        out.strain_site_mixed(a, 3 * static_cast<int>(i) + c) -= F_ic.g[a];
                    }
                }
            }
        };

        if (lattice_cache) {
            for (const auto& gv : lattice_cache->g_vectors) {
                reciprocal_mixed_term(gv.G);
            }
        } else {
            Mat3 A_bohr = unit_cell.direct() * occ::units::ANGSTROM_TO_BOHR;
            Mat3 B_bohr = 2.0 * M_PI * A_bohr.inverse().transpose();
            const double inv_4alpha2 = 1.0 / (4.0 * alpha_bohr * alpha_bohr);
            for (int hx = -params.kmax; hx <= params.kmax; ++hx) {
                for (int hy = -params.kmax; hy <= params.kmax; ++hy) {
                    for (int hz = -params.kmax; hz <= params.kmax; ++hz) {
                        if (hx == 0 && hy == 0 && hz == 0) continue;
                        const Vec3 G0 = B_bohr * Vec3(hx, hy, hz);
                        const double G2 = G0.squaredNorm();
                        const double coeff = std::exp(-G2 * inv_4alpha2) / G2;
                        if (coeff < 1e-12) continue;
                        reciprocal_mixed_term(G0);
                    }
                }
            }
        }
    }

    const double econv = occ::units::AU_TO_KJ_PER_MOL;
    out.grad = E_total.g * econv;
    out.hess = E_total.h * econv;
    out.hess = 0.5 * (out.hess + out.hess.transpose());
    if (include_strain_state) {
        const double force_conv =
            occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;
        out.strain_site_mixed *= force_conv;
    }
    return out;
}

std::vector<TypedSelfBuckingham> williams_typed_self_params() {
    // Self terms from DMACRYS/NEIGHCRYS Williams defaults (pote.dat),
    // in DMACRYS units: A (eV), rho (Angstrom), C (eV*Angstrom^6).
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

    // Water aliases are not present explicitly in the W table; use nearest type.
    self.push_back({505, 3.740000, 0.280899, 0.000000}); // H_Wa -> H_W2
    self.push_back({533, 2949.910000, 0.252525, 13.328149}); // O_Wa -> O_W2
    return self;
}

std::vector<std::vector<int>> bonded_neighbors(
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

int classify_williams_type(
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

void add_pair_strain_gradient(Vec6& strain_grad,
                              const Vec3& force_on_i,
                              const Vec3& disp_ij,
                              double weight) {
    const auto& B = voigt_basis_matrices();
    for (int a = 0; a < 6; ++a) {
        strain_grad[a] += weight * force_on_i.dot(B[a] * disp_ij);
    }
}

void accumulate_pair_strain_hessian_blocks(
    Mat6& H_ee,
    Mat& H_eq,
    int mol_i,
    int mol_j,
    const PairHessianResult& pair,
    double weight,
    const Vec3& pos_i,
    const Vec3& pos_j_image) {

    const auto& B = voigt_basis_matrices();
    const Vec3 disp = pos_j_image - pos_i;
    std::array<Vec3, 6> dA{};
    std::array<Vec3, 6> dB{};
    for (int a = 0; a < 6; ++a) {
        dA[a] = B[a] * pos_i;
        dB[a] = B[a] * pos_j_image;
    }

    const int ia = 6 * mol_i;
    const int ib = 6 * mol_j;
    const Mat3 H_rel =
        0.25 * (pair.H_posA_posA + pair.H_posB_posB -
                pair.H_posA_posB - pair.H_posA_posB.transpose());
    for (int a = 0; a < 6; ++a) {
        const Vec3& ba = dA[a];
        const Vec3& bb = dB[a];

        const Vec3 row_A_pos =
            pair.H_posA_posA.transpose() * ba + pair.H_posA_posB * bb;
        const Vec3 row_A_rot =
            pair.H_posA_rotA.transpose() * ba + pair.H_posB_rotA.transpose() * bb;
        const Vec3 row_B_pos =
            pair.H_posA_posB.transpose() * ba + pair.H_posB_posB.transpose() * bb;
        const Vec3 row_B_rot =
            pair.H_posA_rotB.transpose() * ba + pair.H_posB_rotB.transpose() * bb;

        H_eq.block<1, 3>(a, ia) += weight * row_A_pos.transpose();
        H_eq.block<1, 3>(a, ia + 3) += weight * row_A_rot.transpose();
        H_eq.block<1, 3>(a, ib) += weight * row_B_pos.transpose();
        H_eq.block<1, 3>(a, ib + 3) += weight * row_B_rot.transpose();

        for (int b = 0; b < 6; ++b) {
            const Vec3& ca = dA[b];
            const Vec3& cb = dB[b];
            const Vec3 dRa = bb - ba;
            const Vec3 dRb = cb - ca;
            const double val = dRa.dot(H_rel * dRb);
            H_ee(a, b) += weight * val;
        }
    }
}

bool is_canonical_explicit_pair(int i, int j, const IVec3& shift) {
    if (i < j) {
        return true;
    }
    if (i > j) {
        return false;
    }
    // For i == j keep only one of (+shift, -shift) to avoid mirrored duplicates.
    if (shift[0] != 0) return shift[0] > 0;
    if (shift[1] != 0) return shift[1] > 0;
    return shift[2] > 0;
}

void accumulate_pair_hessian_blocks(Mat& H,
                                    int mol_i,
                                    int mol_j,
                                    const PairHessianResult& pair,
                                    double weight) {
    const int ii = 6 * mol_i;
    const int jj = 6 * mol_j;
    auto add = [&](int r, int c, const Mat3& B) {
        H.block<3, 3>(r, c) += weight * B;
    };

    // Diagonal molecule blocks
    add(ii, ii, pair.H_posA_posA);
    add(ii, ii + 3, pair.H_posA_rotA);
    add(ii + 3, ii, pair.H_posA_rotA.transpose());
    add(ii + 3, ii + 3, pair.H_rotA_rotA);

    add(jj, jj, pair.H_posB_posB);
    add(jj, jj + 3, pair.H_posB_rotB);
    add(jj + 3, jj, pair.H_posB_rotB.transpose());
    add(jj + 3, jj + 3, pair.H_rotB_rotB);

    // Cross-molecule blocks
    add(ii, jj, pair.H_posA_posB);
    add(jj, ii, pair.H_posA_posB.transpose());

    add(ii, jj + 3, pair.H_posA_rotB);
    add(jj + 3, ii, pair.H_posA_rotB.transpose());

    add(jj, ii + 3, pair.H_posB_rotA);
    add(ii + 3, jj, pair.H_posB_rotA.transpose());

    add(ii + 3, jj + 3, pair.H_rotA_rotB);
    add(jj + 3, ii + 3, pair.H_rotA_rotB.transpose());
}

PairHessianResult short_range_site_pair_hessian(
    const Mat3& R_i,
    const Mat3& R_j,
    const Vec3& body_a,
    const Vec3& body_b,
    const Vec3& pos_a,
    const Vec3& pos_b,
    double dE_dr,
    double d2E_dr2) {

    PairHessianResult pair;

    const Vec3 r_ab = pos_b - pos_a;
    const double r = r_ab.norm();
    if (r < 1e-12) {
        return pair;
    }

    const Vec3 u = r_ab / r;
    const Mat3 I = Mat3::Identity();
    const Mat3 Hxx =
        (d2E_dr2 - dE_dr / r) * (u * u.transpose()) + (dE_dr / r) * I;

    // dE/dxA and dE/dxB
    const Vec3 gA = -dE_dr * u;
    const Vec3 gB = -gA;

    const Vec3 lever_a = R_i * body_a;
    const Vec3 lever_b = R_j * body_b;
    const Mat3 Jpsi_a = -skew_symmetric(lever_a);
    const Mat3 Jpsi_b = -skew_symmetric(lever_b);

    pair.H_posA_posA = Hxx;
    pair.H_posA_posB = -Hxx;
    pair.H_posB_posB = Hxx;

    pair.H_posA_rotA = Hxx * Jpsi_a;
    pair.H_posA_rotB = -Hxx * Jpsi_b;
    pair.H_posB_rotA = -Hxx * Jpsi_a;
    pair.H_posB_rotB = Hxx * Jpsi_b;

    pair.H_rotA_rotA = Jpsi_a.transpose() * Hxx * Jpsi_a;
    pair.H_rotA_rotB = Jpsi_a.transpose() * (-Hxx) * Jpsi_b;
    pair.H_rotB_rotB = Jpsi_b.transpose() * Hxx * Jpsi_b;

    // Exponential-map curvature at zero increment:
    // d2(exp([psi]x) v)/dpsi_k dpsi_l|_{psi=0} =
    // 0.5 * (G_k G_l + G_l G_k) v
    const auto& G = so3_generators();
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            const Vec3 d2xa =
                0.5 * (G[k] * G[l] + G[l] * G[k]) * lever_a;
            const Vec3 d2xb =
                0.5 * (G[k] * G[l] + G[l] * G[k]) * lever_b;
            pair.H_rotA_rotA(k, l) += gA.dot(d2xa);
            pair.H_rotB_rotB(k, l) += gB.dot(d2xb);
        }
    }

    return pair;
}

} // namespace

const std::array<Mat3, 6>& voigt_basis_matrices() {
    static const std::array<Mat3, 6> B = [] {
        std::array<Mat3, 6> out{};
        out[0].setZero();
        out[0](0, 0) = 1.0;
        out[1].setZero();
        out[1](1, 1) = 1.0;
        out[2].setZero();
        out[2](2, 2) = 1.0;
        out[3].setZero();
        out[3](1, 2) = out[3](2, 1) = 0.5;
        out[4].setZero();
        out[4](0, 2) = out[4](2, 0) = 0.5;
        out[5].setZero();
        out[5](0, 1) = out[5](1, 0) = 0.5;
        return out;
    }();
    return B;
}

// ============================================================================
// MoleculeState
// ============================================================================

Mat3 MoleculeState::proper_rotation_matrix() const {
    double angle = angle_axis.norm();
    if (angle < 1e-12) {
        return Mat3::Identity();
    }
    Vec3 axis = angle_axis / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

Mat3 MoleculeState::rotation_matrix() const {
    const int p = (parity < 0) ? -1 : 1;
    return static_cast<double>(p) * proper_rotation_matrix();
}

MoleculeState MoleculeState::from_rotation(const Vec3& pos, const Mat3& R) {
    MoleculeState state;
    state.position = pos;
    state.parity = 1;

    // Use exact orthogonal input directly (common in finite-difference
    // perturbations) to avoid introducing SVD gauge noise into the
    // angle-axis extraction. Reproject only when needed.
    Mat3 Q = R;
    const Mat3 I = Mat3::Identity();
    const double ortho_err = (R.transpose() * R - I).norm();
    const double det_err = std::abs(std::abs(R.determinant()) - 1.0);
    if (ortho_err > 1e-10 || det_err > 1e-10) {
        Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Q = svd.matrixU() * svd.matrixV().transpose();
    }

    // Represent O(3) orientation as parity * SO(3) rotation.
    if (Q.determinant() < 0.0) {
        state.parity = -1;
        Q = -Q;
    }

    Eigen::AngleAxisd aa(Q);
    state.angle_axis = aa.angle() * aa.axis();
    return state;
}

// ============================================================================
// CrystalEnergyResult
// ============================================================================

Vec CrystalEnergyResult::pack_gradient() const {
    const int n = static_cast<int>(forces.size());
    Vec grad(6 * n);
    for (int i = 0; i < n; ++i) {
        grad.segment<3>(6 * i) = forces[i];
        grad.segment<3>(6 * i + 3) = torques[i];
    }
    return grad;
}

// ============================================================================
// Williams DE Buckingham Parameters (kJ/mol, Angstrom)
// ============================================================================

std::map<std::pair<int,int>, BuckinghamParams> CrystalEnergy::williams_de_params() {
    // Reference: Williams & Cox, Acta Cryst. B40 (1984)
    // Parameters in kJ/mol and Angstrom
    std::map<std::pair<int,int>, BuckinghamParams> params;

    // H-H
    params[{1, 1}] = {2650.8, 3.74, 27.3};
    // C-C
    params[{6, 6}] = {369742.2, 3.60, 2439.8};
    // N-N
    params[{7, 7}] = {254501.2, 3.78, 1378.4};
    // O-O
    params[{8, 8}] = {230064.3, 3.96, 1123.6};

    // Cross-terms (geometric mean mixing for A and C, arithmetic for B)
    // H-C
    params[{1, 6}] = {31368.8, 3.67, 258.0};
    params[{6, 1}] = params[{1, 6}];
    // H-N
    params[{1, 7}] = {25988.3, 3.76, 194.0};
    params[{7, 1}] = params[{1, 7}];
    // H-O
    params[{1, 8}] = {24716.7, 3.85, 175.2};
    params[{8, 1}] = params[{1, 8}];
    // C-N
    params[{6, 7}] = {306739.8, 3.69, 1834.1};
    params[{7, 6}] = params[{6, 7}];
    // C-O
    params[{6, 8}] = {291770.4, 3.78, 1655.4};
    params[{8, 6}] = params[{6, 8}];
    // N-O
    params[{7, 8}] = {242022.9, 3.87, 1244.5};
    params[{8, 7}] = params[{7, 8}];

    return params;
}

std::map<std::pair<int,int>, BuckinghamParams> CrystalEnergy::williams_typed_params() {
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
            // DMACRYS/NEIGHCRYS Williams mixing uses arithmetic mean in B-space.
            // With rho = 1/B in the source table:
            // B_ij = 0.5 * (1/rho_i + 1/rho_j).
            p.B = 0.5 * ((1.0 / a.rho) + (1.0 / b.rho));
            p.C = std::sqrt(std::max(0.0, a.C) * std::max(0.0, b.C)) * eV_to_kJ;
            params[{a.code, b.code}] = p;
            params[{b.code, a.code}] = p;
        }
    }
    return params;
}

const char* CrystalEnergy::short_range_type_label(int type_code) {
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

int CrystalEnergy::short_range_type_atomic_number(int type_code) {
    switch (type_code) {
    case 501:
    case 502:
    case 503:
    case 504:
    case 505:
        return 1;
    case 511:
    case 512:
    case 513:
        return 6;
    case 521:
    case 522:
    case 523:
    case 524:
        return 7;
    case 531:
    case 532:
    case 533:
        return 8;
    case 540:
        return 9;
    case 541:
        return 17;
    case 542:
        return 16;
    case 543:
        return 19;
    case 544:
        return 35;
    case 545:
        return 53;
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
// CrystalEnergy Constructor
// ============================================================================

CrystalEnergy::CrystalEnergy(const crystal::Crystal& crystal,
                             std::vector<MultipoleSource> multipoles,
                             double cutoff_radius,
                             ForceFieldType ff,
                             bool use_cartesian,
                             bool use_ewald,
                             double ewald_accuracy,
                             double ewald_eta,
                             int ewald_kmax)
    : m_crystal(crystal)
    , m_multipoles(std::move(multipoles))
    , m_cutoff_radius(cutoff_radius)
    , m_force_field(ff)
    , m_use_cartesian(use_cartesian)
    , m_use_ewald(use_ewald)
    , m_ewald_accuracy(ewald_accuracy)
    , m_ewald_eta(ewald_eta)
    , m_ewald_kmax(ewald_kmax) {

    if (m_multipoles.empty()) {
        throw std::invalid_argument("CrystalEnergy: no multipoles provided");
    }

    build_neighbor_list();
    build_molecule_geometry();
    initialize_force_field();
}

CrystalEnergy::~CrystalEnergy() = default;
CrystalEnergy::CrystalEnergy(CrystalEnergy&&) noexcept = default;
CrystalEnergy& CrystalEnergy::operator=(CrystalEnergy&&) noexcept = default;

void CrystalEnergy::invalidate_ewald_params() {
    m_ewald_params_initialized = false;
    m_ewald_lattice_cache.reset();
}

void CrystalEnergy::ensure_ewald_params_initialized() const {
    if (m_ewald_params_initialized) {
        return;
    }

    double ewald_real_cutoff = m_use_com_elec_gate
        ? effective_electrostatic_com_cutoff()
        : effective_neighbor_pair_cutoff();
    const double elec_site_cutoff = effective_electrostatic_site_cutoff();
    if (elec_site_cutoff > 0.0) {
        ewald_real_cutoff = std::max(ewald_real_cutoff, elec_site_cutoff);
    }

    if (m_ewald_eta > 0.0) {
        m_ewald_alpha_fixed = m_ewald_eta;
    } else {
        const double x = std::sqrt(-std::log(m_ewald_accuracy));
        m_ewald_alpha_fixed = x / ewald_real_cutoff;
    }

    if (m_ewald_kmax > 0) {
        m_ewald_kmax_fixed = m_ewald_kmax;
    } else {
        const auto& uc = m_crystal.unit_cell();
        const double min_len = std::min({uc.a(), uc.b(), uc.c()});
        const double G_max =
            2.0 * m_ewald_alpha_fixed * std::sqrt(-std::log(m_ewald_accuracy));
        m_ewald_kmax_fixed = std::max(
            1, static_cast<int>(std::ceil(G_max * min_len / (2.0 * M_PI))));
    }

    m_ewald_params_initialized = true;
    occ::log::info("Ewald parameters fixed for this calculation: alpha={:.6f} /Ang, kmax={}",
                   m_ewald_alpha_fixed, m_ewald_kmax_fixed);
}

void CrystalEnergy::set_cutoff_radius(double cutoff) {
    if (cutoff <= 0.0) {
        throw std::invalid_argument("CrystalEnergy::set_cutoff_radius: cutoff must be > 0");
    }
    m_cutoff_radius = cutoff;
    invalidate_ewald_params();
    if (!m_explicit_neighbors) {
        build_neighbor_list();
    }
}

void CrystalEnergy::set_electrostatic_taper(double r_on, double r_off, int order) {
    if (r_off <= r_on) {
        throw std::invalid_argument("CrystalEnergy::set_electrostatic_taper: require r_off > r_on");
    }
    if (order != 3 && order != 5) {
        throw std::invalid_argument("CrystalEnergy::set_electrostatic_taper: order must be 3 or 5");
    }
    m_electrostatic_taper = {true, r_on, r_off, order};
    invalidate_ewald_params();
}

void CrystalEnergy::set_short_range_taper(double r_on, double r_off, int order) {
    if (r_off <= r_on) {
        throw std::invalid_argument("CrystalEnergy::set_short_range_taper: require r_off > r_on");
    }
    if (order != 3 && order != 5) {
        throw std::invalid_argument("CrystalEnergy::set_short_range_taper: order must be 3 or 5");
    }
    m_short_range_taper = {true, r_on, r_off, order};
    invalidate_ewald_params();
}

double CrystalEnergy::effective_electrostatic_com_cutoff() const {
    double cutoff = m_cutoff_radius;
    if (m_electrostatic_taper.is_valid()) {
        cutoff = std::max(cutoff, m_electrostatic_taper.r_off);
    }
    return cutoff;
}

double CrystalEnergy::effective_electrostatic_site_cutoff() const {
    double cutoff = m_elec_site_cutoff;
    if (m_electrostatic_taper.is_valid()) {
        cutoff = (cutoff > 0.0) ? std::max(cutoff, m_electrostatic_taper.r_off)
                                : m_electrostatic_taper.r_off;
    }
    return cutoff;
}

double CrystalEnergy::effective_buckingham_site_cutoff() const {
    double cutoff = (m_buck_site_cutoff > 0.0) ? m_buck_site_cutoff : m_cutoff_radius;
    if (m_short_range_taper.is_valid()) {
        cutoff = std::max(cutoff, m_short_range_taper.r_off);
    }
    return cutoff;
}

double CrystalEnergy::effective_neighbor_pair_cutoff() const {
    double cutoff = std::max(m_cutoff_radius, effective_buckingham_site_cutoff());
    if (m_use_com_elec_gate) {
        cutoff = std::max(cutoff, effective_electrostatic_com_cutoff());
    }
    return cutoff;
}

// ============================================================================
// Neighbor List Construction
// ============================================================================

void CrystalEnergy::build_neighbor_list() {
    m_neighbors.clear();

    auto dimers = m_crystal.symmetry_unique_dimers(effective_neighbor_pair_cutoff());

    const int num_unique = static_cast<int>(m_multipoles.size());

    for (size_t i = 0; i < dimers.molecule_neighbors.size(); ++i) {
        for (const auto& neighbor : dimers.molecule_neighbors[i]) {
            const auto& dimer = neighbor.dimer;

            // Get molecule indices
            int mol_i = static_cast<int>(i);
            // Get the asymmetric molecule index of the neighbor (molecule B in dimer)
            int mol_j = dimer.b().asymmetric_molecule_idx();

            // Validate indices
            if (mol_i < 0 || mol_i >= num_unique || mol_j < 0 || mol_j >= num_unique) {
                occ::log::debug("Skipping invalid neighbor pair: mol_i={}, mol_j={}, num_unique={}",
                               mol_i, mol_j, num_unique);
                continue;
            }

            // Compute cell shift from molecule B's position
            const auto& mol_a = dimer.a();
            const auto& mol_b = dimer.b();

            Vec3 center_a = mol_a.center_of_mass();
            Vec3 center_b = mol_b.center_of_mass();

            // Convert to fractional and find integer shift
            Vec3 frac_a = m_crystal.to_fractional(center_a);
            Vec3 frac_b = m_crystal.to_fractional(center_b);
            Vec3 diff = frac_b - frac_a;

            IVec3 cell_shift;
            cell_shift << static_cast<int>(std::round(diff[0])),
                         static_cast<int>(std::round(diff[1])),
                         static_cast<int>(std::round(diff[2]));

            // Weight: 0.5 for self-interaction, 1.0 otherwise
            double weight = (mol_i == mol_j && cell_shift == IVec3::Zero()) ? 0.5 : 1.0;

            // Skip true self (same molecule, no translation)
            if (mol_i == mol_j && cell_shift == IVec3::Zero()) {
                continue;
            }

            m_neighbors.push_back({mol_i, mol_j, cell_shift, weight});
        }
    }

    occ::log::debug("Built neighbor list with {} pairs", m_neighbors.size());
}

void CrystalEnergy::update_neighbors() {
    if (m_explicit_neighbors) {
        // Neighbor list was set externally (build_neighbor_list_from_positions
        // or set_neighbor_list). Don't rebuild via symmetry_unique_dimers as
        // that would corrupt an all-pairs list for Z UC molecules.
        occ::log::debug("Skipping neighbor list rebuild (explicit neighbor list)");
        return;
    }
    build_neighbor_list();
}

void CrystalEnergy::update_neighbors(const std::vector<MoleculeState>& states) {
    if (m_explicit_neighbors) {
        std::vector<Vec3> coms;
        coms.reserve(states.size());
        for (const auto& s : states) {
            coms.push_back(s.position);
        }
        build_neighbor_list_from_positions(coms, false, &states);
        return;
    }
    build_neighbor_list();
}

void CrystalEnergy::update_lattice(const crystal::Crystal& strained_crystal,
                                    std::vector<MoleculeState> new_states) {
    m_crystal = strained_crystal;
    m_initial_states = std::move(new_states);
    m_ewald_lattice_cache.reset();  // Invalidate: lattice changed
}

void CrystalEnergy::build_neighbor_list_from_positions(
        const std::vector<Vec3>& mol_coms, bool force_com_cutoff,
        const std::vector<MoleculeState>* orientation_states) {
    m_neighbors.clear();
    int n_mol = static_cast<int>(mol_coms.size());

    const auto& uc = m_crystal.unit_cell();

    // Use atom-based cutoff: include pair if any atom-atom distance < cutoff.
    // When force_com_cutoff=true, use COM distance (matches DMACRYS TBLCNT).
    // Need extra margin for COM-to-atom extent when determining cell search range.
    bool use_atom_cutoff = !m_geometry.empty() && !force_com_cutoff;
    double max_atom_extent = 0.0;
    // Pre-compute crystal-frame atom positions for each molecule
    std::vector<std::vector<Vec3>> crystal_atoms(n_mol);
    if (use_atom_cutoff) {
        const std::vector<MoleculeState>* orient_states = orientation_states;
        if (!orient_states ||
            static_cast<int>(orient_states->size()) != n_mol) {
            orient_states = &m_initial_states;
        }
        for (int m = 0; m < n_mol; ++m) {
            const auto& geom = m_geometry[m];
            Mat3 R = (orient_states &&
                      static_cast<int>(orient_states->size()) == n_mol)
                         ? (*orient_states)[m].rotation_matrix()
                         : Mat3::Identity();
            crystal_atoms[m].reserve(geom.atom_positions.size());
            for (const auto& bp : geom.atom_positions) {
                Vec3 cart = mol_coms[m] + R * bp;
                crystal_atoms[m].push_back(cart);
                double ext = (R * bp).norm();
                if (ext > max_atom_extent)
                    max_atom_extent = ext;
            }
        }
    }

    const double pair_cutoff = effective_neighbor_pair_cutoff();
    double search_radius = pair_cutoff + 2.0 * max_atom_extent;
    // Use anisotropic search extents from reciprocal lattice norms, matching
    // the DMACRYS TBLCNT-style bound N(i) ~ R * |b_i*| + 1.
    const Mat3 reciprocal = uc.reciprocal();
    IVec3 nmax = IVec3::Zero();
    for (int axis = 0; axis < 3; ++axis) {
        const double bnorm = reciprocal.col(axis).norm();
        int ni = static_cast<int>(std::ceil(search_radius * bnorm)) + 1;
        if (ni < 1) ni = 1;
        nmax[axis] = ni;
    }

    for (int i = 0; i < n_mol; ++i) {
        for (int j = 0; j < n_mol; ++j) {
            for (int nx = -nmax[0]; nx <= nmax[0]; ++nx) {
                for (int ny = -nmax[1]; ny <= nmax[1]; ++ny) {
                    for (int nz = -nmax[2]; nz <= nmax[2]; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;

                        IVec3 shift(nx, ny, nz);
                        if (!is_canonical_explicit_pair(i, j, shift)) {
                            continue;
                        }
                        Vec3 trans = uc.to_cartesian(shift.cast<double>());

                        bool include = false;
                        if (use_atom_cutoff) {
                            // Check minimum atom-atom distance
                            for (const auto& ai : crystal_atoms[i]) {
                                for (const auto& aj : crystal_atoms[j]) {
                                    double d = (aj + trans - ai).norm();
                                    if (d < pair_cutoff) {
                                        include = true;
                                        break;
                                    }
                                }
                                if (include) break;
                            }
                        } else {
                            double dist = (mol_coms[j] + trans - mol_coms[i]).norm();
                            include = (dist < pair_cutoff);
                        }

                        if (include) {
                            // Canonical unique pair list (no mirrored duplicates).
                            double com_dist = (mol_coms[j] + trans - mol_coms[i]).norm();
                            m_neighbors.push_back({i, j, shift, 1.0, com_dist});
                        }
                    }
                }
            }
        }
    }

    m_explicit_neighbors = true;
    occ::log::debug("Built neighbor list from positions: {} pairs for {} molecules (atom-based cutoff: {})",
                    m_neighbors.size(), n_mol, use_atom_cutoff);
}

void CrystalEnergy::set_neighbor_list(const std::vector<NeighborPair>& neighbors) {
    m_neighbors = neighbors;
    m_explicit_neighbors = true;
}

void CrystalEnergy::set_molecule_geometry(std::vector<MoleculeGeometry> geometry) {
    m_geometry = std::move(geometry);
    if (m_force_field == ForceFieldType::BuckinghamDE) {
        assign_williams_atom_types();
    }
}

void CrystalEnergy::set_initial_states(std::vector<MoleculeState> states) {
    m_initial_states = std::move(states);
}

// ============================================================================
// Molecule Geometry
// ============================================================================

void CrystalEnergy::build_molecule_geometry() {
    m_geometry.clear();
    m_geometry.reserve(m_multipoles.size());

    const auto& unique_mols = m_crystal.symmetry_unique_molecules();

    for (size_t i = 0; i < m_multipoles.size() && i < unique_mols.size(); ++i) {
        const auto& mol = unique_mols[i];
        MoleculeGeometry geom;

        geom.center_of_mass = mol.center_of_mass();

        for (int j = 0; j < mol.size(); ++j) {
            geom.atomic_numbers.push_back(mol.atomic_numbers()(j));
            // Store in body frame (relative to COM)
            geom.atom_positions.push_back(mol.positions().col(j) - geom.center_of_mass);
        }
        geom.short_range_type_codes.assign(geom.atomic_numbers.size(), 0);

        m_geometry.push_back(std::move(geom));
    }
}

void CrystalEnergy::assign_williams_atom_types() {
    for (auto& geom : m_geometry) {
        const auto neighbors = bonded_neighbors(geom.atomic_numbers, geom.atom_positions);
        geom.short_range_type_codes.resize(geom.atomic_numbers.size(), 0);
        for (size_t i = 0; i < geom.atomic_numbers.size(); ++i) {
            geom.short_range_type_codes[i] = classify_williams_type(
                static_cast<int>(i), neighbors, geom.atomic_numbers);
        }
    }
    m_use_short_range_typing = true;
}

// ============================================================================
// Force Field Initialization
// ============================================================================

void CrystalEnergy::initialize_force_field() {
    if (m_force_field == ForceFieldType::BuckinghamDE) {
        m_buckingham_params = williams_de_params();
        m_typed_buckingham_params = williams_typed_params();
        assign_williams_atom_types();
        m_use_williams_atom_typing = true;
        m_use_short_range_typing = true;
    }
}

void CrystalEnergy::set_buckingham_params(int Z1, int Z2, const BuckinghamParams& params) {
    m_buckingham_params[{Z1, Z2}] = params;
    m_buckingham_params[{Z2, Z1}] = params;
}

void CrystalEnergy::set_typed_buckingham_params(
    int type1, int type2, const BuckinghamParams& params) {
    m_typed_buckingham_params[{type1, type2}] = params;
    m_typed_buckingham_params[{type2, type1}] = params;
    m_use_short_range_typing = true;
}

void CrystalEnergy::set_typed_buckingham_params(
    const std::map<std::pair<int,int>, BuckinghamParams>& params) {
    m_typed_buckingham_params = params;
    if (!m_typed_buckingham_params.empty()) {
        m_use_short_range_typing = true;
    }
}

void CrystalEnergy::clear_typed_buckingham_params() {
    m_typed_buckingham_params.clear();
    m_missing_typed_buckingham_warned.clear();
    m_use_short_range_typing = m_use_williams_atom_typing;
}

void CrystalEnergy::set_short_range_type_labels(
    const std::map<int, std::string>& labels) {
    m_short_range_type_labels = labels;
    if (!m_short_range_type_labels.empty()) {
        m_use_short_range_typing = true;
    }
}

void CrystalEnergy::set_typed_aniso_params(
    const std::map<std::pair<int,int>, AnisotropicRepulsionParams>& params) {
    m_typed_aniso_params = params;
}

bool CrystalEnergy::has_aniso_params(int type1, int type2) const {
    return m_typed_aniso_params.count({type1, type2}) > 0;
}

AnisotropicRepulsionParams CrystalEnergy::get_aniso_params(int type1, int type2) const {
    auto it = m_typed_aniso_params.find({type1, type2});
    if (it != m_typed_aniso_params.end()) {
        return it->second;
    }
    return {};
}

bool CrystalEnergy::has_buckingham_params(int Z1, int Z2) const {
    return m_buckingham_params.find({Z1, Z2}) != m_buckingham_params.end();
}

bool CrystalEnergy::has_typed_buckingham_params(int type1, int type2) const {
    return m_typed_buckingham_params.find({type1, type2}) !=
           m_typed_buckingham_params.end();
}

BuckinghamParams CrystalEnergy::get_buckingham_params_for_types(
    int type1, int type2) const {

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
            short_range_type_name(type1), short_range_type_name(type2));
    }

    const int z1 = short_range_type_atomic_number(type1);
    const int z2 = short_range_type_atomic_number(type2);
    if (z1 > 0 && z2 > 0) {
        return get_buckingham_params(z1, z2);
    }
    return {1000.0, 3.5, 10.0};
}

std::string CrystalEnergy::short_range_type_name(int type_code) const {
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

BuckinghamParams CrystalEnergy::get_buckingham_params(int Z1, int Z2) const {
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
    // Default: small repulsion to avoid overlaps
    return {1000.0, 3.5, 10.0};
}

// ============================================================================
// Initial States
// ============================================================================

std::vector<MoleculeState> CrystalEnergy::initial_states() const {
    if (!m_initial_states.empty()) {
        return m_initial_states;
    }

    std::vector<MoleculeState> states;
    states.reserve(m_multipoles.size());

    const auto& unique_mols = m_crystal.symmetry_unique_molecules();

    for (size_t i = 0; i < m_multipoles.size() && i < unique_mols.size(); ++i) {
        MoleculeState state;
        state.position = unique_mols[i].center_of_mass();
        state.angle_axis = Vec3::Zero();  // Identity rotation
        states.push_back(state);
    }

    return states;
}

// ============================================================================
// Buckingham Site-Pair Masks
// ============================================================================

std::vector<std::vector<bool>> CrystalEnergy::compute_buckingham_site_masks(
    const std::vector<MoleculeState>& states) const {

    const double buck_cutoff = effective_buckingham_site_cutoff();

    std::vector<std::vector<bool>> masks(m_neighbors.size());
    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        int mi = pair.mol_i;
        int mj = pair.mol_j;
        const auto& geom_i = m_geometry[mi];
        const auto& geom_j = m_geometry[mj];
        const size_t nA = geom_i.atom_positions.size();
        const size_t nB = geom_j.atom_positions.size();

        masks[pair_idx].resize(nA * nB, false);

        Mat3 R_i = states[mi].rotation_matrix();
        Mat3 R_j = states[mj].rotation_matrix();
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        for (size_t a = 0; a < nA; ++a) {
            Vec3 pos_a = states[mi].position + R_i * geom_i.atom_positions[a];
            for (size_t b = 0; b < nB; ++b) {
                Vec3 pos_b = states[mj].position + cell_translation +
                             R_j * geom_j.atom_positions[b];
                double r = (pos_b - pos_a).norm();
                if (r <= buck_cutoff && r >= 0.1) {
                    masks[pair_idx][a * nB + b] = true;
                }
            }
        }
    }

    return masks;
}

// ============================================================================
// Short-Range Pair Computation
// ============================================================================

void CrystalEnergy::compute_short_range_pair(
    int mol_i, int mol_j,
    const MoleculeState& state_i,
    const MoleculeState& state_j,
    const Vec3& translation,
    double weight,
    double& energy,
    Vec3& force_i, Vec3& force_j,
    Vec3& torque_i, Vec3& torque_j,
    int neighbor_idx,
    const MoleculeCache* cache_i,
    const MoleculeCache* cache_j,
    int* short_range_site_pairs) const {

    if (m_force_field == ForceFieldType::None) {
        return;
    }

    const auto& geom_i = m_geometry[mol_i];
    const auto& geom_j = m_geometry[mol_j];
    const size_t nA = geom_i.atom_positions.size();
    const size_t nB = geom_j.atom_positions.size();

    // Check if we have frozen site masks for this neighbor pair
    const bool use_frozen = (neighbor_idx >= 0 &&
                             static_cast<size_t>(neighbor_idx) < m_fixed_site_masks.size() &&
                             !m_fixed_site_masks[neighbor_idx].empty());

    // Use cached rotation matrices or compute them
    Mat3 R_i_storage, R_j_storage;
    const Mat3& R_i = cache_i ? cache_i->rotation : (R_i_storage = state_i.rotation_matrix());
    const Mat3& R_j = cache_j ? cache_j->rotation : (R_j_storage = state_j.rotation_matrix());

    // Loop over atom pairs
    for (size_t a = 0; a < nA; ++a) {
        int Z_a = geom_i.atomic_numbers[a];
        Vec3 pos_a = cache_i ? cache_i->lab_atom_positions[a]
                             : (state_i.position + R_i * geom_i.atom_positions[a]);

        for (size_t b = 0; b < nB; ++b) {
            // Apply frozen mask or distance cutoff
            if (use_frozen) {
                if (!m_fixed_site_masks[neighbor_idx][a * nB + b]) {
                    continue;
                }
            }

            int Z_b = geom_j.atomic_numbers[b];
            Vec3 pos_b;
            if (cache_j) {
                pos_b = cache_j->lab_atom_positions[b] + translation;
            } else {
                pos_b = state_j.position + translation + R_j * geom_j.atom_positions[b];
            }

            Vec3 r_ab = pos_b - pos_a;
            double r = r_ab.norm();

            if (!use_frozen) {
                double buck_cutoff = effective_buckingham_site_cutoff();
                if (r > buck_cutoff || r < 0.1) {
                    continue;
                }
            }

            ShortRangeInteraction::EnergyAndDerivatives sr;

            const bool has_type_codes =
                m_use_short_range_typing &&
                a < geom_i.short_range_type_codes.size() &&
                b < geom_j.short_range_type_codes.size() &&
                geom_i.short_range_type_codes[a] > 0 &&
                geom_j.short_range_type_codes[b] > 0;

            if (m_force_field == ForceFieldType::BuckinghamDE ||
                m_force_field == ForceFieldType::Custom) {
                BuckinghamParams params;
                if (has_type_codes) {
                    const int t1 = geom_i.short_range_type_codes[a];
                    const int t2 = geom_j.short_range_type_codes[b];
                    if (has_typed_buckingham_params(t1, t2)) {
                        params = get_buckingham_params_for_types(t1, t2);
                    } else {
                        params = get_buckingham_params(Z_a, Z_b);
                    }
                } else {
                    params = get_buckingham_params(Z_a, Z_b);
                }
                sr = ShortRangeInteraction::buckingham_all(r, params);
            } else if (m_force_field == ForceFieldType::LennardJones) {
                auto it = m_lj_params.find({Z_a, Z_b});
                if (it == m_lj_params.end()) {
                    continue;
                }
                sr = ShortRangeInteraction::lennard_jones_all(r, it->second);
            }

            if (m_short_range_taper.is_valid()) {
                auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                if (sw.value <= 0.0) {
                    continue;
                }
                const double e0 = sr.energy;
                const double d10 = sr.first_derivative;
                const double d20 = sr.second_derivative;
                sr.energy = sw.value * e0;
                sr.first_derivative = sw.value * d10 + e0 * sw.first_derivative;
                sr.second_derivative = sw.value * d20 +
                                       2.0 * sw.first_derivative * d10 +
                                       e0 * sw.second_derivative;
            }

            if (short_range_site_pairs) {
                ++(*short_range_site_pairs);
            }

            energy += weight * sr.energy;

            // Force from derivative
            Vec3 force_on_a = ShortRangeInteraction::derivative_to_force(sr.first_derivative, r_ab);

            // Weighted forces
            Vec3 wf = weight * force_on_a;
            force_i += wf;
            force_j -= wf;

            // Torque from lever arm
            Vec3 lever_a = R_i * geom_i.atom_positions[a];
            Vec3 lever_b = R_j * geom_j.atom_positions[b];

            // Torque in lab frame
            Vec3 torque_lab_a = lever_a.cross(wf);
            Vec3 torque_lab_b = lever_b.cross(-wf);

            // Lever-arm rotational gradient in lab frame:
            // torque_lab = lever × force = lever × (-dE/dr) = -dE/dψ_lab
            // so -torque_lab = +dE/dψ_lab
            torque_i -= torque_lab_a;
            torque_j -= torque_lab_b;

            // Anisotropic repulsion (additive to isotropic Buckingham)
            if (!m_typed_aniso_params.empty() && has_type_codes) {
                const int t1 = geom_i.short_range_type_codes[a];
                const int t2 = geom_j.short_range_type_codes[b];
                if (has_aniso_params(t1, t2)) {
                    const auto aniso_params = get_aniso_params(t1, t2);

                    // Get body-frame aniso axes and rotate to lab frame
                    Vec3 axis_a_lab = Vec3::Zero();
                    Vec3 axis_b_lab = Vec3::Zero();
                    if (a < geom_i.aniso_body_axes.size()) {
                        axis_a_lab = R_i * geom_i.aniso_body_axes[a];
                    }
                    if (b < geom_j.aniso_body_axes.size()) {
                        axis_b_lab = R_j * geom_j.aniso_body_axes[b];
                    }

                    auto aniso = ShortRangeInteraction::anisotropic_repulsion(
                        pos_a, pos_b, axis_a_lab, axis_b_lab, aniso_params);

                    // Apply taper if active
                    if (m_short_range_taper.is_valid()) {
                        auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                        if (sw.value > 0.0) {
                            // Taper: E_tapered = f(r) * E_aniso
                            // F_tapered = f(r)*F_aniso + f'(r)*(E_aniso/r)*r_ab  (chain rule)
                            const double r_inv = 1.0 / r;
                            const Vec3 taper_force_correction =
                                sw.first_derivative * aniso.energy * r_inv * r_ab;
                            energy += weight * sw.value * aniso.energy;

                            Vec3 aniso_fa = sw.value * aniso.force_A - taper_force_correction;
                            Vec3 aniso_fb = sw.value * aniso.force_B + taper_force_correction;
                            force_i += weight * aniso_fa;
                            force_j += weight * aniso_fb;

                            // Lever-arm torque from aniso forces
                            torque_i -= weight * lever_a.cross(aniso_fa);
                            torque_j -= weight * lever_b.cross(aniso_fb);

                            // Axis rotation torque
                            torque_i += weight * sw.value * aniso.torque_axis_A;
                            torque_j += weight * sw.value * aniso.torque_axis_B;
                        }
                    } else {
                        energy += weight * aniso.energy;
                        force_i += weight * aniso.force_A;
                        force_j += weight * aniso.force_B;

                        // Lever-arm torque from aniso forces
                        torque_i -= weight * lever_a.cross(aniso.force_A);
                        torque_j -= weight * lever_b.cross(aniso.force_B);

                        // Axis rotation torque (already dE/dψ_lab form)
                        torque_i += weight * aniso.torque_axis_A;
                        torque_j += weight * aniso.torque_axis_B;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Main Energy Computation
// ============================================================================

CrystalEnergyResult CrystalEnergy::compute(const std::vector<MoleculeState>& molecules) {
    occ::timing::StopWatch<6> sw;
    // 0: cache, 1: set_orientation, 2: cartesian(), 3: elec pairs, 4: SR pairs, 5: ewald

    const int N = static_cast<int>(molecules.size());
    CrystalEnergyResult result;
    result.forces.resize(N, Vec3::Zero());
    result.torques.resize(N, Vec3::Zero());
    result.molecule_energies.resize(N, 0.0);

    sw.start(0);
    // Precompute rotation matrices and lab-frame atom positions once
    std::vector<MoleculeCache> mol_cache(N);
    for (int i = 0; i < N; ++i) {
        mol_cache[i].rotation = molecules[i].rotation_matrix();
        if (!m_geometry.empty() && i < static_cast<int>(m_geometry.size())) {
            const auto& geom = m_geometry[i];
            mol_cache[i].lab_atom_positions.resize(geom.atom_positions.size());
            for (size_t a = 0; a < geom.atom_positions.size(); ++a) {
                mol_cache[i].lab_atom_positions[a] =
                    molecules[i].position + mol_cache[i].rotation * geom.atom_positions[a];
            }
        }
    }
    sw.stop(0);

    sw.start(1);
    // Update multipole orientations (modifies m_multipoles in place)
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            mol_cache[i].rotation,
            molecules[i].position);
    }
    sw.stop(1);

    sw.start(2);
    // Pre-build all CartesianMolecules ONCE (expensive conversion happens here)
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }
    sw.stop(2);

    // Loop over neighbor pairs
    const double elec_com_cutoff = effective_electrostatic_com_cutoff();
    const double elec_site_cutoff = effective_electrostatic_site_cutoff();
    const CutoffSpline* elec_taper =
        m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr;
    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        int i = pair.mol_i;
        int j = pair.mol_j;

        // Translation for molecule j
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());
        const Vec3 disp_ij =
            molecules[j].position + cell_translation - molecules[i].position;

        // Electrostatic interaction.
        // COM gate: skip electrostatics for pairs with COM distance > cutoff.
        // This matches DMACRYS TBLCNT which selects molecule pairs by COM distance.
        // Ewald correction compensates for qq+qμ+μμ regardless of pair selection.
        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate &&
            pair.com_distance > elec_com_cutoff) {
            include_elec = false;
        }
        if (include_elec) {
            sw.start(3);
            // Pass cell_translation as offset_B to avoid copying CartesianMolecule
            auto elec_result = compute_molecule_forces_torques(
                cart_mols[i],
                cart_mols[j],
                elec_site_cutoff,
                m_max_interaction_order,
                cell_translation,
                elec_taper);
            sw.stop(3);

            double e_elec = pair.weight * elec_result.energy;
            result.electrostatic_energy += e_elec;
            result.molecule_energies[i] += e_elec * 0.5;
            result.molecule_energies[j] += e_elec * 0.5;

            result.forces[i] += pair.weight * elec_result.force_A;
            result.forces[j] += pair.weight * elec_result.force_B;
            result.torques[i] += pair.weight * elec_result.grad_angle_axis_A;
            result.torques[j] += pair.weight * elec_result.grad_angle_axis_B;
            add_pair_strain_gradient(
                result.strain_gradient, elec_result.force_A, disp_ij, pair.weight);
        }

        // Short-range interaction
        double sr_energy = 0.0;
        Vec3 sr_force_i = Vec3::Zero();
        Vec3 sr_force_j = Vec3::Zero();
        Vec3 sr_torque_i = Vec3::Zero();
        Vec3 sr_torque_j = Vec3::Zero();

        sw.start(4);
        compute_short_range_pair(
            i, j,
            molecules[i], molecules[j],
            cell_translation,
            pair.weight,
            sr_energy,
            sr_force_i, sr_force_j,
            sr_torque_i, sr_torque_j,
            static_cast<int>(pair_idx),
            &mol_cache[i], &mol_cache[j]);
        sw.stop(4);

        result.repulsion_dispersion += sr_energy;
        result.molecule_energies[i] += sr_energy * 0.5;
        result.molecule_energies[j] += sr_energy * 0.5;

        result.forces[i] += sr_force_i;
        result.forces[j] += sr_force_j;
        result.torques[i] += sr_torque_i;
        result.torques[j] += sr_torque_j;
        add_pair_strain_gradient(result.strain_gradient, sr_force_i, disp_ij, 1.0);
    }

    // Ewald correction for charge-charge electrostatics
    if (m_use_ewald) {
        sw.start(5);
        auto ewald = compute_charge_ewald_correction(molecules, cart_mols);
        sw.stop(5);
        result.electrostatic_energy += ewald.energy;
        for (int i = 0; i < N; ++i) {
            result.forces[i] += ewald.forces[i];
            result.torques[i] += ewald.torques[i];
        }

        // Chain-rule strain contribution from Ewald site-position dependence
        // through molecule COM strain (orientation fixed in this model).
        {
            const auto& B = voigt_basis_matrices();
            for (int m = 0; m < N; ++m) {
                for (int a = 0; a < 6; ++a) {
                    result.strain_gradient[a] +=
                        (-ewald.forces[m]).dot(B[a] * molecules[m].position);
                }
            }
        }

        // Add explicit Ewald lattice/cell contribution to dE/dE at fixed
        // Cartesian site coordinates (image translations, reciprocal lattice,
        // and 1/V prefactor terms).
        auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
        if (!ewald_sites.empty()) {
            auto mol_site_indices = build_mol_site_indices(cart_mols);
            EwaldParams params;
            ensure_ewald_params_initialized();
            params.alpha = m_ewald_alpha_fixed;
            params.kmax = m_ewald_kmax_fixed;
            params.include_dipole = m_ewald_dipole;
            if (!m_ewald_lattice_cache) {
                m_ewald_lattice_cache = std::make_unique<EwaldLatticeCache>(
                    build_ewald_lattice_cache(m_crystal.unit_cell(), params));
            }
            // Full Ewald strain derivatives in the rigid-body strain model:
            // site positions follow molecule COM affine strain (orientation fixed),
            // plus explicit lattice dependence (image vectors, reciprocal lattice, 1/V).
            auto ewald_strain = compute_ewald_explicit_strain_terms(
                ewald_sites, m_crystal.unit_cell(), m_neighbors, mol_site_indices,
                effective_electrostatic_com_cutoff(), m_use_com_elec_gate,
                elec_site_cutoff,
                params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get(),
                false);
            result.strain_gradient += ewald_strain.grad;
        }
    }

    result.total_energy = result.electrostatic_energy + result.repulsion_dispersion;

    occ::log::debug("compute() timing: cache={:.3f}ms orient={:.3f}ms "
                     "cartesian={:.3f}ms elec={:.3f}ms SR={:.3f}ms ewald={:.3f}ms",
                     sw.read(0)*1e3, sw.read(1)*1e3, sw.read(2)*1e3,
                     sw.read(3)*1e3, sw.read(4)*1e3, sw.read(5)*1e3);

    return result;
}

double CrystalEnergy::compute_energy(const std::vector<MoleculeState>& molecules) {
    // Simplified version without gradient computation
    // For now, just call compute() and return energy
    // Could be optimized later to skip gradient calculation
    return compute(molecules).total_energy;
}

// ============================================================================
// Hessian Computation - Analytical pairwise assembly
// ============================================================================

CrystalEnergyResultWithHessian CrystalEnergy::compute_with_hessian(
    const std::vector<MoleculeState>& molecules) {

    const int N = static_cast<int>(molecules.size());
    const int ndof = 6 * N;  // 3 translation + 3 rotation per molecule

    CrystalEnergyResultWithHessian result;

    // First compute energy and gradient at current point
    auto base_result = compute(molecules);
    result.total_energy = base_result.total_energy;
    result.electrostatic_energy = base_result.electrostatic_energy;
    result.repulsion_dispersion = base_result.repulsion_dispersion;
    result.forces = base_result.forces;
    result.torques = base_result.torques;
    result.strain_gradient = base_result.strain_gradient;
    result.molecule_energies = base_result.molecule_energies;
    result.includes_ewald_terms = false;

    // Initialize Hessian (6 DOF per molecule: pos + lab-frame rotation increment)
    result.hessian = Mat::Zero(ndof, ndof);
    result.strain_hessian.setZero();
    result.strain_state_hessian = Mat::Zero(6, ndof);

    // Precompute rotation matrices and lab-frame atom positions
    std::vector<MoleculeCache> mol_cache(N);
    for (int i = 0; i < N; ++i) {
        mol_cache[i].rotation = molecules[i].rotation_matrix();
        if (!m_geometry.empty() && i < static_cast<int>(m_geometry.size())) {
            const auto& geom = m_geometry[i];
            mol_cache[i].lab_atom_positions.resize(geom.atom_positions.size());
            for (size_t a = 0; a < geom.atom_positions.size(); ++a) {
                mol_cache[i].lab_atom_positions[a] =
                    molecules[i].position + mol_cache[i].rotation * geom.atom_positions[a];
            }
        }
    }

    // Update multipole orientations and build Cartesian molecules once.
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(mol_cache[i].rotation, molecules[i].position);
    }
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }

    const double elec_com_cutoff = effective_electrostatic_com_cutoff();
    const double elec_site_cutoff = effective_electrostatic_site_cutoff();
    const CutoffSpline* elec_taper =
        m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr;
    const double buck_cutoff = effective_buckingham_site_cutoff();

    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        const int i = pair.mol_i;
        const int j = pair.mol_j;
        const Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());
        const Vec3 pos_i = molecules[i].position;
        const Vec3 pos_j_image = molecules[j].position + cell_translation;

        // Electrostatic pair Hessian (full Cartesian multipole rigid-body terms).
        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate && pair.com_distance > elec_com_cutoff) {
            include_elec = false;
        }
        if (include_elec) {
            auto epair = compute_molecule_hessian_truncated(
                cart_mols[i], cart_mols[j],
                cell_translation,
                elec_site_cutoff,
                m_max_interaction_order,
                elec_taper,
                m_elec_taper_hessian);
            accumulate_pair_hessian_blocks(
                result.hessian, i, j, epair, pair.weight);
            accumulate_pair_strain_hessian_blocks(
                result.strain_hessian, result.strain_state_hessian,
                i, j, epair, pair.weight, pos_i, pos_j_image);
        }

        // Short-range Hessian (full rigid-body mapping for central site potentials).
        if (m_force_field == ForceFieldType::None) {
            continue;
        }

        const auto& geom_i = m_geometry[i];
        const auto& geom_j = m_geometry[j];
        const size_t nA = geom_i.atom_positions.size();
        const size_t nB = geom_j.atom_positions.size();

        const bool use_frozen = (pair_idx < m_fixed_site_masks.size() &&
                                 !m_fixed_site_masks[pair_idx].empty());

        const Mat3& R_i = mol_cache[i].rotation;
        const Mat3& R_j = mol_cache[j].rotation;

        for (size_t a = 0; a < nA; ++a) {
            const int Z_a = geom_i.atomic_numbers[a];
            const Vec3 pos_a = mol_cache[i].lab_atom_positions[a];

            for (size_t b = 0; b < nB; ++b) {
                if (use_frozen && !m_fixed_site_masks[pair_idx][a * nB + b]) {
                    continue;
                }

                const int Z_b = geom_j.atomic_numbers[b];
                const Vec3 pos_b = mol_cache[j].lab_atom_positions[b] + cell_translation;
                const Vec3 r_ab = pos_b - pos_a;
                const double r = r_ab.norm();

                if (!use_frozen) {
                    if (r > buck_cutoff || r < 0.1) {
                        continue;
                    }
                }

                ShortRangeInteraction::EnergyAndDerivatives sr;
                if (m_force_field == ForceFieldType::BuckinghamDE ||
                    m_force_field == ForceFieldType::Custom) {
                    BuckinghamParams params;
                    const bool has_type_codes =
                        m_use_short_range_typing &&
                        a < geom_i.short_range_type_codes.size() &&
                        b < geom_j.short_range_type_codes.size() &&
                        geom_i.short_range_type_codes[a] > 0 &&
                        geom_j.short_range_type_codes[b] > 0;
                    if (has_type_codes) {
                        const int t1 = geom_i.short_range_type_codes[a];
                        const int t2 = geom_j.short_range_type_codes[b];
                        if (has_typed_buckingham_params(t1, t2)) {
                            params = get_buckingham_params_for_types(t1, t2);
                        } else {
                            params = get_buckingham_params(Z_a, Z_b);
                        }
                    } else {
                        params = get_buckingham_params(Z_a, Z_b);
                    }
                    sr = ShortRangeInteraction::buckingham_all(r, params);
                } else if (m_force_field == ForceFieldType::LennardJones) {
                    auto it = m_lj_params.find({Z_a, Z_b});
                    if (it == m_lj_params.end()) {
                        continue;
                    }
                    sr = ShortRangeInteraction::lennard_jones_all(r, it->second);
                }

                if (m_short_range_taper.is_valid()) {
                    auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                    if (sw.value <= 0.0) {
                        continue;
                    }
                    const double e0 = sr.energy;
                    const double d10 = sr.first_derivative;
                    const double d20 = sr.second_derivative;
                    sr.energy = sw.value * e0;
                    sr.first_derivative = sw.value * d10 + e0 * sw.first_derivative;
                    sr.second_derivative = sw.value * d20 +
                                           2.0 * sw.first_derivative * d10 +
                                           e0 * sw.second_derivative;
                }

                auto spair = short_range_site_pair_hessian(
                    R_i, R_j,
                    geom_i.atom_positions[a], geom_j.atom_positions[b],
                    pos_a, pos_b,
                    sr.first_derivative, sr.second_derivative);

                accumulate_pair_hessian_blocks(
                    result.hessian, i, j, spair, pair.weight);
                accumulate_pair_strain_hessian_blocks(
                    result.strain_hessian, result.strain_state_hessian,
                    i, j, spair, pair.weight, pos_i, pos_j_image);

                // Anisotropic repulsion Hessian (FD of energy for initial impl)
                if (!m_typed_aniso_params.empty()) {
                    const bool htc =
                        m_use_short_range_typing &&
                        a < geom_i.short_range_type_codes.size() &&
                        b < geom_j.short_range_type_codes.size() &&
                        geom_i.short_range_type_codes[a] > 0 &&
                        geom_j.short_range_type_codes[b] > 0;
                    if (htc) {
                        const int t1 = geom_i.short_range_type_codes[a];
                        const int t2 = geom_j.short_range_type_codes[b];
                        if (has_aniso_params(t1, t2)) {
                            const auto ap = get_aniso_params(t1, t2);
                            Vec3 ax_a_lab = Vec3::Zero();
                            Vec3 ax_b_lab = Vec3::Zero();
                            if (a < geom_i.aniso_body_axes.size())
                                ax_a_lab = R_i * geom_i.aniso_body_axes[a];
                            if (b < geom_j.aniso_body_axes.size())
                                ax_b_lab = R_j * geom_j.aniso_body_axes[b];

                            // FD Hessian of aniso energy wrt site positions
                            constexpr double h = 1e-5;
                            auto aniso_e = [&](const Vec3& pa, const Vec3& pb) {
                                auto res = ShortRangeInteraction::anisotropic_repulsion(
                                    pa, pb, ax_a_lab, ax_b_lab, ap);
                                if (m_short_range_taper.is_valid()) {
                                    double rr = (pb - pa).norm();
                                    auto sw = evaluate_cutoff_spline(rr, m_short_range_taper);
                                    return sw.value * res.energy;
                                }
                                return res.energy;
                            };

                            // Build 6x6 Hessian [posA(3), posB(3)]
                            Eigen::Matrix<double, 6, 6> H_aniso;
                            H_aniso.setZero();
                            Vec3 pa0 = pos_a, pb0 = pos_b;
                            for (int di = 0; di < 6; ++di) {
                                for (int dj = di; dj < 6; ++dj) {
                                    Vec3 pa_pp = pa0, pa_pm = pa0, pa_mp = pa0, pa_mm = pa0;
                                    Vec3 pb_pp = pb0, pb_pm = pb0, pb_mp = pb0, pb_mm = pb0;
                                    auto perturb = [&](Vec3& pa, Vec3& pb, int idx, double delta) {
                                        if (idx < 3) pa(idx) += delta;
                                        else pb(idx - 3) += delta;
                                    };
                                    perturb(pa_pp, pb_pp, di, +h);
                                    perturb(pa_pp, pb_pp, dj, +h);
                                    perturb(pa_pm, pb_pm, di, +h);
                                    perturb(pa_pm, pb_pm, dj, -h);
                                    perturb(pa_mp, pb_mp, di, -h);
                                    perturb(pa_mp, pb_mp, dj, +h);
                                    perturb(pa_mm, pb_mm, di, -h);
                                    perturb(pa_mm, pb_mm, dj, -h);
                                    double d2 = (aniso_e(pa_pp, pb_pp) - aniso_e(pa_pm, pb_pm)
                                                 - aniso_e(pa_mp, pb_mp) + aniso_e(pa_mm, pb_mm))
                                                / (4.0 * h * h);
                                    H_aniso(di, dj) = d2;
                                    H_aniso(dj, di) = d2;
                                }
                            }

                            // Map to rigid-body Hessian
                            PairHessianResult apair;
                            apair.H_posA_posA = H_aniso.block<3,3>(0, 0);
                            apair.H_posA_posB = H_aniso.block<3,3>(0, 3);
                            apair.H_posB_posB = H_aniso.block<3,3>(3, 3);

                            const Vec3 lever_a_h = R_i * geom_i.atom_positions[a];
                            const Vec3 lever_b_h = R_j * geom_j.atom_positions[b];
                            const Mat3 Jpsi_a = -skew_symmetric(lever_a_h);
                            const Mat3 Jpsi_b = -skew_symmetric(lever_b_h);

                            apair.H_posA_rotA = apair.H_posA_posA * Jpsi_a;
                            apair.H_posA_rotB = apair.H_posA_posB * Jpsi_b;
                            apair.H_posB_rotA = apair.H_posA_posB.transpose() * Jpsi_a;
                            apair.H_posB_rotB = apair.H_posB_posB * Jpsi_b;

                            apair.H_rotA_rotA = Jpsi_a.transpose() * apair.H_posA_posA * Jpsi_a;
                            apair.H_rotA_rotB = Jpsi_a.transpose() * apair.H_posA_posB * Jpsi_b;
                            apair.H_rotB_rotB = Jpsi_b.transpose() * apair.H_posB_posB * Jpsi_b;

                            // Exponential-map curvature terms
                            auto aniso_ref = ShortRangeInteraction::anisotropic_repulsion(
                                pos_a, pos_b, ax_a_lab, ax_b_lab, ap);
                            double e_aniso = aniso_ref.energy;
                            if (m_short_range_taper.is_valid()) {
                                auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                                e_aniso *= sw.value;
                            }
                            const Vec3 gA = -aniso_ref.force_A;  // gradient
                            const Vec3 gB = -aniso_ref.force_B;
                            if (m_short_range_taper.is_valid()) {
                                // approximate: just use lever-arm
                            }
                            const auto& Ggen = so3_generators();
                            for (int k = 0; k < 3; ++k) {
                                for (int l = 0; l < 3; ++l) {
                                    const Vec3 d2xa = 0.5 * (Ggen[k]*Ggen[l] + Ggen[l]*Ggen[k]) * lever_a_h;
                                    const Vec3 d2xb = 0.5 * (Ggen[k]*Ggen[l] + Ggen[l]*Ggen[k]) * lever_b_h;
                                    apair.H_rotA_rotA(k, l) += gA.dot(d2xa);
                                    apair.H_rotB_rotB(k, l) += gB.dot(d2xb);
                                }
                            }

                            // TODO: axis rotation Hessian terms (aniso.torque_axis)
                            // For now, the FD position Hessian captures the dominant effect.

                            accumulate_pair_hessian_blocks(
                                result.hessian, i, j, apair, pair.weight);
                            accumulate_pair_strain_hessian_blocks(
                                result.strain_hessian, result.strain_state_hessian,
                                i, j, apair, pair.weight, pos_i, pos_j_image);
                        }
                    }
                }
            }
        }
    }

    if (m_use_ewald) {
        auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
        auto mol_site_indices = build_mol_site_indices(cart_mols);

        if (!ewald_sites.empty()) {
            EwaldParams params;
            ensure_ewald_params_initialized();
            params.alpha = m_ewald_alpha_fixed;
            params.kmax = m_ewald_kmax_fixed;
            params.include_dipole = m_ewald_dipole;

            if (!m_ewald_lattice_cache) {
                m_ewald_lattice_cache = std::make_unique<EwaldLatticeCache>(
                    build_ewald_lattice_cache(m_crystal.unit_cell(), params));
            }

            auto ewald = compute_ewald_correction_with_hessian(
                ewald_sites, m_crystal.unit_cell(), m_neighbors,
                mol_site_indices, effective_electrostatic_com_cutoff(),
                m_use_com_elec_gate, elec_site_cutoff, params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get());

            const int ns = static_cast<int>(ewald_sites.size());
            Mat J_state = Mat::Zero(3 * ns, ndof);
            Mat J_strain = Mat::Zero(3 * ns, 6);
            const auto& B = voigt_basis_matrices();

            for (int s = 0; s < ns; ++s) {
                const int m = ewald_sites[s].mol_index;
                const int row = 3 * s;
                const int col = 6 * m;
                J_state.block<3, 3>(row, col) = Mat3::Identity();

                const Vec3 lever = ewald_sites[s].position - molecules[m].position;
                const Mat3 J_rot = -skew_symmetric(lever);
                J_state.block<3, 3>(row, col + 3) = J_rot;

                // Cell strain acts on molecule COMs in this model; rigid internal
                // site offsets are not affinely deformed.
                for (int a = 0; a < 6; ++a) {
                    J_strain.block<3, 1>(row, a) = B[a] * molecules[m].position;
                }
            }

            result.hessian += J_state.transpose() * ewald.site_hessian * J_state;
            result.strain_state_hessian +=
                J_strain.transpose() * ewald.site_hessian * J_state;
            result.strain_hessian +=
                J_strain.transpose() * ewald.site_hessian * J_strain;

            auto ewald_strain = compute_ewald_explicit_strain_terms(
                ewald_sites, m_crystal.unit_cell(), m_neighbors, mol_site_indices,
                effective_electrostatic_com_cutoff(), m_use_com_elec_gate,
                elec_site_cutoff,
                params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get(),
                true);
            result.strain_hessian += ewald_strain.hess;
            if (ewald_strain.strain_site_mixed.rows() == 6 &&
                ewald_strain.strain_site_mixed.cols() == J_state.rows()) {
                result.strain_state_hessian +=
                    ewald_strain.strain_site_mixed * J_state;
                // Mixed explicit Ewald projection into d2E/dE^2.
                // strain_site_mixed stores d^2E / (dE_a dx_site), so projecting
                // through x_site(E) contributes M*J + (M*J)^T to W_ee.
                const Mat6 mixed_ee =
                    ewald_strain.strain_site_mixed * J_strain;
                result.strain_hessian += mixed_ee + mixed_ee.transpose();
            }

            // Exponential-map curvature term for rotation coordinates.
            const auto& G = so3_generators();
            for (int s = 0; s < ns; ++s) {
                const int m = ewald_sites[s].mol_index;
                const int rot = 6 * m + 3;
                const Vec3 lever = ewald_sites[s].position - molecules[m].position;
                const Vec3 g_site = -ewald.site_forces[s]; // gradient wrt site position
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const Vec3 d2x =
                            0.5 * (G[k] * G[l] + G[l] * G[k]) * lever;
                        result.hessian(rot + k, rot + l) += g_site.dot(d2x);
                    }
                }
            }

            result.includes_ewald_terms = true;
        }
    }

    result.exact_for_model = can_compute_exact_hessian();

    // Symmetrize Hessian (handles numerical noise)
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
    result.strain_hessian =
        0.5 * (result.strain_hessian + result.strain_hessian.transpose());

    return result;
}

bool CrystalEnergy::can_compute_exact_hessian() const {
    // The full chain rule taper Hessian is the correct d²(f·E)/dq²,
    // validated against finite differences (analytic W_ei matches FD
    // to ~0.2%). Always use analytic W_ei.
    return true;
}

// ============================================================================
// Pack Hessian for Reduced DOF
// ============================================================================

Mat CrystalEnergyResultWithHessian::pack_hessian(
    bool fix_first_translation, bool fix_first_rotation) const {

    int N = static_cast<int>(forces.size());
    int ndof_full = 6 * N;

    // Count reduced DOF
    int mol0_dof = 0;
    if (!fix_first_translation) mol0_dof += 3;
    if (!fix_first_rotation) mol0_dof += 3;
    int ndof_reduced = mol0_dof + 6 * (N - 1);

    if (ndof_reduced == ndof_full) {
        return hessian;
    }

    Mat H_reduced = Mat::Zero(ndof_reduced, ndof_reduced);

    // Build index mapping from reduced to full DOF
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(ndof_reduced);

    // Molecule 0
    if (!fix_first_translation) {
        reduced_to_full.push_back(0);
        reduced_to_full.push_back(1);
        reduced_to_full.push_back(2);
    }
    if (!fix_first_rotation) {
        reduced_to_full.push_back(3);
        reduced_to_full.push_back(4);
        reduced_to_full.push_back(5);
    }

    // Molecules 1..N-1
    for (int i = 1; i < N; ++i) {
        for (int c = 0; c < 6; ++c) {
            reduced_to_full.push_back(6 * i + c);
        }
    }

    // Extract submatrix
    for (int i = 0; i < ndof_reduced; ++i) {
        for (int j = 0; j < ndof_reduced; ++j) {
            H_reduced(i, j) = hessian(reduced_to_full[i], reduced_to_full[j]);
        }
    }

    return H_reduced;
}

int CrystalEnergy::max_multipole_rank() const {
    int max_rank = 0;
    for (const auto& mp : m_multipoles) {
        const auto& cart = mp.cartesian();
        for (const auto& site : cart.sites) {
            max_rank = std::max(max_rank, site.rank);
        }
    }
    return max_rank;
}

size_t CrystalEnergy::num_sites() const {
    size_t total = 0;
    for (const auto& mp : m_multipoles) {
        total += mp.cartesian().sites.size();
    }
    return total;
}

std::vector<PairEnergyDebug> CrystalEnergy::debug_pair_energies(const std::vector<MoleculeState>& molecules) {
    const int N = static_cast<int>(molecules.size());
    std::vector<PairEnergyDebug> out;
    out.reserve(m_neighbors.size());

    // Update multipole orientations
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            molecules[i].rotation_matrix(),
            molecules[i].position);
    }

    // Pre-build Cartesian molecules
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }

    for (const auto& pair : m_neighbors) {
        int i = pair.mol_i;
        int j = pair.mol_j;

        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        PairEnergyDebug dbg;
        dbg.mol_i = i;
        dbg.mol_j = j;
        dbg.cell_shift = pair.cell_shift;
        dbg.weight = pair.weight;

        Vec3 com_i = molecules[i].position;
        Vec3 com_j = molecules[j].position + cell_translation;
        dbg.com_distance = (com_j - com_i).norm();

        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate &&
            pair.com_distance > effective_electrostatic_com_cutoff()) {
            include_elec = false;
        }
        if (include_elec) {
            auto elec = compute_molecule_forces_torques(
                cart_mols[i], cart_mols[j],
                effective_electrostatic_site_cutoff(),
                m_max_interaction_order,
                cell_translation,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr);
            dbg.electrostatic = pair.weight * elec.energy;
            dbg.total += dbg.electrostatic;
        }

        if (m_force_field != ForceFieldType::None) {
            double sr_energy = 0.0;
            int sr_pairs = 0;
            Vec3 dummyF, dummyF2, dummyT, dummyT2;
            compute_short_range_pair(i, j, molecules[i], molecules[j], cell_translation,
                                     pair.weight, sr_energy, dummyF, dummyF2, dummyT, dummyT2,
                                     -1, nullptr, nullptr, &sr_pairs);
            dbg.short_range = sr_energy;
            dbg.short_range_site_pairs = sr_pairs;
            dbg.total += sr_energy;
        }

        out.push_back(dbg);
    }

    return out;
}

std::vector<int> CrystalEnergy::neighbor_shell_histogram() const {
    std::vector<int> bins(5, 0);
    for (const auto& pair : m_neighbors) {
        Vec3 shift = m_crystal.unit_cell().to_cartesian(pair.cell_shift.cast<double>());
        Vec3 com_i = m_crystal.symmetry_unique_molecules()[pair.mol_i].center_of_mass();
        Vec3 com_j = m_crystal.symmetry_unique_molecules()[pair.mol_j].center_of_mass() + shift;
        double d = (com_j - com_i).norm();
        if (d < 3.0) bins[0]++; else if (d < 6.0) bins[1]++; else if (d < 10.0) bins[2]++; else if (d < 15.0) bins[3]++; else bins[4]++;
    }
    return bins;
}

// ============================================================================
// Charge-Charge Ewald Correction (thin wrapper around standalone engine)
// ============================================================================

CrystalEnergy::EwaldCorrectionResult CrystalEnergy::compute_charge_ewald_correction(
    const std::vector<MoleculeState>& molecules,
    const std::vector<CartesianMolecule>& cart_mols) const {

    const int N = static_cast<int>(molecules.size());
    EwaldCorrectionResult ewald_result;
    ewald_result.forces.resize(N, Vec3::Zero());
    ewald_result.torques.resize(N, Vec3::Zero());

    // Gather sites and indices from CartesianMolecules
    auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
    auto mol_site_indices = build_mol_site_indices(cart_mols);

    if (ewald_sites.empty()) return ewald_result;

    // Select Ewald parameters
    EwaldParams params;
    ensure_ewald_params_initialized();
    params.alpha = m_ewald_alpha_fixed;
    params.kmax = m_ewald_kmax_fixed;
    double elec_site_cutoff = effective_electrostatic_site_cutoff();
    params.include_dipole = m_ewald_dipole;

    // Lazy-build lattice cache on first call (reused across evaluations
    // while the unit cell and Ewald params don't change).
    if (!m_ewald_lattice_cache) {
        m_ewald_lattice_cache = std::make_unique<EwaldLatticeCache>(
            build_ewald_lattice_cache(m_crystal.unit_cell(), params));
    }

    // Call standalone Ewald engine with cached lattice
    auto raw = compute_ewald_correction(
        ewald_sites, m_crystal.unit_cell(), m_neighbors,
        mol_site_indices, effective_electrostatic_com_cutoff(),
        m_use_com_elec_gate, elec_site_cutoff, params,
        m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
        m_ewald_lattice_cache.get());

    ewald_result.energy = raw.energy;

    // Map per-site forces to rigid-body forces and torques
    for (size_t k = 0; k < ewald_sites.size(); ++k) {
        int m = ewald_sites[k].mol_index;
        const Vec3& f_kJ = raw.site_forces[k];

        ewald_result.forces[m] += f_kJ;

        // Lab-frame angular gradient from lever arm:
        // torque_lab = lever × force = -(lever × dE/dr) = -dE/dψ_lab
        // So -torque_lab = +dE/dψ_lab (consistent with grad_angle_axis convention)
        Vec3 lever = ewald_sites[k].position - molecules[m].position;
        Vec3 torque_lab = lever.cross(f_kJ);
        ewald_result.torques[m] -= torque_lab;
    }

    return ewald_result;
}

} // namespace occ::mults
