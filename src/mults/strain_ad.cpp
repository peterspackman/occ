#include <occ/mults/strain_ad.h>
#include <occ/mults/crystal_energy.h>
#include <occ/core/units.h>
#include <array>
#include <cmath>

namespace occ::mults {

namespace {

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

} // anonymous namespace

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
    bool include_strain_state) {

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

} // namespace occ::mults
