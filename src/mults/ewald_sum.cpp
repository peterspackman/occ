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

EwaldLatticeCache build_ewald_lattice_cache(
    const crystal::UnitCell& unit_cell,
    const EwaldParams& params) {
    EwaldLatticeCache cache;

    double alpha_bohr = params.alpha * occ::units::BOHR_TO_ANGSTROM;
    cache.alpha_bohr = alpha_bohr;
    cache.two_alpha_over_sqrt_pi = 2.0 * alpha_bohr / std::sqrt(M_PI);

    Mat3 A_bohr = unit_cell.direct() * occ::units::ANGSTROM_TO_BOHR;
    double volume_bohr = unit_cell.volume() *
        std::pow(occ::units::ANGSTROM_TO_BOHR, 3);
    Mat3 B_bohr = 2.0 * M_PI * A_bohr.inverse().transpose();
    cache.four_pi_over_vol = 4.0 * M_PI / volume_bohr;

    double inv_4alpha2 = 1.0 / (4.0 * alpha_bohr * alpha_bohr);
    int kmax = params.kmax;

    cache.g_vectors.reserve((2*kmax+1)*(2*kmax+1)*(2*kmax+1) - 1);
    for (int hx = -kmax; hx <= kmax; ++hx) {
        for (int hy = -kmax; hy <= kmax; ++hy) {
            for (int hz = -kmax; hz <= kmax; ++hz) {
                if (hx == 0 && hy == 0 && hz == 0) continue;
                Vec3 G = B_bohr * Vec3(hx, hy, hz);
                double G2 = G.squaredNorm();
                double coeff = std::exp(-G2 * inv_4alpha2) / G2;
                // Keep very small reciprocal terms so kmax behaves as requested
                // and high-accuracy parity runs (e.g. DMACRYS benchmarks) can
                // converge cleanly.
                if (coeff < 1e-20) continue;
                cache.g_vectors.push_back({G, coeff});
            }
        }
    }

    return cache;
}

EwaldResult compute_ewald_correction(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params,
    const CutoffSpline* taper,
    const EwaldLatticeCache* lattice_cache) {
    auto full = compute_ewald_correction_with_hessian(
        sites, unit_cell, neighbors, mol_site_indices,
        cutoff_radius, use_com_gate, elec_site_cutoff,
        params, taper, lattice_cache);
    EwaldResult out;
    out.energy = full.energy;
    out.site_forces = std::move(full.site_forces);
    return out;
}

EwaldResultWithHessian compute_ewald_correction_with_hessian(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params,
    const CutoffSpline* taper,
    const EwaldLatticeCache* lattice_cache) {

    const int N = static_cast<int>(mol_site_indices.size());
    const size_t n_sites = sites.size();

    EwaldResultWithHessian result;
    result.site_forces.resize(n_sites, Vec3::Zero());
    result.site_hessian = Mat::Zero(3 * static_cast<int>(n_sites),
                                    3 * static_cast<int>(n_sites));
    if (params.include_dipole) {
        result.site_dipole_gradients = Mat::Zero(static_cast<int>(n_sites), 3);
    }

    if (sites.empty()) return result;

    const double alpha = params.alpha;
    const int kmax = params.kmax;

    const double alpha_bohr = lattice_cache ? lattice_cache->alpha_bohr
                                            : alpha * occ::units::BOHR_TO_ANGSTROM;
    const double two_alpha_over_sqrt_pi = lattice_cache
        ? lattice_cache->two_alpha_over_sqrt_pi
        : 2.0 * alpha_bohr / std::sqrt(M_PI);

    occ::log::debug("Ewald correction: alpha = {:.4f} /Ang ({:.6f} /Bohr), kmax = {}",
                    alpha, alpha_bohr, kmax);

    struct InternalSite {
        Vec3 pos_bohr;
        double charge;
        Vec3 dipole;
    };
    std::vector<InternalSite> isites(n_sites);
    for (size_t i = 0; i < n_sites; ++i) {
        isites[i].pos_bohr = sites[i].position * occ::units::ANGSTROM_TO_BOHR;
        isites[i].charge = sites[i].charge;
        isites[i].dipole = sites[i].dipole;
    }

    std::vector<Vec3> forces_ha_bohr(n_sites, Vec3::Zero());
    Mat dipole_grad_ha(static_cast<int>(n_sites), 3);
    dipole_grad_ha.setZero();
    Mat hess_ha_bohr2 = Mat::Zero(3 * static_cast<int>(n_sites),
                                  3 * static_cast<int>(n_sites));
    double energy_ha = 0.0;

    auto add_hessian_block = [&](size_t i, size_t j, const Mat3& block) {
        hess_ha_bohr2.block<3, 3>(3 * static_cast<int>(i),
                                  3 * static_cast<int>(j)) += block;
    };
    auto erf_pair_correction = [&](const InternalSite& sa, const InternalSite& sb,
                                   const Vec3& R_vec,
                                   Vec3& f_a, Vec3& f_b,
                                   Mat3& H_aa) {
        struct Jet2 {
            double v = 0.0;
            Vec3 g = Vec3::Zero();
            Mat3 h = Mat3::Zero();
        };

        const auto make_const = [](double c) {
            Jet2 out;
            out.v = c;
            return out;
        };
        const auto make_var = [](double x, int idx) {
            Jet2 out;
            out.v = x;
            out.g[idx] = 1.0;
            return out;
        };
        const auto outer = [](const Vec3& a, const Vec3& b) {
            return a * b.transpose();
        };
        const auto add = [&](const Jet2& a, const Jet2& b) {
            Jet2 out;
            out.v = a.v + b.v;
            out.g = a.g + b.g;
            out.h = a.h + b.h;
            return out;
        };
        const auto sub = [&](const Jet2& a, const Jet2& b) {
            Jet2 out;
            out.v = a.v - b.v;
            out.g = a.g - b.g;
            out.h = a.h - b.h;
            return out;
        };
        const auto mul = [&](const Jet2& a, const Jet2& b) {
            Jet2 out;
            out.v = a.v * b.v;
            out.g = a.g * b.v + b.g * a.v;
            out.h = a.h * b.v + b.h * a.v + outer(a.g, b.g) + outer(b.g, a.g);
            return out;
        };
        const auto muls = [&](const Jet2& a, double s) {
            Jet2 out;
            out.v = a.v * s;
            out.g = a.g * s;
            out.h = a.h * s;
            return out;
        };
        const auto inv = [&](const Jet2& a) {
            Jet2 out;
            const double inv_v = 1.0 / a.v;
            const double inv_v2 = inv_v * inv_v;
            out.v = inv_v;
            out.g = -a.g * inv_v2;
            out.h = 2.0 * outer(a.g, a.g) * (inv_v2 * inv_v) - a.h * inv_v2;
            return out;
        };
        const auto div = [&](const Jet2& a, const Jet2& b) {
            return mul(a, inv(b));
        };
        const auto sqrtj = [&](const Jet2& a) {
            Jet2 out;
            const double s = std::sqrt(a.v);
            const double fp = 0.5 / s;
            const double fpp = -0.25 / (a.v * s);
            out.v = s;
            out.g = fp * a.g;
            out.h = fpp * outer(a.g, a.g) + fp * a.h;
            return out;
        };
        const auto expj = [&](const Jet2& a) {
            Jet2 out;
            const double e = std::exp(a.v);
            out.v = e;
            out.g = e * a.g;
            out.h = e * (a.h + outer(a.g, a.g));
            return out;
        };
        const auto erfj = [&](const Jet2& a) {
            Jet2 out;
            const double ev = std::erf(a.v);
            const double fp = 2.0 / std::sqrt(M_PI) * std::exp(-a.v * a.v);
            const double fpp = -2.0 * a.v * fp;
            out.v = ev;
            out.g = fp * a.g;
            out.h = fpp * outer(a.g, a.g) + fp * a.h;
            return out;
        };

        if (R_vec.norm() < 1e-10) return 0.0;

        const Jet2 X = make_var(R_vec[0], 0);
        const Jet2 Y = make_var(R_vec[1], 1);
        const Jet2 Z = make_var(R_vec[2], 2);

        const Jet2 r2 = add(add(mul(X, X), mul(Y, Y)), mul(Z, Z));
        const Jet2 r = sqrtj(r2);
        const Jet2 inv_r = inv(r);
        const Jet2 inv_r2 = inv(r2);
        const Jet2 inv_r3 = mul(inv_r2, inv_r);
        const Jet2 inv_r4 = mul(inv_r2, inv_r2);
        const Jet2 inv_r5 = mul(inv_r4, inv_r);

        const Jet2 ar = muls(r, alpha_bohr);
        const Jet2 erf_ar = erfj(ar);
        const Jet2 ar2 = mul(ar, ar);
        const Jet2 exp_minus_ar2 = expj(muls(ar2, -1.0));

        const double qa = sa.charge;
        const double qb = sb.charge;

        Jet2 E = muls(mul(erf_ar, inv_r), -qa * qb);

        if (params.include_dipole) {
            const Vec3& da = sa.dipole;
            const Vec3& db = sb.dipole;
            const Jet2 muA_dot_R = add(add(muls(X, da[0]), muls(Y, da[1])),
                                       muls(Z, da[2]));
            const Jet2 muB_dot_R = add(add(muls(X, db[0]), muls(Y, db[1])),
                                       muls(Z, db[2]));

            const Jet2 f1 = sub(
                muls(mul(exp_minus_ar2, inv_r2), two_alpha_over_sqrt_pi),
                mul(erf_ar, inv_r3));
            const Jet2 qmu_lin = sub(muls(muB_dot_R, qa), muls(muA_dot_R, qb));
            const Jet2 E_qmu = muls(mul(qmu_lin, f1), -1.0);

            const double a2 = alpha_bohr * alpha_bohr;
            const Jet2 t_inner = add(muls(inv_r2, 2.0 * a2), muls(inv_r4, 3.0));
            const Jet2 Ap_over_r = add(
                muls(mul(exp_minus_ar2, t_inner), -two_alpha_over_sqrt_pi),
                muls(mul(erf_ar, inv_r5), 3.0));
            const Jet2 E_mumu = add(
                muls(f1, da.dot(db)),
                mul(mul(muA_dot_R, muB_dot_R), Ap_over_r));

            E = add(E, add(E_qmu, E_mumu));
        }

        f_a += E.g;
        f_b -= E.g;
        H_aa += 0.5 * (E.h + E.h.transpose());
        return E.v;
    };

    // Analytical dipole gradient for the erf pair correction.
    // dE_erf/d(mu_a) and dE_erf/d(mu_b) from the screened kernel.
    auto erf_pair_dipole_grad = [&](const InternalSite& sa, const InternalSite& sb,
                                     const Vec3& R_vec,
                                     Vec3& dE_dmu_a, Vec3& dE_dmu_b) {
        if (!params.include_dipole) return;
        const double r2 = R_vec.squaredNorm();
        if (r2 < 1e-20) return;
        const double r = std::sqrt(r2);
        const double kr = alpha_bohr * r;
        const double erf_kr = std::erf(kr);
        const double exp_kr2 = std::exp(-kr * kr);
        const double inv_r = 1.0 / r;
        const double inv_r2 = inv_r * inv_r;
        const double inv_r3 = inv_r2 * inv_r;
        const double inv_r4 = inv_r2 * inv_r2;
        const double inv_r5 = inv_r4 * inv_r;

        // b = phi'(r)/r where phi(r) = erf(kr)/r
        const double b = two_alpha_over_sqrt_pi * exp_kr2 * inv_r2
                       - erf_kr * inv_r3;
        // c = Hessian radial coefficient
        const double c = -4.0 * alpha_bohr * alpha_bohr * alpha_bohr
                            / std::sqrt(M_PI) * exp_kr2 * inv_r2
                       - 6.0 * alpha_bohr / std::sqrt(M_PI) * exp_kr2 * inv_r4
                       + 3.0 * erf_kr * inv_r5;

        const double qa = sa.charge, qb = sb.charge;
        const Vec3& da = sa.dipole;
        const Vec3& db = sb.dipole;
        const double miR = da.dot(R_vec);
        const double mjR = db.dot(R_vec);

        // From E_qd = b*(qa*mjR - qb*miR):
        //   dE/d(mu_a) = -b*qb*R
        //   dE/d(mu_b) = b*qa*R
        // From E_dd = -(c*miR*mjR + b*da.db):
        //   dE/d(mu_a) = -(c*R*mjR + b*db)
        //   dE/d(mu_b) = -(c*miR*R + b*da)
        // Note: erf kernel has NEGATIVE sign convention vs erfc kernel
        // E_erf = -E_erfc_equivalent, so signs flip from the erfc formula.
        // The b,c above use erf convention directly.
        dE_dmu_a += -b * qb * R_vec - (c * mjR * R_vec + b * db);
        dE_dmu_b += b * qa * R_vec - (c * miR * R_vec + b * da);
    };

    const bool use_taper = (taper != nullptr && taper->is_valid());
    const bool use_erf_site_cutoff = (elec_site_cutoff > 0.0);
    const double erf_site_cutoff_bohr =
        elec_site_cutoff * occ::units::ANGSTROM_TO_BOHR;

    auto accumulate_real_pair = [&](size_t ai, size_t bj,
                                    const Vec3& R_vec,
                                    double weight) {
        const double r_bohr = R_vec.norm();
        if (r_bohr < 1e-12) return;
        if (use_erf_site_cutoff && r_bohr > erf_site_cutoff_bohr) return;

        Vec3 f_a = Vec3::Zero();
        Vec3 f_b = Vec3::Zero();
        Mat3 H_aa = Mat3::Zero();
        const double e0 = erf_pair_correction(
            isites[ai], isites[bj], R_vec, f_a, f_b, H_aa);

        Vec3 F_a = f_a;
        Vec3 F_b = f_b;
        Mat3 J = H_aa;
        double e_scale = 1.0;

        if (use_taper) {
            const double r_ang = r_bohr * occ::units::BOHR_TO_ANGSTROM;
            const CutoffSplineValue sw = evaluate_cutoff_spline(r_ang, *taper);
            if (sw.value <= 0.0) return;

            e_scale = sw.value;
            F_a = sw.value * f_a;
            F_b = sw.value * f_b;
            J = sw.value * H_aa;

            if (std::abs(sw.first_derivative) > 0.0) {
                const double dwdR_bohr =
                    sw.first_derivative * occ::units::BOHR_TO_ANGSTROM;
                const double d2wdR2_bohr2 =
                    sw.second_derivative *
                    occ::units::BOHR_TO_ANGSTROM *
                    occ::units::BOHR_TO_ANGSTROM;
                const Vec3 u = R_vec / r_bohr;
                const Mat3 uuT = u * u.transpose();
                const Mat3 I = Mat3::Identity();

                const Vec3 extra = e0 * dwdR_bohr * u;
                F_a += extra;
                F_b -= extra;

                J += dwdR_bohr *
                     (u * f_a.transpose() + f_a * u.transpose());
                J += e0 *
                     (d2wdR2_bohr2 * uuT +
                      (dwdR_bohr / r_bohr) * (I - uuT));
            }
        }

        energy_ha += weight * e_scale * e0;
        forces_ha_bohr[ai] += weight * F_a;
        forces_ha_bohr[bj] += weight * F_b;

        add_hessian_block(ai, ai, weight * J);
        add_hessian_block(ai, bj, -weight * J);
        add_hessian_block(bj, ai, -weight * J);
        add_hessian_block(bj, bj, weight * J);

        // Dipole gradient for this pair (analytical, no taper derivative needed
        // because taper depends on distance not dipoles)
        if (params.include_dipole) {
            Vec3 dE_dmu_a = Vec3::Zero(), dE_dmu_b = Vec3::Zero();
            erf_pair_dipole_grad(isites[ai], isites[bj], R_vec,
                                  dE_dmu_a, dE_dmu_b);
            dipole_grad_ha.row(static_cast<int>(ai)) +=
                (weight * e_scale * dE_dmu_a).transpose();
            dipole_grad_ha.row(static_cast<int>(bj)) +=
                (weight * e_scale * dE_dmu_b).transpose();
        }
    };

    for (const auto& pair : neighbors) {
        if (use_com_gate && pair.com_distance > cutoff_radius) continue;

        const int mi = pair.mol_i;
        const int mj = pair.mol_j;
        const Vec3 cell_trans_bohr = unit_cell.to_cartesian(
            pair.cell_shift.cast<double>()) * occ::units::ANGSTROM_TO_BOHR;

        for (size_t ai : mol_site_indices[mi]) {
            for (size_t bj : mol_site_indices[mj]) {
                const Vec3 R_vec = isites[bj].pos_bohr + cell_trans_bohr -
                                   isites[ai].pos_bohr;
                accumulate_real_pair(ai, bj, R_vec, pair.weight);
            }
        }
    }

    for (int m = 0; m < N; ++m) {
        const auto& indices = mol_site_indices[m];
        for (size_t ii = 0; ii < indices.size(); ++ii) {
            const size_t ai = indices[ii];
            for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
                const size_t bj = indices[jj];
                const Vec3 R_vec = isites[bj].pos_bohr - isites[ai].pos_bohr;
                accumulate_real_pair(ai, bj, R_vec, 1.0);
            }
        }
    }

    double four_pi_over_vol;
    if (lattice_cache) {
        four_pi_over_vol = lattice_cache->four_pi_over_vol;
    } else {
        Mat3 A_bohr = unit_cell.direct() * occ::units::ANGSTROM_TO_BOHR;
        const double volume_bohr = unit_cell.volume() *
            std::pow(occ::units::ANGSTROM_TO_BOHR, 3);
        four_pi_over_vol = 4.0 * M_PI / volume_bohr;
    }

    std::vector<double> cos_phase(n_sites), sin_phase(n_sites), mu_dot_G(n_sites);
    std::vector<double> dP_dphi(n_sites), dQ_dphi(n_sites);

    auto process_g_vector = [&](const Vec3& G, double coeff) {
        double Sq_re = 0.0;
        double Sq_im = 0.0;
        double Smu_re = 0.0;
        double Smu_im = 0.0;

        for (size_t k = 0; k < n_sites; ++k) {
            const double phase = G.dot(isites[k].pos_bohr);
#if defined(__APPLE__) && defined(__arm64__)
            __sincos(phase, &sin_phase[k], &cos_phase[k]);
#else
            sin_phase[k] = std::sin(phase);
            cos_phase[k] = std::cos(phase);
#endif
            Sq_re += isites[k].charge * cos_phase[k];
            Sq_im += isites[k].charge * sin_phase[k];
            mu_dot_G[k] = params.include_dipole ? isites[k].dipole.dot(G) : 0.0;
            if (params.include_dipole) {
                Smu_re += mu_dot_G[k] * cos_phase[k];
                Smu_im += mu_dot_G[k] * sin_phase[k];
            }
        }

        const double qq_recip = Sq_re * Sq_re + Sq_im * Sq_im;
        const double qmu_cross = params.include_dipole
            ? -2.0 * (Sq_re * Smu_im - Sq_im * Smu_re)
            : 0.0;
        const double mumu_recip = params.include_dipole
            ? Smu_re * Smu_re + Smu_im * Smu_im
            : 0.0;
        const double pref_energy = 0.5 * four_pi_over_vol * coeff;
        energy_ha += pref_energy * (qq_recip + qmu_cross + mumu_recip);

        const double P = Sq_re - Smu_im;
        const double Q = Sq_im + Smu_re;
        const double pref_force = four_pi_over_vol * coeff;
        const Mat3 GGT = G * G.transpose();

        for (size_t j = 0; j < n_sites; ++j) {
            const double qj = isites[j].charge;
            const double mj = mu_dot_G[j];
            dP_dphi[j] = -qj * sin_phase[j] - mj * cos_phase[j];
            dQ_dphi[j] =  qj * cos_phase[j] - mj * sin_phase[j];
        }

        for (size_t i = 0; i < n_sites; ++i) {
            const double qi = isites[i].charge;
            const double mi = mu_dot_G[i];
            const double si = sin_phase[i];
            const double ci = cos_phase[i];

            const double A = qi * (P * si - Q * ci) +
                             mi * (P * ci + Q * si);
            forces_ha_bohr[i] += pref_force * A * G;

            // Reciprocal dipole gradient:
            // dE/d(mu_i_d) = pref_force * G_d * (Q*cos - P*sin)
            if (params.include_dipole) {
                const double dmu_coeff = pref_force * (Q * ci - P * si);
                dipole_grad_ha.row(static_cast<int>(i)) +=
                    (dmu_coeff * G).transpose();
            }

            for (size_t j = 0; j < n_sites; ++j) {
                const bool delta = (i == j);
                const double dB = dP_dphi[j] * si +
                                  (delta ? P * ci : 0.0) -
                                  dQ_dphi[j] * ci +
                                  (delta ? Q * si : 0.0);
                const double dC = dP_dphi[j] * ci -
                                  (delta ? P * si : 0.0) +
                                  dQ_dphi[j] * si +
                                  (delta ? Q * ci : 0.0);
                const double dA = qi * dB + mi * dC;
                const Mat3 Hij = -pref_force * dA * GGT;
                add_hessian_block(i, j, Hij);
            }
        }
    };

    if (lattice_cache) {
        for (const auto& gv : lattice_cache->g_vectors) {
            process_g_vector(gv.G, gv.coeff);
        }
    } else {
        Mat3 A_bohr = unit_cell.direct() * occ::units::ANGSTROM_TO_BOHR;
        Mat3 B_bohr = 2.0 * M_PI * A_bohr.inverse().transpose();
        const double inv_4alpha2 = 1.0 / (4.0 * alpha_bohr * alpha_bohr);
        for (int hx = -kmax; hx <= kmax; ++hx) {
            for (int hy = -kmax; hy <= kmax; ++hy) {
                for (int hz = -kmax; hz <= kmax; ++hz) {
                    if (hx == 0 && hy == 0 && hz == 0) continue;
                    const Vec3 G = B_bohr * Vec3(hx, hy, hz);
                    const double G2 = G.squaredNorm();
                    const double coeff = std::exp(-G2 * inv_4alpha2) / G2;
                    if (coeff < 1e-12) continue;
                    process_g_vector(G, coeff);
                }
            }
        }
    }

    const double alpha3 = alpha_bohr * alpha_bohr * alpha_bohr;
    const double self_dipole_coeff = 2.0 * alpha3 / (3.0 * std::sqrt(M_PI));
    for (size_t k = 0; k < n_sites; ++k) {
        energy_ha -= (alpha_bohr / std::sqrt(M_PI)) *
            isites[k].charge * isites[k].charge;
        if (params.include_dipole) {
            energy_ha -= self_dipole_coeff * isites[k].dipole.squaredNorm();
            // Self dipole gradient: dE_self/d(mu_i) = -2*coeff*mu_i
            dipole_grad_ha.row(static_cast<int>(k)) -=
                (2.0 * self_dipole_coeff * isites[k].dipole).transpose();
        }
    }

    result.energy = energy_ha * occ::units::AU_TO_KJ_PER_MOL;
    const double force_conv =
        occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;
    const double hessian_conv =
        occ::units::AU_TO_KJ_PER_MOL /
        (occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM);

    for (size_t k = 0; k < n_sites; ++k) {
        result.site_forces[k] = forces_ha_bohr[k] * force_conv;
    }
    result.site_hessian = hess_ha_bohr2 * hessian_conv;

    // Dipole gradient: convert from Hartree/(e*Bohr) to kJ/mol/(e*Bohr)
    if (params.include_dipole) {
        result.site_dipole_gradients = dipole_grad_ha * occ::units::AU_TO_KJ_PER_MOL;
    }

    occ::log::debug("Ewald correction: {:.4f} kJ/mol ({} sites, {} neighbor pairs)",
                    result.energy, sites.size(), neighbors.size());

    return result;
}

} // namespace occ::mults
