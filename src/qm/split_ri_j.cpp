#include "split_ri_j_kernels.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/ints/boys.h>
#include <occ/ints/kernels.h>
#include <occ/qm/coulomb_metric.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/split_ri_j.h>
#include <stdexcept>

namespace occ::qm {

namespace {

using ints::AuxShellData;
using ints::Boys;
using ints::hermite_index;
using ints::ncart;
using ints::nhermsum;
using ints::ShellPairData;

inline int nsph(int l) { return 2 * l + 1; }

// Cartesian -> spherical for the E-matrices of a shell pair: rows ab become
// rows of kron(T_a, T_b).
void transform_shell_pair_to_spherical(ShellPairData<double> &data, int la,
                                       int lb) {
  const Mat T_a = gto::cartesian_to_spherical_transformation_matrix(la);
  const Mat T_b = gto::cartesian_to_spherical_transformation_matrix(lb);
  const int nsph_a = nsph(la), nsph_b = nsph(lb);
  const int ncart_a = ncart(la), ncart_b = ncart(lb);
  MatRM T_ab(nsph_a * nsph_b, ncart_a * ncart_b);
  for (int i = 0; i < nsph_a; ++i)
    for (int j = 0; j < nsph_b; ++j)
      for (int k = 0; k < ncart_a; ++k)
        for (int l = 0; l < ncart_b; ++l)
          T_ab(i * nsph_b + j, k * ncart_b + l) = T_a(i, k) * T_b(j, l);

  const int nab_cart = ncart_a * ncart_b, nab_sph = nsph_a * nsph_b;
  const int nherm = data.nherm();
  for (auto &prim : data.primitives) {
    const MatRM E_sph =
        T_ab * Eigen::Map<MatRM>(prim.E_matrix.data(), nab_cart, nherm);
    prim.E_matrix.resize(nab_sph * nherm);
    prim.nab = nab_sph;
    Eigen::Map<MatRM>(prim.E_matrix.data(), nab_sph, nherm) = E_sph;
  }
}

void transform_aux_shell_to_spherical(AuxShellData<double> &data) {
  const int lc = data.lc;
  const int nc_cart = ncart(lc), nc_sph = nsph(lc);
  const int nherm = data.nherm();
  const Mat T_c = gto::cartesian_to_spherical_transformation_matrix(lc);
  for (auto &prim : data.primitives) {
    const MatRM E_sph =
        T_c * Eigen::Map<MatRM>(prim.E_matrix.data(), nc_cart, nherm);
    prim.E_matrix.resize(nc_sph * nherm);
    prim.nc = nc_sph;
    Eigen::Map<MatRM>(prim.E_matrix.data(), nc_sph, nherm) = E_sph;
  }
}

struct HermiteTUV {
  int8_t t, u, v;
};

std::vector<HermiteTUV> build_hermite_tuv_table(int L_max) {
  std::vector<HermiteTUV> table(nhermsum(L_max));
  for (int L = 0; L <= L_max; ++L)
    for (int t = 0; t <= L; ++t)
      for (int u = 0; u <= L - t; ++u) {
        const int v = L - t - u;
        table[hermite_index(t, u, v)] = {static_cast<int8_t>(t),
                                         static_cast<int8_t>(u),
                                         static_cast<int8_t>(v)};
      }
  return table;
}

// 1D Hermite expansion coefficients E(i, j, t), i <= imax, j <= jmax, of a
// primitive pair (exponents a, b, separation XAB = A_x - B_x).
struct E1D {
  // imax <= 5, jmax <= 4 (AO l <= 4, one up for the derivative)
  static constexpr int MAXI = 6, MAXJ = 5, MAXT = 10;
  int imax, jmax, tmax;
  std::array<double, MAXI * MAXJ * MAXT> e{};
  double operator()(int i, int j, int t) const {
    if (i < 0 || t < 0 || t > i + j)
      return 0.0;
    return e[(i * (jmax + 1) + j) * (tmax + 1) + t];
  }
  double &at(int i, int j, int t) {
    return e[(i * (jmax + 1) + j) * (tmax + 1) + t];
  }
  E1D(int im, int jm, double a, double b, double XAB)
      : imax(im), jmax(jm), tmax(im + jm) {
    const double p = a + b;
    const double mu = a * b / p;
    const double inv_2p = 0.5 / p;
    const double XPA = -b / p * XAB;
    const double XPB = a / p * XAB;
    at(0, 0, 0) = std::exp(-mu * XAB * XAB);
    for (int i = 0; i <= imax; ++i) {
      for (int j = 0; j <= jmax; ++j) {
        if (i == 0 && j == 0)
          continue;
        // build (i, j) from (i-1, j) when i > 0, else from (0, j-1)
        const bool up_i = i > 0;
        const int pi = up_i ? i - 1 : i, pj = up_i ? j : j - 1;
        const double X = up_i ? XPA : XPB;
        for (int t = 0; t <= i + j; ++t) {
          double v = X * (*this)(pi, pj, t);
          if (t > 0)
            v += inv_2p * (*this)(pi, pj, t - 1);
          v += (t + 1) * (*this)(pi, pj, t + 1);
          at(i, j, t) = v;
        }
      }
    }
  }
};

std::vector<std::array<int, 3>> cart_powers(int l) {
  std::vector<std::array<int, 3>> out;
  for (int i = l; i >= 0; --i)
    for (int j = l - i; j >= 0; --j)
      out.push_back({i, j, l - i - j});
  return out;
}

} // namespace

struct SplitRIJ::Impl {
  gto::AOBasis ao_basis;
  gto::AOBasis aux_basis;
  ShellPairList shellpair_list; // Significant shell pairs from IntegralEngine
  Mat schwarz;                  // Schwarz screening matrix (nshells x nshells)
  Vec V_diag;                   // sqrt of the diagonal of V
  double screening_threshold = 1e-12;

  CoulombMetric V_LLt;
  Boys<> boys;

  struct SignificantPair {
    size_t sa, sb;
    size_t sp_data_idx;
  };
  std::vector<SignificantPair> significant_pairs;
  std::vector<ShellPairData<double>> shell_pair_data;
  std::vector<AuxShellData<double>> aux_shells;

  // Primitive pairs of each significant pair, flattened: pair idx owns
  // pair_prims[pair_prim_offset[idx] .. pair_prim_offset[idx + 1]).
  std::vector<srij::PrimPair> pair_prims;
  std::vector<size_t> pair_prim_offset;
  // Pair processing order, most expensive first (load balance).
  std::vector<size_t> pair_order;

  // Auxiliary primitives grouped by angular momentum, structure-of-arrays,
  // sorted by decreasing sqrt(V_cc) of their shell so the per-pair screen
  // keeps a prefix. Hermite arrays (Y, T) of class l are [t'][k], row
  // stride npad, at offset `herm_offset` in one flat buffer, with room for
  // degree l + 1 (the gradient).
  struct AuxClass {
    int lc = 0;
    int n = 0, npad = 0;
    std::vector<double> gamma, cpref, cx, cy, cz, vmax;
    std::vector<int> shell, prim;
    size_t herm_offset = 0;
    srij::AuxView view() const {
      return {gamma.data(), cpref.data(), cx.data(), cy.data(), cz.data(),
              npad};
    }
  };
  std::vector<AuxClass> classes;
  size_t herm_size = 0;

  // Largest D/J block and per-pair Hermite vector (all primitives).
  int max_nab = 0;
  size_t max_pair_herm = 0;

  void precompute_shell_data() {
    const auto &ao_shells_vec = ao_basis.shells();
    const auto &aux_shells_vec = aux_basis.shells();
    const bool ao_is_spherical = ao_basis.is_pure();

    for (size_t sa = 0; sa < ao_shells_vec.size(); ++sa) {
      const auto &shell_a = ao_shells_vec[sa];
      for (size_t sb : shellpair_list[sa]) {
        const auto &shell_b = ao_shells_vec[sb];
        auto sp_data = ints::precompute_shell_pair_dispatch<double>(
            shell_a.l, shell_b.l, shell_a.num_primitives(),
            shell_b.num_primitives(), shell_a.exponents.data(),
            shell_b.exponents.data(),
            shell_a.contraction_coefficients.col(0).data(),
            shell_b.contraction_coefficients.col(0).data(),
            shell_a.origin.data(), shell_b.origin.data());
        // Every primitive pair below the overlap cutoff: nothing to add.
        if (sp_data.primitives.empty())
          continue;
        if (ao_is_spherical)
          transform_shell_pair_to_spherical(sp_data, shell_a.l, shell_b.l);
        significant_pairs.push_back({sa, sb, shell_pair_data.size()});
        shell_pair_data.push_back(std::move(sp_data));
      }
    }

    occ::log::debug("Split-RI-J: {} significant shell pairs ({})",
                    significant_pairs.size(),
                    ao_is_spherical ? "spherical" : "cartesian");

    const bool aux_is_spherical = aux_basis.is_pure();
    aux_shells.reserve(aux_shells_vec.size());
    for (const auto &shell_c : aux_shells_vec) {
      auto aux_data = ints::precompute_aux_shell_dispatch<double>(
          shell_c.l, shell_c.num_primitives(), shell_c.exponents.data(),
          shell_c.contraction_coefficients.col(0).data(),
          shell_c.origin.data());
      if (aux_is_spherical)
        transform_aux_shell_to_spherical(aux_data);
      aux_shells.push_back(std::move(aux_data));
    }
  }

  void build_kernel_layout() {
    const double pi_1p5 = std::pow(ints::BoysConstants<double>::pi, 1.5);
    const auto &aux_first_bf = aux_basis.first_bf();

    // AO primitive pairs, flattened per significant pair
    pair_prim_offset.resize(significant_pairs.size() + 1, 0);
    for (size_t idx = 0; idx < significant_pairs.size(); ++idx) {
      const auto &sp = shell_pair_data[significant_pairs[idx].sp_data_idx];
      pair_prim_offset[idx] = pair_prims.size();
      for (const auto &prim : sp.primitives)
        pair_prims.push_back(
            {prim.p, prim.Px, prim.Py, prim.Pz, prim.prefactor});
      max_nab = std::max(max_nab, sp.primitives[0].nab);
      max_pair_herm =
          std::max(max_pair_herm, sp.primitives.size() * sp.nherm());
    }
    pair_prim_offset[significant_pairs.size()] = pair_prims.size();

    // Aux primitives by angular momentum
    const int max_lc = aux_basis.l_max();
    if (!SplitRIJ::supports(ao_basis, aux_basis))
      throw std::runtime_error(
          "Split-RI-J: angular momentum beyond the compiled kernels (AO l <= "
          "4, auxiliary l <= 6)");
    std::vector<std::vector<std::pair<int, int>>> members(max_lc + 1);
    for (size_t sc = 0; sc < aux_shells.size(); ++sc)
      for (size_t k = 0; k < aux_shells[sc].primitives.size(); ++k)
        members[aux_shells[sc].lc].push_back(
            {static_cast<int>(sc), static_cast<int>(k)});
    std::vector<double> shell_vmax(aux_shells.size());
    for (size_t sc = 0; sc < aux_shells.size(); ++sc)
      shell_vmax[sc] =
          V_diag.segment(aux_first_bf[sc], aux_basis[sc].size()).maxCoeff();
    classes.assign(max_lc + 1, AuxClass{});
    herm_size = 0;
    for (int lc = 0; lc <= max_lc; ++lc) {
      auto &m = members[lc];
      std::stable_sort(m.begin(), m.end(), [&](const auto &x, const auto &y) {
        return shell_vmax[x.first] > shell_vmax[y.first];
      });
      AuxClass &c = classes[lc];
      c.lc = lc;
      c.n = static_cast<int>(m.size());
      c.npad = (c.n + srij::NP - 1) / srij::NP * srij::NP;
      c.gamma.assign(c.npad, 1.0);
      c.cpref.assign(c.npad, 0.0);
      c.cx.assign(c.npad, 0.0);
      c.cy.assign(c.npad, 0.0);
      c.cz.assign(c.npad, 0.0);
      c.vmax.assign(c.npad, 0.0);
      const double sph = ints::spherical_harmonic_factor<double>(lc);
      for (int i = 0; i < c.n; ++i) {
        const auto &aux = aux_shells[m[i].first];
        const auto &prim = aux.primitives[m[i].second];
        c.gamma[i] = prim.gamma;
        c.cpref[i] = prim.coeff * sph * pi_1p5 / prim.gamma;
        c.cx[i] = aux.C[0];
        c.cy[i] = aux.C[1];
        c.cz[i] = aux.C[2];
        c.vmax[i] = shell_vmax[m[i].first];
        c.shell.push_back(m[i].first);
        c.prim.push_back(m[i].second);
      }
      c.herm_offset = herm_size;
      herm_size += static_cast<size_t>(nhermsum(lc + 1)) * c.npad;
    }

    // Most expensive pairs first
    pair_order.resize(significant_pairs.size());
    std::vector<double> cost(significant_pairs.size());
    for (size_t idx = 0; idx < significant_pairs.size(); ++idx) {
      const auto &sp = shell_pair_data[significant_pairs[idx].sp_data_idx];
      cost[idx] = double(sp.primitives.size()) * nhermsum(sp.la + sp.lb + 2);
      pair_order[idx] = idx;
    }
    std::stable_sort(pair_order.begin(), pair_order.end(),
                     [&](size_t x, size_t y) { return cost[x] > cost[y]; });

    if (spdlog::should_log(spdlog::level::debug)) {
      std::vector<size_t> nprim_lab(2 * ao_basis.l_max() + 1, 0);
      for (const auto &sp : shell_pair_data)
        nprim_lab[sp.la + sp.lb] += sp.primitives.size();
      for (size_t l = 0; l < nprim_lab.size(); ++l)
        occ::log::debug("Split-RI-J: L_ab = {}: {} primitive pairs", l,
                        nprim_lab[l]);
      for (const auto &c : classes)
        occ::log::debug("Split-RI-J: lc = {}: {} aux primitives", c.lc, c.n);
    }
  }

  Impl(const gto::AOBasis &ao, const gto::AOBasis &aux,
       const ShellPairList &spl, const Mat &schwarz_in)
      : ao_basis(ao), aux_basis(aux), shellpair_list(spl),
        schwarz(schwarz_in) {
    occ::timing::start(occ::timing::category::df);
    IntegralEngine aux_engine(aux_basis);
    Mat V = aux_engine.one_electron_operator(cint::Operator::coulomb, false);
    occ::timing::stop(occ::timing::category::df);

    V_diag = V.diagonal().array().sqrt();

    occ::timing::start(occ::timing::category::la);
    V_LLt.compute(V);
    if (!V_LLt.uses_cholesky()) {
      occ::log::warn("Split-RI-J: Coulomb metric is not positive definite, "
                     "falling back to an eigendecomposition ({} of {} "
                     "auxiliary vectors discarded)",
                     V_LLt.num_discarded(), V.rows());
    }
    occ::timing::stop(occ::timing::category::la);

    precompute_shell_data();
    build_kernel_layout();
  }

  Impl(const gto::AOBasis &ao, const gto::AOBasis &aux,
       const ShellPairList &spl, const Mat &schwarz_in,
       const CoulombMetric &metric, const Vec &V_diag_sqrt)
      : ao_basis(ao), aux_basis(aux), shellpair_list(spl),
        schwarz(schwarz_in), V_diag(V_diag_sqrt), V_LLt(metric) {
    precompute_shell_data();
    build_kernel_layout();
  }

  // Number of class-lc aux primitives (a multiple of NP) a pair with
  // screening value `sv` must see: those with sv * sqrt(V_cc) >= threshold.
  int class_extent(const AuxClass &c, double sv) const {
    if (sv <= 0.0)
      return 0;
    const double vmin = screening_threshold / sv;
    const auto it =
        std::partition_point(c.vmax.begin(), c.vmax.begin() + c.n,
                             [vmin](double v) { return v >= vmin; });
    const int n = static_cast<int>(it - c.vmax.begin());
    return (n + srij::NP - 1) / srij::NP * srij::NP;
  }

  // D block of a pair (off-diagonal pairs doubled), and its max |D|.
  double gather_density(const Mat &D, const SignificantPair &sig,
                        double *D_block) const {
    const auto &ao_shells_vec = ao_basis.shells();
    const auto &ao_first_bf = ao_basis.first_bf();
    const int bf_a = ao_first_bf[sig.sa];
    const int bf_b = ao_first_bf[sig.sb];
    const int na = ao_shells_vec[sig.sa].size();
    const int nb = ao_shells_vec[sig.sb].size();
    const double fac = sig.sa != sig.sb ? 2.0 : 1.0;
    double d_max = 0.0;
    for (int a = 0; a < na; ++a) {
      for (int b = 0; b < nb; ++b) {
        const double v = fac * D(bf_a + a, bf_b + b);
        D_block[a * nb + b] = v;
        d_max = std::max(d_max, std::abs(v));
      }
    }
    return d_max;
  }

  // X_i = E_i^T D_block for every primitive pair i of a pair, X[i * nherm].
  static void hermite_density(const ShellPairData<double> &sp,
                              const double *D_block, double *X) {
    const int nab = sp.primitives[0].nab;
    const int nherm_ab = sp.nherm();
    Eigen::Map<const Eigen::VectorXd> D_vec(D_block, nab);
    for (size_t i = 0; i < sp.primitives.size(); ++i) {
      Eigen::Map<const MatRM> E_ab(sp.primitives[i].E_matrix.data(), nab,
                                   nherm_ab);
      Eigen::Map<Eigen::VectorXd>(X + i * nherm_ab, nherm_ab).noalias() =
          E_ab.transpose() * D_vec;
    }
  }

  // Per-thread scratch: a D/J block and Hermite vectors for one pair.
  struct Scratch {
    std::vector<double> block, x, u, xa;
  };
  Scratch make_scratch() const {
    return Scratch{std::vector<double>(max_nab),
                   std::vector<double>(max_pair_herm), {}, {}};
  }

  // Forward pass, summed over pairs: Y_k[t'] = sum S_k R_{t+t'} X[t] in the
  // class layout, with the aux side `dlc` Hermite degrees above its shells.
  std::vector<double> forward(const Mat &D, int dlc) const {
    const double *boys_table = boys.table();
    const bool use_screening = schwarz.size() > 0;
    occ::parallel::thread_local_storage<Scratch> scratch_local(
        [this]() { return make_scratch(); });
    occ::parallel::thread_local_storage<std::vector<double>> Y_local(
        [this]() { return std::vector<double>(herm_size, 0.0); });

    occ::parallel::parallel_for(size_t(0), significant_pairs.size(),
                                [&](size_t ord) {
      const size_t idx = pair_order[ord];
      auto &sc = scratch_local.local();
      const auto &sig = significant_pairs[idx];
      const auto &sp = shell_pair_data[sig.sp_data_idx];
      const double d_max = gather_density(D, sig, sc.block.data());
      const double sv =
          d_max * (use_screening ? schwarz(sig.sa, sig.sb) : 1.0);
      if (use_screening && sv < screening_threshold)
        return;
      hermite_density(sp, sc.block.data(), sc.x.data());

      auto &Y = Y_local.local();
      const int nprim = static_cast<int>(sp.primitives.size());
      const srij::PrimPair *pp = pair_prims.data() + pair_prim_offset[idx];
      for (const auto &c : classes) {
        const int kend = use_screening ? class_extent(c, sv) : c.npad;
        if (kend == 0)
          continue;
        srij::forward_kernel(sp.la + sp.lb, c.lc + dlc)(
            boys_table, pp, nprim, sc.x.data(), sp.nherm(), c.view(), kend,
            Y.data() + c.herm_offset);
      }
    });

    std::vector<double> Y(herm_size, 0.0);
    for (const auto &local_Y : Y_local)
      for (size_t i = 0; i < herm_size; ++i)
        Y[i] += local_Y[i];
    return Y;
  }

  // Fitted coefficients d = V^-1 g, with g = E_c Y per aux primitive.
  Vec fit(const std::vector<double> &Y) const {
    const auto &aux_first_bf = aux_basis.first_bf();
    Vec g = Vec::Zero(aux_basis.nbf());
    for (const auto &c : classes) {
      const int nherm_c = nhermsum(c.lc);
      Eigen::VectorXd Yk(nherm_c);
      for (int i = 0; i < c.n; ++i) {
        const int shell = c.shell[i];
        const auto &prim = aux_shells[shell].primitives[c.prim[i]];
        for (int h = 0; h < nherm_c; ++h)
          Yk(h) = Y[c.herm_offset + h * c.npad + i];
        Eigen::Map<const MatRM> E_c(prim.E_matrix.data(), prim.nc, nherm_c);
        g.segment(aux_first_bf[shell], prim.nc).noalias() += E_c * Yk;
      }
    }
    return V_LLt.solve(g);
  }

  // T = E_c^T d per aux primitive, in the class layout.
  std::vector<double> hermite_coefficients(const Vec &d) const {
    const auto &aux_first_bf = aux_basis.first_bf();
    std::vector<double> T(herm_size, 0.0);
    for (const auto &c : classes) {
      const int nherm_c = nhermsum(c.lc);
      for (int i = 0; i < c.n; ++i) {
        const int shell = c.shell[i];
        const auto &prim = aux_shells[shell].primitives[c.prim[i]];
        Eigen::Map<const MatRM> E_c(prim.E_matrix.data(), prim.nc, nherm_c);
        const Eigen::VectorXd Tk =
            E_c.transpose() * d.segment(aux_first_bf[shell], prim.nc);
        for (int h = 0; h < nherm_c; ++h)
          T[c.herm_offset + h * c.npad + i] = Tk(h);
      }
    }
    return T;
  }

  Mat coulomb(const Mat &D) const {
    occ::timing::start(occ::timing::category::df);
    const double *boys_table = boys.table();
    const auto &ao_shells_vec = ao_basis.shells();
    const auto &ao_first_bf = ao_basis.first_bf();
    const size_t nbf = ao_basis.nbf();

    const std::vector<double> T_all = hermite_coefficients(fit(forward(D, 0)));

    // Backward pass: U = sum_k S R T_k, J_ab = E U. Each pair owns its
    // block(s) of J, so they are written directly.
    occ::parallel::thread_local_storage<Scratch> scratch_local(
        [this]() { return make_scratch(); });
    Mat J = Mat::Zero(nbf, nbf);
    occ::parallel::parallel_for(size_t(0), significant_pairs.size(),
                                [&](size_t ord) {
      const size_t idx = pair_order[ord];
      auto &sc = scratch_local.local();
      const auto &sig = significant_pairs[idx];
      const auto &sp = shell_pair_data[sig.sp_data_idx];
      const int nab = sp.primitives[0].nab;
      const int nherm_ab = sp.nherm();
      const int L_ab = sp.la + sp.lb;
      const int nprim = static_cast<int>(sp.primitives.size());

      double *U = sc.x.data();
      std::fill(U, U + nprim * nherm_ab, 0.0);
      const srij::PrimPair *pp = pair_prims.data() + pair_prim_offset[idx];
      for (const auto &c : classes) {
        if (c.n == 0)
          continue;
        srij::backward_kernel(L_ab, c.lc)(boys_table, pp, nprim, c.view(),
                                          c.npad, T_all.data() + c.herm_offset,
                                          U, nherm_ab);
      }

      Eigen::Map<Eigen::VectorXd> J_vec(sc.block.data(), nab);
      J_vec.setZero();
      for (int i = 0; i < nprim; ++i) {
        Eigen::Map<const MatRM> E_ab(sp.primitives[i].E_matrix.data(), nab,
                                     nherm_ab);
        J_vec.noalias() +=
            E_ab * Eigen::Map<const Eigen::VectorXd>(U + i * nherm_ab,
                                                     nherm_ab);
      }
      const int bf_a = ao_first_bf[sig.sa];
      const int bf_b = ao_first_bf[sig.sb];
      const int na = ao_shells_vec[sig.sa].size();
      const int nb = ao_shells_vec[sig.sb].size();
      for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
          J(bf_a + a, bf_b + b) += J_vec[a * nb + b];
          if (sig.sa != sig.sb)
            J(bf_b + b, bf_a + a) += J_vec[a * nb + b];
        }
      }
    });

    occ::timing::stop(occ::timing::category::df);
    return 2.0 * J;
  }

  // ---------------------------------------------------------------------
  // Nuclear gradient of d^T g, g_P = sum_mn (mn|P) D_mn, at fixed fitted
  // coefficients d. With S R_{t+t'}(P - C) the Hermite integrals,
  //   d/dC_x : -R_{t+t'+e_x}   (aux centre)
  //   d/dP_x : +R_{t+t'+e_x}   (both AO centres moved together)
  // and the A-centre derivative from the Hermite expansion of
  // d/dA_x [phi_a phi_b] = (2a phi_{a+1x} - a_x phi_{a-1x}) phi_b, the
  // B-centre one being the P derivative less the A one. So the aux side
  // needs a forward pass one Hermite degree higher on the aux side (which
  // also gives g), and the AO side a backward pass one degree higher on the
  // pair side.
  // ---------------------------------------------------------------------
  Mat3N gradient(const Mat &D, Vec *fitted) const {
    occ::timing::start(occ::timing::category::df);
    const double *boys_table = boys.table();
    const auto &ao_shells_vec = ao_basis.shells();
    const auto &ao_atom = ao_basis.shell_to_atom();
    const auto &aux_atom = aux_basis.shell_to_atom();
    const int natom = static_cast<int>(ao_basis.atoms().size());
    const bool pure = ao_basis.is_pure();
    const int max_lc = static_cast<int>(classes.size()) - 1;
    const int max_la = ao_basis.l_max();
    const auto tuv = build_hermite_tuv_table(2 * max_la + max_lc + 2);
    std::vector<Mat> cart_to_sph(max_la + 1);
    for (int l = 0; l <= max_la; ++l)
      cart_to_sph[l] = gto::cartesian_to_spherical_transformation_matrix(l);

    // Forward pass with the aux side one degree up; its degree-lc part gives
    // g and d.
    const std::vector<double> Y_all = forward(D, 1);
    const Vec d = fit(Y_all);
    const std::vector<double> T_all = hermite_coefficients(d);

    // Aux-centre gradient: -sum_t' T[t'] Y[t' + e_x]
    Mat3N grad = Mat3N::Zero(3, natom);
    for (const auto &c : classes) {
      const int nherm_c = nhermsum(c.lc);
      for (int i = 0; i < c.n; ++i) {
        Vec3 s = Vec3::Zero();
        for (int h = 0; h < nherm_c; ++h) {
          const double Tk = T_all[c.herm_offset + h * c.npad + i];
          const auto [t, u, v] = tuv[h];
          auto Y = [&](int k) { return Y_all[c.herm_offset + k * c.npad + i]; };
          s(0) += Tk * Y(hermite_index(t + 1, u, v));
          s(1) += Tk * Y(hermite_index(t, u + 1, v));
          s(2) += Tk * Y(hermite_index(t, u, v + 1));
        }
        grad.col(aux_atom[c.shell[i]]) -= s;
      }
    }

    // Backward pass, pair side one degree up
    occ::parallel::thread_local_storage<Scratch> scratch_local(
        [this]() { return make_scratch(); });
    occ::parallel::thread_local_storage<Mat3N> grad_local(
        Mat3N::Zero(3, natom));
    occ::parallel::parallel_for(size_t(0), significant_pairs.size(),
                                [&](size_t ord) {
      const size_t idx = pair_order[ord];
      auto &sc = scratch_local.local();
      const auto &sig = significant_pairs[idx];
      const auto &sp = shell_pair_data[sig.sp_data_idx];
      const int la = sp.la, lb = sp.lb;
      const int na = ao_shells_vec[sig.sa].size(),
                nb = ao_shells_vec[sig.sb].size();
      const double d_max = gather_density(D, sig, sc.block.data());
      if (d_max == 0.0)
        return;
      hermite_density(sp, sc.block.data(), sc.x.data());
      const int nprim = static_cast<int>(sp.primitives.size());
      const int nh = sp.nherm();
      const int nh1 = nhermsum(la + lb + 1);
      sc.u.assign(static_cast<size_t>(nprim) * nh1, 0.0);
      const srij::PrimPair *pp = pair_prims.data() + pair_prim_offset[idx];
      for (const auto &c : classes) {
        if (c.n == 0)
          continue;
        srij::backward_kernel(la + lb + 1, c.lc)(
            boys_table, pp, nprim, c.view(), c.npad,
            T_all.data() + c.herm_offset, sc.u.data(), nh1);
      }

      // Cartesian density block
      const int nca = ncart(la), ncb = ncart(lb);
      Mat Dc(nca, ncb);
      {
        Eigen::Map<const MatRM> Db(sc.block.data(), na, nb);
        if (pure)
          Dc = cart_to_sph[la].transpose() * Db * cart_to_sph[lb];
        else
          Dc = Db;
      }
      const auto pa = cart_powers(la), pb = cart_powers(lb);
      const auto &shell_a = ao_shells_vec[sig.sa];
      const auto &shell_b = ao_shells_vec[sig.sb];
      Vec3 gP = Vec3::Zero(), gA = Vec3::Zero();
      sc.xa.resize(3 * nh1);
      for (int i = 0; i < nprim; ++i) {
        const auto &prim = sp.primitives[i];
        const double *Xi = sc.x.data() + i * nh;
        const double *Ui = sc.u.data() + i * nh1;
        for (int h = 0; h < nh; ++h) {
          const auto [t, u, v] = tuv[h];
          gP(0) += Xi[h] * Ui[hermite_index(t + 1, u, v)];
          gP(1) += Xi[h] * Ui[hermite_index(t, u + 1, v)];
          gP(2) += Xi[h] * Ui[hermite_index(t, u, v + 1)];
        }
        // A-centre derivative distribution, contracted with Dc
        const double a = prim.a, b = prim.b;
        const std::array<E1D, 3> E = {
            E1D(la + 1, lb, a, b, shell_a.origin(0) - shell_b.origin(0)),
            E1D(la + 1, lb, a, b, shell_a.origin(1) - shell_b.origin(1)),
            E1D(la + 1, lb, a, b, shell_a.origin(2) - shell_b.origin(2))};
        std::fill(sc.xa.begin(), sc.xa.end(), 0.0);
        for (int ia = 0; ia < nca; ++ia) {
          for (int ib = 0; ib < ncb; ++ib) {
            const double dab = Dc(ia, ib);
            if (dab == 0.0)
              continue;
            const auto &A = pa[ia];
            const auto &B = pb[ib];
            for (int dir = 0; dir < 3; ++dir) {
              // direction dir differentiated, the others plain
              const int d1 = (dir + 1) % 3, d2 = (dir + 2) % 3;
              for (int t = 0; t <= A[dir] + B[dir] + 1; ++t) {
                const double f = 2.0 * a * E[dir](A[dir] + 1, B[dir], t) -
                                 A[dir] * E[dir](A[dir] - 1, B[dir], t);
                if (f == 0.0)
                  continue;
                for (int u = 0; u <= A[d1] + B[d1]; ++u) {
                  const double fu = dab * f * E[d1](A[d1], B[d1], u);
                  for (int v = 0; v <= A[d2] + B[d2]; ++v) {
                    int q[3];
                    q[dir] = t;
                    q[d1] = u;
                    q[d2] = v;
                    sc.xa[dir * nh1 + hermite_index(q[0], q[1], q[2])] +=
                        fu * E[d2](A[d2], B[d2], v);
                  }
                }
              }
            }
          }
        }
        for (int dir = 0; dir < 3; ++dir) {
          double s = 0.0;
          for (int h = 0; h < nh1; ++h)
            s += sc.xa[dir * nh1 + h] * Ui[h];
          gA(dir) += s;
        }
      }
      auto &gl = grad_local.local();
      gl.col(ao_atom[sig.sa]) += gA;
      gl.col(ao_atom[sig.sb]) += gP - gA;
    });
    for (const auto &gl : grad_local)
      grad += gl;

    if (fitted)
      *fitted = d;
    occ::timing::stop(occ::timing::category::df);
    return grad;
  }
};

SplitRIJ::SplitRIJ(const gto::AOBasis &ao_basis, const gto::AOBasis &aux_basis,
                   const ShellPairList &shellpairs, const Mat &schwarz)
    : impl_(std::make_unique<Impl>(ao_basis, aux_basis, shellpairs, schwarz)) {
}

SplitRIJ::SplitRIJ(const gto::AOBasis &ao_basis, const gto::AOBasis &aux_basis,
                   const ShellPairList &shellpairs, const Mat &schwarz,
                   const CoulombMetric &metric, const Vec &V_diag_sqrt)
    : impl_(std::make_unique<Impl>(ao_basis, aux_basis, shellpairs, schwarz,
                                   metric, V_diag_sqrt)) {}

SplitRIJ::~SplitRIJ() = default;

bool SplitRIJ::supports(const gto::AOBasis &ao_basis,
                        const gto::AOBasis &aux_basis) {
  // Shell-pair E-matrices go up to g-g; the gradient's kernels run one
  // Hermite degree above the Fock build's on either side.
  return ao_basis.l_max() <= 4 && 2 * ao_basis.l_max() + 1 <= srij::MAX_LA &&
         aux_basis.l_max() + 1 <= srij::MAX_LC &&
         2 * ao_basis.l_max() + aux_basis.l_max() + 1 <= srij::MAX_L;
}

SplitRIJ::SplitRIJ(SplitRIJ &&) noexcept = default;
SplitRIJ &SplitRIJ::operator=(SplitRIJ &&) noexcept = default;

Mat SplitRIJ::coulomb(const MolecularOrbitals &mo) const {
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    return coulomb_from_density(mo.D);
  case SpinorbitalKind::Unrestricted: {
    Mat D_total = qm::block::a(mo.D) + qm::block::b(mo.D);
    Mat J = coulomb_from_density(D_total);
    Mat result(2 * nbf(), nbf());
    qm::block::a(result) = J;
    qm::block::b(result) = J;
    return result;
  }
  case SpinorbitalKind::General: {
    Mat D_total = qm::block::aa(mo.D) + qm::block::bb(mo.D);
    Mat J = coulomb_from_density(D_total);
    Mat result = Mat::Zero(2 * nbf(), 2 * nbf());
    qm::block::aa(result) = J;
    qm::block::bb(result) = J;
    return result;
  }
  default:
    throw std::runtime_error("SplitRIJ::coulomb: unknown SpinorbitalKind");
  }
}

Mat SplitRIJ::coulomb_from_density(const Mat &D) const {
  return impl_->coulomb(D);
}

Mat3N SplitRIJ::three_center_gradient(const Mat &D, Vec *fitted) const {
  return impl_->gradient(D, fitted);
}

const gto::AOBasis &SplitRIJ::aux_basis() const { return impl_->aux_basis; }

const gto::AOBasis &SplitRIJ::ao_basis() const { return impl_->ao_basis; }

size_t SplitRIJ::naux() const { return impl_->aux_basis.nbf(); }

size_t SplitRIJ::nbf() const { return impl_->ao_basis.nbf(); }

} // namespace occ::qm
