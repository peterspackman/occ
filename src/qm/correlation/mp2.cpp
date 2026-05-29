
#include <algorithm>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/opmatrix.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm {

namespace {
// Accumulate the restricted MP2 same-spin/opposite-spin energy for one (i,j)
// occupied pair, given K(a,b) = (ia|jb). `factor` lets callers exploit the
// i<->j summand symmetry (2 for off-diagonal pairs, 1 otherwise).
inline void accumulate_mp2_pair(const Mat &K, double eps_i, double eps_j,
                                const Vec &eps_v, double factor, double &ss,
                                double &os) {
  const Eigen::Index v = eps_v.size();
  const double eij = eps_i + eps_j;
  for (Eigen::Index a = 0; a < v; ++a) {
    const double eija = eij - eps_v(a);
    for (Eigen::Index b = 0; b < v; ++b) {
      const double denom = eija - eps_v(b);
      if (std::abs(denom) < 1e-12)
        continue;
      const double kab = K(a, b);
      const double kba = K(b, a);
      const double inv = factor / denom;
      os += 2.0 * kab * kab * inv;
      ss -= kab * kba * inv;
    }
  }
}

// Opposite-spin (αβ) energy for one (iα, jβ) pair: K(a,b) = (iα aα | jβ bβ),
// a indexing α-virtuals, b indexing β-virtuals.  Returns Σ_ab K(a,b)²/D.
inline double os_pair_energy(const Mat &K, double eps_i, double eps_j,
                             const Vec &eps_va, const Vec &eps_vb) {
  const Eigen::Index va = eps_va.size();
  const Eigen::Index vb = eps_vb.size();
  const double eij = eps_i + eps_j;
  double e = 0.0;
  for (Eigen::Index a = 0; a < va; ++a) {
    const double eija = eij - eps_va(a);
    for (Eigen::Index b = 0; b < vb; ++b) {
      const double denom = eija - eps_vb(b);
      if (std::abs(denom) < 1e-12)
        continue;
      const double kab = K(a, b);
      e += kab * kab / denom;
    }
  }
  return e;
}

// Same-spin (σσ) energy for one (i, j) pair, both spin σ: K(a,b) = (ia|jb).
// Returns ¼ Σ_ab [(ia|jb)-(ib|ja)]²/D; the caller sums over the full i,j and
// a,b ranges (the ¼ accounts for the ordered double counting).
inline double ss_pair_energy(const Mat &K, double eps_i, double eps_j,
                             const Vec &eps_v) {
  const Eigen::Index v = eps_v.size();
  const double eij = eps_i + eps_j;
  double e = 0.0;
  for (Eigen::Index a = 0; a < v; ++a) {
    const double eija = eij - eps_v(a);
    for (Eigen::Index b = 0; b < v; ++b) {
      const double denom = eija - eps_v(b);
      if (std::abs(denom) < 1e-12)
        continue;
      const double m = K(a, b) - K(b, a);
      e += 0.25 * m * m / denom;
    }
  }
  return e;
}
} // namespace

MP2::MP2(const AOBasis &basis, const MolecularOrbitals &mo, double scf_energy)
    : PostHFMethod(basis, mo, scf_energy) {
  occ::log::debug("MP2 initialized (conventional)");
  m_algorithm = Conventional;
}

MP2::MP2(const AOBasis &basis, const AOBasis &aux_basis,
         const MolecularOrbitals &mo, double scf_energy)
    : PostHFMethod(basis, mo, scf_energy) {
  occ::log::debug(
      "MP2 initialized (RI): {} AO functions, {} auxiliary functions",
      basis.nbf(), aux_basis.nbf());
  m_algorithm = RI;

  // Initialize DF engine with auxiliary basis
  const auto &atoms = basis.atoms();
  const auto &ao_shells = basis.shells();
  const auto &aux_shells = aux_basis.shells();

  m_df_engine =
      std::make_unique<IntegralEngineDF>(atoms, ao_shells, aux_shells);
}

void MP2::set_frozen_core_auto() {
  constexpr int scandium_z = 21;
  constexpr int sodium_z = 11;
  constexpr int lithium_z = 3;
  constexpr int core_orbitals_sc_and_above = 9;
  constexpr int core_orbitals_na_ar = 5;
  constexpr int core_orbitals_li_ne = 1;

  const auto &atoms = m_mo_engine->ao_engine().aobasis().atoms();
  size_t frozen_count = 0;

  for (const auto &atom : atoms) {
    int z = atom.atomic_number;
    if (z >= scandium_z) {
      frozen_count += core_orbitals_sc_and_above;
    } else if (z >= sodium_z) {
      frozen_count += core_orbitals_na_ar;
    } else if (z >= lithium_z) {
      frozen_count += core_orbitals_li_ne;
    }
  }

  m_n_frozen_core = std::min(frozen_count, n_occupied() - 1);

  occ::log::debug("Automatic frozen core: {} orbitals", m_n_frozen_core);
}

double MP2::compute_correlation_energy() {
  occ::timing::start(occ::timing::category::post_hf);

  const size_t n_occ_total = n_occupied();
  const size_t n_virt_total = n_virtual();
  const size_t n_ao = m_mo.n_ao;
  const Vec &orbital_energies = m_mo.energies;

  auto active_ranges = get_active_orbital_ranges();
  size_t n_occ_active = active_ranges.first;
  size_t n_virt_active = active_ranges.second;

  occ::log::debug("Active space: {}/{} occupied, {}/{} virtual orbitals",
                  n_occ_active, n_occ_total, n_virt_active, n_virt_total);

  log_frozen_core_info(n_occ_total, n_occ_active, orbital_energies);
  log_virtual_truncation_info(n_occ_total, n_virt_total, n_virt_active,
                              orbital_energies);

  if (m_algorithm == Conventional) {
    estimate_memory_requirements(n_ao, n_occ_active, n_virt_active);
  }

  if (m_algorithm == RI && m_df_engine) {
    m_correlation_energy = compute_ri_mp2_energy();
  } else {
    m_correlation_energy = compute_conventional_mp2_energy();
  }

  store_results(n_occ_total, n_virt_total, n_occ_active, n_virt_active);

  occ::timing::stop(occ::timing::category::post_hf);
  return m_correlation_energy;
}

std::pair<size_t, size_t> MP2::get_active_orbital_ranges() const {
  const size_t n_occ_total = n_occupied();
  const size_t n_virt_total = n_virtual();
  const Vec &orbital_energies = m_mo.energies;

  size_t n_frozen_by_energy = 0;
  for (size_t i = 0; i < n_occ_total; ++i) {
    if (orbital_energies(i) < m_e_min) {
      n_frozen_by_energy++;
    }
  }

  size_t n_frozen_total = std::max(m_n_frozen_core, n_frozen_by_energy);
  size_t n_occ_active = n_occ_total - n_frozen_total;

  size_t n_virt_active = 0;
  for (size_t a = 0; a < n_virt_total; ++a) {
    double virt_energy = orbital_energies(n_occ_total + a);
    if (virt_energy <= m_e_max && virt_energy <= m_virtual_cutoff &&
        n_virt_active < m_max_virtuals) {
      n_virt_active++;
    } else if (virt_energy > m_e_max) {
      break;
    }
  }

  return {n_occ_active, n_virt_active};
}

double MP2::compute_ri_mp2_energy() {
  if (!m_df_engine) {
    return compute_conventional_mp2_energy();
  }

  constexpr auto R = SpinorbitalKind::Restricted;
  if (m_mo.kind == SpinorbitalKind::Unrestricted) {
    return compute_unrestricted_ri_energy();
  }
  if (m_mo.kind != R) {
    throw std::runtime_error("General RI-MP2 is not implemented");
  }

  const auto active_ranges = get_active_orbital_ranges();
  const size_t n_occ_active = active_ranges.first;
  const size_t n_virt_active = active_ranges.second;
  const size_t n_occ_total = n_occupied();
  const size_t n_frozen = n_occ_total - n_occ_active;
  const Vec &eps = m_mo.energies;

  if (n_occ_active == 0 || n_virt_active == 0) {
    m_results.same_spin_correlation = 0.0;
    m_results.opposite_spin_correlation = 0.0;
    return 0.0;
  }

  // Active MO coefficient blocks. The MP2 energy is even in the sign of each MO
  // coefficient, so no phase convention is needed here.
  const Mat C_occ = m_mo.C.middleCols(n_frozen, n_occ_active);     // nbf x o
  const Mat C_virt = m_mo.C.middleCols(n_occ_total, n_virt_active); // nbf x v
  const Vec eps_o = eps.segment(n_frozen, n_occ_active);
  const Vec eps_v = eps.segment(n_occ_total, n_virt_active);

  DFIntegrals df(*m_df_engine, m_memory_budget);
  const size_t naux = df.naux();
  const Eigen::Index v = static_cast<Eigen::Index>(n_virt_active);

  // Memory-bounded occupied block size: we hold at most two metric-folded B
  // blocks (B_I and B_J) at a time, each (block * v * naux) doubles.
  const size_t bytes_per_occ =
      static_cast<size_t>(n_virt_active) * naux * sizeof(double);
  size_t block = (bytes_per_occ > 0)
                     ? std::max<size_t>(1, m_memory_budget / (2 * bytes_per_occ))
                     : n_occ_active;
  block = std::min(block, n_occ_active);

  occ::log::debug("RI-MP2: {} active occ, {} active virt, {} aux; occ block {}",
                  n_occ_active, n_virt_active, naux, block);

  struct Acc {
    double ss = 0.0;
    double os = 0.0;
  };
  occ::parallel::thread_local_storage<Acc> acc_local;

  // Loop over occupied blocks using i>=j symmetry: the restricted MP2 summand
  // is invariant under (i,a)<->(j,b), so off-diagonal pairs are counted once
  // with a factor of 2.
  occ::timing::start(occ::timing::category::mp2_energy);
  for (size_t Istart = 0; Istart < n_occ_active; Istart += block) {
    const size_t bI = std::min(block, n_occ_active - Istart);
    const Mat B_I = df.build_b_tilde(C_occ.middleCols(Istart, bI), C_virt);

    for (size_t Jstart = 0; Jstart <= Istart; Jstart += block) {
      const size_t bJ = std::min(block, n_occ_active - Jstart);
      const bool diag_block = (Jstart == Istart);
      Mat B_J_storage;
      if (!diag_block) {
        B_J_storage = df.build_b_tilde(C_occ.middleCols(Jstart, bJ), C_virt);
      }
      const Mat &B_J = diag_block ? B_I : B_J_storage;

      occ::parallel::parallel_for(size_t(0), bI, [&](size_t il) {
        auto &acc = acc_local.local();
        const size_t i_glob = Istart + il;
        const double eps_i = eps_o(static_cast<Eigen::Index>(i_glob));
        const auto Bi = B_I.middleRows(static_cast<Eigen::Index>(il) * v, v);

        const size_t jl_max = diag_block ? il : (bJ - 1);
        for (size_t jl = 0; jl <= jl_max; ++jl) {
          const size_t j_glob = Jstart + jl;
          const double eps_j = eps_o(static_cast<Eigen::Index>(j_glob));
          const double factor = (i_glob == j_glob) ? 1.0 : 2.0;
          const auto Bj = B_J.middleRows(static_cast<Eigen::Index>(jl) * v, v);

          const Mat K = Bi * Bj.transpose(); // K(a,b) = (ia|jb)
          accumulate_mp2_pair(K, eps_i, eps_j, eps_v, factor, acc.ss, acc.os);
        }
      });
    }
  }
  occ::timing::stop(occ::timing::category::mp2_energy);

  double same_spin = 0.0;
  double opposite_spin = 0.0;
  for (const auto &a : acc_local) {
    same_spin += a.ss;
    opposite_spin += a.os;
  }
  m_results.same_spin_correlation = same_spin;
  m_results.opposite_spin_correlation = opposite_spin;
  return same_spin + opposite_spin;
}

double MP2::compute_conventional_mp2_energy() {
  constexpr auto R = SpinorbitalKind::Restricted;
  if (m_mo.kind == SpinorbitalKind::Unrestricted) {
    return compute_unrestricted_conventional_energy();
  }
  if (m_mo.kind != R) {
    throw std::runtime_error("General conventional MP2 is not implemented");
  }

  const auto active_ranges = get_active_orbital_ranges();
  const size_t n_occ_active = active_ranges.first;
  const size_t n_virt_active = active_ranges.second;
  const size_t n_occ_total = n_occupied();
  const size_t n_frozen = n_occ_total - n_occ_active;
  const Vec &eps = m_mo.energies;

  if (n_occ_active == 0 || n_virt_active == 0) {
    m_results.same_spin_correlation = 0.0;
    m_results.opposite_spin_correlation = 0.0;
    return 0.0;
  }

  const Eigen::Index N = static_cast<Eigen::Index>(m_mo.n_ao);
  const Eigen::Index v = static_cast<Eigen::Index>(n_virt_active);
  const Eigen::Index o = static_cast<Eigen::Index>(n_occ_active);

  // Active MO coefficient blocks (raw signs; the energy is even in MO signs).
  const Mat C_occ = m_mo.C.middleCols(n_frozen, n_occ_active);      // N x o
  const Mat C_virt = m_mo.C.middleCols(n_occ_total, n_virt_active); // N x v
  const Vec eps_o = eps.segment(n_frozen, n_occ_active);
  const Vec eps_v = eps.segment(n_occ_total, n_virt_active);

  Eigen::TensorMap<const Eigen::Tensor<double, 2>> Cv(C_virt.data(), N, v);
  Eigen::TensorMap<const Eigen::Tensor<double, 2>> Co(C_occ.data(), N, o);
  const Eigen::array<Eigen::IndexPair<int>, 1> con0 = {
      Eigen::IndexPair<int>(0, 0)};
  const Eigen::array<Eigen::IndexPair<int>, 1> con1 = {
      Eigen::IndexPair<int>(1, 0)};

  // Memory-bounded occupied block size. The AO-direct half-transform holds one
  // (b_o x N x N x N) buffer per thread plus the reduced buffer; pick b_o so
  // that footprint fits the budget (b_o=o => one AO pass / semidirect; small
  // b_o => more AO passes / fully direct). Never N⁴.
  const int nthreads = std::max(1, occ::parallel::get_num_threads());
  const size_t bytes_per_i = static_cast<size_t>(N) * N * N * sizeof(double);
  const size_t denom_bytes =
      static_cast<size_t>(nthreads + 1) * std::max<size_t>(1, bytes_per_i);
  size_t block = std::max<size_t>(1, m_memory_budget / denom_bytes);
  block = std::min(block, n_occ_active);

  occ::log::debug("Conventional MP2 (AO-direct): {} active occ, {} active virt, "
                  "N={}, occ block {} ({} threads)",
                  n_occ_active, n_virt_active, N, block, nthreads);

  struct Acc {
    double ss = 0.0;
    double os = 0.0;
  };
  occ::parallel::thread_local_storage<Acc> acc_local;

  using T3 = Eigen::Tensor<double, 3>;

  occ::timing::start(occ::timing::category::mp2_energy);
  for (size_t Istart = 0; Istart < n_occ_active; Istart += block) {
    const Eigen::Index bI =
        static_cast<Eigen::Index>(std::min(block, n_occ_active - Istart));

    // First quarter transform (AO-direct): H1(i, ν, ρ, σ) for i in this block.
    const Eigen::Tensor<double, 4> H1 = m_ao_engine->ao_direct_half_transform(
        C_occ.middleCols(static_cast<Eigen::Index>(Istart), bI));

    // Complete the transform per occupied i and accumulate the pair energy.
    occ::parallel::parallel_for(Eigen::Index(0), bI, [&](Eigen::Index il) {
      auto &acc = acc_local.local();
      const size_t i_glob = Istart + static_cast<size_t>(il);
      const double eps_i = eps_o(static_cast<Eigen::Index>(i_glob));

      T3 H1_i = H1.chip(il, 0);        // (ν, ρ, σ)
      T3 t2 = H1_i.contract(Cv, con0); // contract ν -> (ρ, σ, a)
      T3 t3 = t2.contract(Cv, con1);   // contract σ -> (ρ, a, b)
      T3 K = t3.contract(Co, con0);    // contract ρ -> (a, b, j)

      for (Eigen::Index j = 0; j < o; ++j) {
        Eigen::Map<const Mat> Kmat(K.data() + static_cast<size_t>(j) * v * v, v,
                                   v); // Kmat(a,b) = (ia|jb)
        accumulate_mp2_pair(Kmat, eps_i, eps_o(j), eps_v, 1.0, acc.ss, acc.os);
      }
    });
  }
  occ::timing::stop(occ::timing::category::mp2_energy);

  double same_spin = 0.0;
  double opposite_spin = 0.0;
  for (const auto &a : acc_local) {
    same_spin += a.ss;
    opposite_spin += a.os;
  }
  m_results.same_spin_correlation = same_spin;
  m_results.opposite_spin_correlation = opposite_spin;
  return same_spin + opposite_spin;
}

void MP2::log_frozen_core_info(size_t n_occ_total, size_t n_occ_active,
                               const Vec &orbital_energies) const {
  size_t n_frozen_total = n_occ_total - n_occ_active;
  if (n_frozen_total == 0)
    return;

  size_t n_frozen_by_energy = 0;
  for (size_t i = 0; i < n_occ_total; ++i) {
    if (orbital_energies(i) < m_e_min)
      n_frozen_by_energy++;
  }

  occ::log::debug("Frozen core: {} total ({} by energy, {} manual)",
                  n_frozen_total, n_frozen_by_energy,
                  n_frozen_total - n_frozen_by_energy);
}

void MP2::log_virtual_truncation_info(size_t n_occ_total, size_t n_virt_total,
                                      size_t n_virt_active,
                                      const Vec &orbital_energies) const {
  if (n_virt_active == n_virt_total)
    return;

  double highest_included = orbital_energies(n_occ_total + n_virt_active - 1);
  occ::log::debug(
      "Virtual truncation: {}/{} orbitals, highest energy {:.4f} Hartree",
      n_virt_active, n_virt_total, highest_included);
}

void MP2::estimate_memory_requirements(size_t n_ao, size_t n_occ_active,
                                       size_t n_virt_active) const {
  constexpr double gb_factor = 1.0 / (1024.0 * 1024.0 * 1024.0);
  constexpr double memory_warning_threshold = 8.0;

  double ao_tensor_gb = static_cast<double>(n_ao * n_ao * n_ao * n_ao) *
                        sizeof(double) * gb_factor;
  double ovov_tensor_gb = static_cast<double>(n_occ_active * n_virt_active *
                                              n_occ_active * n_virt_active) *
                          sizeof(double) * gb_factor;
  double intermediate_gb =
      static_cast<double>(n_occ_active * n_ao * n_ao * n_ao) * sizeof(double) *
      gb_factor;
  double total_peak_gb = ao_tensor_gb + intermediate_gb + ovov_tensor_gb;

  if (total_peak_gb > memory_warning_threshold) {
    occ::log::warn("High memory usage estimated: {:.1f} GB. Consider RI-MP2.",
                   total_peak_gb);
  }
}

void MP2::store_results(size_t n_occ_total, size_t n_virt_total,
                        size_t n_occ_active, size_t n_virt_active) {
  m_results.total_correlation = m_correlation_energy;
  m_results.scs_mp2_correlation = m_c_ss * m_results.same_spin_correlation +
                                  m_c_os * m_results.opposite_spin_correlation;
  m_results.n_frozen_core = n_occ_total - n_occ_active;
  m_results.n_active_occ = n_occ_active;
  m_results.n_active_virt = n_virt_active;
  m_results.n_total_occ = n_occ_total;
  m_results.n_total_virt = n_virt_total;
  m_results.e_min_used = m_e_min;
  m_results.e_max_used = m_e_max;
}

std::array<size_t, 3>
MP2::spin_active_ranges(size_t n_occ_spin, Eigen::Ref<const Vec> eps) const {
  const size_t nbf = m_mo.n_ao;
  const size_t n_virt_total = nbf - n_occ_spin;

  size_t n_frozen_energy = 0;
  for (size_t i = 0; i < n_occ_spin; ++i)
    if (eps(static_cast<Eigen::Index>(i)) < m_e_min)
      ++n_frozen_energy;
  size_t n_frozen = std::min(std::max(m_n_frozen_core, n_frozen_energy),
                             n_occ_spin == 0 ? 0 : n_occ_spin);
  const size_t n_occ_active = n_occ_spin - n_frozen;

  size_t n_virt_active = 0;
  for (size_t a = 0; a < n_virt_total; ++a) {
    const double ve = eps(static_cast<Eigen::Index>(n_occ_spin + a));
    if (ve <= m_e_max && ve <= m_virtual_cutoff &&
        n_virt_active < m_max_virtuals)
      ++n_virt_active;
    else if (ve > m_e_max)
      break;
  }
  return {n_frozen, n_occ_active, n_virt_active};
}

double MP2::compute_unrestricted_ri_energy() {
  const size_t nbf = m_mo.n_ao;
  const Vec eps_a = m_mo.energies.head(static_cast<Eigen::Index>(nbf));
  const Vec eps_b =
      m_mo.energies.segment(static_cast<Eigen::Index>(nbf),
                            static_cast<Eigen::Index>(nbf));
  const Mat Ca = block::a(m_mo.C); // nbf x nbf (alpha)
  const Mat Cb = block::b(m_mo.C); // nbf x nbf (beta)

  const auto ra = spin_active_ranges(m_mo.n_alpha, eps_a);
  const auto rb = spin_active_ranges(m_mo.n_beta, eps_b);
  const size_t oa = ra[1], vaa = ra[2];
  const size_t ob = rb[1], vab = rb[2];

  const Mat Coa = Ca.middleCols(ra[0], oa);
  const Mat Cva = Ca.middleCols(m_mo.n_alpha, vaa);
  const Mat Cob = Cb.middleCols(rb[0], ob);
  const Mat Cvb = Cb.middleCols(m_mo.n_beta, vab);
  const Vec eoa = eps_a.segment(static_cast<Eigen::Index>(ra[0]), oa);
  const Vec eva = eps_a.segment(static_cast<Eigen::Index>(m_mo.n_alpha), vaa);
  const Vec eob = eps_b.segment(static_cast<Eigen::Index>(rb[0]), ob);
  const Vec evb = eps_b.segment(static_cast<Eigen::Index>(m_mo.n_beta), vab);

  occ::log::debug("UHF RI-MP2: active occ (a/b) {}/{}, virt {}/{}", oa, ob, vaa,
                  vab);

  DFIntegrals df(*m_df_engine, m_memory_budget);
  const bool have_a = (oa > 0 && vaa > 0);
  const bool have_b = (ob > 0 && vab > 0);
  // Metric-folded B tensors per spin (minimal DF object, o*v*naux each).
  Mat Ba = have_a ? df.build_b_tilde(Coa, Cva) : Mat();
  Mat Bb = have_b ? df.build_b_tilde(Cob, Cvb) : Mat();

  const Eigen::Index VA = static_cast<Eigen::Index>(vaa);
  const Eigen::Index VB = static_cast<Eigen::Index>(vab);

  struct Acc {
    double ss = 0.0;
    double os = 0.0;
  };
  occ::parallel::thread_local_storage<Acc> acc_local;

  occ::timing::start(occ::timing::category::mp2_energy);
  // Same-spin αα and opposite-spin αβ are both driven by the α-occupied loop.
  if (have_a) {
    occ::parallel::parallel_for(size_t(0), oa, [&](size_t i) {
      auto &acc = acc_local.local();
      const auto Bi = Ba.middleRows(static_cast<Eigen::Index>(i) * VA, VA);
      for (size_t j = 0; j < oa; ++j) {
        const auto Bj = Ba.middleRows(static_cast<Eigen::Index>(j) * VA, VA);
        const Mat K = Bi * Bj.transpose(); // (iα aα|jα bα)
        acc.ss += ss_pair_energy(K, eoa(static_cast<Eigen::Index>(i)),
                                 eoa(static_cast<Eigen::Index>(j)), eva);
      }
      if (have_b) {
        for (size_t j = 0; j < ob; ++j) {
          const auto Bj = Bb.middleRows(static_cast<Eigen::Index>(j) * VB, VB);
          const Mat K = Bi * Bj.transpose(); // (iα aα|jβ bβ)
          acc.os += os_pair_energy(K, eoa(static_cast<Eigen::Index>(i)),
                                   eob(static_cast<Eigen::Index>(j)), eva, evb);
        }
      }
    });
  }
  // Same-spin ββ.
  if (have_b) {
    occ::parallel::parallel_for(size_t(0), ob, [&](size_t i) {
      auto &acc = acc_local.local();
      const auto Bi = Bb.middleRows(static_cast<Eigen::Index>(i) * VB, VB);
      for (size_t j = 0; j < ob; ++j) {
        const auto Bj = Bb.middleRows(static_cast<Eigen::Index>(j) * VB, VB);
        const Mat K = Bi * Bj.transpose(); // (iβ aβ|jβ bβ)
        acc.ss += ss_pair_energy(K, eob(static_cast<Eigen::Index>(i)),
                                 eob(static_cast<Eigen::Index>(j)), evb);
      }
    });
  }
  occ::timing::stop(occ::timing::category::mp2_energy);

  double ss = 0.0, os = 0.0;
  for (const auto &a : acc_local) {
    ss += a.ss;
    os += a.os;
  }
  m_results.same_spin_correlation = ss;
  m_results.opposite_spin_correlation = os;
  return ss + os;
}

double MP2::compute_unrestricted_conventional_energy() {
  using T3 = Eigen::Tensor<double, 3>;
  const Eigen::Index N = static_cast<Eigen::Index>(m_mo.n_ao);
  const size_t nbf = m_mo.n_ao;
  const Vec eps_a = m_mo.energies.head(static_cast<Eigen::Index>(nbf));
  const Vec eps_b =
      m_mo.energies.segment(static_cast<Eigen::Index>(nbf),
                            static_cast<Eigen::Index>(nbf));
  const Mat Ca = block::a(m_mo.C);
  const Mat Cb = block::b(m_mo.C);

  const auto ra = spin_active_ranges(m_mo.n_alpha, eps_a);
  const auto rb = spin_active_ranges(m_mo.n_beta, eps_b);
  const size_t oa = ra[1], vaa = ra[2];
  const size_t ob = rb[1], vab = rb[2];

  const Mat Coa = Ca.middleCols(ra[0], oa);
  const Mat Cva = Ca.middleCols(m_mo.n_alpha, vaa);
  const Mat Cob = Cb.middleCols(rb[0], ob);
  const Mat Cvb = Cb.middleCols(m_mo.n_beta, vab);
  const Vec eoa = eps_a.segment(static_cast<Eigen::Index>(ra[0]), oa);
  const Vec eva = eps_a.segment(static_cast<Eigen::Index>(m_mo.n_alpha), vaa);
  const Vec eob = eps_b.segment(static_cast<Eigen::Index>(rb[0]), ob);
  const Vec evb = eps_b.segment(static_cast<Eigen::Index>(m_mo.n_beta), vab);

  Eigen::TensorMap<const Eigen::Tensor<double, 2>> CvA(
      Cva.data(), N, static_cast<Eigen::Index>(vaa));
  Eigen::TensorMap<const Eigen::Tensor<double, 2>> CvB(
      Cvb.data(), N, static_cast<Eigen::Index>(vab));
  Eigen::TensorMap<const Eigen::Tensor<double, 2>> CoA(
      Coa.data(), N, static_cast<Eigen::Index>(oa));
  Eigen::TensorMap<const Eigen::Tensor<double, 2>> CoB(
      Cob.data(), N, static_cast<Eigen::Index>(ob));
  const Eigen::array<Eigen::IndexPair<int>, 1> con0 = {
      Eigen::IndexPair<int>(0, 0)};
  const Eigen::array<Eigen::IndexPair<int>, 1> con1 = {
      Eigen::IndexPair<int>(1, 0)};
  const Eigen::Index VA = static_cast<Eigen::Index>(vaa);
  const Eigen::Index VB = static_cast<Eigen::Index>(vab);

  const int nthreads = std::max(1, occ::parallel::get_num_threads());
  const size_t bytes_per_i = static_cast<size_t>(N) * N * N * sizeof(double);
  const size_t denom_bytes =
      static_cast<size_t>(nthreads + 1) * std::max<size_t>(1, bytes_per_i);
  const size_t block = std::max<size_t>(1, m_memory_budget / denom_bytes);

  struct Acc {
    double ss = 0.0;
    double os = 0.0;
  };
  occ::parallel::thread_local_storage<Acc> acc_local;

  occ::timing::start(occ::timing::category::mp2_energy);
  // α-occupied drives same-spin αα and opposite-spin αβ.
  if (oa > 0 && vaa > 0) {
    for (size_t Istart = 0; Istart < oa; Istart += block) {
      const Eigen::Index bI =
          static_cast<Eigen::Index>(std::min(block, oa - Istart));
      const Eigen::Tensor<double, 4> H1 = m_ao_engine->ao_direct_half_transform(
          Coa.middleCols(static_cast<Eigen::Index>(Istart), bI));
      occ::parallel::parallel_for(Eigen::Index(0), bI, [&](Eigen::Index il) {
        auto &acc = acc_local.local();
        const double ei = eoa(static_cast<Eigen::Index>(Istart) + il);
        T3 H1_i = H1.chip(il, 0);
        T3 t2 = H1_i.contract(CvA, con0); // ν→aα : (ρ,σ,aα)
        // αα
        {
          T3 t3 = t2.contract(CvA, con1); // σ→bα : (ρ,aα,bα)
          T3 Kt = t3.contract(CoA, con0); // ρ→jα : (aα,bα,jα)
          for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(oa); ++j) {
            Eigen::Map<const Mat> K(Kt.data() + static_cast<size_t>(j) * VA * VA,
                                    VA, VA);
            acc.ss += ss_pair_energy(K, ei, eoa(j), eva);
          }
        }
        // αβ
        if (ob > 0 && vab > 0) {
          T3 t3 = t2.contract(CvB, con1); // σ→bβ : (ρ,aα,bβ)
          T3 Kt = t3.contract(CoB, con0); // ρ→jβ : (aα,bβ,jβ)
          for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(ob); ++j) {
            Eigen::Map<const Mat> K(Kt.data() + static_cast<size_t>(j) * VA * VB,
                                    VA, VB);
            acc.os += os_pair_energy(K, ei, eob(j), eva, evb);
          }
        }
      });
    }
  }
  // β-occupied drives same-spin ββ.
  if (ob > 0 && vab > 0) {
    for (size_t Istart = 0; Istart < ob; Istart += block) {
      const Eigen::Index bI =
          static_cast<Eigen::Index>(std::min(block, ob - Istart));
      const Eigen::Tensor<double, 4> H1 = m_ao_engine->ao_direct_half_transform(
          Cob.middleCols(static_cast<Eigen::Index>(Istart), bI));
      occ::parallel::parallel_for(Eigen::Index(0), bI, [&](Eigen::Index il) {
        auto &acc = acc_local.local();
        const double ei = eob(static_cast<Eigen::Index>(Istart) + il);
        T3 H1_i = H1.chip(il, 0);
        T3 t2 = H1_i.contract(CvB, con0); // ν→aβ : (ρ,σ,aβ)
        T3 t3 = t2.contract(CvB, con1);   // σ→bβ : (ρ,aβ,bβ)
        T3 Kt = t3.contract(CoB, con0);   // ρ→jβ : (aβ,bβ,jβ)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(ob); ++j) {
          Eigen::Map<const Mat> K(Kt.data() + static_cast<size_t>(j) * VB * VB,
                                  VB, VB);
          acc.ss += ss_pair_energy(K, ei, eob(j), evb);
        }
      });
    }
  }
  occ::timing::stop(occ::timing::category::mp2_energy);

  double ss = 0.0, os = 0.0;
  for (const auto &a : acc_local) {
    ss += a.ss;
    os += a.os;
  }
  m_results.same_spin_correlation = ss;
  m_results.opposite_spin_correlation = os;
  return ss + os;
}

} // namespace occ::qm