#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/seminumerical_exchange.h>
#include <occ/qm/spatial_grid_hierarchy.h>
#include <occ/gto/gto.h>
#include <occ/ints/boys.h>

namespace occ::qm::cosx {

using ShellPairList = std::vector<std::vector<size_t>>;
using ShellList = std::vector<gto::Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellKind = gto::Shell::Kind;
using Op = qm::cint::Operator;
using Buffer = std::vector<double>;
using IntegralResult = qm::IntegralEngine::IntegralResult<3>;

// Compute bounding sphere for a batch of grid points
// pts_block is a 3 x N matrix where each column is a grid point
GridBatchInfo compute_batch_info(const occ::Mat3N &pts_block) {
    GridBatchInfo info;

    // Compute centroid as mean of all points
    info.center = pts_block.rowwise().mean();

    // Compute radius as max distance from center to any point
    info.radius = 0.0;
    for (Eigen::Index i = 0; i < pts_block.cols(); ++i) {
        double dist = (pts_block.col(i) - info.center).norm();
        info.radius = std::max(info.radius, dist);
    }

    return info;
}

// Screen shell pairs based on distance from grid batch
// Returns indices of shell pairs that should be evaluated
// screening_extents: shell extents calculated with looser threshold for screening
ScreenedShellPairs screen_shell_pairs(
    const GridBatchInfo &batch,
    const std::vector<occ::gto::Shell> &shells,
    const Vec &screening_extents,
    double margin
) {
    ScreenedShellPairs result;
    const size_t nshells = shells.size();

    // Precompute which shells are "near" the batch
    std::vector<bool> shell_is_near(nshells);
    for (size_t s = 0; s < nshells; ++s) {
        const auto &shell = shells[s];
        double dist = (shell.origin - batch.center).norm();
        // Use screening extents (calculated with 1e-6 threshold) not shell.extent (1e-12)
        double effective_dist = dist - screening_extents(s);
        shell_is_near[s] = (effective_dist < batch.radius + margin);
    }

    // Classify shell pairs
    size_t pair_idx = 0;
    for (size_t p = 0; p < nshells; ++p) {
        for (size_t q = 0; q <= p; ++q, ++pair_idx) {
            bool p_near = shell_is_near[p];
            bool q_near = shell_is_near[q];

            if (p_near && q_near) {
                result.list1.push_back(pair_idx);
            } else if (p_near || q_near) {
                result.list2.push_back(pair_idx);
            }
            // else: list3 - skip entirely
        }
    }

    return result;
}

// Helper function to convert pair_idx to shell indices (p, q)
// Inverse of: pair_idx = p*(p+1)/2 + q where p >= q
inline std::pair<size_t, size_t> pair_index_to_shells(size_t pair_idx) {
    // Solve for p: p = floor((sqrt(8*pair_idx + 1) - 1) / 2)
    size_t p = static_cast<size_t>((std::sqrt(8.0 * pair_idx + 1.0) - 1.0) / 2.0);
    size_t q = pair_idx - p * (p + 1) / 2;
    return {p, q};
}

// =============================================================================
// ESP shell PAIR screening (for seminumerical exchange)
// Screen based on shell pair overlap region, not individual shell extents
// =============================================================================

std::vector<size_t> screen_shell_pairs_esp(
    const GridBatchInfo &batch,
    const occ::ints::ESPEvaluator<double> &esp,
    const ankerl::unordered_dense::map<size_t, size_t> &shell_pair_map,
    double margin  // additional margin in Bohr
) {
    std::vector<size_t> result;
    result.reserve(shell_pair_map.size());

    for (const auto& [flat_idx, esp_idx] : shell_pair_map) {
        auto center = esp.pair_center(esp_idx);
        double extent = esp.pair_extent(esp_idx);

        // Distance from pair center to batch center
        occ::Vec3 P(center[0], center[1], center[2]);
        double dist = (P - batch.center).norm();

        // Include if pair overlaps batch (considering extents and margin)
        // Condition: dist - extent - batch_radius < margin
        if (dist - extent < batch.radius + margin) {
            result.push_back(flat_idx);
        }
    }
    return result;
}

// =============================================================================
// ORCA-style SHARK shell list screening (shell-based, not pair-based)
// See: Helmich-Paris et al. J. Chem. Phys. 155, 104109 (2021), Section II.C
// =============================================================================

std::vector<size_t> screen_shells_geometric(
    const GridBatchInfo &batch,
    const std::vector<occ::gto::Shell> &shells,
    const Vec &screening_extents,
    double margin
) {
    std::vector<size_t> list1;
    const size_t nshells = shells.size();
    list1.reserve(nshells);  // Upper bound

    for (size_t s = 0; s < nshells; ++s) {
        const auto &shell = shells[s];
        double dist = (shell.origin - batch.center).norm();
        // Shell is "near" if its sphere (origin + extent) overlaps with batch sphere
        double effective_dist = dist - screening_extents(s);
        if (effective_dist < batch.radius + margin) {
            list1.push_back(s);
        }
    }
    return list1;
}

std::vector<size_t> screen_shells_density(
    const std::vector<size_t> &list1,
    const Mat &Fg,
    const std::vector<occ::gto::Shell> &shells,
    const std::vector<int> &first_bf,
    double threshold
) {
    std::vector<size_t> list2;
    list2.reserve(list1.size());

    // For each shell in list1, check if any basis function has significant F value
    for (size_t s : list1) {
        int bf_start = first_bf[s];
        int bf_end = bf_start + shells[s].size();

        // Find max |F| across all grid points for this shell's basis functions
        double max_F = 0.0;
        for (int bf = bf_start; bf < bf_end; ++bf) {
            double shell_max = Fg.col(bf).cwiseAbs().maxCoeff();
            max_F = std::max(max_F, shell_max);
        }

        if (max_F > threshold) {
            list2.push_back(s);
        }
    }
    return list2;
}

std::vector<size_t> screen_shells_overlap(
    const std::vector<size_t> &list2,
    const GridBatchInfo &batch,
    const std::vector<occ::gto::Shell> &shells,
    const Vec &screening_extents,
    double threshold
) {
    // For list-3, use the screening_extents (computed with looser threshold)
    // Using shell.extent (tighter threshold, smaller extent) was too aggressive
    // and incorrectly excluded compact shells that still contribute via ESP integrals
    std::vector<size_t> list3;
    list3.reserve(list2.size());

    for (size_t s : list2) {
        const auto &shell = shells[s];
        double dist = (shell.origin - batch.center).norm();
        // Use screening_extents (looser, 1e-6 threshold) not shell.extent (1e-12)
        double effective_dist = dist - screening_extents(s);
        if (effective_dist < batch.radius) {
            list3.push_back(s);
        }
    }
    return list3;
}

ShellLists build_shell_lists(
    const GridBatchInfo &batch,
    const Mat &Fg,
    const std::vector<occ::gto::Shell> &shells,
    const std::vector<int> &first_bf,
    const Vec &screening_extents,
    double f_threshold,
    double overlap_threshold,
    double margin
) {
    ShellLists result;

    // Step 1: Geometric screening (list-1)
    result.list1 = screen_shells_geometric(batch, shells, screening_extents, margin);

    // Step 2: Density-based screening (list-2) - subset of list-1
    result.list2 = screen_shells_density(result.list1, Fg, shells, first_bf, f_threshold);

    // Step 3: Overlap screening (list-3) - subset of list-2
    result.list3 = screen_shells_overlap(result.list2, batch, shells, screening_extents, overlap_threshold);

    return result;
}

// Helper function to create a basis with shell cutoffs calculated
static gto::AOBasis prepare_basis_with_cutoffs(const gto::AOBasis &basis) {
  gto::AOBasis result = basis;
  result.calculate_shell_cutoffs();
  return result;
}

SemiNumericalExchange::SemiNumericalExchange(const gto::AOBasis &basis,
                                             const GridSettings &settings)
    : m_atoms(basis.atoms()),
      m_basis(prepare_basis_with_cutoffs(basis)),
      m_grid(m_basis, settings),
      m_engine(m_basis.atoms(), m_basis.shells()) {
  for (size_t i = 0; i < m_atoms.size(); i++) {
    m_atom_grids.push_back(m_grid.get_partitioned_atom_grid(i));
  }
  m_overlap = m_engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
  m_numerical_overlap = compute_overlap_matrix();
  fmt::print("Max error |Sn - S|: {:12.8f}\n",
             (m_numerical_overlap - m_overlap).array().cwiseAbs().maxCoeff());
  m_overlap_projector = m_numerical_overlap.ldlt().solve(m_overlap);

  // Initialize ESP evaluator (will be populated with shell pairs when used)
  m_esp_evaluator = std::make_unique<occ::ints::ESPEvaluator<double>>(
      occ::ints::boys().table());

  // Calculate screening extents with threshold (default 1e-4)
  // Looser threshold = smaller extents = more aggressive screening
  m_screening_extents = occ::gto::evaluate_decay_cutoff(m_basis, m_settings.screen_threshold);

  // Log screening extent statistics
  occ::log::info("COSX shell screening extents (threshold={:.0e}): min={:.2f}, max={:.2f}, mean={:.2f} Bohr",
                 m_settings.screen_threshold,
                 m_screening_extents.minCoeff(),
                 m_screening_extents.maxCoeff(),
                 m_screening_extents.mean());
}

void SemiNumericalExchange::set_settings(const Settings &settings) {
  // If threshold changed, recompute screening extents
  if (settings.screen_threshold != m_settings.screen_threshold) {
    m_screening_extents = occ::gto::evaluate_decay_cutoff(m_basis, settings.screen_threshold);
    occ::log::info("COSX updated extents (threshold={:.0e}): min={:.2f}, max={:.2f}, mean={:.2f} Bohr",
                   settings.screen_threshold,
                   m_screening_extents.minCoeff(),
                   m_screening_extents.maxCoeff(),
                   m_screening_extents.mean());
  }
  m_settings = settings;
}

// Precompute ESP shell pairs for significant pairs only
// Returns: shell_pair_map mapping (p, q) flat index to ESP evaluator index
//          significant_pairs list of (p, q) pairs
void precompute_significant_shell_pairs(
    const gto::AOBasis &basis,
    const ShellPairList &shellpairs,
    occ::ints::ESPEvaluator<double> &esp,
    ankerl::unordered_dense::map<size_t, size_t> &shell_pair_map,
    std::vector<std::pair<size_t, size_t>> &significant_pairs) {

  shell_pair_map.clear();
  significant_pairs.clear();
  const auto &shells = basis.shells();
  bool spherical = basis.is_pure();

  // Only precompute significant shell pairs
  for (size_t p = 0; p < shellpairs.size(); p++) {
    for (size_t q : shellpairs[p]) {
      // Ensure p >= q for canonical ordering
      size_t pp = std::max(p, q);
      size_t qq = std::min(p, q);
      size_t flat_idx = pp * (pp + 1) / 2 + qq;

      // Skip if already added
      if (shell_pair_map.count(flat_idx) > 0) continue;

      const auto &sh_p = shells[pp];
      const auto &sh_q = shells[qq];

      size_t esp_idx = esp.add_shell_pair(
          sh_p.l, sh_q.l,
          sh_p.num_primitives(), sh_q.num_primitives(),
          sh_p.exponents.data(), sh_q.exponents.data(),
          sh_p.contraction_coefficients.col(0).data(),
          sh_q.contraction_coefficients.col(0).data(),
          sh_p.origin.data(), sh_q.origin.data(),
          spherical);

      shell_pair_map[flat_idx] = esp_idx;
      significant_pairs.push_back({pp, qq});
    }
  }
}

Mat SemiNumericalExchange::compute_overlap_matrix() const {
  const auto &basis = m_engine.aobasis();
  size_t nbf = basis.nbf();
  constexpr size_t BLOCKSIZE = 64;

  occ::parallel::thread_local_storage<Mat> S_local(Mat::Zero(nbf, nbf));

  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();
    const size_t num_blocks = npt_total / BLOCKSIZE + 1;

    occ::parallel::parallel_for(size_t(0), num_blocks, [&](size_t block) {
      auto &S = S_local.local();
      Mat rho(BLOCKSIZE, 1);

      Eigen::Index l = block * BLOCKSIZE;
      Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
      Eigen::Index npt = u - l;
      if (npt <= 0)
        return;

      const auto &pts_block = atom_pts.middleCols(l, npt);
      const auto &weights_block = atom_weights.segment(l, npt);
      occ::gto::GTOValues ao;
      occ::gto::evaluate_basis(basis, pts_block, ao, 0);
      S.noalias() +=
          ao.phi.transpose() *
          (ao.phi.array().colwise() * weights_block.array()).matrix();
    });
  }

  // Reduce results from all threads
  Mat S_result = Mat::Zero(nbf, nbf);
  for (const auto &S_thread : S_local) {
    S_result.noalias() += S_thread;
  }
  return S_result;
}

template <ShellKind kind, typename Lambda>
void three_center_screened_aux_kernel_tbb(
    Lambda &f, qm::cint::IntegralEnvironment &env, const gto::AOBasis &aobasis,
    const gto::AOBasis &auxbasis, const ShellPairList &shellpairs) noexcept {
  
  // Build a list of auxiliary shells to process in parallel
  std::vector<int> aux_shells_to_process;
  for (int auxP = 0; auxP < auxbasis.size(); auxP++) {
    aux_shells_to_process.push_back(auxP);
  }
  
  // Thread-local storage for optimizer and buffer
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
    [&env]() { return occ::qm::cint::Optimizer(env, Op::coulomb, 3); }
  );
  
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
    [&aobasis, &auxbasis]() {
      size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                       auxbasis.max_shell_size();
      return std::make_unique<double[]>(bufsize);
    }
  );
  
  occ::parallel::parallel_for(size_t(0), aux_shells_to_process.size(), [&](size_t idx) {
    auto &opt = opt_local.local();
    auto &buffer = buffer_local.local();
    
    int auxP = aux_shells_to_process[idx];
    const auto &shauxP = auxbasis[auxP];
    const auto &first_bf_ao = aobasis.first_bf();
    const auto &first_bf_aux = auxbasis.first_bf();
    
    IntegralResult args;
    args.thread = 0; // Not used in TBB version
    args.buffer = buffer.get();
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    
    for (int p = 0; p < aobasis.size(); p++) {
      args.bf[0] = first_bf_ao[p];
      args.shell[0] = p;
      const auto &shp = aobasis[p];
      const auto &plist = shellpairs[p];
      if ((shp.extent > 0.0) &&
          (shp.origin - shauxP.origin).norm() > shp.extent) {
        continue;
      }
      for (const int q : plist) {
        args.bf[1] = first_bf_ao[q];
        args.shell[1] = q;
        const auto &shq = aobasis[q];
        std::array<int, 3> shell_idx = {p, q, auxP + static_cast<int>(aobasis.size())};
        if ((shq.extent > 0.0) &&
            (shq.origin - shauxP.origin).norm() > shq.extent) {
          continue;
        }
        args.dims = env.three_center_helper<Op::coulomb, kind>(
            shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  });
}

// Legacy function kept for compatibility
template <ShellKind kind, typename Lambda>
void three_center_screened_aux_kernel(
    Lambda &f, qm::cint::IntegralEnvironment &env, const gto::AOBasis &aobasis,
    const gto::AOBasis &auxbasis, const ShellPairList &shellpairs,
    occ::qm::cint::Optimizer &opt, int thread_id = 0) noexcept {
  // This function is deprecated - use three_center_screened_aux_kernel_tbb instead
}

size_t SemiNumericalExchange::num_grid_points() const {
  return m_grid.get_molecular_grid_points().num_points();
}

size_t SemiNumericalExchange::num_batches() const {
  const auto& gp = m_grid.get_molecular_grid_points();
  if (gp.has_hierarchy()) {
    return gp.get_hierarchy().num_leaves();
  }
  // Fallback: estimate based on default batch size
  constexpr size_t default_batch_size = 128;
  return (gp.num_points() + default_batch_size - 1) / default_batch_size;
}

size_t SemiNumericalExchange::num_atoms() const {
  return m_grid.get_molecular_grid_points().num_atoms();
}

// =============================================================================
// Spinorbital dispatch for compute_K
// =============================================================================

Mat SemiNumericalExchange::compute_K(const qm::MolecularOrbitals &mo,
                                     double precision,
                                     const Mat &Schwarz) const {
  occ::timing::start(occ::timing::category::cosx);

  Mat K;
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    K = compute_K_restricted(mo, precision);
    break;
  case SpinorbitalKind::Unrestricted:
    K = compute_K_unrestricted(mo, precision);
    break;
  case SpinorbitalKind::General:
    K = compute_K_general(mo, precision);
    break;
  }

  occ::timing::stop(occ::timing::category::cosx);
  return K;
}

// =============================================================================
// Restricted spinorbital implementation
// =============================================================================

Mat SemiNumericalExchange::compute_K_restricted(const qm::MolecularOrbitals &mo,
                                                 double precision) const {
  // Use ESP-based approach if enabled
  if (m_use_esp) {
    // Precompute shell pairs if not already done
    if (m_shell_pair_map.empty()) {
      precompute_significant_shell_pairs(m_basis, m_engine.shellpairs(),
                                         *m_esp_evaluator, m_shell_pair_map,
                                         m_significant_pairs);
    }

    // D2 = 2 * D, then project: D2q = S^{-1} * D2
    Mat D2 = 2.0 * mo.D;
    Mat D2q = m_overlap_projector * D2;

    // Use helper to compute K
    Mat K = compute_K_for_density(D2q, precision);
    return 0.25 * (K + K.transpose());
  }

  // Original implementation (backward compatible)
  size_t nbf = m_basis.nbf();
  const auto &D = mo.D;
  Mat D2 = 2 * D;
  constexpr size_t BLOCKSIZE = 128;

  Mat D2q = m_overlap_projector * D2;
  Mat K = Mat::Zero(nbf, nbf);
  const auto &basis = m_engine.aobasis();
  const auto shell2bf = basis.first_bf();
  occ::qm::cint::Optimizer opt(m_engine.env(), Op::coulomb, 3);

  Mat Fg(BLOCKSIZE, nbf);
  Mat Gg(BLOCKSIZE, nbf);
  Mat rho(BLOCKSIZE, 1);

  auto f = [&Gg, &Fg](const qm::IntegralEngine::IntegralResult<3> &args) {
    int n = args.shell[2];
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    Gg.block(n, args.bf[1], 1, args.dims[1]) +=
        Fg.block(n, args.bf[0], 1, args.dims[0]) * tmp;
    if (args.shell[0] != args.shell[1]) {
      Gg.block(n, args.bf[0], 1, args.dims[0]) +=
          Fg.block(n, args.bf[1], 1, args.dims[1]) * tmp.transpose();
    }
  };
  // Lambda function removed - now using TBB directly in the loop below

  std::vector<occ::core::Atom> dummy_atoms;
  std::vector<gto::Shell> aux_shells;

  // compute J, K
  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();
    const size_t num_blocks = npt_total / BLOCKSIZE + 1;
    occ::gto::GTOValues ao;
    for (size_t block = 0; block < num_blocks; block++) {
      Eigen::Index l = block * BLOCKSIZE;
      Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
      Eigen::Index npt = u - l;
      if (npt <= 0)
        continue;

      const auto &pts_block = atom_pts.middleCols(l, npt);
      const auto &weights_block = atom_weights.segment(l, npt);
      occ::gto::evaluate_basis(basis, pts_block, ao, 0);
      if (ao.phi.maxCoeff() < precision)
        continue;

      dummy_atoms.resize(npt);
      aux_shells.resize(npt);
      for (size_t pt = 0; pt < npt; pt++) {
        dummy_atoms[pt] = {0, pts_block(0, pt), pts_block(1, pt),
                           pts_block(2, pt)};
        aux_shells[pt].origin = pts_block.col(pt);
      }

      m_engine.set_dummy_basis(dummy_atoms, aux_shells);

      Mat wao = ao.phi.array().colwise() * weights_block.array();
      Fg = wao * D2q;
      Gg.setZero();

      if (m_engine.is_spherical()) {
        three_center_screened_aux_kernel_tbb<gto::Shell::Kind::Spherical>(
            f, m_engine.env(), m_engine.aobasis(), m_engine.auxbasis(),
            m_engine.shellpairs());
      } else {
        three_center_screened_aux_kernel_tbb<gto::Shell::Kind::Cartesian>(
            f, m_engine.env(), m_engine.aobasis(), m_engine.auxbasis(),
            m_engine.shellpairs());
      }
      K.noalias() -= ao.phi.transpose() * Gg.block(0, 0, npt, nbf);
    }
  }

  return 0.25 * (K + K.transpose());
}

// =============================================================================
// Unrestricted spinorbital implementation
// =============================================================================

Mat SemiNumericalExchange::compute_K_unrestricted(const qm::MolecularOrbitals &mo,
                                                   double precision) const {
  const size_t nbf = m_basis.nbf();

  // Use ESP-based approach if enabled
  if (m_use_esp) {
    // Precompute shell pairs if not already done
    if (m_shell_pair_map.empty()) {
      precompute_significant_shell_pairs(m_basis, m_engine.shellpairs(),
                                         *m_esp_evaluator, m_shell_pair_map,
                                         m_significant_pairs);
    }

    // Extract alpha and beta density blocks: D is (2*nbf, nbf)
    // UHF stores Da = 0.5 * Ca*Ca^T. Multiply by 2 to get true density Ca*Ca^T.
    // Unlike RHF which uses D2 = 2*D (for double occupation), UHF orbitals
    // are singly occupied so we don't need that extra factor.
    Mat Da = 2.0 * qm::block::a(mo.D);  // (nbf, nbf)
    Mat Db = 2.0 * qm::block::b(mo.D);  // (nbf, nbf)

    // Apply overlap projector to each block
    Mat Daq = m_overlap_projector * Da;
    Mat Dbq = m_overlap_projector * Db;

    // Compute K for each spin block
    Mat Ka = compute_K_for_density(Daq, precision);
    Mat Kb = compute_K_for_density(Dbq, precision);

    // Symmetrize with factor 0.5 (vs 0.25 for RHF which has 2x density)
    Ka = 0.5 * (Ka + Ka.transpose());
    Kb = 0.5 * (Kb + Kb.transpose());

    // Combine into (2*nbf, nbf) result
    Mat K(2 * nbf, nbf);
    qm::block::a(K) = Ka;
    qm::block::b(K) = Kb;
    return K;
  }

  // Fallback: non-ESP mode not implemented for unrestricted
  throw std::runtime_error("COSX unrestricted path requires ESP mode (set_use_esp(true))");
}

// =============================================================================
// General spinorbital implementation
// =============================================================================

Mat SemiNumericalExchange::compute_K_general(const qm::MolecularOrbitals &mo,
                                              double precision) const {
  const size_t nbf = m_basis.nbf();

  // Use ESP-based approach if enabled
  if (m_use_esp) {
    // Precompute shell pairs if not already done
    if (m_shell_pair_map.empty()) {
      precompute_significant_shell_pairs(m_basis, m_engine.shellpairs(),
                                         *m_esp_evaluator, m_shell_pair_map,
                                         m_significant_pairs);
    }

    // Extract 4 density blocks: D is (2*nbf, 2*nbf)
    // General D is stored with factor 0.5, so multiply by 2 to get true density
    Mat Daa = 2.0 * qm::block::aa(mo.D);  // (nbf, nbf)
    Mat Dab = 2.0 * qm::block::ab(mo.D);  // (nbf, nbf)
    Mat Dba = 2.0 * qm::block::ba(mo.D);  // (nbf, nbf)
    Mat Dbb = 2.0 * qm::block::bb(mo.D);  // (nbf, nbf)

    // Apply overlap projector to each block
    Mat Daaq = m_overlap_projector * Daa;
    Mat Dabq = m_overlap_projector * Dab;
    Mat Dbaq = m_overlap_projector * Dba;
    Mat Dbbq = m_overlap_projector * Dbb;

    // Compute K for each block
    Mat Kaa = compute_K_for_density(Daaq, precision);
    Mat Kab = compute_K_for_density(Dabq, precision);
    Mat Kba = compute_K_for_density(Dbaq, precision);
    Mat Kbb = compute_K_for_density(Dbbq, precision);

    // Symmetrize diagonal blocks - use 0.5 factor (same as unrestricted)
    Kaa = 0.5 * (Kaa + Kaa.transpose());
    Kbb = 0.5 * (Kbb + Kbb.transpose());
    // Off-diagonal blocks: Kab and Kba are transposes of each other
    Mat Kab_sym = 0.5 * (Kab + Kba.transpose());
    Mat Kba_sym = 0.5 * (Kba + Kab.transpose());

    // Combine into (2*nbf, 2*nbf) result
    Mat K(2 * nbf, 2 * nbf);
    qm::block::aa(K) = Kaa;
    qm::block::ab(K) = Kab_sym;
    qm::block::ba(K) = Kba_sym;
    qm::block::bb(K) = Kbb;
    return K;
  }

  // Fallback: non-ESP mode not implemented for general
  throw std::runtime_error("COSX general spinorbital path requires ESP mode (set_use_esp(true))");
}

// =============================================================================
// Core K computation for a projected density matrix
// =============================================================================

Mat SemiNumericalExchange::compute_K_for_density(const Mat &D2q,
                                                  double precision) const {
  const size_t nbf = m_basis.nbf();
  const auto &basis = m_basis;
  const auto &shells = basis.shells();
  const auto &first_bf = basis.first_bf();
  constexpr size_t BLOCKSIZE = 128;

  Mat K = Mat::Zero(nbf, nbf);

  // Build batch list
  struct BatchInfo {
    size_t atom_idx;
    size_t block_idx;
    Eigen::Index start;
    Eigen::Index count;
  };
  std::vector<BatchInfo> batches;

  for (size_t atom_idx = 0; atom_idx < m_atom_grids.size(); ++atom_idx) {
    const auto& atom_grid = m_atom_grids[atom_idx];
    const size_t npt_total = atom_grid.points.cols();
    const size_t num_blocks = (npt_total + BLOCKSIZE - 1) / BLOCKSIZE;

    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
      Eigen::Index start = block_idx * BLOCKSIZE;
      Eigen::Index end = std::min(static_cast<Eigen::Index>(npt_total),
                                   static_cast<Eigen::Index>((block_idx + 1) * BLOCKSIZE));
      Eigen::Index count = end - start;
      if (count > 0) {
        batches.push_back({atom_idx, block_idx, start, count});
      }
    }
  }

  // Thread-local storage for K matrices
  occ::parallel::thread_local_storage<Mat> K_local([nbf]() { return Mat::Zero(nbf, nbf); });

  // Thread-local buffers for ESP evaluator
  using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  occ::parallel::thread_local_storage<MatRM> esp_workspace_local;
  occ::parallel::thread_local_storage<MatRM> esp_integrals_local;

  // Process batches in parallel
  occ::parallel::parallel_for(size_t(0), batches.size(), [&](size_t batch_idx) {
    const auto& batch = batches[batch_idx];
    const auto& atom_grid = m_atom_grids[batch.atom_idx];

    auto pts_block = atom_grid.points.middleCols(batch.start, batch.count);
    auto weights_block = atom_grid.weights.segment(batch.start, batch.count);
    Eigen::Index npt = batch.count;

    // Quick screening using traditional evaluation
    occ::gto::GTOValues ao;
    occ::gto::evaluate_basis(basis, pts_block, ao, 0);

    if (ao.phi.maxCoeff() < precision)
      return;

    Mat wao = ao.phi.array().colwise() * weights_block.array();
    Mat Fg = wao * D2q;
    Mat Gg = Mat::Zero(npt, nbf);

    constexpr double density_threshold = 1e-6;

    // Get thread-local buffers for this batch
    auto& workspace = esp_workspace_local.local();
    auto& integrals = esp_integrals_local.local();

    // Process all shell pairs sequentially within this batch
    for (const auto& [pair_idx, esp_idx] : m_shell_pair_map) {
      auto [p, q] = pair_index_to_shells(pair_idx);

      int bf_p = first_bf[p];
      int bf_q = first_bf[q];
      int size_p = shells[p].size();
      int size_q = shells[q].size();

      // Density screening based on input density
      double max_D = D2q.block(bf_p, bf_q, size_p, size_q).cwiseAbs().maxCoeff();
      if (p != q) {
        max_D = std::max(max_D, D2q.block(bf_q, bf_p, size_q, size_p).cwiseAbs().maxCoeff());
      }
      if (max_D < density_threshold) continue;

      int nab = m_esp_evaluator->nab(esp_idx);
      int nherm = m_esp_evaluator->nherm(esp_idx);

      // Resize buffers if needed
      if (workspace.rows() != npt || workspace.cols() != nherm) {
        workspace.resize(npt, nherm);
      }
      if (integrals.rows() != npt || integrals.cols() != nab) {
        integrals.resize(npt, nab);
      }

      m_esp_evaluator->evaluate(esp_idx, pts_block, integrals, workspace);

      // Contract integrals with Fg and accumulate into Gg
      for (Eigen::Index pt = 0; pt < npt; pt++) {
        Eigen::Map<const MatRM> V(integrals.row(pt).data(), size_p, size_q);
        Gg.row(pt).segment(bf_q, size_q).noalias() -=
            Fg.row(pt).segment(bf_p, size_p) * V;
        if (p != q) {
          Gg.row(pt).segment(bf_p, size_p).noalias() -=
              Fg.row(pt).segment(bf_q, size_q) * V.transpose();
        }
      }
    }

    K_local.local().noalias() -= ao.phi.transpose() * Gg;
  });

  // Combine results from all threads
  for (const auto& K_thread : K_local) {
    K.noalias() += K_thread;
  }

  return K;
}

} // namespace occ::qm::cosx
