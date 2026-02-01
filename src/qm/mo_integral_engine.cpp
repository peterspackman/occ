#include <chrono>
#include <occ/core/log.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mo_integral_engine.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm {

MOIntegralEngine::MOIntegralEngine(const IntegralEngine &ao_engine,
                                   const MolecularOrbitals &mo)
    : m_ao_engine(ao_engine), m_mo(mo) {
  setup_mo_coefficients();
}

MOIntegralEngine::MOIntegralEngine(const IntegralEngine &ao_engine,
                                   const MolecularOrbitals &mo,
                                   const IntegralEngineDF *df_engine)
    : m_ao_engine(ao_engine), m_mo(mo), m_df_engine(df_engine) {
  setup_mo_coefficients();
}

void MOIntegralEngine::setup_mo_coefficients() {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;

  // Input validation
  if (m_mo.n_alpha > m_mo.n_ao) {
    occ::log::error("Invalid orbital counts: n_alpha ({}) > n_ao ({})",
                    m_mo.n_alpha, m_mo.n_ao);
    m_n_occ = 0;
    m_n_virt = 0;
    return;
  }

  if (m_mo.C.rows() != static_cast<Eigen::Index>(m_mo.n_ao) ||
      m_mo.C.cols() < static_cast<Eigen::Index>(m_mo.n_ao)) {
    occ::log::error("Invalid MO coefficient matrix dimensions: {}x{}, expected "
                    "at least {}x{}",
                    m_mo.C.rows(), m_mo.C.cols(), m_mo.n_ao, m_mo.n_ao);
    m_n_occ = 0;
    m_n_virt = 0;
    return;
  }

  switch (m_mo.kind) {
  case R:
    m_n_occ = m_mo.n_alpha;
    m_n_virt = m_mo.n_ao - m_mo.n_alpha;
    if (m_n_occ > 0 && m_n_virt > 0) {
      // MO coefficient extraction: occupied = first n_occ columns, virtual =
      // remaining columns
      m_C_occ = -m_mo.C.leftCols(m_n_occ);
      m_C_virt = -m_mo.C.middleCols(m_n_occ, m_n_virt);
    }
    break;
  case U:
    m_n_occ = m_mo.n_alpha;
    m_n_virt = m_mo.n_ao - m_mo.n_alpha;
    if (m_n_occ > 0 && m_n_virt > 0) {
      // For unrestricted, use alpha orbitals by default (can be extended for
      // spin-specific methods)
      m_C_occ = -block::a(m_mo.C).leftCols(m_n_occ);
      m_C_virt = -block::a(m_mo.C).middleCols(m_n_occ, m_n_virt);
    }
    break;
  case G:
    m_n_occ = m_mo.n_alpha;
    m_n_virt = m_mo.n_ao - m_mo.n_alpha;
    if (m_n_occ > 0 && m_n_virt > 0) {
      // For general, the orbitals are stored in 2x2 block structure
      m_C_occ = -m_mo.C.leftCols(m_n_occ);
      m_C_virt = -m_mo.C.middleCols(m_n_occ, m_n_virt);
    }
    break;
  }

  occ::log::debug("MOIntegralEngine: {} occupied, {} virtual orbitals", m_n_occ,
                  m_n_virt);
}

double MOIntegralEngine::compute_mo_eri(size_t i, size_t j, size_t k,
                                        size_t l) const {
  const auto &basis = m_ao_engine.aobasis();
  const auto &first_bf = basis.first_bf();
  const size_t nsh = basis.size();

  double result = 0.0;

  for (size_t p_sh = 0; p_sh < nsh; ++p_sh) {
    size_t p_bf_start = first_bf[p_sh];
    size_t p_bf_end = p_bf_start + basis[p_sh].size();

    for (size_t q_sh = 0; q_sh < nsh; ++q_sh) {
      size_t q_bf_start = first_bf[q_sh];
      size_t q_bf_end = q_bf_start + basis[q_sh].size();

      for (size_t r_sh = 0; r_sh < nsh; ++r_sh) {
        size_t r_bf_start = first_bf[r_sh];
        size_t r_bf_end = r_bf_start + basis[r_sh].size();

        for (size_t s_sh = 0; s_sh < nsh; ++s_sh) {
          size_t s_bf_start = first_bf[s_sh];
          size_t s_bf_end = s_bf_start + basis[s_sh].size();

          constexpr auto op = cint::Operator::coulomb;

          std::array<int, 4> shell_idxs{
              static_cast<int>(p_sh), static_cast<int>(q_sh),
              static_cast<int>(r_sh), static_cast<int>(s_sh)};

          auto env = m_ao_engine.env();
          occ::qm::cint::Optimizer opt(env, op, 4);
          auto buffer = std::make_unique<double[]>(env.buffer_size_2e());

          std::array<int, 4> dims;
          if (basis.is_pure()) {
            dims = env.four_center_helper<op, Shell::Kind::Spherical>(
                shell_idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
          } else {
            dims = env.four_center_helper<op, Shell::Kind::Cartesian>(
                shell_idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
          }

          if (dims[0] < 0)
            continue;

          for (size_t p_bf = p_bf_start, p_idx = 0; p_bf < p_bf_end;
               ++p_bf, ++p_idx) {
            for (size_t q_bf = q_bf_start, q_idx = 0; q_bf < q_bf_end;
                 ++q_bf, ++q_idx) {
              for (size_t r_bf = r_bf_start, r_idx = 0; r_bf < r_bf_end;
                   ++r_bf, ++r_idx) {
                for (size_t s_bf = s_bf_start, s_idx = 0; s_bf < s_bf_end;
                     ++s_bf, ++s_idx) {
                  size_t integral_idx = p_idx * dims[1] * dims[2] * dims[3] +
                                        q_idx * dims[2] * dims[3] +
                                        r_idx * dims[3] + s_idx;

                  double ao_integral = buffer[integral_idx];

                  result += m_mo.C(p_bf, i) * m_mo.C(q_bf, j) *
                            m_mo.C(r_bf, k) * m_mo.C(s_bf, l) * ao_integral;
                }
              }
            }
          }
        }
      }
    }
  }

  return result;
}

Tensor4D MOIntegralEngine::transform_first_index(const Tensor4D &ao_tensor,
                                                 size_t n_ao) const {
  Tensor4D half1(m_n_occ, n_ao, n_ao, n_ao);
  half1.setZero();

  auto start_time = std::chrono::high_resolution_clock::now();
  occ::log::debug("Step 1: Transform first index (μν|ρσ) -> (iν|ρσ)");

  for (size_t i = 0; i < m_n_occ; ++i) {
    for (size_t nu = 0; nu < n_ao; ++nu) {
      for (size_t rho = 0; rho < n_ao; ++rho) {
        for (size_t sigma = 0; sigma < n_ao; ++sigma) {
          double sum = 0.0;

          for (size_t mu = 0; mu < n_ao; ++mu) {
            size_t mu_can = std::min(mu, nu);
            size_t nu_can = std::max(mu, nu);
            size_t rho_can = std::min(rho, sigma);
            size_t sigma_can = std::max(rho, sigma);

            size_t munu = mu_can * n_ao + nu_can;
            size_t rhosigma = rho_can * n_ao + sigma_can;

            double integral_value;
            if (munu <= rhosigma) {
              integral_value = ao_tensor(mu_can, nu_can, rho_can, sigma_can);
            } else {
              integral_value = ao_tensor(rho_can, sigma_can, mu_can, nu_can);
            }

            sum += m_C_occ(mu, i) * integral_value;
          }

          half1(i, nu, rho, sigma) = sum;
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time).count();
  occ::log::debug("Step 1 completed in {:.3f} seconds", duration);

  return half1;
}

Tensor4D MOIntegralEngine::transform_second_index(const Tensor4D &half1,
                                                  size_t n_ao) const {
  Tensor4D half2(m_n_occ, m_n_virt, n_ao, n_ao);
  half2.setZero();

  auto start_time = std::chrono::high_resolution_clock::now();
  occ::log::debug("Step 2: Transform second index (iν|ρσ) -> (ia|ρσ)");

  for (size_t i = 0; i < m_n_occ; ++i) {
    for (size_t a = 0; a < m_n_virt; ++a) {
      for (size_t rho = 0; rho < n_ao; ++rho) {
        for (size_t sigma = 0; sigma < n_ao; ++sigma) {
          double sum = 0.0;
          for (size_t nu = 0; nu < n_ao; ++nu) {
            sum += m_C_virt(nu, a) * half1(i, nu, rho, sigma);
          }
          half2(i, a, rho, sigma) = sum;
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time).count();
  occ::log::debug("Step 2 completed in {:.3f} seconds", duration);

  return half2;
}

Tensor4D MOIntegralEngine::transform_third_index(const Tensor4D &half2,
                                                 size_t n_ao) const {
  Tensor4D half3(m_n_occ, m_n_virt, m_n_occ, n_ao);
  half3.setZero();

  auto start_time = std::chrono::high_resolution_clock::now();
  occ::log::debug("Step 3: Transform third index (ia|ρσ) -> (ia|jσ)");

  for (size_t i = 0; i < m_n_occ; ++i) {
    for (size_t a = 0; a < m_n_virt; ++a) {
      for (size_t j = 0; j < m_n_occ; ++j) {
        for (size_t sigma = 0; sigma < n_ao; ++sigma) {
          double sum = 0.0;
          for (size_t rho = 0; rho < n_ao; ++rho) {
            sum += m_C_occ(rho, j) * half2(i, a, rho, sigma);
          }
          half3(i, a, j, sigma) = sum;
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time).count();
  occ::log::debug("Step 3 completed in {:.3f} seconds", duration);

  return half3;
}

void MOIntegralEngine::transform_fourth_index(const Tensor4D &half3,
                                              size_t n_ao,
                                              Tensor4D &result) const {
  auto start_time = std::chrono::high_resolution_clock::now();
  occ::log::debug("Step 4: Transform fourth index (ia|jσ) -> (ia|jb)");

  for (size_t i = 0; i < m_n_occ; ++i) {
    for (size_t a = 0; a < m_n_virt; ++a) {
      for (size_t j = 0; j < m_n_occ; ++j) {
        for (size_t b = 0; b < m_n_virt; ++b) {
          double sum = 0.0;
          for (size_t sigma = 0; sigma < n_ao; ++sigma) {
            sum += m_C_virt(sigma, b) * half3(i, a, j, sigma);
          }
          result(i, a, j, b) = sum;
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time).count();
  occ::log::debug("Step 4 completed in {:.3f} seconds", duration);
}

Tensor4D MOIntegralEngine::compute_ovov_tensor() const {
  // Input validation
  if (m_n_occ == 0 || m_n_virt == 0) {
    occ::log::warn("compute_ovov_tensor: zero orbitals (n_occ={}, n_virt={})",
                   m_n_occ, m_n_virt);
    return Tensor4D(0, 0, 0, 0);
  }

  // Initialize 4D tensor with dimensions [n_occ][n_virt][n_occ][n_virt]
  Tensor4D result(m_n_occ, m_n_virt, m_n_occ, m_n_virt);
  result.setZero();

  const auto &basis = m_ao_engine.aobasis();
  const size_t n_ao = basis.nbf();

  if (n_ao == 0) {
    occ::log::warn("compute_ovov_tensor: zero basis functions");
    return Tensor4D(0, 0, 0, 0);
  }

  // Validate MO coefficient dimensions
  if (m_C_occ.rows() != static_cast<Eigen::Index>(n_ao) ||
      m_C_occ.cols() != static_cast<Eigen::Index>(m_n_occ) ||
      m_C_virt.rows() != static_cast<Eigen::Index>(n_ao) ||
      m_C_virt.cols() != static_cast<Eigen::Index>(m_n_virt)) {
    occ::log::error("compute_ovov_tensor: MO coefficient dimension mismatch");
    occ::log::error("  n_ao={}, n_occ={}, n_virt={}", n_ao, m_n_occ, m_n_virt);
    occ::log::error("  C_occ: {}x{}, C_virt: {}x{}", m_C_occ.rows(),
                    m_C_occ.cols(), m_C_virt.rows(), m_C_virt.cols());
    return result;
  }

  occ::log::debug("Starting compute_ovov_tensor with dimensions: n_ao={}, "
                  "n_occ={}, n_virt={}",
                  n_ao, m_n_occ, m_n_virt);

  auto start_time = std::chrono::high_resolution_clock::now();

  Tensor4D ao_tensor;
  if (m_df_engine) {
    occ::log::debug("Computing AO integrals using DF approximation");
    ao_tensor = m_df_engine->four_center_integrals_tensor();
  } else {
    occ::log::debug(
        "Computing Schwarz screening matrix for shell pair screening");
    auto schwarz_matrix = m_ao_engine.schwarz();
    occ::log::debug(
        "Computing AO integrals (parallelized over shell quartets)");
    ao_tensor = m_ao_engine.four_center_integrals_tensor(schwarz_matrix);
  }

  // Validate AO tensor dimensions
  if (ao_tensor.dimension(0) != n_ao || ao_tensor.dimension(1) != n_ao ||
      ao_tensor.dimension(2) != n_ao || ao_tensor.dimension(3) != n_ao) {
    occ::log::error("compute_ovov_tensor: AO tensor dimension mismatch");
    occ::log::error("  Expected: {}x{}x{}x{}", n_ao, n_ao, n_ao, n_ao);
    occ::log::error("  Got: {}x{}x{}x{}", ao_tensor.dimension(0),
                    ao_tensor.dimension(1), ao_tensor.dimension(2),
                    ao_tensor.dimension(3));
    return result;
  }

  auto ao_time = std::chrono::high_resolution_clock::now();
  auto ao_duration =
      std::chrono::duration<double>(ao_time - start_time).count();
  occ::log::debug("AO integral computation took {:.3f} seconds", ao_duration);

  occ::log::debug("Starting MO transformation (using separate functions)");

  // Perform 4-step transformation using separate functions
  auto half1 = transform_first_index(ao_tensor, n_ao);
  occ::log::debug("First transformation completed");

  auto half2 = transform_second_index(half1, n_ao);
  occ::log::debug("Second transformation completed");

  auto half3 = transform_third_index(half2, n_ao);
  occ::log::debug("Third transformation completed");

  transform_fourth_index(half3, n_ao, result);
  occ::log::debug("Fourth transformation completed");

  auto total_time = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration<double>(total_time - start_time).count();
  auto mo_transform_duration = total_duration - ao_duration;

  occ::log::debug("=== MO Integral Transformation Summary ===");
  occ::log::debug("AO integrals:     {:.3f} seconds", ao_duration);
  occ::log::debug("MO transform:     {:.3f} seconds", mo_transform_duration);
  occ::log::debug("Total time:       {:.3f} seconds", total_duration);

  return result;
}

Mat MOIntegralEngine::compute_ovov_block() const {
  // Use the tensor version and convert to matrix format
  auto tensor = compute_ovov_tensor();

  // Input validation
  if (m_n_occ == 0 || m_n_virt == 0) {
    return Mat::Zero(0, 0);
  }

  Mat result = Mat::Zero(m_n_occ * m_n_virt, m_n_occ * m_n_virt);

  for (size_t i = 0; i < m_n_occ; ++i) {
    for (size_t a = 0; a < m_n_virt; ++a) {
      for (size_t j = 0; j < m_n_occ; ++j) {
        for (size_t b = 0; b < m_n_virt; ++b) {
          size_t ia = i * m_n_virt + a;
          size_t jb = j * m_n_virt + b;
          result(ia, jb) = tensor(i, a, j, b);
        }
      }
    }
  }

  return result;
}

Mat MOIntegralEngine::compute_oovv_block() const {
  Mat result = Mat::Zero(m_n_occ * m_n_occ, m_n_virt * m_n_virt);

  occ::log::debug("Computing (oo|vv) block: {}x{}", m_n_occ * m_n_occ,
                  m_n_virt * m_n_virt);

  return result;
}

Mat MOIntegralEngine::compute_ovvv_block() const {
  Mat result = Mat::Zero(m_n_occ * m_n_virt, m_n_virt * m_n_virt);

  occ::log::debug("Computing (ov|vv) block: {}x{}", m_n_occ * m_n_virt,
                  m_n_virt * m_n_virt);

  return result;
}

Tensor4D MOIntegralEngine::transform_block(
    const std::array<IndexRange, 4> &ranges) const {
  Tensor4D result;

  size_t total_size =
      ranges[0].size() * ranges[1].size() * ranges[2].size() * ranges[3].size();

  occ::log::debug("Transforming general block of size {}", total_size);

  return result;
}

} // namespace occ::qm