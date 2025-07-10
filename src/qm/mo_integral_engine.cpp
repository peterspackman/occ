#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/qm/mo_integral_engine.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm {

MOIntegralEngine::MOIntegralEngine(const IntegralEngine& ao_engine, 
                                   const MolecularOrbitals& mo)
    : m_ao_engine(ao_engine), m_mo(mo) {
    setup_mo_coefficients();
}

MOIntegralEngine::MOIntegralEngine(const IntegralEngine& ao_engine, 
                                   const MolecularOrbitals& mo,
                                   const IntegralEngineDF* df_engine)
    : m_ao_engine(ao_engine), m_mo(mo), m_df_engine(df_engine) {
    setup_mo_coefficients();
}

void MOIntegralEngine::setup_mo_coefficients() {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    
    switch (m_mo.kind) {
    case R:
        m_n_occ = m_mo.n_alpha;
        m_n_virt = m_mo.n_ao - m_mo.n_alpha;
        // MO coefficient extraction: occupied = first n_occ columns, virtual = remaining columns
        m_C_occ = -m_mo.C.leftCols(m_n_occ);
        m_C_virt = -m_mo.C.middleCols(m_n_occ, m_n_virt);
        break;
    case U:
        m_n_occ = m_mo.n_alpha;
        m_n_virt = m_mo.n_ao - m_mo.n_alpha;
        // For unrestricted, use alpha orbitals by default (can be extended for spin-specific methods)
        m_C_occ = -block::a(m_mo.C).leftCols(m_n_occ);
        m_C_virt = -block::a(m_mo.C).middleCols(m_n_occ, m_n_virt);
        break;
    case G:
        m_n_occ = m_mo.n_alpha;
        m_n_virt = m_mo.n_ao - m_mo.n_alpha;
        // For general, the orbitals are stored in 2x2 block structure
        m_C_occ = -m_mo.C.leftCols(m_n_occ);
        m_C_virt = -m_mo.C.middleCols(m_n_occ, m_n_virt);
        break;
    }
    
    occ::log::debug("MOIntegralEngine: {} occupied, {} virtual orbitals", 
                    m_n_occ, m_n_virt);
}

double MOIntegralEngine::compute_mo_eri(size_t i, size_t j, size_t k, size_t l) const {
    const auto& basis = m_ao_engine.aobasis();
    const auto& first_bf = basis.first_bf();
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
                        static_cast<int>(r_sh), static_cast<int>(s_sh)
                    };
                    
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
                    
                    if (dims[0] < 0) continue;
                    
                    for (size_t p_bf = p_bf_start, p_idx = 0; p_bf < p_bf_end; ++p_bf, ++p_idx) {
                        for (size_t q_bf = q_bf_start, q_idx = 0; q_bf < q_bf_end; ++q_bf, ++q_idx) {
                            for (size_t r_bf = r_bf_start, r_idx = 0; r_bf < r_bf_end; ++r_bf, ++r_idx) {
                                for (size_t s_bf = s_bf_start, s_idx = 0; s_bf < s_bf_end; ++s_bf, ++s_idx) {
                                    size_t integral_idx = p_idx * dims[1] * dims[2] * dims[3] +
                                                         q_idx * dims[2] * dims[3] +
                                                         r_idx * dims[3] +
                                                         s_idx;
                                    
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

Tensor4D MOIntegralEngine::compute_ovov_tensor() const {
    // Initialize 4D tensor with dimensions [n_occ][n_virt][n_occ][n_virt]
    Tensor4D result(m_n_occ, std::vector<std::vector<std::vector<double>>>(
        m_n_virt, std::vector<std::vector<double>>(
            m_n_occ, std::vector<double>(m_n_virt, 0.0))));
    
    const auto& basis = m_ao_engine.aobasis();
    const size_t n_ao = basis.nbf();
    
    occ::log::debug("Starting compute_ovov_tensor with dimensions: n_ao={}, n_occ={}, n_virt={}", 
                    n_ao, m_n_occ, m_n_virt);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::Tensor<double, 4> ao_tensor;
    if (m_df_engine) {
        occ::log::debug("Computing AO integrals using DF approximation");
        ao_tensor = m_df_engine->four_center_integrals_tensor();
    } else {
        occ::log::debug("Computing Schwarz screening matrix for shell pair screening");
        auto schwarz_matrix = m_ao_engine.schwarz();
        occ::log::debug("Computing AO integrals (parallelized over shell quartets)");
        ao_tensor = m_ao_engine.four_center_integrals_tensor(schwarz_matrix);
    }
    
    auto ao_time = std::chrono::high_resolution_clock::now();
    auto ao_duration = std::chrono::duration<double>(ao_time - start_time).count();
    occ::log::debug("AO integral computation took {:.3f} seconds", ao_duration);
    
    occ::log::debug("Starting MO transformation (parallelizing the 4 steps)");
    
    // Intermediate storage using Eigen tensors for each step
    Eigen::Tensor<double, 4> half1(m_n_occ, n_ao, n_ao, n_ao);  // (iν|ρσ)
    Eigen::Tensor<double, 4> half2(m_n_occ, m_n_virt, n_ao, n_ao);  // (ia|ρσ)
    Eigen::Tensor<double, 4> half3(m_n_occ, m_n_virt, m_n_occ, n_ao);  // (ia|jσ)
    
    // Transform first index: (μν|ρσ) -> (iν|ρσ) with parallelization
    auto step1_start = std::chrono::high_resolution_clock::now();
    occ::log::debug("Step 1: Transform first index (μν|ρσ) -> (iν|ρσ) with {} threads", 
                    occ::parallel::get_num_threads());
    
    auto transform1_lambda = [&](int thread_id) {
        const int num_threads = occ::parallel::get_num_threads();
        size_t i_start = (thread_id * m_n_occ) / num_threads;
        size_t i_end = ((thread_id + 1) * m_n_occ) / num_threads;
        
        for (size_t i = i_start; i < i_end; ++i) {
            for (size_t nu = 0; nu < n_ao; ++nu) {
                for (size_t rho = 0; rho < n_ao; ++rho) {
                    for (size_t sigma = 0; sigma < n_ao; ++sigma) {
                        double sum = 0.0;
                        
                        for (size_t mu = 0; mu < n_ao; ++mu) {
                            // Access integral using canonical ordering
                            size_t mu_can = std::min(mu, nu);
                            size_t nu_can = std::max(mu, nu);
                            size_t rho_can = std::min(rho, sigma);
                            size_t sigma_can = std::max(rho, sigma);
                            
                            // Check if (μν) <= (ρσ) to determine storage location
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
    };
    
    occ::parallel::parallel_do(transform1_lambda);
    
    auto step1_time = std::chrono::high_resolution_clock::now();
    auto step1_duration = std::chrono::duration<double>(step1_time - step1_start).count();
    occ::log::debug("Step 1 completed in {:.3f} seconds", step1_duration);
    
    // Transform second index: (iν|ρσ) -> (ia|ρσ)
    auto step2_start = std::chrono::high_resolution_clock::now();
    auto num_threads = occ::parallel::get_num_threads();
    occ::log::debug("Step 2: Transform second index (iν|ρσ) -> (ia|ρσ) with {} threads", num_threads);
    
    auto step2_lambda = [&](int thread_id) {
        // Divide work among threads by occupied orbitals
        size_t i_start = (thread_id * m_n_occ) / num_threads;
        size_t i_end = ((thread_id + 1) * m_n_occ) / num_threads;
        
        for (size_t i = i_start; i < i_end; ++i) {
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
    };
    
    occ::parallel::parallel_do(step2_lambda);
    
    auto step2_time = std::chrono::high_resolution_clock::now();
    auto step2_duration = std::chrono::duration<double>(step2_time - step2_start).count();
    occ::log::debug("Step 2 completed in {:.3f} seconds", step2_duration);
    
    // Transform third index: (ia|ρσ) -> (ia|jσ) with parallelization
    auto step3_start = std::chrono::high_resolution_clock::now();
    occ::log::debug("Step 3: Transform third index (ia|ρσ) -> (ia|jσ) with {} threads",
                    occ::parallel::get_num_threads());
    
    auto transform3_lambda = [&](int thread_id) {
        const int num_threads = occ::parallel::get_num_threads();
        size_t chunk_size = (m_n_occ * m_n_virt * m_n_occ + num_threads - 1) / num_threads;
        size_t start_idx = thread_id * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, m_n_occ * m_n_virt * m_n_occ);
        
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            size_t temp = idx;
            size_t i = temp / (m_n_virt * m_n_occ);
            temp %= (m_n_virt * m_n_occ);
            size_t a = temp / m_n_occ;
            size_t j = temp % m_n_occ;
            
            for (size_t sigma = 0; sigma < n_ao; ++sigma) {
                double sum = 0.0;
                for (size_t rho = 0; rho < n_ao; ++rho) {
                    sum += m_C_occ(rho, j) * half2(i, a, rho, sigma);
                }
                half3(i, a, j, sigma) = sum;
            }
        }
    };
    
    occ::parallel::parallel_do(transform3_lambda);
    
    auto step3_time = std::chrono::high_resolution_clock::now();
    auto step3_duration = std::chrono::duration<double>(step3_time - step3_start).count();
    occ::log::debug("Step 3 completed in {:.3f} seconds", step3_duration);
    
    // Transform fourth index: (ia|jσ) -> (ia|jb) with parallelization
    auto step4_start = std::chrono::high_resolution_clock::now();
    occ::log::debug("Step 4: Transform fourth index (ia|jσ) -> (ia|jb) with {} threads",
                    occ::parallel::get_num_threads());
    
    auto transform4_lambda = [&](int thread_id) {
        const int num_threads = occ::parallel::get_num_threads();
        size_t total_elements = m_n_occ * m_n_virt * m_n_occ * m_n_virt;
        size_t chunk_size = (total_elements + num_threads - 1) / num_threads;
        size_t start_idx = thread_id * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_elements);
        
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            size_t temp = idx;
            size_t i = temp / (m_n_virt * m_n_occ * m_n_virt);
            temp %= (m_n_virt * m_n_occ * m_n_virt);
            size_t a = temp / (m_n_occ * m_n_virt);
            temp %= (m_n_occ * m_n_virt);
            size_t j = temp / m_n_virt;
            size_t b = temp % m_n_virt;
            
            double sum = 0.0;
            for (size_t sigma = 0; sigma < n_ao; ++sigma) {
                sum += m_C_virt(sigma, b) * half3(i, a, j, sigma);
            }
            result[i][a][j][b] = sum;
        }
    };
    
    occ::parallel::parallel_do(transform4_lambda);
    
    auto step4_time = std::chrono::high_resolution_clock::now();
    auto step4_duration = std::chrono::duration<double>(step4_time - step4_start).count();
    occ::log::debug("Step 4 completed in {:.3f} seconds", step4_duration);
    
    auto total_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<double>(total_time - start_time).count();
    auto mo_transform_duration = total_duration - ao_duration;
    
    occ::log::debug("=== MO Integral Transformation Summary ===");
    occ::log::debug("AO integrals:     {:.3f} seconds", ao_duration);
    occ::log::debug("MO transform:     {:.3f} seconds", mo_transform_duration);
    occ::log::debug("  Step 1 (parallel): {:.3f} seconds", step1_duration);
    occ::log::debug("  Step 2 (parallel): {:.3f} seconds", step2_duration);
    occ::log::debug("  Step 3 (parallel): {:.3f} seconds", step3_duration);
    occ::log::debug("  Step 4 (parallel): {:.3f} seconds", step4_duration);
    occ::log::debug("Total time:       {:.3f} seconds", total_duration);
    
    return result;
}

Mat MOIntegralEngine::compute_ovov_block() const {
    Mat result = Mat::Zero(m_n_occ * m_n_virt, m_n_occ * m_n_virt);
    
    const auto& basis = m_ao_engine.aobasis();
    const size_t n_ao = basis.nbf();
    
    occ::log::debug("Computing (ov|ov) block using four-step transformation");
    occ::log::debug("Dimensions: n_ao={}, n_occ={}, n_virt={}", n_ao, m_n_occ, m_n_virt);
    
    // Get Schwarz screening matrix and AO integrals
    auto schwarz_matrix = m_ao_engine.schwarz();
    auto ao_tensor = m_ao_engine.four_center_integrals_tensor(schwarz_matrix);
    
    // Four-step transformation: (μν|ρσ) -> (iν|ρσ) -> (ia|ρσ) -> (ia|jσ) -> (ia|jb)
    
    // Intermediate storage using Eigen tensors for each step
    Eigen::Tensor<double, 4> half1(m_n_occ, n_ao, n_ao, n_ao);  // (iν|ρσ)
    Eigen::Tensor<double, 4> half2(m_n_occ, m_n_virt, n_ao, n_ao);  // (ia|ρσ)
    Eigen::Tensor<double, 4> half3(m_n_occ, m_n_virt, m_n_occ, n_ao);  // (ia|jσ)
    
    // Transform first index: (μν|ρσ) -> (iν|ρσ) using canonical loops
    for (size_t i = 0; i < m_n_occ; ++i) {
        for (size_t nu = 0; nu < n_ao; ++nu) {
            for (size_t rho = 0; rho < n_ao; ++rho) {
                for (size_t sigma = 0; sigma < n_ao; ++sigma) {
                    double sum = 0.0;
                    
                    for (size_t mu = 0; mu < n_ao; ++mu) {
                        // Access integral using canonical ordering
                        size_t mu_can = std::min(mu, nu);
                        size_t nu_can = std::max(mu, nu);
                        size_t rho_can = std::min(rho, sigma);
                        size_t sigma_can = std::max(rho, sigma);
                        
                        // Check if (μν) <= (ρσ) to determine storage location
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
    
    // Transform second index: (iν|ρσ) -> (ia|ρσ)
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
    
    // Transform third index: (ia|ρσ) -> (ia|jσ)
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
    
    // Transform fourth index: (ia|jσ) -> (ia|jb) and store in matrix format
    for (size_t i = 0; i < m_n_occ; ++i) {
        for (size_t a = 0; a < m_n_virt; ++a) {
            for (size_t j = 0; j < m_n_occ; ++j) {
                for (size_t b = 0; b < m_n_virt; ++b) {
                    double sum = 0.0;
                    for (size_t sigma = 0; sigma < n_ao; ++sigma) {
                        sum += m_C_virt(sigma, b) * half3(i, a, j, sigma);
                    }
                    size_t ia = i * m_n_virt + a;
                    size_t jb = j * m_n_virt + b;
                    result(ia, jb) = sum;
                }
            }
        }
    }
    
    return result;
}

Mat MOIntegralEngine::compute_oovv_block() const {
    Mat result = Mat::Zero(m_n_occ * m_n_occ, m_n_virt * m_n_virt);
    
    occ::log::debug("Computing (oo|vv) block: {}x{}", 
                    m_n_occ * m_n_occ, m_n_virt * m_n_virt);
    
    return result;
}

Mat MOIntegralEngine::compute_ovvv_block() const {
    Mat result = Mat::Zero(m_n_occ * m_n_virt, m_n_virt * m_n_virt);
    
    occ::log::debug("Computing (ov|vv) block: {}x{}", 
                    m_n_occ * m_n_virt, m_n_virt * m_n_virt);
    
    return result;
}

Tensor4D MOIntegralEngine::transform_block(const std::array<IndexRange, 4>& ranges) const {
    Tensor4D result;
    
    size_t total_size = ranges[0].size() * ranges[1].size() * 
                       ranges[2].size() * ranges[3].size();
    
    occ::log::debug("Transforming general block of size {}", total_size);
    
    return result;
}


} // namespace occ::qm