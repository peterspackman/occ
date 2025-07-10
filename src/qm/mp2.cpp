#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/parallel.h>
#include <occ/qm/mp2.h>
#include <occ/qm/mp2_components.h>
#include <occ/qm/opmatrix.h>
#include <mutex>
#include <atomic>
#include <numeric>

namespace occ::qm {

MP2::MP2(const AOBasis& basis,
          const MolecularOrbitals& mo,
          double scf_energy) : PostHFMethod(basis, mo, scf_energy) {
    occ::log::debug("MP2 initialized (conventional)");
    m_algorithm = Conventional;
}

MP2::MP2(const AOBasis& basis,
          const AOBasis& aux_basis,
          const MolecularOrbitals& mo,
          double scf_energy) : PostHFMethod(basis, mo, scf_energy) {
    occ::log::debug("MP2 initialized (RI): {} AO functions, {} auxiliary functions", 
                   basis.nbf(), aux_basis.nbf());
    m_algorithm = RI;
    
    // Initialize DF engine with auxiliary basis
    const auto& atoms = basis.atoms();
    const auto& ao_shells = basis.shells();
    const auto& aux_shells = aux_basis.shells();
    
    m_df_engine = std::make_unique<IntegralEngineDF>(atoms, ao_shells, aux_shells);
    
    // Replace the MO engine with DF-enabled version
    m_mo_engine = std::make_unique<MOIntegralEngine>(*m_ao_engine, m_mo, m_df_engine.get());
}

void MP2::set_frozen_core_auto() {
    constexpr int scandium_z = 21;
    constexpr int sodium_z = 11;
    constexpr int lithium_z = 3;
    constexpr int core_orbitals_sc_and_above = 9;
    constexpr int core_orbitals_na_ar = 5;
    constexpr int core_orbitals_li_ne = 1;
    
    const auto& atoms = m_mo_engine->ao_engine().aobasis().atoms();
    size_t frozen_count = 0;
    
    for (const auto& atom : atoms) {
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
    const Vec& orbital_energies = m_mo.energies;
    
    auto [n_occ_active, n_virt_active] = get_active_orbital_ranges();
    
    occ::log::debug("Active space: {}/{} occupied, {}/{} virtual orbitals", 
                    n_occ_active, n_occ_total, n_virt_active, n_virt_total);
    
    log_frozen_core_info(n_occ_total, n_occ_active, orbital_energies);
    log_virtual_truncation_info(n_occ_total, n_virt_total, n_virt_active, orbital_energies);
    
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
    const Vec& orbital_energies = m_mo.energies;
    
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
        if (virt_energy <= m_e_max && virt_energy <= m_virtual_cutoff && n_virt_active < m_max_virtuals) {
            n_virt_active++;
        } else if (virt_energy > m_e_max) {
            break;
        }
    }
    
    return {n_occ_active, n_virt_active};
}

double MP2::compute_ri_mp2_energy() {
    auto [n_occ_active, n_virt_active] = get_active_orbital_ranges();
    const size_t n_occ_total = n_occupied();
    const size_t n_virt_total = n_virtual();
    
    if (m_df_engine) {
        MP2OrbitalSpec orbital_spec;
        orbital_spec.n_frozen_core = n_occ_total - n_occ_active;
        orbital_spec.n_active_occ = n_occ_active;
        orbital_spec.n_active_virt = n_virt_active;
        orbital_spec.n_total_occ = n_occ_total;
        orbital_spec.n_total_virt = n_virt_total;
        orbital_spec.e_min = m_e_min;
        orbital_spec.e_max = m_e_max;
        
        auto mp2_components = m_df_engine->compute_df_mp2_energy(m_mo, m_mo.energies, orbital_spec);
        
        m_results.same_spin_correlation = mp2_components.same_spin_correlation;
        m_results.opposite_spin_correlation = mp2_components.opposite_spin_correlation;
        m_results.n_frozen_core = mp2_components.orbital_info.n_frozen_core;
        m_results.n_active_occ = mp2_components.orbital_info.n_active_occ;
        m_results.n_active_virt = mp2_components.orbital_info.n_active_virt;
        m_results.n_total_occ = mp2_components.orbital_info.n_total_occ;
        m_results.n_total_virt = mp2_components.orbital_info.n_total_virt;
        m_results.e_min_used = mp2_components.orbital_info.e_min_used;
        m_results.e_max_used = mp2_components.orbital_info.e_max_used;
        
        return mp2_components.total_correlation;
    } else {
        return compute_conventional_mp2_energy();
    }
}

double MP2::compute_conventional_mp2_energy() {
    auto [n_occ_active, n_virt_active] = get_active_orbital_ranges();
    const size_t n_occ_total = n_occupied();
    
    double same_spin_energy = 0.0;
    double opposite_spin_energy = 0.0;
    double total_correlation = 0.0;
    
    const Vec& orbital_energies = m_mo.energies;
    
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    
    if (m_mo.kind == R) {
        // Compute (ov|ov) integrals using four-step transformation
        auto ovov_tensor = m_mo_engine->compute_ovov_tensor();
        
        // Clear pair energies for this calculation
        m_results.pair_energies.clear();
        
        occ::log::debug("Computing MP2 energy with {} threads", occ::parallel::get_num_threads());
        
        // Thread-local storage for results with cache line alignment
        const int num_threads = occ::parallel::get_num_threads();
        constexpr size_t cache_line_size = 64;
        
        struct alignas(cache_line_size) ThreadData {
            double total = 0.0;
            double same_spin = 0.0;
            double opposite_spin = 0.0;
            std::map<std::pair<size_t,size_t>, double> pair_energies;
            char padding[cache_line_size - sizeof(double) * 3];
        };
        
        std::vector<ThreadData> thread_data(num_threads);
        
        auto dense_mp2_lambda = [&](int thread_id) {
            
            size_t i_start = (thread_id * n_occ_active) / num_threads;
            size_t i_end = ((thread_id + 1) * n_occ_active) / num_threads;
            
            auto& local_data = thread_data[thread_id];
            double& local_total = local_data.total;
            double& local_same_spin = local_data.same_spin;
            double& local_opposite_spin = local_data.opposite_spin;
            auto& local_pair_energies = local_data.pair_energies;
            
            for (size_t i = i_start; i < i_end; ++i) {
                for (size_t j = 0; j < n_occ_active; ++j) {
                    for (size_t a = 0; a < n_virt_active; ++a) {
                        for (size_t b = 0; b < n_virt_active; ++b) {
                            
                            // Map to full orbital space indices
                            size_t i_full = i + m_n_frozen_core;
                            size_t j_full = j + m_n_frozen_core;
                            
                            // Get integrals (ia|jb) and (ib|ja) from tensor
                            double integral_iajb = ovov_tensor[i_full][a][j_full][b];
                            double integral_ibja = ovov_tensor[i_full][b][j_full][a];
                            
                            double eps_i = orbital_energies(i_full);
                            double eps_j = orbital_energies(j_full);
                            double eps_a = orbital_energies(n_occ_total + a);
                            double eps_b = orbital_energies(n_occ_total + b);
                            double denominator = eps_i + eps_j - eps_a - eps_b;
                            
                            constexpr double denominator_threshold = 1e-12;
                            if (std::abs(denominator) < denominator_threshold) {
                                continue;
                            }
                            
                            double numerator = integral_iajb * (2.0 * integral_iajb - integral_ibja);
                            double mp2_contribution = numerator / denominator;
                            local_total += mp2_contribution;
                            
                            double opposite_spin_contrib = 2.0 * integral_iajb * integral_iajb / denominator;
                            double same_spin_contrib = -integral_iajb * integral_ibja / denominator;
                            
                            local_opposite_spin += opposite_spin_contrib;
                            local_same_spin += same_spin_contrib;
                            
                            std::pair<size_t,size_t> pair_key = (i <= j) ? std::make_pair(i, j) : std::make_pair(j, i);
                            local_pair_energies[pair_key] += mp2_contribution;
                        }
                    }
                }
            }
        };
        
        occ::parallel::parallel_do_timed(dense_mp2_lambda, occ::timing::category::mp2_energy);
        
        for (const auto& data : thread_data) {
            total_correlation += data.total;
            same_spin_energy += data.same_spin;
            opposite_spin_energy += data.opposite_spin;
            for (const auto& [pair_key, energy] : data.pair_energies) {
                m_results.pair_energies[pair_key] += energy;
            }
        }
        
    } else if (m_mo.kind == U) {
        throw std::runtime_error("Unrestricted MP2 is not implemented");
    } else {
        throw std::runtime_error("General spinorbital MP2 is not implemented");
    }
    
    m_results.same_spin_correlation = same_spin_energy;
    m_results.opposite_spin_correlation = opposite_spin_energy;
    
    return total_correlation;
}

void MP2::log_frozen_core_info(size_t n_occ_total, size_t n_occ_active, const Vec& orbital_energies) const {
    size_t n_frozen_total = n_occ_total - n_occ_active;
    if (n_frozen_total == 0) return;
    
    size_t n_frozen_by_energy = 0;
    for (size_t i = 0; i < n_occ_total; ++i) {
        if (orbital_energies(i) < m_e_min) n_frozen_by_energy++;
    }
    
    occ::log::debug("Frozen core: {} total ({} by energy, {} manual)", 
                    n_frozen_total, n_frozen_by_energy, n_frozen_total - n_frozen_by_energy);
}

void MP2::log_virtual_truncation_info(size_t n_occ_total, size_t n_virt_total, 
                                      size_t n_virt_active, const Vec& orbital_energies) const {
    if (n_virt_active == n_virt_total) return;
    
    double highest_included = orbital_energies(n_occ_total + n_virt_active - 1);
    occ::log::debug("Virtual truncation: {}/{} orbitals, highest energy {:.4f} Hartree", 
                    n_virt_active, n_virt_total, highest_included);
}

void MP2::estimate_memory_requirements(size_t n_ao, size_t n_occ_active, size_t n_virt_active) const {
    constexpr double gb_factor = 1.0 / (1024.0 * 1024.0 * 1024.0);
    constexpr double memory_warning_threshold = 8.0;
    
    double ao_tensor_gb = static_cast<double>(n_ao * n_ao * n_ao * n_ao) * sizeof(double) * gb_factor;
    double ovov_tensor_gb = static_cast<double>(n_occ_active * n_virt_active * n_occ_active * n_virt_active) * sizeof(double) * gb_factor;
    double intermediate_gb = static_cast<double>(n_occ_active * n_ao * n_ao * n_ao) * sizeof(double) * gb_factor;
    double total_peak_gb = ao_tensor_gb + intermediate_gb + ovov_tensor_gb;
    
    if (total_peak_gb > memory_warning_threshold) {
        occ::log::warn("High memory usage estimated: {:.1f} GB. Consider RI-MP2.", total_peak_gb);
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

} // namespace occ::qm