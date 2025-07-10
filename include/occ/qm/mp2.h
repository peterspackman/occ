#pragma once
#include <occ/qm/post_hf_method.h>
#include <occ/qm/integral_engine_df.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <map>
#include <cstddef>

namespace occ::qm {

class MP2 : public PostHFMethod {
public:
    enum Algorithm { Conventional, RI };
    
    MP2(const AOBasis& basis,
        const MolecularOrbitals& mo,
        double scf_energy);
    
    // Constructor for RI-MP2 with auxiliary basis
    MP2(const AOBasis& basis,
        const AOBasis& aux_basis,
        const MolecularOrbitals& mo,
        double scf_energy);
    
    double compute_correlation_energy() override;
    
    // Set frozen core and virtual truncation options
    void set_frozen_core(size_t n_frozen) { m_n_frozen_core = n_frozen; }
    void set_frozen_core_auto(); // Automatically determine frozen core based on atoms
    void set_virtual_cutoff_energy(double cutoff_hartree) { m_virtual_cutoff = cutoff_hartree; }
    void set_max_virtuals(size_t max_virt) { m_max_virtuals = max_virt; }
    
    // Set orbital energy cutoffs (similar to ORCA defaults)
    void set_orbital_energy_cutoffs(double e_min = -1.5, double e_max = 1000.0) {
        m_e_min = e_min;
        m_e_max = e_max;
    }
    
    // Set algorithm choice
    void set_algorithm(Algorithm alg) { m_algorithm = alg; }
    Algorithm algorithm() const { return m_algorithm; }
    
    
    struct Results {
        double same_spin_correlation = 0.0;
        double opposite_spin_correlation = 0.0; 
        double total_correlation = 0.0;
        double scs_mp2_correlation = 0.0;  // Spin-component scaled
        std::map<std::pair<size_t,size_t>, double> pair_energies;  // MP2 pair energies
        
        // Information about approximations used
        size_t n_frozen_core = 0;
        size_t n_active_occ = 0;
        size_t n_active_virt = 0;
        size_t n_total_occ = 0;
        size_t n_total_virt = 0;
        double e_min_used = 0.0;
        double e_max_used = 0.0;
    };
    
    const Results& results() const { return m_results; }
    
    void set_scs_parameters(double c_ss = 1.0/3.0, double c_os = 1.2) {
        m_c_ss = c_ss;
        m_c_os = c_os;
    }
    
private:
    double compute_conventional_mp2_energy();
    double compute_ri_mp2_energy();
    std::pair<size_t, size_t> get_active_orbital_ranges() const;
    
    void log_frozen_core_info(size_t n_occ_total, size_t n_occ_active, const Vec& orbital_energies) const;
    void log_virtual_truncation_info(size_t n_occ_total, size_t n_virt_total, 
                                     size_t n_virt_active, const Vec& orbital_energies) const;
    void estimate_memory_requirements(size_t n_ao, size_t n_occ_active, size_t n_virt_active) const;
    void store_results(size_t n_occ_total, size_t n_virt_total, 
                       size_t n_occ_active, size_t n_virt_active);
    
    Results m_results;
    
    double m_c_ss = 1.0/3.0;
    double m_c_os = 1.2;
    
    size_t m_n_frozen_core = 0;
    double m_virtual_cutoff = 1000.0;
    size_t m_max_virtuals = SIZE_MAX;
    double m_e_min = -1.5;
    double m_e_max = 1000.0;
    Algorithm m_algorithm = Conventional;
    std::unique_ptr<IntegralEngineDF> m_df_engine;
};

} // namespace occ::qm