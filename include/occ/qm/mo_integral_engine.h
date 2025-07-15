#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/mo.h>

namespace occ::qm {

struct IndexRange {
    size_t start;
    size_t end;
    
    IndexRange(size_t s, size_t e) : start(s), end(e) {}
    size_t size() const { return end - start; }
};

using Tensor4D = Eigen::Tensor<double, 4>;

class MOIntegralEngine {
public:
    explicit MOIntegralEngine(const IntegralEngine& ao_engine, 
                             const MolecularOrbitals& mo);
    
    // Constructor for DF mode
    explicit MOIntegralEngine(const IntegralEngine& ao_engine, 
                             const MolecularOrbitals& mo,
                             const class IntegralEngineDF* df_engine);

    double compute_mo_eri(size_t i, size_t j, size_t k, size_t l) const;
    
    Tensor4D compute_ovov_tensor() const;
    Mat compute_ovov_block() const;
    Mat compute_oovv_block() const;
    Mat compute_ovvv_block() const;
    
    Tensor4D transform_block(const std::array<IndexRange, 4>& ranges) const;
    
    size_t n_occupied() const { return m_n_occ; }
    size_t n_virtual() const { return m_n_virt; }
    size_t n_ao() const { return m_mo.n_ao; }
    
    const MolecularOrbitals& molecular_orbitals() const { return m_mo; }
    const IntegralEngine& ao_engine() const { return m_ao_engine; }

private:
    void setup_mo_coefficients();
    
    double transform_eri_quarter(size_t p, size_t q, size_t r, size_t s,
                                const Mat& C_p, const Mat& C_q,
                                const Mat& C_r, const Mat& C_s) const;
    
    void transform_shell_quartet(const std::array<size_t, 4>& shell_indices,
                                const std::array<Mat*, 4>& coeffs,
                                const std::array<IndexRange, 4>& ranges,
                                Tensor4D& result) const;

    // Helper functions for MO transformation steps
    Tensor4D transform_first_index(const Tensor4D& ao_tensor, size_t n_ao) const;
    Tensor4D transform_second_index(const Tensor4D& half1, size_t n_ao) const;
    Tensor4D transform_third_index(const Tensor4D& half2, size_t n_ao) const;
    void transform_fourth_index(const Tensor4D& half3, size_t n_ao, Tensor4D& result) const;

    const IntegralEngine& m_ao_engine;
    const MolecularOrbitals& m_mo;
    const class IntegralEngineDF* m_df_engine = nullptr;
    
    Mat m_C_occ;
    Mat m_C_virt; 
    size_t m_n_occ;
    size_t m_n_virt;
};

} // namespace occ::qm