#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <vector>
#include <string>
#include <occ/dft/grid.h>
#include <occ/dft/functional.h>
#include <occ/qm/ints.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/hf.h>
#include <occ/gto/gto.h>
#include <occ/gto/density.h>
#include <occ/qm/basisset.h>
#include <occ/core/energy_components.h>
#include <occ/dft/xc_potential_matrix.h>

namespace occ::dft {
using occ::qm::SpinorbitalKind;
using occ::qm::expectation;
using occ::qm::BasisSet;

using occ::Mat3N;
using occ::MatN4;
using occ::Vec;
using occ::IVec;
using occ::ints::BasisSet;
using occ::ints::compute_1body_ints;
using occ::ints::compute_1body_ints_deriv;
using occ::ints::Operator;
using occ::ints::shellpair_data_t;
using occ::ints::shellpair_list_t;


std::vector<DensityFunctional> parse_method(const std::string& method_string, bool polarized = false);

namespace impl {

template <SpinorbitalKind spinorbital_kind, int derivative_order>
void set_params(DensityFunctional::Params& params, const Mat& rho)
{
    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
        params.rho.col(0) = rho.col(0);
    }
    else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        // correct assignment
        params.rho.col(0) = rho.col(0).alpha();
        params.rho.col(1) = rho.col(0).beta();
    }

    if constexpr(derivative_order > 0) {
        if constexpr(spinorbital_kind == SpinorbitalKind::Restricted)
        {
            params.sigma.col(0) = (
                rho.block(0, 1, rho.rows(), 3).array() * rho.block(0, 1, rho.rows(), 3).array() 
            ).rowwise().sum();
        }
        else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted)
        {
            const auto& rho_alpha = rho.alpha();
            const auto& rho_beta = rho.beta();
            const auto& dx_rho_a = rho_alpha.col(1).array();
            const auto& dy_rho_a = rho_alpha.col(2).array();
            const auto& dz_rho_a = rho_alpha.col(3).array();
            const auto& dx_rho_b = rho_beta.col(1).array();
            const auto& dy_rho_b = rho_beta.col(2).array();
            const auto& dz_rho_b = rho_beta.col(3).array();
            params.sigma.col(0) = dx_rho_a * dx_rho_a + dy_rho_a * dy_rho_a + dz_rho_a * dz_rho_a;
            params.sigma.col(1) = dx_rho_a * dx_rho_b + dy_rho_a * dy_rho_b + dz_rho_a * dz_rho_b;
            params.sigma.col(2) = dx_rho_b * dx_rho_b + dy_rho_b * dy_rho_b + dz_rho_b * dz_rho_b;
        }
    }
    if constexpr(derivative_order > 1)
    {
        if constexpr(spinorbital_kind == SpinorbitalKind::Restricted)
        {
            params.laplacian.col(0) = rho.col(4);
            params.tau.col(0) = rho.col(5);
        }
        else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted)
        {
            params.laplacian.col(0) = rho.col(4).alpha();
            params.laplacian.col(1) = rho.col(4).beta();
            params.tau.col(0) = rho.col(5).alpha();
            params.tau.col(1) = rho.col(5).beta();
        }
    }

}

}

class DFT {

public:
    DFT(const std::string&, const BasisSet&, const std::vector<occ::core::Atom>&, const SpinorbitalKind kind = SpinorbitalKind::Restricted);
    const auto &shellpair_list() const { return m_hf.shellpair_list(); }
    const auto &shellpair_data() const { return m_hf.shellpair_data(); }
    const auto &atoms() const { return m_hf.atoms(); }
    const auto &basis() const { return m_hf.basis(); }

    void set_system_charge(int charge) {
        m_hf.set_system_charge(charge);
    }
    int system_charge() const { return m_hf.system_charge(); }
    int num_e() const { return m_hf.num_e(); }

    double two_electron_energy() const { return m_two_electron_energy; }
    double exchange_correlation_energy() const { return m_exc_dft; }

    bool usual_scf_energy() const { return false; }
    void update_scf_energy(occ::core::EnergyComponents &energy, bool incremental) const
    { 
        if(incremental)
        {
            energy["electronic.2e"] += two_electron_energy();
            energy["electronic"]  += two_electron_energy();
            energy["electronic.dft_xc"] += exchange_correlation_energy();
        }
        else
        {
            energy["electronic"] = energy["electronic.1e"];
            energy["electronic.2e"] = two_electron_energy();
            energy["electronic"]  += two_electron_energy();
            energy["electronic.dft_xc"] = exchange_correlation_energy();
        }
        energy["total"] = energy["electronic"] + energy["nuclear.repulsion"];
    }
    bool supports_incremental_fock_build() const { return false; }

    int density_derivative() const;
    double exact_exchange_factor() const {
        return std::accumulate(m_funcs.begin(), m_funcs.end(), 0.0,
                               [&](double a, const auto& v) { return a + v.exact_exchange_factor(); });
    }

    double nuclear_repulsion_energy() const { return m_hf.nuclear_repulsion_energy(); }

    auto compute_kinetic_matrix() const {
      return m_hf.compute_kinetic_matrix();
    }

    auto compute_overlap_matrix() const {
      return m_hf.compute_overlap_matrix();
    }

    auto compute_nuclear_attraction_matrix() const {
      return m_hf.compute_nuclear_attraction_matrix();
    }
    
    auto compute_point_charge_interaction_matrix(const std::vector<std::pair<double, std::array<double, 3>>> &point_charges) const
    {
        return m_hf.compute_point_charge_interaction_matrix(point_charges);
    }


    auto compute_kinetic_energy_derivatives(unsigned derivative) const {
      return m_hf.compute_kinetic_energy_derivatives(derivative);
    }

    auto compute_nuclear_attraction_derivatives(unsigned derivative) const {
      return m_hf.compute_nuclear_attraction_derivatives(derivative);
    }

    auto compute_overlap_derivatives(unsigned derivative) const {
      return m_hf.compute_overlap_derivatives(derivative);
    }

    auto compute_shellblock_norm(const Mat &A) const {
        return m_hf.compute_shellblock_norm(A);
    }

    auto compute_schwarz_ints() const {
      return m_hf.compute_schwarz_ints();
    }

    template<unsigned int order = 1>
    inline auto compute_electronic_multipole_matrices(const Vec3 &o = {0.0, 0.0, 0.0}) const
    {
        return m_hf.template compute_electronic_multipole_matrices<order>(o);
    }

    template<unsigned int order = 1>
    inline auto compute_electronic_multipoles(SpinorbitalKind k, const Mat& D, const Vec3 &o = {0.0, 0.0, 0.0}) const
    {
        return m_hf.template compute_electronic_multipoles<order>(k, D, o);
    }

    template<unsigned int order = 1>
    inline auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0, 0.0}) const
    {
        return m_hf.template compute_nuclear_multipoles<order>(o);
    }

    template<int derivative_order, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
    Mat compute_fock_dft(const Mat &D, double precision, const Mat& Schwarz)
    {
        using occ::parallel::nthreads;
        const auto& basis = m_hf.basis();
        const auto& atoms = m_hf.atoms();
        size_t F_rows, F_cols;
        size_t nbf = occ::qm::nbf(basis);
        std::tie(F_rows, F_cols) = occ::qm::matrix_dimensions<spinorbital_kind>(nbf);
        Mat F = Mat::Zero(F_rows, F_cols);
        m_two_electron_energy = 0.0;
        m_exc_dft = 0.0;
        double ecoul{0.0}, exc{0.0};
        double exchange_factor = exact_exchange_factor();

        constexpr size_t BLOCKSIZE = 128;
        double total_density_a{0.0}, total_density_b{0.0};
        const auto& D2 = 2 * D;
        DensityFunctional::Family family{DensityFunctional::Family::LDA};
        if constexpr (derivative_order == 1) {
            family = DensityFunctional::Family::GGA;
        }
        if constexpr (derivative_order == 2)
        {
            family = DensityFunctional::Family::MGGA;
        }

        std::vector<Mat> Kt(occ::parallel::nthreads, Mat::Zero(D.rows(), D.cols()));
        std::vector<double> energies(occ::parallel::nthreads, 0.0);
        std::vector<double> alpha_densities(occ::parallel::nthreads, 0.0);
        std::vector<double> beta_densities(occ::parallel::nthreads, 0.0);
        const auto& funcs = m_funcs;
        if(m_gto_values_cache.size() < m_atom_grids.size())
        {
            m_gto_values_cache.resize(m_atom_grids.size());
        }
        size_t atom_grid_idx{0};
        for(const auto& atom_grid : m_atom_grids) {
            const auto& atom_pts = atom_grid.points;
            const auto& atom_weights = atom_grid.weights;
            const size_t npt_total = atom_pts.cols();
            const size_t num_blocks = npt_total / BLOCKSIZE + 1;

            auto &cache = m_gto_values_cache[atom_grid_idx];
            if(cache.size() != num_blocks)
            {
                cache.resize(num_blocks);
            }
            auto lambda = [&](int thread_id)
            {
                Mat rho(BLOCKSIZE, occ::density::num_components(derivative_order));
                for(size_t block = 0; block < num_blocks; block++) {
                    if(block % nthreads != thread_id) continue;
                    Eigen::Index l = block * BLOCKSIZE;
                    Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
                    Eigen::Index npt = u - l;
                    if(npt == 0) continue;
                    auto& k = Kt[thread_id];
                    const auto& pts_block = atom_pts.middleCols(l, npt);
                    const auto& weights_block = atom_weights.segment(l, npt);
                    if(cache[block].phi.rows() == 0)
                    {
                        occ::gto::evaluate_basis(basis, atoms, pts_block, cache[block], derivative_order);
                    }
                    const auto &gto_vals = cache[block];
                    occ::density::evaluate_density<derivative_order, spinorbital_kind>(D2, gto_vals, rho);

                    double max_density_block = rho.col(0).maxCoeff();
                    if constexpr(spinorbital_kind == SpinorbitalKind::Restricted)
                    {
                        alpha_densities[thread_id] += rho.col(0).dot(weights_block);
                    }
                    else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted)
                    {
                        double tot_density_a = rho.col(0).alpha().dot(weights_block);
                        double tot_density_b = rho.col(0).beta().dot(weights_block);
                        alpha_densities[thread_id] += tot_density_a;
                        beta_densities[thread_id] += tot_density_b;
                    }
                    //if(max_density_block < m_density_threshold) continue;

                    DensityFunctional::Params params(npt, family, spinorbital_kind);
                    impl::set_params<spinorbital_kind, derivative_order>(params, rho);

                    DensityFunctional::Result res(npt, family, spinorbital_kind);
                    for(const auto& func: funcs)
                    {
                        res += func.evaluate(params);
                    }

                    Mat KK = Mat::Zero(k.rows(), k.cols());

                    // Weight the arrays by the grid weights
                    res.weight_by(weights_block);
                    xc_potential_matrix<spinorbital_kind, derivative_order>(res, rho, gto_vals, KK, energies[thread_id]);
                    k.noalias() += KK;
                }
            };
            occ::timing::start(occ::timing::category::dft_xc);
            occ::parallel::parallel_do(lambda);
            occ::timing::stop(occ::timing::category::dft_xc);
            atom_grid_idx++;
        }
        double exc_dft{0.0};
        for(size_t i = 0; i < nthreads; i++) {
            F += Kt[i];
            exc_dft += energies[i];
            total_density_a += alpha_densities[i];
            total_density_b += beta_densities[i];
        }
        occ::log::debug("Total density: alpha = {} beta = {}", total_density_a, total_density_b);

        if(exchange_factor != 0.0) {
            Mat J, K;
            std::tie(J, K) = m_hf.compute_JK(spinorbital_kind, D, precision, Schwarz);
            ecoul = expectation<spinorbital_kind>(D, J);
            exc = - expectation<spinorbital_kind>(D, K) * exchange_factor;
            F.noalias() += J;
            F.noalias() -= K * exchange_factor;
        }
        else {
            Mat J = m_hf.compute_J(spinorbital_kind, D, precision, Schwarz);
            ecoul = expectation<spinorbital_kind>(D, J);
            F.noalias() += J;
        }
        occ::log::debug("EXC_dft = {}, EXC = {}, E_coul = {}\n", exc_dft, exc, ecoul);
        m_two_electron_energy += exc_dft + exc + ecoul;
        m_exc_dft += exc_dft;
        return F;
    }


    Mat compute_fock(SpinorbitalKind kind, const Mat& D, double precision, const Mat& Schwarz)
    {
        int deriv = density_derivative();
        switch (kind) {
        case SpinorbitalKind::Unrestricted: {
            switch (deriv) {
                case 0: return compute_fock_dft<0, SpinorbitalKind::Unrestricted>(D, precision, Schwarz);
                case 1: return compute_fock_dft<1, SpinorbitalKind::Unrestricted>(D, precision, Schwarz);
                case 2: return compute_fock_dft<2, SpinorbitalKind::Unrestricted>(D, precision, Schwarz);
                default: throw std::runtime_error("Not implemented: DFT for derivative order > 2");
            }
        }
        case SpinorbitalKind::Restricted: {
            switch (deriv) {
                case 0: return compute_fock_dft<0, SpinorbitalKind::Restricted>(D, precision, Schwarz);
                case 1: return compute_fock_dft<1, SpinorbitalKind::Restricted>(D, precision, Schwarz);
                case 2: return compute_fock_dft<2, SpinorbitalKind::Restricted>(D, precision, Schwarz);
                default: throw std::runtime_error("Not implemented: DFT for derivative order > 2");
            }
        }
        default: throw std::runtime_error("Not implemented: DFT for General spinorbitals");
        }
    }
    const auto& hf() const { return m_hf; }

    Vec electronic_electric_potential_contribution(occ::qm::SpinorbitalKind kind, const Mat &D, const Mat3N &pts) const
    {
        return m_hf.electronic_electric_potential_contribution(kind, D, pts);
    }

    Vec nuclear_electric_potential_contribution(const Mat3N &pts) const
    {
        return m_hf.nuclear_electric_potential_contribution(pts);
    }

    void update_core_hamiltonian(occ::qm::SpinorbitalKind k, const Mat &D, Mat &H) { return; }
    
private:

    SpinorbitalKind m_spinorbital_kind;
    occ::hf::HartreeFock m_hf;
    MolecularGrid m_grid;
    std::vector<DensityFunctional> m_funcs;
    std::vector<AtomGrid> m_atom_grids;
    mutable std::vector<std::vector<occ::gto::GTOValues>> m_gto_values_cache{};
    mutable double m_two_electron_energy{0.0};
    mutable double m_exc_dft{0.0};
    double m_density_threshold{1e-10};
};
}
