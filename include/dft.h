#pragma once
#include "linear_algebra.h"
#include <vector>
#include <string>
#include "dft_grid.h"
#include "density_functional.h"
#include "ints.h"
#include "spinorbital.h"
#include "hf.h"

namespace libint2 {
class BasisSet;
class Atom;
}

namespace tonto::dft {
using tonto::qm::SpinorbitalKind;
using tonto::Mat3N;
using tonto::MatRM;
using tonto::MatN4;
using tonto::Vec;
using tonto::IVec;
using tonto::MatRM;
using tonto::ints::BasisSet;
using tonto::ints::compute_1body_ints;
using tonto::ints::compute_1body_ints_deriv;
using tonto::ints::Operator;
using tonto::ints::shellpair_data_t;
using tonto::ints::shellpair_list_t;


std::vector<DensityFunctional> parse_method(const std::string& method_string);

class DFT {

public:
    DFT(const std::string&, const libint2::BasisSet&, const std::vector<libint2::Atom>&);
    const auto &shellpair_list() const { return m_hf.shellpair_list(); }
    const auto &shellpair_data() const { return m_hf.shellpair_data(); }
    const auto &atoms() const { return m_hf.atoms(); }
    const auto &basis() const { return m_hf.basis(); }

    void set_system_charge(int charge) {
        m_hf.set_system_charge(charge);
    }
    int system_charge() const { return m_hf.system_charge(); }
    int num_e() const { return m_hf.num_e(); }

    double two_electron_energy() const { return m_e_alpha + m_e_beta; }
    double two_electron_energy_alpha() const { return m_e_alpha; }
    double two_electron_energy_beta() const { return m_e_beta; }
    bool usual_scf_energy() const { return false; }

    int density_derivative() const;
    double exact_exchange_factor() const {
        return std::accumulate(m_funcs.begin(), m_funcs.end(), 0.0,
                               [&](double a, const auto& v) { return a + v.exact_exchange_factor(); });
    }

    double nuclear_repulsion_energy() const { return m_hf.nuclear_repulsion_energy(); }
    auto compute_kinetic_matrix() {
      return m_hf.compute_kinetic_matrix();
    }
    auto compute_overlap_matrix() {
      return m_hf.compute_overlap_matrix();
    }
    auto compute_nuclear_attraction_matrix() {
      return m_hf.compute_nuclear_attraction_matrix();
    }

    auto compute_kinetic_energy_derivatives(unsigned derivative) {
      return m_hf.compute_kinetic_energy_derivatives(derivative);
    }

    auto compute_nuclear_attraction_derivatives(unsigned derivative) {
      return m_hf.compute_nuclear_attraction_derivatives(derivative);
    }

    auto compute_overlap_derivatives(unsigned derivative) {
      return m_hf.compute_overlap_derivatives(derivative);
    }

    MatRM compute_shellblock_norm(const MatRM &A) const {
        return m_hf.compute_shellblock_norm(A);
    }

    auto compute_schwarz_ints() {
      return m_hf.compute_schwarz_ints();
    }

    MatRM compute_fock(SpinorbitalKind kind, const MatRM &D,
                      double precision = std::numeric_limits<double>::epsilon(),
                      const MatRM &Schwarz = MatRM())
    {
        if(kind == SpinorbitalKind::General) throw std::runtime_error("GKS not implemented");
        if(kind == SpinorbitalKind::Unrestricted) throw std::runtime_error("UKS not implemented");
        return compute_2body_fock(D, precision, Schwarz);
    }

    MatRM
    compute_2body_fock(const MatRM &D,
                       double precision = std::numeric_limits<double>::epsilon(),
                       const MatRM &Schwarz = MatRM()) const;

    std::pair<MatRM, MatRM>
    compute_JK(const MatRM &D,
               double precision = std::numeric_limits<double>::epsilon(),
               const MatRM &Schwarz = MatRM()) const;
private:
    MatRM compute_2body_fock_d0(const MatRM&, double, const MatRM&) const;
    MatRM compute_2body_fock_d1(const MatRM&, double, const MatRM&) const;
    MatRM compute_2body_fock_d2(const MatRM&, double, const MatRM&) const;

    tonto::hf::HartreeFock m_hf;
    DFTGrid m_grid;
    std::vector<DensityFunctional> m_funcs;
    std::vector<std::pair<tonto::Mat3N, tonto::Vec>> m_atom_grids;
    mutable double m_e_alpha;
    mutable double m_e_beta;
};
}
