#pragma once
#include "ints.h"

namespace craso::hf {

using craso::ints::BasisSet;
using craso::ints::Operator;
using craso::ints::shellpair_list_t;
using craso::ints::shellpair_data_t;
using craso::ints::RowMajorMatrix;
using craso::ints::compute_1body_ints;

/// to use precomputed shell pair data must decide on max precision a priori
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const BasisSet& bs1,
                   const BasisSet& bs2 = BasisSet(),
                   double threshold = 1e-12);

class HartreeFock {
public:
    HartreeFock(const std::vector<libint2::Atom>& atoms, const BasisSet& basis);
    const auto& shellpair_list() const { return m_shellpair_list; }
    const auto& shellpair_data() const { return m_shellpair_data; }
    void set_system_charge(int charge) { m_num_e += m_charge; m_charge = charge; m_num_e -= m_charge; }
    int system_charge() const { return m_charge; }
    int num_e() const { return m_num_e; }
    double nuclear_repulsion_energy() const;
    RowMajorMatrix compute_soad() const;
    auto compute_kinetic_energy_integrals() const { return compute_1body_ints<Operator::kinetic>(m_basis, m_shellpair_list)[0]; }
    auto compute_overlap_integrals() const { return compute_1body_ints<Operator::overlap>(m_basis, m_shellpair_list)[0]; }
    auto compute_nuclear_attraction_integrals() const {
        return compute_1body_ints<Operator::nuclear>(m_basis,  m_shellpair_list, libint2::make_point_charges(m_atoms))[0];
    }

private:
    std::vector<libint2::Atom> m_atoms;
    BasisSet m_basis;
    int m_charge{0};
    int m_num_e{0};
    shellpair_list_t m_shellpair_list;  // shellpair list for OBS
    shellpair_data_t m_shellpair_data;  // shellpair data for OBS

};

}