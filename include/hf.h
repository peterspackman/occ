#pragma once
#include <libint2.hpp>
#include <unordered_map>
#include <vector>

namespace craso::hf {

using libint2::BasisSet;
using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
using RowMajorMatrix =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// to use precomputed shell pair data must decide on max precision a priori
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const BasisSet& bs1,
                   const BasisSet& bs2 = BasisSet(),
                   double threshold = 1e-12);

class HartreeFock {
public:
    HartreeFock(const BasisSet& basis);
    const auto& shellpair_list() const { return m_shellpair_list; }
    const auto& shellpair_data() const { return m_shellpair_data; }
private:
    BasisSet m_basis;
    shellpair_list_t m_shellpair_list;  // shellpair list for OBS
    shellpair_data_t m_shellpair_data;  // shellpair data for OBS

};

}