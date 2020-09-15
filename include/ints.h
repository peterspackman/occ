#pragma once
#include <unordered_map>
#include <vector>
#include <array>
#include <libint2.hpp>

namespace craso::ints {
    using libint2::BasisSet;
    using libint2::Operator;
    using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
    using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
    using RowMajorMatrix =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    template <Operator obtype, typename OperatorParams = typename libint2::operator_traits<obtype>::oper_params_type>
    std::array<RowMajorMatrix, libint2::operator_traits<obtype>::nopers> 
    compute_1body_ints(const BasisSet& obs, const shellpair_list_t shellpair_list, OperatorParams oparams = OperatorParams());
}