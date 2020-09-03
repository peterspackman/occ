#include "spacegroup.h"

namespace craso::crystal {

SpaceGroup::SpaceGroup(const std::string& symbol)
{
    m_sgdata = gemmi::find_spacegroup_by_name(symbol);
}

SpaceGroup::SpaceGroup(const std::vector<SymmetryOperation>& symops) : m_symops(symops)
{
    gemmi::GroupOps ops;
    for(const auto& symop: symops) {
        ops.sym_ops.push_back(gemmi::parse_triplet(symop.to_string()));
    }
    m_sgdata = gemmi::find_spacegroup_by_ops(ops);
}

bool SpaceGroup::has_H_R_choice() const
{
    switch(number()) {
    case 146:
    case 148:
    case 155:
    case 160:
    case 161:
    case 166:
    case 167:
        return true;
    default:
        return false;
    }
}

std::pair<VectorXi, Matrix3Xd> SpaceGroup::apply_all_symmetry_operations(const Eigen::Matrix3Xd& frac) const
{
    int nSites = frac.cols();
    int nSymops = m_symops.size();
    Eigen::Matrix3Xd transformed(3, nSites * nSymops);
    Eigen::VectorXi generators(nSites * nSymops);
    transformed.block(0, 0, 3, nSites) = frac;
    for(int i = 0; i < nSites; i++) {
        generators(i) = 16484;
    }
    int offset = nSites;
    for(const auto& symop : m_symops) {
        if(symop.is_identity()) continue;
        int code = symop.to_int();
        generators.block(offset, 0, nSites, 1).setConstant(code);
        transformed.block(0, offset, frac.rows(), frac.cols()) = symop(frac);
        offset += nSites;
    }
    return {generators, transformed};
}
}
