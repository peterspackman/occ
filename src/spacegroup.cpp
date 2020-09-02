#include "spacegroup.h"

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

std::pair<Eigen::VectorXi, Eigen::Matrix3Xd> SpaceGroup::applyAllSymmetryOperations(const Eigen::Matrix3Xd& frac) const
{
    int nSites = frac.cols();
    int nSymops = m_symops.length();
    Eigen::Matrix3Xd transformed(3, nSites * nSymops);
    Eigen::VectorXi generators(nSites * nSymops);
    transformed.block(0, 0, 3, nSites) = frac;
    for(int i = 0; i < nSites; i++) {
        generators(i) = 16484;
    }
    int offset = nSites;
    for(const auto& symop : m_symops) {
        if(symop.isIdentity()) continue;
        int code = symop.toInt();
        generators.block(offset, 0, nSites, 1).setConstant(code);
        transformed.block(0, offset, frac.rows(), frac.cols()) = symop(frac);
        offset += nSites;
    }
    return {generators, transformed};
}
