#include <occ/core/log.h>
#include <occ/crystal/spacegroup.h>

namespace occ::crystal {

SpaceGroup::SpaceGroup(int number) {
    m_sgdata = gemmi::find_spacegroup_by_number(number);
    for (const auto &op : m_sgdata->operations()) {
        m_symops.push_back(SymmetryOperation(op.triplet()));
    }
}

SpaceGroup::SpaceGroup(const std::string &symbol) {
    m_sgdata = gemmi::find_spacegroup_by_name(symbol);
    occ::log::debug("Initializing space group from symbol: {}", symbol);
    if (m_sgdata != nullptr) {
        occ::log::debug("Found space group: {}", m_sgdata->hm);
        for (const auto &op : m_sgdata->operations()) {
            m_symops.push_back(SymmetryOperation(op.triplet()));
        }
    } else {
        occ::log::error(
            "Could not find matching space group: some data will be missing");
    }
}

SpaceGroup::SpaceGroup(const std::vector<std::string> &symops) {
    occ::log::debug(
        "Initializing space group from symops (std::vector<std::string>)");
    gemmi::GroupOps ops;
    for (const auto &symop : symops) {
        ops.sym_ops.push_back(gemmi::parse_triplet(symop));
    }
    m_sgdata = gemmi::find_spacegroup_by_ops(ops);
    if (m_sgdata != nullptr) {
        occ::log::debug("Found space group: {}", m_sgdata->hm);
        for (const auto &op : m_sgdata->operations()) {
            m_symops.push_back(SymmetryOperation(op.triplet()));
        }
    } else {
        occ::log::error(
            "Could not find matching space group: some data will be missing");
        for (const auto &op : symops) {
            m_symops.push_back(SymmetryOperation(op));
        }
    }
}

SpaceGroup::SpaceGroup(const std::vector<SymmetryOperation> &symops)
    : m_symops(symops) {
    occ::log::debug("Initializing space group from symops "
                    "(std::vector<SymmetryOperation>)");
    gemmi::GroupOps ops;
    for (const auto &symop : symops) {
        ops.sym_ops.push_back(gemmi::parse_triplet(symop.to_string()));
    }
    m_sgdata = gemmi::find_spacegroup_by_ops(ops);
    if (m_sgdata != nullptr) {
        occ::log::debug("Found space group: {}", m_sgdata->hm);
        for (const auto &op : m_sgdata->operations()) {
            m_symops.push_back(SymmetryOperation(op.triplet()));
        }
    } else {
        occ::log::error(
            "Could not find matching space group: some data will be missing");
    }
}

bool SpaceGroup::has_H_R_choice() const {
    switch (number()) {
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

std::pair<IVec, Mat3N>
SpaceGroup::apply_all_symmetry_operations(const Mat3N &frac) const {
    int nSites = frac.cols();
    int nSymops = m_symops.size();
    Mat3N transformed(3, nSites * nSymops);
    IVec generators(nSites * nSymops);
    transformed.block(0, 0, 3, nSites) = frac.block(0, 0, 3, nSites);
    for (int i = 0; i < nSites; i++) {
        generators(i) = 16484;
    }
    int offset = nSites;
    for (const auto &symop : m_symops) {
        if (symop.is_identity())
            continue;
        int code = symop.to_int();
        generators.block(offset, 0, nSites, 1).setConstant(code);
        transformed.block(0, offset, frac.rows(), frac.cols()) = symop(frac);
        offset += nSites;
    }
    return {generators, transformed};
}
} // namespace occ::crystal
