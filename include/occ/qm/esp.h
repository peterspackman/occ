#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>
#include <occ/qm/shellpair.h>
#include <occ/qm/spinorbital.h>
#include <occ/core/atom.h>

namespace occ::ints {

using occ::qm::BasisSet;
using occ::qm::ShellPairData;
using occ::qm::ShellPairList;

Vec compute_electric_potential2(const Mat &D, const BasisSet &obs,
                                const ShellPairList &shellpair_list,
                                const occ::Mat3N &positions);
}
