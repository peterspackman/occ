#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/shellpair.h>


namespace occ::ints {

using occ::qm::BasisSet;
using occ::qm::ShellPairList;
using occ::qm::ShellPairData;


Vec compute_electric_potential(const Mat &D, const BasisSet &obs,
                                    const ShellPairList &shellpair_list,
                                    const occ::Mat3N &positions);
}
