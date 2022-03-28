#include <occ/qm/esp.h>

namespace occ::ints {

Vec compute_electric_potential(const Mat &D, const BasisSet &obs,
                                    const ShellPairList &shellpair_list,
                                    const occ::Mat3N &positions) {
    Vec result(positions.size());
}


}
