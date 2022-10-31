#include <cmath>
#include <occ/core/constants.h>
#include <occ/core/units.h>
#include <occ/interaction/wolf.h>
#include <unsupported/Eigen/SpecialFunctions>

namespace occ::interaction {

double wolf_coulomb_energy(double qi, const Vec3 &pi, Eigen::Ref<const Vec> qj,
                           Eigen::Ref<const Mat3N> pj,
                           const WolfParams &params) {
    using occ::constants::sqrt_pi;
    using std::erfc;
    double eta = params.eta / occ::units::ANGSTROM_TO_BOHR;
    double rc = params.cutoff * occ::units::ANGSTROM_TO_BOHR;
    double trc = erfc(eta * rc) / rc;

    double self_term = qi * qi * (0.5 * trc + eta / sqrt_pi<double>);
    Vec rij =
        (pj.colwise() - pi).colwise().norm() * occ::units::ANGSTROM_TO_BOHR;

    double pair_term =
        qi *
        (qj.array() * ((eta * rij).array().erfc() / rij.array() - trc)).sum();
    return 0.5 * pair_term - self_term;
}

} // namespace occ::interaction
