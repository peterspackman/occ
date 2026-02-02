#include <occ/mults/cartesian_interaction.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_rotation.h>

namespace occ::mults {

CartesianInteractions::CartesianInteractions(const Config &config)
    : m_config(config) {}

double CartesianInteractions::compute_interaction_energy(
    const occ::dma::Mult &mult1, const Vec3 &pos1,
    const occ::dma::Mult &mult2, const Vec3 &pos2) const {

    CartesianSite sA, sB;
    spherical_to_cartesian<4>(mult1, sA.cart);
    spherical_to_cartesian<4>(mult2, sB.cart);
    sA.position = pos1;
    sA.rank = sA.cart.effective_rank();
    sB.position = pos2;
    sB.rank = sB.cart.effective_rank();
    return compute_site_pair_energy(sA, sB);
}

double CartesianInteractions::compute_interaction_energy(
    const occ::dma::Mult &mult1, const Vec3 &pos1, const Mat3 &rot1,
    const occ::dma::Mult &mult2, const Vec3 &pos2, const Mat3 &rot2) const {

    CartesianMultipole<4> bodyA, bodyB;
    spherical_to_cartesian<4>(mult1, bodyA);
    spherical_to_cartesian<4>(mult2, bodyB);

    CartesianSite sA, sB;
    rotate_cartesian_multipole<4>(bodyA, rot1, sA.cart);
    rotate_cartesian_multipole<4>(bodyB, rot2, sB.cart);
    sA.position = pos1;
    sA.rank = sA.cart.effective_rank();
    sB.position = pos2;
    sB.rank = sB.cart.effective_rank();
    return compute_site_pair_energy(sA, sB);
}

} // namespace occ::mults
