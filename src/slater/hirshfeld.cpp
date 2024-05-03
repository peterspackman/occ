#include <occ/slater/hirshfeld.h>

namespace occ::slater {

StockholderWeight::StockholderWeight(const PromoleculeDensity &inside, const PromoleculeDensity &outside) :
    m_inside(inside), m_outside(outside) {}


void StockholderWeight::set_background_density(float rho) {
    m_background_density = rho;
}

}
