#pragma once
#include "ints.h"
#include "wavefunction.h"
#include <memory>

namespace tonto::interaction {
using tonto::MatRM;


class PairInteraction
{
public:
    PairInteraction(const std::shared_ptr<tonto::qm::Wavefunction>& w1,
                    const std::shared_ptr<tonto::qm::Wavefunction>& w2);

private:
    void merge_molecular_orbitals();
    std::shared_ptr<tonto::qm::Wavefunction> m_wfn_a, m_wfn_b;
    tonto::qm::Wavefunction m_wfn;
};

}
