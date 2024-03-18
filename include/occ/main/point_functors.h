#pragma once
#include <occ/core/atom.h>
#include <vector>
#include <occ/core/linear_algebra.h>
#include <occ/qm/wavefunction.h>
#include <occ/qm/hf.h>

namespace occ::main {

using AtomList = std::vector<occ::core::Atom>;
using occ::qm::Wavefunction;

// Be careful with lifetimes here, these are designed
// to be used as short lived objects/temporaries

struct EEQEspFunctor {
    EEQEspFunctor(const AtomList &a, double charge = 0.0);
    void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

    const AtomList &atoms;
    Vec charges;
};

struct EspFunctor {
    EspFunctor(const Wavefunction &wfn);
    void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

    const Wavefunction &wfn;
    qm::HartreeFock hf;
};

struct ElectronDensityFunctor {
    enum class Spin {
	Total,
	Alpha,
	Beta
    };


    ElectronDensityFunctor(const Wavefunction &wfn, Spin spin = Spin::Total);
    void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

    const Wavefunction &wfn;
    Spin spin{Spin::Total};
};


}
