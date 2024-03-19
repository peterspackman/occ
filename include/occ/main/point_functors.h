#pragma once
#include <occ/core/atom.h>
#include <vector>
#include <occ/core/linear_algebra.h>
#include <occ/qm/wavefunction.h>
#include <occ/qm/hf.h>
#include <occ/core/interpolator.h>
#include <ankerl/unordered_dense.h>

namespace occ::main {

namespace pfimpl {
using LinearInterpolatorFloat =
    occ::core::Interpolator1D<float, occ::core::DomainMapping::Linear>;

struct AtomInterpolator {
    LinearInterpolatorFloat interpolator;
    Eigen::Matrix3Xf positions;
    float threshold{144.0};
};

struct InterpolatorParams {
    int num_points{8192};
    float domain_lower{0.04};
    float domain_upper{144.0};
};

}

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

struct PromolDensityFunctor {
    PromolDensityFunctor(const AtomList &a);
    void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

    AtomList atoms;
    Vec charges;

    pfimpl::InterpolatorParams interpolator_params;
    std::vector<pfimpl::AtomInterpolator> atom_interpolators;
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
    int mo_index{-1};
};


struct DeformationDensityFunctor {
    DeformationDensityFunctor(const Wavefunction &wfn, ElectronDensityFunctor::Spin = ElectronDensityFunctor::Spin::Total);
    void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

    PromolDensityFunctor pro_func;
    ElectronDensityFunctor rho_func;
};

}
