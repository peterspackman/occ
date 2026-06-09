#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/dft/dft.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>
#include <occ/slater/promolecule.h>
#include <occ/isosurface/volume_data.h>
#include <vector>

namespace occ::isosurface {

using AtomList = std::vector<occ::core::Atom>;
using occ::qm::Wavefunction;

// Pointwise property evaluators: each operator()(points, dest) accumulates a
// scalar property at an arbitrary batch of points. They live in a sub-namespace
// because two of them (ElectronDensityFunctor, DeformationDensityFunctor) share
// names with the grid / marching-cubes field functors in electron_density.h and
// deformation_density.h.
namespace pointwise {

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
  PromolDensityFunctor(const AtomList &atoms);
  void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

  slater::PromoleculeDensity promol;
};

struct ElectronDensityFunctor {
  ElectronDensityFunctor(const Wavefunction &wfn,
                         SpinComponent spin = SpinComponent::Total);
  void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

  const Wavefunction &wfn;
  SpinComponent spin{SpinComponent::Total};
  int mo_index{-1};
};

struct DeformationDensityFunctor {
  DeformationDensityFunctor(const Wavefunction &wfn,
                            SpinComponent = SpinComponent::Total);
  void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);

  PromolDensityFunctor pro_func;
  ElectronDensityFunctor rho_func;
};

struct XCDensityFunctor {
  XCDensityFunctor(const Wavefunction &wfn, const std::string &functional,
                   SpinComponent = SpinComponent::Total);
  void operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest);
  const Wavefunction &wfn;
  dft::DFT ks;
};

} // namespace pointwise
} // namespace occ::isosurface
