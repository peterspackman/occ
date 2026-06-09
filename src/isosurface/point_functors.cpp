#include <occ/core/eeq.h>
#include <occ/core/units.h>
#include <occ/gto/density.h>
#include <occ/isosurface/point_functors.h>

namespace occ::isosurface::pointwise {

std::pair<IVec, Mat3N> atom_nums_positions(const AtomList &atoms) {
  IVec n(atoms.size());
  Mat3N p(3, atoms.size());
  for (int i = 0; i < atoms.size(); i++) {
    n(i) = atoms[i].atomic_number;
    p(0, i) = atoms[i].x * occ::units::BOHR_TO_ANGSTROM;
    p(1, i) = atoms[i].y * occ::units::BOHR_TO_ANGSTROM;
    p(2, i) = atoms[i].z * occ::units::BOHR_TO_ANGSTROM;
  }
  return {n, p};
}

EspFunctor::EspFunctor(const Wavefunction &w) : wfn(w), hf(w.basis) {}

void EspFunctor::operator()(Eigen::Ref<const Mat3N> points,
                            Eigen::Ref<Vec> result) {
  result += hf.electronic_electric_potential_contribution(wfn.mo, points);
  result += hf.nuclear_electric_potential_contribution(points);
}

EEQEspFunctor::EEQEspFunctor(const AtomList &a, double charge) : atoms(a) {
  auto [p, n] = atom_nums_positions(a);
  charges = occ::core::charges::eeq_partial_charges(p, n, 0.0);
}

void EEQEspFunctor::operator()(Eigen::Ref<const Mat3N> points,
                               Eigen::Ref<Vec> result) {
  for (int i = 0; i < atoms.size(); i++) {
    Vec3 pos(atoms[i].x, atoms[i].y, atoms[i].z);
    result.array() +=
        charges(i) / (points.colwise() - pos).colwise().norm().array();
  }
}

namespace {
// Atom coordinates are stored in bohr, which is what
// slater::PromoleculeDensity's raw constructor expects (the Molecule
// overload converts from Angstrom). Build it directly from the atom list.
slater::PromoleculeDensity make_promolecule_density(const AtomList &atoms) {
  IVec numbers(atoms.size());
  FMat3N positions(3, atoms.size());
  for (int i = 0; i < static_cast<int>(atoms.size()); i++) {
    numbers(i) = atoms[i].atomic_number;
    positions(0, i) = static_cast<float>(atoms[i].x);
    positions(1, i) = static_cast<float>(atoms[i].y);
    positions(2, i) = static_cast<float>(atoms[i].z);
  }
  return slater::PromoleculeDensity(numbers, positions);
}
} // namespace

PromolDensityFunctor::PromolDensityFunctor(const AtomList &atoms)
    : promol(make_promolecule_density(atoms)) {}

void PromolDensityFunctor::operator()(Eigen::Ref<const Mat3N> points,
                                      Eigen::Ref<Vec> dest) {
  for (int pt = 0; pt < points.cols(); pt++) {
    const FVec3 pos = points.col(pt).cast<float>();
    dest(pt) += promol(pos);
  }
}

ElectronDensityFunctor::ElectronDensityFunctor(const Wavefunction &w,
                                               SpinComponent s)
    : wfn(w), spin(s) {}

void ElectronDensityFunctor::operator()(Eigen::Ref<const Mat3N> points,
                                        Eigen::Ref<Vec> dest) {
  if (mo_index >= 0) {
    dest += wfn.electron_density_mo(points, mo_index);
  } else {
    dest += wfn.electron_density(points, spin);
  }
}

DeformationDensityFunctor::DeformationDensityFunctor(const Wavefunction &wfn,
                                                     SpinComponent spin)
    : pro_func(wfn.atoms), rho_func(wfn, spin) {}

void DeformationDensityFunctor::operator()(Eigen::Ref<const Mat3N> points,
                                           Eigen::Ref<Vec> dest) {
  Vec tmp = Vec::Zero(dest.rows(), dest.cols());
  rho_func(points, dest);
  pro_func(points, tmp);
  if (rho_func.spin != SpinComponent::Total) {
    dest -= 0.5 * tmp;
  } else {
    dest -= tmp;
  }
}

XCDensityFunctor::XCDensityFunctor(const Wavefunction &w,
                                   const std::string &functional,
                                   SpinComponent s)
    : wfn(w), ks(functional, w.basis) {}

void XCDensityFunctor::operator()(Eigen::Ref<const Mat3N> points,
                                  Eigen::Ref<Vec> dest) {

  Mat D2 = wfn.mo.D * 2;
  Mat rho = occ::density::evaluate_density_on_grid<2>(wfn, points);

  constexpr auto R = qm::SpinorbitalKind::Restricted;
  constexpr auto U = qm::SpinorbitalKind::Unrestricted;

  dft::DensityFunctional::Family family{dft::DensityFunctional::Family::MGGA};
  dft::DensityFunctional::Params params(points.cols(), family, wfn.mo.kind);
  // just always do MGGA for now
  switch (wfn.mo.kind) {
  case R: {
    dft::impl::set_params<R, 2>(params, rho);
    break;
  }
  case U: {
    dft::impl::set_params<U, 2>(params, rho);
    break;
  }
  default: {
    throw std::runtime_error("Not implemented: general spinorbitals with DFT");
  }
  }
  dft::DensityFunctional::Result res(points.cols(), family, wfn.mo.kind);
  const auto &funcs = ks.functionals();
  for (const auto &func : funcs) {
    res += func.evaluate(params);
  }
  dest += res.exc;
}

} // namespace occ::isosurface::pointwise
