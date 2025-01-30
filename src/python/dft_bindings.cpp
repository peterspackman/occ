#include "dft_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <occ/dft/dft.h>
#include <occ/qm/scf.h>

using namespace nb::literals;
using occ::dft::DFT;
using occ::io::BeckeGridSettings;
using occ::qm::AOBasis;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;

nb::module_ register_dft_bindings(nb::module_ &m) {

  nb::class_<BeckeGridSettings>(m, "BeckeGridSettings")
      .def(nb::init<>())
      .def_rw("max_angular_points", &BeckeGridSettings::max_angular_points)
      .def_rw("min_angular_points", &BeckeGridSettings::min_angular_points)
      .def_rw("radial_points", &BeckeGridSettings::radial_points)
      .def_rw("radial_precision", &BeckeGridSettings::radial_precision)
      .def("__repr__", [](const BeckeGridSettings &settings) {
        return fmt::format(
            "<BeckeGridSettings ang=({},{}) radial={}, prec={:.2g}>",
            settings.min_angular_points, settings.max_angular_points,
            settings.radial_points, settings.radial_precision);
      });

  using KS = SCF<DFT>;

  nb::class_<KS>(m, "KS")
      .def(nb::init<DFT &>())
      .def(nb::init<DFT &, SpinorbitalKind>())
      .def_rw("convergence_settings", &KS::convergence_settings)
      .def("set_charge_multiplicity", &KS::set_charge_multiplicity)
      .def("set_initial_guess", &KS::set_initial_guess_from_wfn)
      .def("scf_kind", &KS::scf_kind)
      .def("run", &KS::compute_scf_energy)
      .def("compute_scf_energy", &KS::compute_scf_energy)
      .def("wavefunction", &KS::wavefunction)
      .def("__repr__", [](const KS &ks) {
        return fmt::format("<SCF(KS) ({}, {} atoms)>",
                           ks.m_procedure.aobasis().name(),
                           ks.m_procedure.atoms().size());
      });

  nb::class_<DFT>(m, "DFT")
      .def(nb::init<const std::string &, const AOBasis &>())
      .def(nb::init<const std::string &, const AOBasis &,
                    const BeckeGridSettings &>())
      .def("nuclear_attraction_matrix", &DFT::compute_nuclear_attraction_matrix)
      .def("kinetic_matrix", &DFT::compute_kinetic_matrix)
      .def("set_density_fitting_basis", &DFT::set_density_fitting_basis)
      .def("overlap_matrix", &DFT::compute_overlap_matrix)
      .def("nuclear_repulsion", &DFT::nuclear_repulsion_energy)
      .def("set_precision", &DFT::set_precision)
      .def("set_method", &DFT::set_method)
      .def("fock_matrix",
           [](DFT &dft, const MolecularOrbitals &mo) {
             return dft.compute_fock(mo);
           })
      .def(
          "scf",
          [](DFT &dft, SpinorbitalKind kind = SpinorbitalKind::Restricted) {
            return KS(dft, kind);
          },
          "unrestricted"_a = SpinorbitalKind::Restricted)
      .def("__repr__", [](const DFT &dft) {
        return fmt::format("<DFT {} ({}, {} atoms)>", dft.method_string(),
                           dft.aobasis().name(), dft.atoms().size());
      });

  return m;
}
