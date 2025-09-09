#include "dft_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <occ/dft/dft.h>
#include <occ/qm/scf.h>
#include <occ/qm/hessians.h>
#include <occ/qm/gradients.h>

using namespace nb::literals;
using occ::dft::DFT;
using occ::io::GridSettings;
using occ::qm::AOBasis;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;

nb::module_ register_dft_bindings(nb::module_ &m) {

  nb::class_<GridSettings>(m, "GridSettings")
      .def(nb::init<>())
      .def_rw("max_angular_points", &GridSettings::max_angular_points)
      .def_rw("min_angular_points", &GridSettings::min_angular_points)
      .def_rw("radial_points", &GridSettings::radial_points)
      .def_rw("radial_precision", &GridSettings::radial_precision)
      .def("__repr__", [](const GridSettings &settings) {
        return fmt::format("<GridSettings ang=({},{}) radial={}, prec={:.2g}>",
                           settings.min_angular_points,
                           settings.max_angular_points, settings.radial_points,
                           settings.radial_precision);
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
                    const GridSettings &>())
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
      .def("compute_gradient",
           [](DFT &dft, const MolecularOrbitals &mo) {
             occ::qm::GradientEvaluator<DFT> grad(dft);
             return grad(mo);
           }, "mo"_a, "Compute atomic gradients for the given molecular orbitals")
      .def("hessian_evaluator",
           [](DFT &dft) {
             return occ::qm::HessianEvaluator<DFT>(dft);
           }, "Create a Hessian evaluator for this DFT object")
      .def("__repr__", [](const DFT &dft) {
        return fmt::format("<DFT {} ({}, {} atoms)>", dft.method_string(),
                           dft.aobasis().name(), dft.atoms().size());
      });

  // Hessian evaluator for DFT
  nb::class_<occ::qm::HessianEvaluator<DFT>>(m, "HessianEvaluatorDFT")
      .def(nb::init<DFT &>(), "dft"_a)
      .def("set_method", &occ::qm::HessianEvaluator<DFT>::set_method, "method"_a)
      .def("set_step_size", &occ::qm::HessianEvaluator<DFT>::set_step_size, "h"_a,
           "Set finite differences step size in Bohr")
      .def("set_use_acoustic_sum_rule", &occ::qm::HessianEvaluator<DFT>::set_use_acoustic_sum_rule, "use"_a,
           "Enable/disable acoustic sum rule optimization")
      .def("step_size", &occ::qm::HessianEvaluator<DFT>::step_size,
           "Get current step size")
      .def("use_acoustic_sum_rule", &occ::qm::HessianEvaluator<DFT>::use_acoustic_sum_rule,
           "Check if acoustic sum rule is enabled")
      .def("nuclear_repulsion", &occ::qm::HessianEvaluator<DFT>::nuclear_repulsion,
           "Compute nuclear repulsion Hessian")
      .def("__call__", &occ::qm::HessianEvaluator<DFT>::operator(), "mo"_a,
           "Compute the full molecular Hessian")
      .def("__repr__", [](const occ::qm::HessianEvaluator<DFT> &hess) {
        return fmt::format("<HessianEvaluatorDFT step_size={:.4f} acoustic_sum_rule={}>",
                           hess.step_size(), hess.use_acoustic_sum_rule());
      });

  return m;
}
