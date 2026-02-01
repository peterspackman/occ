#include "dft_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/dft/dft.h>
#include <occ/qm/scf.h>
#include <occ/qm/hessians.h>
#include <occ/qm/gradients.h>
#include <occ/qm/external_potential.h>
#include <occ/xdm/xdm.h>

using namespace nb::literals;
using occ::dft::DFT;
using occ::io::GridSettings;
using occ::gto::AOBasis;
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

  // Point charge corrected DFT procedures
  using PointChargeList = std::vector<occ::core::PointCharge>;
  using PointChargeDFT = occ::qm::PointChargeCorrectedProcedure<DFT>;
  using SCF_PointChargeDFT = SCF<PointChargeDFT>;

  nb::class_<PointChargeDFT>(m, "PointChargeDFT")
      .def(nb::init<DFT &, const PointChargeList &>(),
           "dft"_a, "point_charges"_a,
           "Create DFT procedure with external point charge potential")
      .def("nuclear_repulsion", &PointChargeDFT::nuclear_repulsion_energy)
      .def("atoms", &PointChargeDFT::atoms)
      .def("aobasis", &PointChargeDFT::aobasis)
      .def("scf",
           [](PointChargeDFT &proc,
              SpinorbitalKind kind = SpinorbitalKind::Restricted) {
             return SCF_PointChargeDFT(proc, kind);
           },
           "kind"_a = SpinorbitalKind::Restricted,
           "Create SCF driver for point-charge-corrected DFT")
      .def("__repr__", [](const PointChargeDFT &proc) {
        return fmt::format("<PointChargeDFT ({}, {} atoms)>",
                           proc.aobasis().name(), proc.atoms().size());
      });

  nb::class_<SCF_PointChargeDFT>(m, "SCF_PointChargeDFT")
      .def(nb::init<PointChargeDFT &>())
      .def(nb::init<PointChargeDFT &, SpinorbitalKind>())
      .def_rw("convergence_settings", &SCF_PointChargeDFT::convergence_settings)
      .def("set_charge_multiplicity", &SCF_PointChargeDFT::set_charge_multiplicity)
      .def("set_initial_guess", &SCF_PointChargeDFT::set_initial_guess_from_wfn)
      .def("scf_kind", &SCF_PointChargeDFT::scf_kind)
      .def("run", &SCF_PointChargeDFT::compute_scf_energy)
      .def("compute_scf_energy", &SCF_PointChargeDFT::compute_scf_energy)
      .def("wavefunction", &SCF_PointChargeDFT::wavefunction)
      .def("__repr__", [](const SCF_PointChargeDFT &scf) {
        return fmt::format("<SCF(PointChargeDFT) ({}, {} atoms)>",
                           scf.m_procedure.aobasis().name(),
                           scf.m_procedure.atoms().size());
      });

  // Wolf sum corrected DFT procedures
  using WolfDFT = occ::qm::WolfSumCorrectedProcedure<DFT>;
  using SCF_WolfDFT = SCF<WolfDFT>;

  nb::class_<WolfDFT>(m, "WolfDFT")
      .def(nb::init<DFT &, const PointChargeList &,
                    const std::vector<double> &, double, double>(),
           "dft"_a, "point_charges"_a, "molecular_charges"_a, "alpha"_a,
           "cutoff"_a,
           "Create DFT procedure with Wolf sum external potential")
      .def("nuclear_repulsion", &WolfDFT::nuclear_repulsion_energy)
      .def("atoms", &WolfDFT::atoms)
      .def("aobasis", &WolfDFT::aobasis)
      .def("scf",
           [](WolfDFT &proc, SpinorbitalKind kind = SpinorbitalKind::Restricted) {
             return SCF_WolfDFT(proc, kind);
           },
           "kind"_a = SpinorbitalKind::Restricted,
           "Create SCF driver for Wolf-corrected DFT")
      .def("__repr__", [](const WolfDFT &proc) {
        return fmt::format("<WolfDFT ({}, {} atoms)>", proc.aobasis().name(),
                           proc.atoms().size());
      });

  nb::class_<SCF_WolfDFT>(m, "SCF_WolfDFT")
      .def(nb::init<WolfDFT &>())
      .def(nb::init<WolfDFT &, SpinorbitalKind>())
      .def_rw("convergence_settings", &SCF_WolfDFT::convergence_settings)
      .def("set_charge_multiplicity", &SCF_WolfDFT::set_charge_multiplicity)
      .def("set_initial_guess", &SCF_WolfDFT::set_initial_guess_from_wfn)
      .def("scf_kind", &SCF_WolfDFT::scf_kind)
      .def("run", &SCF_WolfDFT::compute_scf_energy)
      .def("compute_scf_energy", &SCF_WolfDFT::compute_scf_energy)
      .def("wavefunction", &SCF_WolfDFT::wavefunction)
      .def("__repr__", [](const SCF_WolfDFT &scf) {
        return fmt::format("<SCF(WolfDFT) ({}, {} atoms)>",
                           scf.m_procedure.aobasis().name(),
                           scf.m_procedure.atoms().size());
      });

  // XDM - Exchange-hole dipole moment dispersion
  nb::class_<occ::xdm::XDM::Parameters>(m, "XDMParameters")
      .def(nb::init<>())
      .def_rw("a1", &occ::xdm::XDM::Parameters::a1,
              "XDM damping parameter a1")
      .def_rw("a2", &occ::xdm::XDM::Parameters::a2,
              "XDM damping parameter a2 (Angstroms)")
      .def("__repr__", [](const occ::xdm::XDM::Parameters &p) {
        return fmt::format("<XDMParameters a1={:.4f} a2={:.4f}>", p.a1, p.a2);
      });

  nb::class_<occ::xdm::XDM>(m, "XDM")
      .def(nb::init<const occ::gto::AOBasis &, int,
                    const occ::xdm::XDM::Parameters &>(),
           "basis"_a, "charge"_a = 0, "params"_a = occ::xdm::XDM::Parameters{},
           "Create XDM dispersion calculator")
      .def("energy", &occ::xdm::XDM::energy, "mo"_a,
           "Compute XDM dispersion energy for given molecular orbitals")
      .def("forces", &occ::xdm::XDM::forces, "mo"_a,
           "Compute XDM dispersion forces")
      .def("moments", &occ::xdm::XDM::moments,
           "Get atomic moments (M2 values)")
      .def("hirshfeld_charges", &occ::xdm::XDM::hirshfeld_charges,
           "Get Hirshfeld partial charges")
      .def("atom_volume", &occ::xdm::XDM::atom_volume,
           "Get atomic volumes")
      .def("free_atom_volume", &occ::xdm::XDM::free_atom_volume,
           "Get free atom volumes")
      .def("polarizabilities", &occ::xdm::XDM::polarizabilities,
           "Get atomic polarizabilities")
      .def("parameters", &occ::xdm::XDM::parameters,
           "Get XDM parameters")
      .def("__repr__", [](const occ::xdm::XDM &xdm) {
        const auto &p = xdm.parameters();
        return fmt::format("<XDM a1={:.4f} a2={:.4f}>", p.a1, p.a2);
      });

  return m;
}
