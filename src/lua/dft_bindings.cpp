#include "dft_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/dft/dft.h>
#include <occ/qm/external_potential.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hessians.h>
#include <occ/qm/scf.h>
#include <occ/xdm/xdm.h>

namespace sol {
template <> struct is_automagical<occ::dft::DFT> : std::false_type {};
template <>
struct is_automagical<occ::qm::SCF<occ::dft::DFT>> : std::false_type {};
template <>
struct is_automagical<occ::qm::HessianEvaluator<occ::dft::DFT>>
    : std::false_type {};
template <> struct is_automagical<occ::xdm::XDM> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using occ::dft::DFT;
using occ::gto::AOBasis;
using occ::io::GridSettings;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;

void register_dft_bindings(sol::state_view, sol::table &m) {
  m.new_usertype<GridSettings>(
      "GridSettings",
      sol::call_constructor, sol::factories([]() { return GridSettings{}; }),
      "max_angular_points", &GridSettings::max_angular_points,
      "min_angular_points", &GridSettings::min_angular_points,
      "radial_points", &GridSettings::radial_points,
      "radial_precision", &GridSettings::radial_precision,
      sol::meta_function::to_string, [](const GridSettings &s) {
        return fmt::format("<GridSettings ang=({},{}) radial={}, prec={:.2g}>",
                           s.min_angular_points, s.max_angular_points,
                           s.radial_points, s.radial_precision);
      });

  using KS = SCF<DFT>;

  m.new_usertype<KS>(
      "KS",
      sol::call_constructor,
      sol::factories([](DFT &dft) { return KS(dft); },
                     [](DFT &dft, SpinorbitalKind k) { return KS(dft, k); }),
      "convergence_settings", &KS::convergence_settings,
      "set_charge_multiplicity", &KS::set_charge_multiplicity,
      "set_initial_guess", &KS::set_initial_guess_from_wfn,
      "set_external_potential",
      sol::overload(
          [](KS &scf, const sol::table &V_ext_table, double nuc_e,
             const std::string &label) {
            const int n = static_cast<int>(V_ext_table.size());
            occ::Mat V(n, n);
            for (int i = 0; i < n; ++i) {
              sol::table row = V_ext_table.get<sol::table>(i + 1);
              for (int j = 0; j < n; ++j) V(i, j) = row.get<double>(j + 1);
            }
            scf.set_external_potential(V, nuc_e, label);
          },
          [](KS &scf, const occ::qm::PointChargePotential &pot) {
            scf.set_external_potential(pot);
          },
          [](KS &scf, const occ::qm::WolfPointChargePotential &pot) {
            scf.set_external_potential(pot);
          }),
      "scf_kind", &KS::scf_kind,
      "run", &KS::compute_scf_energy,
      "compute_scf_energy", &KS::compute_scf_energy,
      "wavefunction", &KS::wavefunction,
      sol::meta_function::to_string, [](const KS &ks) {
        return fmt::format("<SCF(KS) ({}, {} atoms)>",
                           ks.m_procedure.aobasis().name(),
                           ks.m_procedure.atoms().size());
      });

  m.new_usertype<DFT>(
      "DFT",
      sol::call_constructor,
      sol::factories(
          [](const std::string &method, const AOBasis &basis) {
            return DFT(method, basis);
          },
          [](const std::string &method, const AOBasis &basis,
             const GridSettings &grid) { return DFT(method, basis, grid); }),
      "nuclear_attraction_matrix",
      [](DFT &dft, sol::this_state s) {
        return dft.compute_nuclear_attraction_matrix();
      },
      "kinetic_matrix",
      [](DFT &dft, sol::this_state s) {
        return dft.compute_kinetic_matrix();
      },
      "overlap_matrix",
      [](DFT &dft, sol::this_state s) {
        return dft.compute_overlap_matrix();
      },
      // The C++ default arg (auto_aux_threshold = 1e-4) is invisible to
      // sol2 — under SOL_ALL_SAFETIES_ON it would demand the second arg.
      // Wrap so Lua callers can omit it.
      "set_density_fitting_basis",
      [](DFT &dft, const std::string &name,
         sol::optional<double> threshold) {
        dft.set_density_fitting_basis(name, threshold.value_or(1e-4));
      },
      "nuclear_repulsion", &DFT::nuclear_repulsion_energy,
      "set_precision", &DFT::set_precision,
      "set_method", &DFT::set_method,
      "fock_matrix",
      [](DFT &dft, const MolecularOrbitals &mo, sol::this_state s) {
        return dft.compute_fock(mo);
      },
      "scf",
      [](DFT &dft, sol::optional<SpinorbitalKind> kind) {
        return KS(dft, kind.value_or(SpinorbitalKind::Restricted));
      },
      "compute_gradient",
      [](DFT &dft, const MolecularOrbitals &mo, sol::this_state s) {
        occ::qm::GradientEvaluator<DFT> grad(dft);
        return grad(mo);
      },
      "hessian_evaluator",
      [](DFT &dft) { return occ::qm::HessianEvaluator<DFT>(dft); },
      sol::meta_function::to_string, [](const DFT &dft) {
        return fmt::format("<DFT {} ({}, {} atoms)>", dft.method_string(),
                           dft.aobasis().name(), dft.atoms().size());
      });

  using HessEvalDFT = occ::qm::HessianEvaluator<DFT>;
  m.new_usertype<HessEvalDFT>(
      "HessianEvaluatorDFT",
      sol::call_constructor, sol::constructors<HessEvalDFT(DFT &)>(),
      "set_method", &HessEvalDFT::set_method,
      "set_step_size", &HessEvalDFT::set_step_size,
      "set_use_acoustic_sum_rule", &HessEvalDFT::set_use_acoustic_sum_rule,
      "step_size", &HessEvalDFT::step_size,
      "use_acoustic_sum_rule", &HessEvalDFT::use_acoustic_sum_rule,
      "nuclear_repulsion",
      [](HessEvalDFT &h, sol::this_state s) {
        return h.nuclear_repulsion();
      },
      "compute",
      [](HessEvalDFT &h, const occ::qm::Wavefunction &wfn, sol::this_state s) {
        return h(wfn);
      },
      sol::meta_function::to_string, [](const HessEvalDFT &h) {
        return fmt::format(
            "<HessianEvaluatorDFT step_size={:.4f} acoustic_sum_rule={}>",
            h.step_size(), h.use_acoustic_sum_rule());
      });

  m.new_usertype<occ::xdm::XDM::Parameters>(
      "XDMParameters",
      sol::call_constructor,
      sol::factories([]() { return occ::xdm::XDM::Parameters{}; }),
      "a1", &occ::xdm::XDM::Parameters::a1,
      "a2", &occ::xdm::XDM::Parameters::a2,
      sol::meta_function::to_string, [](const occ::xdm::XDM::Parameters &p) {
        return fmt::format("<XDMParameters a1={:.4f} a2={:.4f}>", p.a1, p.a2);
      });

  m.new_usertype<occ::xdm::XDM>(
      "XDM",
      sol::call_constructor,
      sol::factories(
          [](const AOBasis &basis) { return occ::xdm::XDM(basis); },
          [](const AOBasis &basis, int charge) {
            return occ::xdm::XDM(basis, charge);
          },
          [](const AOBasis &basis, int charge,
             const occ::xdm::XDM::Parameters &p) {
            return occ::xdm::XDM(basis, charge, p);
          }),
      "energy", &occ::xdm::XDM::energy,
      "forces",
      [](occ::xdm::XDM &x, const MolecularOrbitals &mo, sol::this_state s) {
        return x.forces(mo);
      },
      "moments",
      [](const occ::xdm::XDM &x, sol::this_state s) {
        return x.moments();
      },
      "hirshfeld_charges",
      [](const occ::xdm::XDM &x, sol::this_state s) {
        return x.hirshfeld_charges();
      },
      "atom_volume",
      [](const occ::xdm::XDM &x, sol::this_state s) {
        return x.atom_volume();
      },
      "free_atom_volume",
      [](const occ::xdm::XDM &x, sol::this_state s) {
        return x.free_atom_volume();
      },
      "polarizabilities",
      [](const occ::xdm::XDM &x, sol::this_state s) {
        return x.polarizabilities();
      },
      "parameters", &occ::xdm::XDM::parameters,
      sol::meta_function::to_string, [](const occ::xdm::XDM &x) {
        const auto &p = x.parameters();
        return fmt::format("<XDM a1={:.4f} a2={:.4f}>", p.a1, p.a2);
      });
}

} // namespace occ::lua_bindings
