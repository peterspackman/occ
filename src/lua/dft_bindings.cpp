#include "dft_bindings.h"
#include "eigen_conv.h"
#include "enum_stacks.h"
#include <fmt/core.h>
#include <occ/dft/dft.h>
#include <occ/qm/external_potential.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hessians.h>
#include <occ/qm/scf.h>
#include <occ/xdm/xdm.h>

namespace occ::lua_bindings {

using occ::dft::DFT;
using occ::gto::AOBasis;
using occ::io::GridSettings;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;
namespace lb = luabridge;

void register_dft_bindings(lua_State *L) {
  using KS = SCF<DFT>;
  using HessEvalDFT = occ::qm::HessianEvaluator<DFT>;

  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginClass<GridSettings>("GridSettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("max_angular_points",
                            &GridSettings::max_angular_points)
      .addPropertyReadWrite("min_angular_points",
                            &GridSettings::min_angular_points)
      .addPropertyReadWrite("radial_points", &GridSettings::radial_points)
      .addPropertyReadWrite("radial_precision", &GridSettings::radial_precision)
      .addFunction(
          "__tostring",
          +[](const GridSettings *s) {
            return fmt::format(
                "<GridSettings ang=({},{}) radial={}, prec={:.2g}>",
                s->min_angular_points, s->max_angular_points, s->radial_points,
                s->radial_precision);
          })
      .endClass()

      .beginClass<KS>("KS")
      // Two construction shapes — default kind is Restricted.
      .addConstructor<void (*)(DFT &, SpinorbitalKind)>()
      .addStaticFunction(
          "new_restricted",
          +[](DFT &dft) { return new KS(dft, SpinorbitalKind::Restricted); })
      .addProperty("convergence_settings", &KS::convergence_settings)
      .addFunction("set_charge_multiplicity", &KS::set_charge_multiplicity)
      .addFunction("set_initial_guess", &KS::set_initial_guess_from_wfn)
      // sol::overload split — pick the matrix-based form as canonical
      // and expose the model variants under distinct names.
      .addFunction(
          "set_external_potential",
          +[](KS *scf, const lb::LuaRef &V_ext_table, double nuc_e,
              const std::string &label) {
            const int n = V_ext_table.length();
            occ::Mat V(n, n);
            for (int i = 0; i < n; ++i) {
              lb::LuaRef row = lua_get_table(V_ext_table, i + 1);
              for (int j = 0; j < n; ++j)
                V(i, j) = lua_get_num(row, j + 1);
            }
            scf->set_external_potential(V, nuc_e, label);
          })
      .addFunction(
          "set_external_point_charge_potential",
          +[](KS *scf, const occ::qm::PointChargePotential &pot) {
            scf->set_external_potential(pot);
          })
      .addFunction(
          "set_external_wolf_point_charge_potential",
          +[](KS *scf, const occ::qm::WolfPointChargePotential &pot) {
            scf->set_external_potential(pot);
          })
      .addFunction("scf_kind", &KS::scf_kind)
      .addFunction("run", &KS::compute_scf_energy)
      .addFunction("compute_scf_energy", &KS::compute_scf_energy)
      .addFunction("wavefunction", &KS::wavefunction)
      .addFunction(
          "__tostring",
          +[](const KS *ks) {
            return fmt::format("<SCF(KS) ({}, {} atoms)>",
                               ks->m_procedure.aobasis().name(),
                               ks->m_procedure.atoms().size());
          })
      .endClass()

      .beginClass<DFT>("DFT")
      // Two construction shapes — basis-only vs grid-configured.
      .addConstructor<void (*)(const std::string &, const AOBasis &)>()
      .addStaticFunction(
          "new_with_grid",
          +[](const std::string &method, const AOBasis &basis,
              const GridSettings &grid) {
            return new DFT(method, basis, grid);
          })
      .addFunction(
          "nuclear_attraction_matrix",
          +[](DFT *dft) { return dft->compute_nuclear_attraction_matrix(); })
      .addFunction(
          "kinetic_matrix",
          +[](DFT *dft) { return dft->compute_kinetic_matrix(); })
      .addFunction(
          "overlap_matrix",
          +[](DFT *dft) { return dft->compute_overlap_matrix(); })
      // Optional auto_aux_threshold (C++ default 1e-4).
      .addFunction(
          "set_density_fitting_basis",
          +[](DFT *dft, const std::string &name, const lb::LuaRef &threshold) {
            const double t =
                threshold.isNumber() ? threshold.unsafe_cast<double>() : 1e-4;
            dft->set_density_fitting_basis(name, t);
          })
      // nuclear_repulsion_energy is inherited from SCFMethodBase; LB3's
      // addProperty can't deduce the inherited member-fn pointer here,
      // so it stays as a method call.
      .addFunction("nuclear_repulsion", &DFT::nuclear_repulsion_energy)
      .addFunction("set_precision", &DFT::set_precision)
      .addFunction("set_method", &DFT::set_method)
      .addFunction(
          "fock_matrix",
          +[](DFT *dft, const MolecularOrbitals &mo) {
            return dft->compute_fock(mo);
          })
      // scf(kind?) — optional SpinorbitalKind defaulting to
      // Restricted. Pass as LuaRef so a no-arg call is a nil ref.
      .addFunction(
          "scf",
          +[](DFT *dft, const lb::LuaRef &kind) {
            SpinorbitalKind k = kind.isNil()
                                    ? SpinorbitalKind::Restricted
                                    : kind.unsafe_cast<SpinorbitalKind>();
            return KS(*dft, k);
          })
      .addFunction(
          "compute_gradient",
          +[](DFT *dft, const MolecularOrbitals &mo) {
            occ::qm::GradientEvaluator<DFT> grad(*dft);
            return grad(mo);
          })
      .addFunction(
          "hessian_evaluator", +[](DFT *dft) { return HessEvalDFT(*dft); })
      .addFunction(
          "__tostring",
          +[](const DFT *dft) {
            return fmt::format("<DFT {} ({}, {} atoms)>", dft->method_string(),
                               dft->aobasis().name(), dft->atoms().size());
          })
      .endClass()

      .beginClass<HessEvalDFT>("HessianEvaluatorDFT")
      .addConstructor<void (*)(DFT &)>()
      .addFunction("set_method", &HessEvalDFT::set_method)
      .addFunction("set_step_size", &HessEvalDFT::set_step_size)
      .addFunction("set_use_acoustic_sum_rule",
                   &HessEvalDFT::set_use_acoustic_sum_rule)
      .addProperty("step_size", &HessEvalDFT::step_size)
      .addProperty("use_acoustic_sum_rule", &HessEvalDFT::use_acoustic_sum_rule)
      .addFunction(
          "nuclear_repulsion",
          +[](HessEvalDFT *h) { return h->nuclear_repulsion(); })
      .addFunction(
          "compute",
          +[](HessEvalDFT *h, const occ::qm::Wavefunction &wfn) {
            return (*h)(wfn);
          })
      .addFunction(
          "__tostring",
          +[](const HessEvalDFT *h) {
            return fmt::format(
                "<HessianEvaluatorDFT step_size={:.4f} acoustic_sum_rule={}>",
                h->step_size(), h->use_acoustic_sum_rule());
          })
      .endClass()

      .beginClass<occ::xdm::XDM::Parameters>("XDMParameters")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("a1", &occ::xdm::XDM::Parameters::a1)
      .addPropertyReadWrite("a2", &occ::xdm::XDM::Parameters::a2)
      .addFunction(
          "__tostring",
          +[](const occ::xdm::XDM::Parameters *p) {
            return fmt::format("<XDMParameters a1={:.4f} a2={:.4f}>", p->a1,
                               p->a2);
          })
      .endClass()

      .beginClass<occ::xdm::XDM>("XDM")
      // Three construction shapes — pick the most-complete as canonical.
      .addConstructor<void (*)(const AOBasis &, int,
                               const occ::xdm::XDM::Parameters &)>()
      .addStaticFunction(
          "new_with_basis",
          +[](const AOBasis &basis) { return new occ::xdm::XDM(basis); })
      .addStaticFunction(
          "new_with_charge",
          +[](const AOBasis &basis, int charge) {
            return new occ::xdm::XDM(basis, charge);
          })
      .addFunction("energy", &occ::xdm::XDM::energy)
      .addFunction(
          "forces", +[](occ::xdm::XDM *x,
                        const MolecularOrbitals &mo) { return x->forces(mo); })
      .addProperty(
          "moments", +[](const occ::xdm::XDM *x) { return x->moments(); })
      .addProperty(
          "hirshfeld_charges",
          +[](const occ::xdm::XDM *x) { return x->hirshfeld_charges(); })
      .addProperty(
          "atom_volume",
          +[](const occ::xdm::XDM *x) { return x->atom_volume(); })
      .addProperty(
          "free_atom_volume",
          +[](const occ::xdm::XDM *x) { return x->free_atom_volume(); })
      .addProperty(
          "polarizabilities",
          +[](const occ::xdm::XDM *x) { return x->polarizabilities(); })
      .addProperty("parameters", &occ::xdm::XDM::parameters)
      .addFunction(
          "__tostring",
          +[](const occ::xdm::XDM *x) {
            const auto &p = x->parameters();
            return fmt::format("<XDM a1={:.4f} a2={:.4f}>", p.a1, p.a2);
          })
      .endClass()

      .endNamespace();
}

} // namespace occ::lua_bindings
