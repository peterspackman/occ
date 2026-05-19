#include "qm_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/vibration.h>
#include <occ/driver/vibrational_analysis.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/gto/shell.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/expectation.h>
#include <occ/qm/external_potential.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hessians.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/io/fchkreader.h>
#include <occ/qm/io/fchkwriter.h>
#include <occ/qm/io/moldenreader.h>
#include <occ/qm/scf.h>
#include <occ/qm/spinorbital.h>

namespace occ::lua_bindings {

using occ::Mat;
using occ::Vec;
using occ::Vec3;
using occ::core::Atom;
using namespace occ::qm;
namespace lb = luabridge;

namespace {

std::string chemical_formula_from_atoms(const std::vector<Atom> &atoms) {
  std::vector<occ::core::Element> elements;
  elements.reserve(atoms.size());
  for (const auto &a : atoms) elements.emplace_back(a.atomic_number);
  return occ::core::chemical_formula(elements);
}

// Convert a Lua table-of-tables to a square Mat. n is taken from the outer
// length; rows are inner tables of equal length.
Mat table_to_square_mat(const lb::LuaRef &t) {
  const int n = t.length();
  Mat m(n, n);
  for (int i = 0; i < n; ++i) {
    lb::LuaRef row = lua_get_table(t, i + 1);
    for (int j = 0; j < n; ++j) m(i, j) = lua_get_num(row, j + 1);
  }
  return m;
}

// ---- enums + small data types ------------------------------------------

void register_enums_and_small_types(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

        .beginNamespace("SpinorbitalKind")
          .addProperty("Restricted",
                       +[]() { return static_cast<int>(SpinorbitalKind::Restricted); })
          .addProperty("Unrestricted",
                       +[]() { return static_cast<int>(SpinorbitalKind::Unrestricted); })
          .addProperty("General",
                       +[]() { return static_cast<int>(SpinorbitalKind::General); })
        .endNamespace()

        .beginNamespace("Operator")
          .addProperty("Overlap",
                       +[]() { return static_cast<int>(cint::Operator::overlap); })
          .addProperty("Nuclear",
                       +[]() { return static_cast<int>(cint::Operator::nuclear); })
          .addProperty("Kinetic",
                       +[]() { return static_cast<int>(cint::Operator::kinetic); })
          .addProperty("Coulomb",
                       +[]() { return static_cast<int>(cint::Operator::coulomb); })
          .addProperty("Dipole",
                       +[]() { return static_cast<int>(cint::Operator::dipole); })
          .addProperty("Quadrupole",
                       +[]() { return static_cast<int>(cint::Operator::quadrupole); })
          .addProperty("Octapole",
                       +[]() { return static_cast<int>(cint::Operator::octapole); })
          .addProperty("Hexadecapole",
                       +[]() { return static_cast<int>(cint::Operator::hexadecapole); })
          .addProperty("Rinv",
                       +[]() { return static_cast<int>(cint::Operator::rinv); })
        .endNamespace()

        .beginClass<JKPair>("JKPair")
          .addConstructor<void (*)()>()
          .addProperty("get_J",
                       +[](const JKPair *p) -> Mat { return p->J; })
          .addFunction("set_J",
                       +[](JKPair *p, const lb::LuaRef &t) {
                         p->J = table_to_square_mat(t);
                       })
          .addProperty("get_K",
                       +[](const JKPair *p) -> Mat { return p->K; })
          .addFunction("set_K",
                       +[](JKPair *p, const lb::LuaRef &t) {
                         p->K = table_to_square_mat(t);
                       })
        .endClass()

        .beginClass<JKTriple>("JKTriple")
          .addConstructor<void (*)()>()
        .endClass()
      .endNamespace();
}

// ---- basis / shells ----------------------------------------------------

void register_basis(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<Shell>("Shell")
          .addConstructor<void (*)(occ::core::PointCharge, double)>()
          .addProperty("origin",
                       +[](const Shell *s) -> Vec3 { return s->origin; })
          .addProperty("exponents",
                       +[](const Shell *s) -> Vec { return s->exponents; })
          .addProperty("contraction_coefficients",
                       +[](const Shell *s) -> Mat { return s->contraction_coefficients; })
          .addProperty("num_contractions", &Shell::num_contractions)
          .addProperty("num_primitives", &Shell::num_primitives)
          .addProperty("norm", &Shell::norm)
          .addFunction("__tostring",
                       +[](const Shell *s) {
                         return fmt::format(
                             "<Shell l={} [{:.5f}, {:.5f}, {:.5f}]>", s->l,
                             s->origin(0), s->origin(1), s->origin(2));
                       })
        .endClass()

        .beginClass<AOBasis>("AOBasis")
          .addFunction("shells",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &shells = b->shells();
                         for (size_t i = 0; i < shells.size(); ++i) {
                           t[static_cast<int>(i + 1)] = shells[i];
                         }
                         return t;
                       })
          .addFunction("set_pure", &AOBasis::set_pure)
          .addProperty("size", &AOBasis::size)
          .addProperty("nbf", &AOBasis::nbf)
          .addFunction("atoms",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &atoms = b->atoms();
                         for (size_t i = 0; i < atoms.size(); ++i) {
                           t[static_cast<int>(i + 1)] = atoms[i];
                         }
                         return t;
                       })
          .addFunction("first_bf",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &v = b->first_bf();
                         for (size_t i = 0; i < v.size(); ++i) {
                           t[static_cast<int>(i + 1)] = v[i];
                         }
                         return t;
                       })
          .addFunction("bf_to_shell",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &v = b->bf_to_shell();
                         for (size_t i = 0; i < v.size(); ++i) {
                           t[static_cast<int>(i + 1)] = v[i];
                         }
                         return t;
                       })
          .addFunction("bf_to_atom",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &v = b->bf_to_atom();
                         for (size_t i = 0; i < v.size(); ++i) {
                           t[static_cast<int>(i + 1)] = v[i];
                         }
                         return t;
                       })
          .addFunction("shell_to_atom",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &v = b->shell_to_atom();
                         for (size_t i = 0; i < v.size(); ++i) {
                           t[static_cast<int>(i + 1)] = v[i];
                         }
                         return t;
                       })
          .addFunction("atom_to_shell",
                       +[](const AOBasis *b, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         const auto &v = b->atom_to_shell();
                         for (size_t i = 0; i < v.size(); ++i) {
                           lb::LuaRef inner = lb::newTable(S);
                           for (size_t j = 0; j < v[i].size(); ++j) {
                             inner[static_cast<int>(j + 1)] = v[i][j];
                           }
                           t[static_cast<int>(i + 1)] = inner;
                         }
                         return t;
                       })
          .addProperty("l_max", &AOBasis::l_max)
          .addProperty("name", &AOBasis::name)
          // `evaluate` had a sol::optional<int> defaulted to 0; split into
          // two named functions.
          .addFunction(
              "evaluate",
              +[](const AOBasis *basis, const lb::LuaRef &points, int derivatives) {
                if (derivatives < 0 || derivatives > 2) {
                  throw std::runtime_error(
                      "Invalid max derivative (must be 0, 1, 2)");
                }
                return occ::gto::evaluate_basis(*basis, table_to_mat3n(points),
                                                derivatives);
              })
          .addFunction(
              "evaluate_default",
              +[](const AOBasis *basis, const lb::LuaRef &points) {
                return occ::gto::evaluate_basis(*basis, table_to_mat3n(points), 0);
              })
          .addFunction("__tostring",
                       +[](const AOBasis *basis) {
                         return fmt::format(
                             "<AOBasis ({}) nsh={} nbf={} natoms={}>",
                             basis->name(), basis->nsh(), basis->nbf(),
                             basis->atoms().size());
                       })
        .endClass()

        // `AOBasis::load` is a static factory; expose as a free function.
        .addFunction("AOBasis_load", &AOBasis::load)
      .endNamespace();
}

// ---- molecular orbitals + wavefunction ---------------------------------

void register_mo_and_wfn(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<MolecularOrbitals>("MolecularOrbitals")
          .addConstructor<void (*)()>()
          .addProperty("kind", &MolecularOrbitals::kind)
          .addProperty("num_alpha", &MolecularOrbitals::n_alpha)
          .addProperty("num_beta", &MolecularOrbitals::n_beta)
          .addProperty("num_ao", &MolecularOrbitals::n_ao)
          .addProperty("orbital_coeffs",
                       +[](const MolecularOrbitals *mo) -> Mat { return mo->C; })
          .addProperty("occupied_orbital_coeffs",
                       +[](const MolecularOrbitals *mo) -> Mat { return mo->Cocc; })
          .addProperty("density_matrix",
                       +[](const MolecularOrbitals *mo) -> Mat { return mo->D; })
          .addProperty("orbital_energies",
                       +[](const MolecularOrbitals *mo) -> Vec { return mo->energies; })
          .addFunction(
              "expectation_value",
              +[](const MolecularOrbitals *mo, const lb::LuaRef &op) {
                Mat opmat = table_to_square_mat(op);
                return 2 * occ::qm::expectation(mo->kind, mo->D, opmat);
              })
          .addFunction("__tostring",
                       +[](const MolecularOrbitals *mo) {
                         return fmt::format(
                             "<MolecularOrbitals kind={} nao={} nalpha={} "
                             "nbeta={}>",
                             spinorbital_kind_to_string(mo->kind), mo->n_ao,
                             mo->n_alpha, mo->n_beta);
                       })
        .endClass()

        .beginClass<Wavefunction>("Wavefunction")
          .addFunction("atoms",
                       +[](const Wavefunction *w, lua_State *S) {
                         lb::LuaRef t = lb::newTable(S);
                         for (size_t i = 0; i < w->atoms.size(); ++i) {
                           t[static_cast<int>(i + 1)] = w->atoms[i];
                         }
                         return t;
                       })
          .addProperty("molecular_orbitals",
                       +[](const Wavefunction *w) -> const MolecularOrbitals & {
                         return w->mo;
                       })
          .addProperty("basis",
                       +[](const Wavefunction *w) -> const AOBasis & {
                         return w->basis;
                       })
          .addProperty("charge", &Wavefunction::charge)
          .addProperty("multiplicity", &Wavefunction::multiplicity)
          .addProperty("mulliken_charges",
                       +[](const Wavefunction *w) -> Vec { return w->mulliken_charges(); })
          .addFunction("rotate",
                       +[](Wavefunction *w, const lb::LuaRef &rotation) {
                         w->apply_rotation(table_to_mat3(rotation));
                       })
          .addFunction("translate",
                       +[](Wavefunction *w, const lb::LuaRef &translation) {
                         w->apply_translation(table_to_vec3(translation));
                       })
          // C++ signature is (Mat3 rotation, Vec3 translation) — Lua callers
          // pass two tables.
          .addFunction(
              "transform",
              +[](Wavefunction *w, const lb::LuaRef &rotation,
                  const lb::LuaRef &translation) {
                w->apply_transformation(table_to_mat3(rotation),
                                         table_to_vec3(translation));
              })
          .addFunction("save",
                       +[](Wavefunction *w, const std::string &filename) {
                         w->save(filename);
                       })
          .addFunction("copy",
                       +[](const Wavefunction *w) { return Wavefunction(*w); })
          // `electron_density` had `sol::optional<int>` defaulted to 0; split
          // electron_density(points[, derivatives]) — derivatives is
          // optional, defaults to 0 (density only).
          .addFunction(
              "electron_density",
              +[](const Wavefunction *wfn, const lb::LuaRef &points,
                  const lb::LuaRef &derivatives) {
                const int d = derivatives.isNumber()
                                   ? derivatives.unsafe_cast<int>()
                                   : 0;
                return mat_to_table(
                    points.state(),
                    occ::density::evaluate_density_on_grid(
                        *wfn, table_to_mat3n(points), d));
              })
          .addFunction(
              "electron_density_default",
              +[](const Wavefunction *wfn, lua_State *S,
                  const lb::LuaRef &points) {
                return mat_to_table(
                    S, occ::density::evaluate_density_on_grid(
                           *wfn, table_to_mat3n(points), 0));
              })
          .addProperty("chelpg_charges",
                       +[](const Wavefunction *wfn) -> Vec { return chelpg_charges(*wfn); })
          .addFunction(
              "to_fchk",
              +[](Wavefunction *wfn, const std::string &filename) {
                auto writer = occ::io::FchkWriter(filename);
                wfn->save(writer);
                writer.write();
              })
          .addFunction("__tostring",
                       +[](const Wavefunction *wfn) {
                         return fmt::format(
                             "<Wavefunction {} {}/{} kind={} nbf={} charge={}>",
                             chemical_formula_from_atoms(wfn->atoms),
                             wfn->method, wfn->basis.name(),
                             spinorbital_kind_to_string(wfn->mo.kind),
                             wfn->basis.nbf(), wfn->charge());
                       })
        .endClass()

        .addFunction("Wavefunction_load", &Wavefunction::load)
        .addFunction("Wavefunction_from_fchk",
                     +[](const std::string &filename) {
                       auto reader = occ::io::FchkReader(filename);
                       return Wavefunction(reader);
                     })
        .addFunction("Wavefunction_from_molden",
                     +[](const std::string &filename) {
                       auto reader = occ::io::MoldenReader(filename);
                       return Wavefunction(reader);
                     })
      .endNamespace();
}

// ---- HartreeFock + SCF -------------------------------------------------

void register_hf_and_scf(lua_State *L) {
  using HF = SCF<HartreeFock>;

  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<SCFConvergenceSettings>("SCFConvergenceSettings")
          .addConstructor<void (*)()>()
          .addPropertyReadWrite("energy_threshold",
                                &SCFConvergenceSettings::energy_threshold)
          .addPropertyReadWrite("commutator_threshold",
                                &SCFConvergenceSettings::commutator_threshold)
          .addPropertyReadWrite(
              "incremental_fock_threshold",
              &SCFConvergenceSettings::incremental_fock_threshold)
          .addFunction("energy_converged",
                       &SCFConvergenceSettings::energy_converged)
          .addFunction("commutator_converged",
                       &SCFConvergenceSettings::commutator_converged)
          .addFunction("energy_and_commutator_converged",
                       &SCFConvergenceSettings::energy_and_commutator_converged)
          .addFunction("start_incremental_fock",
                       &SCFConvergenceSettings::start_incremental_fock)
        .endClass()

        .beginClass<HF>("HF")
          // sol2 had two factories: HF(HartreeFock&) and HF(HartreeFock&, kind).
          // LuaBridge3 doesn't auto-overload — pick the single-arg canonical
          // and expose the kind-taking shape as a static factory.
          .addConstructor<void (*)(HartreeFock &)>()
          .addStaticFunction(
              "with_kind",
              +[](HartreeFock &hf, SpinorbitalKind kind) {
                return new HF(hf, kind);
              })
          .addProperty("convergence_settings", &HF::convergence_settings)
          .addFunction("set_charge_multiplicity", &HF::set_charge_multiplicity)
          .addFunction("set_initial_guess", &HF::set_initial_guess_from_wfn)
          // sol::overload for set_external_potential — split into three.
          .addFunction(
              "set_external_potential_matrix",
              +[](HF *scf, const lb::LuaRef &V_ext_table,
                  double nuclear_energy, const std::string &label) {
                Mat V = table_to_square_mat(V_ext_table);
                scf->set_external_potential(V, nuclear_energy, label);
              })
          .addFunction(
              "set_external_potential_point_charge",
              +[](HF *scf, const PointChargePotential &pot) {
                scf->set_external_potential(pot);
              })
          .addFunction(
              "set_external_potential_wolf",
              +[](HF *scf, const WolfPointChargePotential &pot) {
                scf->set_external_potential(pot);
              })
          .addProperty("scf_kind", &HF::scf_kind)
          .addFunction("run", &HF::compute_scf_energy)
          .addFunction("compute_scf_energy", &HF::compute_scf_energy)
          .addFunction("wavefunction", &HF::wavefunction)
          .addFunction("__tostring",
                       +[](const HF *hf) {
                         return fmt::format(
                             "<SCF(HF) ({}, {} atoms)>",
                             hf->m_procedure.aobasis().name(),
                             hf->m_procedure.atoms().size());
                       })
        .endClass()

        .beginClass<HartreeFock>("HartreeFock")
          .addConstructor<void (*)(const AOBasis &)>()
          .addFunction("point_charge_interaction_energy",
                       &HartreeFock::nuclear_point_charge_interaction_energy)
          .addFunction("wolf_point_charge_interaction_energy",
                       &HartreeFock::wolf_point_charge_interaction_energy)
          // sol::optional<double> alpha → split.
          .addFunction(
              "point_charge_interaction_matrix",
              +[](HartreeFock *hf, lua_State *S,
                  const std::vector<occ::core::PointCharge> &charges,
                  double alpha) {
                return mat_to_table(
                    S, hf->compute_point_charge_interaction_matrix(charges,
                                                                   alpha));
              })
          .addFunction(
              "point_charge_interaction_matrix_default",
              +[](HartreeFock *hf, lua_State *S,
                  const std::vector<occ::core::PointCharge> &charges) {
                return mat_to_table(
                    S, hf->compute_point_charge_interaction_matrix(charges,
                                                                   1e16));
              })
          .addFunction("nuclear_attraction_matrix",
                       +[](HartreeFock *hf, lua_State *S) {
                         return mat_to_table(
                             S, hf->compute_nuclear_attraction_matrix());
                       })
          // sol::optional<double> threshold → split.
          .addFunction(
              "set_density_fitting_basis",
              +[](HartreeFock *hf, const std::string &name, double threshold) {
                hf->set_density_fitting_basis(name, threshold);
              })
          .addFunction(
              "set_density_fitting_basis_default",
              +[](HartreeFock *hf, const std::string &name) {
                hf->set_density_fitting_basis(name, 1e-4);
              })
          .addFunction("kinetic_matrix",
                       +[](HartreeFock *hf, lua_State *S) {
                         return mat_to_table(S, hf->compute_kinetic_matrix());
                       })
          .addFunction("overlap_matrix",
                       +[](HartreeFock *hf, lua_State *S) {
                         return mat_to_table(S, hf->compute_overlap_matrix());
                       })
          .addFunction(
              "overlap_matrix_for_basis",
              +[](HartreeFock *hf, lua_State *S, const AOBasis &other) {
                return mat_to_table(S, hf->compute_overlap_matrix_for_basis(other));
              })
          // nuclear_repulsion_energy is inherited from SCFMethodBase; LB3's
          // addProperty can't deduce the inherited member-fn pointer here,
          // so it stays as a method call.
          .addFunction("nuclear_repulsion",
                       &HartreeFock::nuclear_repulsion_energy)
          // Optional kind: 2-arg call uses given SpinorbitalKind; no-arg
          // call defaults to Restricted. We accept the kind as a LuaRef
          // so the missing-arg case is just nil and we pick the default.
          .addFunction(
              "scf",
              +[](HartreeFock *hf, const lb::LuaRef &kind) {
                SpinorbitalKind k = kind.isNil()
                                         ? SpinorbitalKind::Restricted
                                         : kind.unsafe_cast<SpinorbitalKind>();
                return HF(*hf, k);
              })
          .addFunction("set_precision", &HartreeFock::set_precision)
          .addFunction(
              "coulomb_matrix",
              +[](const HartreeFock *hf, lua_State *S,
                  const MolecularOrbitals &mo) {
                return mat_to_table(S, hf->compute_J(mo));
              })
          .addFunction(
              "fock_matrix",
              +[](const HartreeFock *hf, lua_State *S,
                  const MolecularOrbitals &mo) {
                return mat_to_table(S, hf->compute_fock(mo));
              })
          .addFunction(
              "compute_gradient",
              +[](HartreeFock *hf, const MolecularOrbitals &mo) {
                GradientEvaluator<HartreeFock> grad(*hf);
                return grad(mo);
              })
          .addFunction(
              "nuclear_repulsion_gradient",
              +[](HartreeFock *hf, lua_State *S) {
                return hf->nuclear_repulsion_gradient();
              })
          .addFunction(
              "hessian_evaluator",
              +[](HartreeFock *hf) {
                return occ::qm::HessianEvaluator<HartreeFock>(*hf);
              })
          .addFunction("__tostring",
                       +[](const HartreeFock *hf) {
                         return fmt::format("<HartreeFock ({}, {} atoms)>",
                                            hf->aobasis().name(),
                                            hf->atoms().size());
                       })
        .endClass()
      .endNamespace();
}

// ---- external potentials -----------------------------------------------

void register_external_potentials(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<PointChargePotential>("PointChargePotential")
          // sol2 had two factories: default and from a PointChargeList.
          // Pick default as canonical; expose list-taking shape as static.
          .addConstructor<void (*)()>()
          .addStaticFunction(
              "from_charges",
              +[](PointChargeList charges) {
                return new PointChargePotential{std::move(charges)};
              })
          .addProperty("charges", &PointChargePotential::charges)
          .addFunction(
              "compute_potential_matrix",
              +[](const PointChargePotential *pot, lua_State *S,
                  HartreeFock &hf) {
                return mat_to_table(S, pot->compute_potential_matrix(hf));
              })
          .addFunction(
              "nuclear_interaction_energy",
              +[](const PointChargePotential *pot, const HartreeFock &hf) {
                return pot->nuclear_interaction_energy(hf);
              })
          .addFunction("label",
                       +[](const PointChargePotential *p) {
                         return std::string(p->label());
                       })
          .addFunction("__tostring",
                       +[](const PointChargePotential *p) {
                         return fmt::format("<{}>", p->descriptor());
                       })
        .endClass()

        .beginClass<WolfPointChargePotential>("WolfPointChargePotential")
          .addConstructor<void (*)()>()
          .addStaticFunction(
              "from_charges",
              +[](PointChargeList charges,
                  std::vector<double> molecular_charges, double alpha,
                  double cutoff) {
                return new WolfPointChargePotential{
                    std::move(charges), std::move(molecular_charges), alpha,
                    cutoff};
              })
          .addProperty("charges", &WolfPointChargePotential::charges)
          .addProperty("molecular_charges",
                       &WolfPointChargePotential::molecular_charges)
          .addProperty("alpha", &WolfPointChargePotential::alpha)
          .addProperty("cutoff", &WolfPointChargePotential::cutoff)
          .addFunction(
              "compute_potential_matrix",
              +[](const WolfPointChargePotential *pot, lua_State *S,
                  HartreeFock &hf) {
                return mat_to_table(S, pot->compute_potential_matrix(hf));
              })
          .addFunction(
              "nuclear_interaction_energy",
              +[](const WolfPointChargePotential *pot, const HartreeFock &hf) {
                return pot->nuclear_interaction_energy(hf);
              })
          .addFunction("label",
                       +[](const WolfPointChargePotential *p) {
                         return std::string(p->label());
                       })
          .addFunction("__tostring",
                       +[](const WolfPointChargePotential *p) {
                         return fmt::format("<{}>", p->descriptor());
                       })
        .endClass()
      .endNamespace();
}

// ---- hessian + vibrations ---------------------------------------------

void register_hessian_and_vibrations(lua_State *L) {
  using HessEval = occ::qm::HessianEvaluator<HartreeFock>;
  using VM = occ::core::VibrationalModes;

  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<HessEval>("HessianEvaluatorHF")
          .addConstructor<void (*)(HartreeFock &)>()
          .addFunction("set_method", &HessEval::set_method)
          .addFunction("set_step_size", &HessEval::set_step_size)
          .addFunction("set_use_acoustic_sum_rule",
                       &HessEval::set_use_acoustic_sum_rule)
          .addProperty("step_size", &HessEval::step_size)
          .addProperty("use_acoustic_sum_rule",
                       &HessEval::use_acoustic_sum_rule)
          .addFunction("nuclear_repulsion",
                       +[](HessEval *h) {
                         return h->nuclear_repulsion();
                       })
          // HessianEvaluator::operator() takes a Wavefunction in this version
          // of occ, not a MolecularOrbitals directly.
          .addFunction("compute",
                       +[](HessEval *h, const Wavefunction &wfn) {
                         return (*h)(wfn);
                       })
          .addFunction(
              "__tostring",
              +[](const HessEval *h) {
                return fmt::format(
                    "<HessianEvaluatorHF step_size={:.4f} acoustic_sum_rule={}>",
                    h->step_size(), h->use_acoustic_sum_rule());
              })
        .endClass()

        .beginClass<VM>("VibrationalModes")
          .addProperty("frequencies_cm",
                       +[](const VM *v) -> Vec { return v->frequencies_cm; })
          .addProperty("frequencies_hartree",
                       +[](const VM *v) -> Vec { return v->frequencies_hartree; })
          .addProperty("normal_modes",
                       +[](const VM *v) -> Mat { return v->normal_modes; })
          .addProperty("mass_weighted_hessian",
                       +[](const VM *v) -> Mat { return v->mass_weighted_hessian; })
          .addProperty("hessian",
                       +[](const VM *v) -> Mat { return v->hessian; })
          .addProperty("n_modes", &VM::n_modes)
          .addProperty("n_atoms", &VM::n_atoms)
          .addProperty("summary_string", &VM::summary_string)
          .addProperty("frequencies_string", &VM::frequencies_string)
          // sol::optional<double> threshold → split.
          .addFunction("normal_modes_string",
                       +[](const VM *v, double threshold) {
                         return v->normal_modes_string(threshold);
                       })
          .addFunction("normal_modes_string_default",
                       +[](const VM *v) {
                         return v->normal_modes_string(0.1);
                       })
          .addProperty("get_all_frequencies",
                       +[](const VM *v) -> Vec { return v->get_all_frequencies(); })
          .addFunction("__tostring",
                       +[](const VM *v) {
                         return fmt::format(
                             "<VibrationalModes n_modes={} n_atoms={}>",
                             v->n_modes(), v->n_atoms());
                       })
        .endClass()
      .endNamespace();
}

// ---- IntegralEngine + DF + smearing + GTO values ----------------------

void register_integral_engines(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<IntegralEngine>("IntegralEngine")
          // sol2 had two factories: from AOBasis, and from atoms+shells.
          // Pick AOBasis as canonical; expose atoms+shells as static factory.
          .addConstructor<void (*)(const AOBasis &)>()
          .addStaticFunction(
              "from_atoms_and_shells",
              +[](const std::vector<Atom> &atoms,
                  const std::vector<Shell> &shells) {
                return new IntegralEngine(atoms, shells);
              })
          .addProperty("schwarz",
                       +[](const IntegralEngine *e) -> Mat { return e->schwarz(); })
          .addFunction("set_precision", &IntegralEngine::set_precision)
          .addFunction("set_range_separated_omega",
                       &IntegralEngine::set_range_separated_omega)
          .addProperty("range_separated_omega",
                       &IntegralEngine::range_separated_omega)
          .addProperty("is_spherical", &IntegralEngine::is_spherical)
          .addProperty("have_auxiliary_basis",
                       &IntegralEngine::have_auxiliary_basis)
          // sol::optional<bool> dummy → split.
          .addFunction(
              "set_auxiliary_basis",
              +[](IntegralEngine *e, const std::vector<Shell> &basis,
                  bool dummy) {
                e->set_auxiliary_basis(basis, dummy);
              })
          .addFunction(
              "set_auxiliary_basis_default",
              +[](IntegralEngine *e, const std::vector<Shell> &basis) {
                e->set_auxiliary_basis(basis, false);
              })
          .addFunction("clear_auxiliary_basis",
                       &IntegralEngine::clear_auxiliary_basis)
          // sol::optional<bool> use_shellpair → split.
          .addFunction(
              "one_electron_operator",
              +[](const IntegralEngine *e, lua_State *S, int op,
                  bool use_shellpair) {
                return mat_to_table(
                    S, e->one_electron_operator(
                           static_cast<cint::Operator>(op), use_shellpair));
              })
          .addFunction(
              "one_electron_operator_default",
              +[](const IntegralEngine *e, lua_State *S, int op) {
                return mat_to_table(
                    S, e->one_electron_operator(
                           static_cast<cint::Operator>(op), true));
              })
          .addProperty("nbf", &IntegralEngine::nbf)
          .addProperty("nsh", &IntegralEngine::nsh)
          .addProperty("aobasis", &IntegralEngine::aobasis)
          .addProperty("auxbasis", &IntegralEngine::auxbasis)
          .addProperty("nbf_aux", &IntegralEngine::nbf_aux)
          .addProperty("nsh_aux", &IntegralEngine::nsh_aux)
          .addProperty("have_effective_core_potentials",
                       &IntegralEngine::have_effective_core_potentials)
          .addFunction("set_effective_core_potentials",
                       &IntegralEngine::set_effective_core_potentials)
          .addFunction("__tostring",
                       +[](const IntegralEngine *engine) {
                         return fmt::format(
                             "<IntegralEngine nbf={} nsh={} spherical={}>",
                             engine->nbf(), engine->nsh(),
                             engine->is_spherical() ? "true" : "false");
                       })
        .endClass()

        .beginNamespace("OrbitalSmearingKind")
          .addProperty("None_",
                       +[]() {
                         return static_cast<int>(OrbitalSmearing::Kind::None);
                       })
          .addProperty("Fermi",
                       +[]() {
                         return static_cast<int>(OrbitalSmearing::Kind::Fermi);
                       })
          .addProperty(
              "Gaussian",
              +[]() {
                return static_cast<int>(OrbitalSmearing::Kind::Gaussian);
              })
          .addProperty(
              "Linear",
              +[]() {
                return static_cast<int>(OrbitalSmearing::Kind::Linear);
              })
        .endNamespace()

        .beginClass<OrbitalSmearing>("OrbitalSmearing")
          .addConstructor<void (*)()>()
          .addPropertyReadWrite("kind", &OrbitalSmearing::kind)
          .addPropertyReadWrite("mu", &OrbitalSmearing::mu)
          .addPropertyReadWrite("fermi_level", &OrbitalSmearing::fermi_level)
          .addPropertyReadWrite("sigma", &OrbitalSmearing::sigma)
          .addPropertyReadWrite("entropy", &OrbitalSmearing::entropy)
          .addFunction("calculate_entropy", &OrbitalSmearing::calculate_entropy)
          .addProperty("ec_entropy", &OrbitalSmearing::ec_entropy)
          .addFunction(
              "__tostring",
              +[](const OrbitalSmearing *os) {
                const char *k = "?";
                switch (os->kind) {
                case OrbitalSmearing::Kind::None: k = "None"; break;
                case OrbitalSmearing::Kind::Fermi: k = "Fermi"; break;
                case OrbitalSmearing::Kind::Gaussian: k = "Gaussian"; break;
                case OrbitalSmearing::Kind::Linear: k = "Linear"; break;
                }
                return fmt::format(
                    "<OrbitalSmearing kind={} sigma={:.6f} mu={:.6f}>", k,
                    os->sigma, os->mu);
              })
        .endClass()

        .beginNamespace("IntegralEngineDFPolicy")
          .addProperty(
              "Choose",
              +[]() {
                return static_cast<int>(IntegralEngineDF::Policy::Choose);
              })
          .addProperty(
              "Direct",
              +[]() {
                return static_cast<int>(IntegralEngineDF::Policy::Direct);
              })
          .addProperty(
              "Stored",
              +[]() {
                return static_cast<int>(IntegralEngineDF::Policy::Stored);
              })
        .endNamespace()

        .beginClass<IntegralEngineDF>("IntegralEngineDF")
          .addConstructor<void (*)(const std::vector<Atom> &,
                                    const std::vector<Shell> &,
                                    const std::vector<Shell> &)>()
          .addFunction("set_integral_policy",
                       &IntegralEngineDF::set_integral_policy)
          .addFunction("set_range_separated_omega",
                       &IntegralEngineDF::set_range_separated_omega)
          .addFunction("set_precision", &IntegralEngineDF::set_precision)
          .addFunction("__tostring",
                       +[](const IntegralEngineDF *e) {
                         const char *p = "?";
                         switch (e->integral_policy()) {
                         case IntegralEngineDF::Policy::Choose:
                           p = "Choose";
                           break;
                         case IntegralEngineDF::Policy::Direct:
                           p = "Direct";
                           break;
                         case IntegralEngineDF::Policy::Stored:
                           p = "Stored";
                           break;
                         }
                         return fmt::format(
                             "<IntegralEngineDF policy={} precision={:.2e}>",
                             p, e->precision());
                       })
        .endClass()

        .beginClass<occ::gto::GTOValues>("GTOValues")
          .addConstructor<void (*)()>()
          .addFunction("reserve", &occ::gto::GTOValues::reserve)
          .addFunction("set_zero", &occ::gto::GTOValues::set_zero)
          .addProperty("phi",
                       +[](const occ::gto::GTOValues *g) -> Mat { return g->phi; })
          .addProperty("phi_x",
                       +[](const occ::gto::GTOValues *g) -> Mat { return g->phi_x; })
          .addProperty("phi_y",
                       +[](const occ::gto::GTOValues *g) -> Mat { return g->phi_y; })
          .addProperty("phi_z",
                       +[](const occ::gto::GTOValues *g) -> Mat { return g->phi_z; })
        .endClass()
      .endNamespace();
}

// ---- free vibrational-analysis helpers --------------------------------

void register_vibrational_free_functions(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        // Optional project (default false). Hessian may be passed as
        // either a Mat userdata or a nested Lua table.
        .addFunction(
            "compute_vibrational_modes",
            +[](const lb::LuaRef &hessian, const occ::core::Molecule &mol,
                const lb::LuaRef &project) {
              const bool p =
                  project.isBool() ? project.unsafe_cast<bool>() : false;
              Mat H;
              if (hessian.isUserdata()) {
                H = hessian.unsafe_cast<const Mat &>();
              } else {
                H = table_to_square_mat(hessian);
              }
              return occ::core::compute_vibrational_modes(H, mol, p);
            })

        .addFunction(
            "eigenvalues_to_frequencies_cm",
            +[](lua_State *S, const lb::LuaRef &eigenvalues) {
              Vec ev = table_to_vecx(eigenvalues);
              return vec_to_table(
                  S, occ::core::eigenvalues_to_frequencies_cm(ev));
            })

        .addFunction(
            "frequencies_cm_to_hartree",
            +[](lua_State *S, const lb::LuaRef &freqs_cm) {
              Vec f = table_to_vecx(freqs_cm);
              return vec_to_table(S, occ::core::frequencies_cm_to_hartree(f));
            })
      .endNamespace();
}

} // namespace

void register_qm_bindings(lua_State *L) {
  register_enums_and_small_types(L);
  register_basis(L);
  register_mo_and_wfn(L);
  register_hf_and_scf(L);
  register_external_potentials(L);
  register_hessian_and_vibrations(L);
  register_integral_engines(L);
  register_vibrational_free_functions(L);
}

} // namespace occ::lua_bindings
