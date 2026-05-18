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

// HartreeFock has `operator()` overloads that take/return Eigen matrices.
// sol2's automagic enrollment for that triggers the same "cannot form a
// reference to void" Eigen-iterator cascade described in
// `feedback-sol2-gotchas`. We bind everything we need explicitly, so opt
// the type out of automagic. Same applies to a handful of other types
// here that have callable / spaceship-style overloads.
namespace sol {
// Disable both automagic enrollment AND container probing for every qm
// type sol2 might try to introspect. Without these, sol2's automagic
// pairs/index machinery instantiates `as_container_t<Eigen::Matrix<...>>`
// (because Wavefunction/MO carry Eigen-bearing fields), which fails to
// compile through Eigen 3.4's iterator interface ("cannot form a
// reference to void").
template <>
struct is_automagical<occ::qm::HartreeFock> : std::false_type {};
template <>
struct is_automagical<occ::qm::IntegralEngine> : std::false_type {};
template <>
struct is_automagical<occ::qm::HessianEvaluator<occ::qm::HartreeFock>>
    : std::false_type {};
template <>
struct is_automagical<occ::qm::Wavefunction> : std::false_type {};
template <>
struct is_automagical<occ::qm::MolecularOrbitals> : std::false_type {};
template <>
struct is_automagical<occ::qm::SCF<occ::qm::HartreeFock>>
    : std::false_type {};
template <>
struct is_automagical<occ::qm::AOBasis> : std::false_type {};

template <>
struct is_container<occ::qm::Wavefunction> : std::false_type {};
template <>
struct is_container<occ::qm::MolecularOrbitals> : std::false_type {};
template <>
struct is_container<occ::qm::AOBasis> : std::false_type {};
template <>
struct is_container<occ::qm::HartreeFock> : std::false_type {};
template <>
struct is_container<occ::qm::IntegralEngine> : std::false_type {};
template <>
struct is_container<occ::qm::Shell> : std::false_type {};
template <>
struct is_container<occ::qm::SCF<occ::qm::HartreeFock>>
    : std::false_type {};
template <>
struct is_container<occ::qm::HessianEvaluator<occ::qm::HartreeFock>>
    : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using occ::Mat;
using occ::Vec;
using occ::Vec3;
using occ::core::Atom;
using namespace occ::qm;

namespace {

std::string chemical_formula_from_atoms(const std::vector<Atom> &atoms) {
  std::vector<occ::core::Element> elements;
  elements.reserve(atoms.size());
  for (const auto &a : atoms) elements.emplace_back(a.atomic_number);
  return occ::core::chemical_formula(elements);
}

// ---- enums + small data types ------------------------------------------

void register_enums_and_small_types(sol::table &m) {
  m.new_enum<SpinorbitalKind>(
      "SpinorbitalKind", {{"Restricted", SpinorbitalKind::Restricted},
                          {"Unrestricted", SpinorbitalKind::Unrestricted},
                          {"General", SpinorbitalKind::General}});

  m.new_enum<cint::Operator>(
      "Operator", {{"Overlap", cint::Operator::overlap},
                   {"Nuclear", cint::Operator::nuclear},
                   {"Kinetic", cint::Operator::kinetic},
                   {"Coulomb", cint::Operator::coulomb},
                   {"Dipole", cint::Operator::dipole},
                   {"Quadrupole", cint::Operator::quadrupole},
                   {"Octapole", cint::Operator::octapole},
                   {"Hexadecapole", cint::Operator::hexadecapole},
                   {"Rinv", cint::Operator::rinv}});

  m.new_usertype<JKPair>(
      "JKPair",
      sol::call_constructor, sol::factories([]() { return JKPair{}; }),
      "J",
      sol::property(
          [](const JKPair &p, sol::this_state s) {
            return p.J;
          },
          [](JKPair &p, const sol::table &t) {
            const int n = static_cast<int>(t.size());
            Mat m(n, n);
            for (int i = 0; i < n; ++i) {
              sol::table row = t.get<sol::table>(i + 1);
              for (int j = 0; j < n; ++j) m(i, j) = row.get<double>(j + 1);
            }
            p.J = m;
          }),
      "K",
      sol::property(
          [](const JKPair &p, sol::this_state s) {
            return p.K;
          },
          [](JKPair &p, const sol::table &t) {
            const int n = static_cast<int>(t.size());
            Mat m(n, n);
            for (int i = 0; i < n; ++i) {
              sol::table row = t.get<sol::table>(i + 1);
              for (int j = 0; j < n; ++j) m(i, j) = row.get<double>(j + 1);
            }
            p.K = m;
          }));

  m.new_usertype<JKTriple>(
      "JKTriple",
      sol::call_constructor, sol::factories([]() { return JKTriple{}; }));
}

// ---- basis / shells ----------------------------------------------------

void register_basis(sol::table &m) {
  m.new_usertype<Shell>(
      "Shell",
      sol::call_constructor,
      sol::constructors<Shell(occ::core::PointCharge, double)>(),
      "origin",
      sol::readonly_property([](const Shell &s, sol::this_state st) {
        return vec_to_table(st, s.origin);
      }),
      "exponents",
      sol::readonly_property([](const Shell &s, sol::this_state st) {
        return vec_to_table(st, s.exponents);
      }),
      "contraction_coefficients",
      sol::readonly_property([](const Shell &s, sol::this_state st) {
        return mat_to_table(st, s.contraction_coefficients);
      }),
      "num_contractions", &Shell::num_contractions,
      "num_primitives", &Shell::num_primitives,
      "norm", &Shell::norm,
      sol::meta_function::to_string, [](const Shell &s) {
        return fmt::format("<Shell l={} [{:.5f}, {:.5f}, {:.5f}]>", s.l,
                           s.origin(0), s.origin(1), s.origin(2));
      });

  m.new_usertype<AOBasis>(
      "AOBasis", sol::no_constructor,
      "shells", [](const AOBasis &b) { return sol::as_table(b.shells()); },
      "set_pure", &AOBasis::set_pure,
      "size", &AOBasis::size,
      "nbf", &AOBasis::nbf,
      "atoms", [](const AOBasis &b) { return sol::as_table(b.atoms()); },
      "first_bf",
      [](const AOBasis &b) { return sol::as_table(b.first_bf()); },
      "bf_to_shell",
      [](const AOBasis &b) { return sol::as_table(b.bf_to_shell()); },
      "bf_to_atom",
      [](const AOBasis &b) { return sol::as_table(b.bf_to_atom()); },
      "shell_to_atom",
      [](const AOBasis &b) { return sol::as_table(b.shell_to_atom()); },
      "atom_to_shell",
      [](const AOBasis &b) { return sol::as_table(b.atom_to_shell()); },
      "l_max", &AOBasis::l_max,
      "name", &AOBasis::name,
      "evaluate",
      [](const AOBasis &basis, const sol::table &points,
         sol::optional<int> derivatives) {
        const int d = derivatives.value_or(0);
        if (d < 0 || d > 2) {
          throw std::runtime_error(
              "Invalid max derivative (must be 0, 1, 2)");
        }
        return occ::gto::evaluate_basis(basis, table_to_mat3n(points), d);
      },
      sol::meta_function::to_string, [](const AOBasis &basis) {
        return fmt::format("<AOBasis ({}) nsh={} nbf={} natoms={}>",
                           basis.name(), basis.nsh(), basis.nbf(),
                           basis.atoms().size());
      });

  // `AOBasis::load` is a static factory; expose as a free function (Lua
  // has no native `static method` syntax in sol2 usertypes).
  m.set_function("AOBasis_load", &AOBasis::load);
}

// ---- molecular orbitals + wavefunction ---------------------------------

void register_mo_and_wfn(sol::table &m) {
  m.new_usertype<MolecularOrbitals>(
      "MolecularOrbitals",
      sol::call_constructor,
      sol::factories([]() { return MolecularOrbitals{}; }),
      "kind", &MolecularOrbitals::kind,
      "num_alpha", &MolecularOrbitals::n_alpha,
      "num_beta", &MolecularOrbitals::n_beta,
      "num_ao", &MolecularOrbitals::n_ao,
      "orbital_coeffs",
      sol::readonly_property([](const MolecularOrbitals &mo, sol::this_state s) {
        return mo.C;
      }),
      "occupied_orbital_coeffs",
      sol::readonly_property([](const MolecularOrbitals &mo, sol::this_state s) {
        return mo.Cocc;
      }),
      "density_matrix",
      sol::readonly_property([](const MolecularOrbitals &mo, sol::this_state s) {
        return mo.D;
      }),
      "orbital_energies",
      sol::readonly_property([](const MolecularOrbitals &mo, sol::this_state s) {
        return mo.energies;
      }),
      "expectation_value",
      [](const MolecularOrbitals &mo, const sol::table &op) {
        const int n = static_cast<int>(op.size());
        Mat opmat(n, n);
        for (int i = 0; i < n; ++i) {
          sol::table row = op.get<sol::table>(i + 1);
          for (int j = 0; j < n; ++j) opmat(i, j) = row.get<double>(j + 1);
        }
        return 2 * occ::qm::expectation(mo.kind, mo.D, opmat);
      },
      sol::meta_function::to_string, [](const MolecularOrbitals &mo) {
        return fmt::format(
            "<MolecularOrbitals kind={} nao={} nalpha={} nbeta={}>",
            spinorbital_kind_to_string(mo.kind), mo.n_ao, mo.n_alpha,
            mo.n_beta);
      });

  m.new_usertype<Wavefunction>(
      "Wavefunction", sol::no_constructor,
      "atoms",
      sol::readonly_property(
          [](const Wavefunction &w) { return sol::as_table(w.atoms); }),
      "molecular_orbitals",
      sol::readonly_property(
          [](const Wavefunction &w) -> const MolecularOrbitals & {
            return w.mo;
          }),
      "basis",
      sol::readonly_property(
          [](const Wavefunction &w) -> const AOBasis & { return w.basis; }),
      "charge", &Wavefunction::charge,
      "multiplicity", &Wavefunction::multiplicity,
      "mulliken_charges",
      [](const Wavefunction &w, sol::this_state s) {
        return w.mulliken_charges();
      },
      "rotate",
      [](Wavefunction &w, const sol::table &rotation) {
        w.apply_rotation(table_to_mat3(rotation));
      },
      "translate",
      [](Wavefunction &w, const sol::table &translation) {
        w.apply_translation(table_to_vec3(translation));
      },
      // C++ signature is (Mat3 rotation, Vec3 translation) — Lua callers
      // pass two tables. (`apply_transformation` doesn't take a Mat4.)
      "transform",
      [](Wavefunction &w, const sol::table &rotation,
         const sol::table &translation) {
        w.apply_transformation(table_to_mat3(rotation),
                                table_to_vec3(translation));
      },
      "save",
      [](Wavefunction &w, const std::string &filename) { w.save(filename); },
      "copy", [](const Wavefunction &w) { return Wavefunction(w); },
      "electron_density",
      [](const Wavefunction &wfn, const sol::table &points,
         sol::optional<int> derivatives, sol::this_state s) {
        return mat_to_table(
            s, occ::density::evaluate_density_on_grid(
                   wfn, table_to_mat3n(points), derivatives.value_or(0)));
      },
      "chelpg_charges",
      [](const Wavefunction &wfn, sol::this_state s) {
        return chelpg_charges(wfn);
      },
      "to_fchk",
      [](Wavefunction &wfn, const std::string &filename) {
        auto writer = occ::io::FchkWriter(filename);
        wfn.save(writer);
        writer.write();
      },
      sol::meta_function::to_string, [](const Wavefunction &wfn) {
        return fmt::format("<Wavefunction {} {}/{} kind={} nbf={} charge={}>",
                           chemical_formula_from_atoms(wfn.atoms), wfn.method,
                           wfn.basis.name(),
                           spinorbital_kind_to_string(wfn.mo.kind),
                           wfn.basis.nbf(), wfn.charge());
      });

  m.set_function("Wavefunction_load", &Wavefunction::load);
  m.set_function("Wavefunction_from_fchk", [](const std::string &filename) {
    auto reader = occ::io::FchkReader(filename);
    return Wavefunction(reader);
  });
  m.set_function("Wavefunction_from_molden", [](const std::string &filename) {
    auto reader = occ::io::MoldenReader(filename);
    return Wavefunction(reader);
  });
}

// ---- HartreeFock + SCF -------------------------------------------------

void register_hf_and_scf(sol::table &m) {
  using HF = SCF<HartreeFock>;

  m.new_usertype<SCFConvergenceSettings>(
      "SCFConvergenceSettings",
      sol::call_constructor,
      sol::factories([]() { return SCFConvergenceSettings{}; }),
      "energy_threshold", &SCFConvergenceSettings::energy_threshold,
      "commutator_threshold", &SCFConvergenceSettings::commutator_threshold,
      "incremental_fock_threshold",
      &SCFConvergenceSettings::incremental_fock_threshold,
      "energy_converged", &SCFConvergenceSettings::energy_converged,
      "commutator_converged", &SCFConvergenceSettings::commutator_converged,
      "energy_and_commutator_converged",
      &SCFConvergenceSettings::energy_and_commutator_converged,
      "start_incremental_fock",
      &SCFConvergenceSettings::start_incremental_fock);

  m.new_usertype<HF>(
      "HF",
      sol::call_constructor,
      sol::factories(
          [](HartreeFock &hf) { return HF(hf); },
          [](HartreeFock &hf, SpinorbitalKind kind) { return HF(hf, kind); }),
      "convergence_settings", &HF::convergence_settings,
      "set_charge_multiplicity", &HF::set_charge_multiplicity,
      "set_initial_guess", &HF::set_initial_guess_from_wfn,
      "set_external_potential",
      sol::overload(
          [](HF &scf, const sol::table &V_ext_table, double nuclear_energy,
             const std::string &label) {
            // Accept the external potential matrix as a Lua table-of-tables.
            const int n = static_cast<int>(V_ext_table.size());
            Mat V(n, n);
            for (int i = 0; i < n; ++i) {
              sol::table row = V_ext_table.get<sol::table>(i + 1);
              for (int j = 0; j < n; ++j) V(i, j) = row.get<double>(j + 1);
            }
            scf.set_external_potential(V, nuclear_energy, label);
          },
          [](HF &scf, const PointChargePotential &pot) {
            scf.set_external_potential(pot);
          },
          [](HF &scf, const WolfPointChargePotential &pot) {
            scf.set_external_potential(pot);
          }),
      "scf_kind", &HF::scf_kind,
      "run", &HF::compute_scf_energy,
      "compute_scf_energy", &HF::compute_scf_energy,
      "wavefunction", &HF::wavefunction,
      sol::meta_function::to_string, [](const HF &hf) {
        return fmt::format("<SCF(HF) ({}, {} atoms)>",
                           hf.m_procedure.aobasis().name(),
                           hf.m_procedure.atoms().size());
      });

  m.new_usertype<HartreeFock>(
      "HartreeFock",
      sol::call_constructor,
      sol::constructors<HartreeFock(const AOBasis &)>(),
      "point_charge_interaction_energy",
      &HartreeFock::nuclear_point_charge_interaction_energy,
      "wolf_point_charge_interaction_energy",
      &HartreeFock::wolf_point_charge_interaction_energy,
      "point_charge_interaction_matrix",
      [](HartreeFock &hf, const std::vector<occ::core::PointCharge> &charges,
         sol::optional<double> alpha, sol::this_state s) {
        return mat_to_table(s, hf.compute_point_charge_interaction_matrix(
                                    charges, alpha.value_or(1e16)));
      },
      "nuclear_attraction_matrix",
      [](HartreeFock &hf, sol::this_state s) {
        return hf.compute_nuclear_attraction_matrix();
      },
      "set_density_fitting_basis",
      [](HartreeFock &hf, const std::string &name,
         sol::optional<double> threshold) {
        hf.set_density_fitting_basis(name, threshold.value_or(1e-4));
      },
      "kinetic_matrix",
      [](HartreeFock &hf, sol::this_state s) {
        return hf.compute_kinetic_matrix();
      },
      "overlap_matrix",
      [](HartreeFock &hf, sol::this_state s) {
        return hf.compute_overlap_matrix();
      },
      "overlap_matrix_for_basis",
      [](HartreeFock &hf, const AOBasis &other, sol::this_state s) {
        return hf.compute_overlap_matrix_for_basis(other);
      },
      "nuclear_repulsion", &HartreeFock::nuclear_repulsion_energy,
      "scf",
      [](HartreeFock &hf, sol::optional<SpinorbitalKind> kind) {
        return HF(hf, kind.value_or(SpinorbitalKind::Restricted));
      },
      "set_precision", &HartreeFock::set_precision,
      "coulomb_matrix",
      [](const HartreeFock &hf, const MolecularOrbitals &mo,
         sol::this_state s) {
        return hf.compute_J(mo);
      },
      "fock_matrix",
      [](const HartreeFock &hf, const MolecularOrbitals &mo,
         sol::this_state s) {
        return hf.compute_fock(mo);
      },
      "compute_gradient",
      [](HartreeFock &hf, const MolecularOrbitals &mo, sol::this_state s) {
        GradientEvaluator<HartreeFock> grad(hf);
        return grad(mo);
      },
      "nuclear_repulsion_gradient",
      [](HartreeFock &hf, sol::this_state s) {
        return hf.nuclear_repulsion_gradient();
      },
      "hessian_evaluator",
      [](HartreeFock &hf) {
        return occ::qm::HessianEvaluator<HartreeFock>(hf);
      },
      sol::meta_function::to_string, [](const HartreeFock &hf) {
        return fmt::format("<HartreeFock ({}, {} atoms)>",
                           hf.aobasis().name(), hf.atoms().size());
      });
}

// ---- external potentials -----------------------------------------------

void register_external_potentials(sol::table &m) {
  m.new_usertype<PointChargePotential>(
      "PointChargePotential",
      sol::call_constructor,
      sol::factories(
          []() { return PointChargePotential{}; },
          [](PointChargeList charges) {
            return PointChargePotential{std::move(charges)};
          }),
      "charges", &PointChargePotential::charges,
      "compute_potential_matrix",
      [](const PointChargePotential &pot, HartreeFock &hf,
         sol::this_state s) {
        return pot.compute_potential_matrix(hf);
      },
      "nuclear_interaction_energy",
      [](const PointChargePotential &pot, const HartreeFock &hf) {
        return pot.nuclear_interaction_energy(hf);
      },
      "label",
      [](const PointChargePotential &p) { return std::string(p.label()); },
      sol::meta_function::to_string, [](const PointChargePotential &p) {
        return fmt::format("<{}>", p.descriptor());
      });

  m.new_usertype<WolfPointChargePotential>(
      "WolfPointChargePotential",
      sol::call_constructor,
      sol::factories(
          []() { return WolfPointChargePotential{}; },
          [](PointChargeList charges, std::vector<double> molecular_charges,
             double alpha, double cutoff) {
            return WolfPointChargePotential{std::move(charges),
                                             std::move(molecular_charges),
                                             alpha, cutoff};
          }),
      "charges", &WolfPointChargePotential::charges,
      "molecular_charges", &WolfPointChargePotential::molecular_charges,
      "alpha", &WolfPointChargePotential::alpha,
      "cutoff", &WolfPointChargePotential::cutoff,
      "compute_potential_matrix",
      [](const WolfPointChargePotential &pot, HartreeFock &hf,
         sol::this_state s) {
        return pot.compute_potential_matrix(hf);
      },
      "nuclear_interaction_energy",
      [](const WolfPointChargePotential &pot, const HartreeFock &hf) {
        return pot.nuclear_interaction_energy(hf);
      },
      "label",
      [](const WolfPointChargePotential &p) {
        return std::string(p.label());
      },
      sol::meta_function::to_string, [](const WolfPointChargePotential &p) {
        return fmt::format("<{}>", p.descriptor());
      });
}

// ---- hessian + vibrations ---------------------------------------------

void register_hessian_and_vibrations(sol::table &m) {
  using HessEval = occ::qm::HessianEvaluator<HartreeFock>;
  m.new_usertype<HessEval>(
      "HessianEvaluatorHF",
      sol::call_constructor,
      sol::constructors<HessEval(HartreeFock &)>(),
      "set_method", &HessEval::set_method,
      "set_step_size", &HessEval::set_step_size,
      "set_use_acoustic_sum_rule", &HessEval::set_use_acoustic_sum_rule,
      "step_size", &HessEval::step_size,
      "use_acoustic_sum_rule", &HessEval::use_acoustic_sum_rule,
      "nuclear_repulsion",
      [](HessEval &h, sol::this_state s) {
        return h.nuclear_repulsion();
      },
      // HessianEvaluator::operator() takes a Wavefunction in this version
      // of occ, not a MolecularOrbitals directly.
      "compute",
      [](HessEval &h, const Wavefunction &wfn, sol::this_state s) {
        return h(wfn);
      },
      sol::meta_function::to_string, [](const HessEval &h) {
        return fmt::format(
            "<HessianEvaluatorHF step_size={:.4f} acoustic_sum_rule={}>",
            h.step_size(), h.use_acoustic_sum_rule());
      });

  using VM = occ::core::VibrationalModes;
  m.new_usertype<VM>(
      "VibrationalModes", sol::no_constructor,
      "frequencies_cm",
      sol::readonly_property([](const VM &v, sol::this_state s) {
        return v.frequencies_cm;
      }),
      "frequencies_hartree",
      sol::readonly_property([](const VM &v, sol::this_state s) {
        return v.frequencies_hartree;
      }),
      "normal_modes",
      sol::readonly_property([](const VM &v, sol::this_state s) {
        return v.normal_modes;
      }),
      "mass_weighted_hessian",
      sol::readonly_property([](const VM &v, sol::this_state s) {
        return v.mass_weighted_hessian;
      }),
      "hessian",
      sol::readonly_property([](const VM &v, sol::this_state s) {
        return v.hessian;
      }),
      "n_modes", &VM::n_modes,
      "n_atoms", &VM::n_atoms,
      "summary_string", &VM::summary_string,
      "frequencies_string", &VM::frequencies_string,
      "normal_modes_string",
      [](const VM &v, sol::optional<double> threshold) {
        return v.normal_modes_string(threshold.value_or(0.1));
      },
      "get_all_frequencies",
      [](const VM &v, sol::this_state s) {
        return v.get_all_frequencies();
      },
      sol::meta_function::to_string, [](const VM &v) {
        return fmt::format("<VibrationalModes n_modes={} n_atoms={}>",
                           v.n_modes(), v.n_atoms());
      });
}

// ---- IntegralEngine + DF + smearing + GTO values ----------------------

void register_integral_engines(sol::table &m) {
  m.new_usertype<IntegralEngine>(
      "IntegralEngine",
      sol::call_constructor,
      sol::factories(
          [](const AOBasis &basis) { return IntegralEngine(basis); },
          [](const std::vector<Atom> &atoms,
             const std::vector<Shell> &shells) {
            return IntegralEngine(atoms, shells);
          }),
      "schwarz",
      [](const IntegralEngine &e, sol::this_state s) {
        return e.schwarz();
      },
      "set_precision", &IntegralEngine::set_precision,
      "set_range_separated_omega", &IntegralEngine::set_range_separated_omega,
      "range_separated_omega", &IntegralEngine::range_separated_omega,
      "is_spherical", &IntegralEngine::is_spherical,
      "have_auxiliary_basis", &IntegralEngine::have_auxiliary_basis,
      "set_auxiliary_basis",
      [](IntegralEngine &e, const std::vector<Shell> &basis,
         sol::optional<bool> dummy) {
        e.set_auxiliary_basis(basis, dummy.value_or(false));
      },
      "clear_auxiliary_basis", &IntegralEngine::clear_auxiliary_basis,
      "one_electron_operator",
      [](const IntegralEngine &e, cint::Operator op,
         sol::optional<bool> use_shellpair, sol::this_state s) {
        return mat_to_table(s, e.one_electron_operator(
                                    op, use_shellpair.value_or(true)));
      },
      "nbf", &IntegralEngine::nbf,
      "nsh", &IntegralEngine::nsh,
      "aobasis", &IntegralEngine::aobasis,
      "auxbasis", &IntegralEngine::auxbasis,
      "nbf_aux", &IntegralEngine::nbf_aux,
      "nsh_aux", &IntegralEngine::nsh_aux,
      "have_effective_core_potentials",
      &IntegralEngine::have_effective_core_potentials,
      "set_effective_core_potentials",
      &IntegralEngine::set_effective_core_potentials,
      sol::meta_function::to_string, [](const IntegralEngine &engine) {
        return fmt::format("<IntegralEngine nbf={} nsh={} spherical={}>",
                           engine.nbf(), engine.nsh(),
                           engine.is_spherical() ? "true" : "false");
      });

  m.new_enum<OrbitalSmearing::Kind>(
      "OrbitalSmearingKind", {{"None_", OrbitalSmearing::Kind::None},
                              {"Fermi", OrbitalSmearing::Kind::Fermi},
                              {"Gaussian", OrbitalSmearing::Kind::Gaussian},
                              {"Linear", OrbitalSmearing::Kind::Linear}});

  m.new_usertype<OrbitalSmearing>(
      "OrbitalSmearing",
      sol::call_constructor,
      sol::factories([]() { return OrbitalSmearing{}; }),
      "kind", &OrbitalSmearing::kind,
      "mu", &OrbitalSmearing::mu,
      "fermi_level", &OrbitalSmearing::fermi_level,
      "sigma", &OrbitalSmearing::sigma,
      "entropy", &OrbitalSmearing::entropy,
      "calculate_entropy", &OrbitalSmearing::calculate_entropy,
      "ec_entropy", &OrbitalSmearing::ec_entropy,
      sol::meta_function::to_string, [](const OrbitalSmearing &os) {
        const char *k = "?";
        switch (os.kind) {
        case OrbitalSmearing::Kind::None: k = "None"; break;
        case OrbitalSmearing::Kind::Fermi: k = "Fermi"; break;
        case OrbitalSmearing::Kind::Gaussian: k = "Gaussian"; break;
        case OrbitalSmearing::Kind::Linear: k = "Linear"; break;
        }
        return fmt::format(
            "<OrbitalSmearing kind={} sigma={:.6f} mu={:.6f}>", k, os.sigma,
            os.mu);
      });

  m.new_enum<IntegralEngineDF::Policy>(
      "IntegralEngineDFPolicy",
      {{"Choose", IntegralEngineDF::Policy::Choose},
       {"Direct", IntegralEngineDF::Policy::Direct},
       {"Stored", IntegralEngineDF::Policy::Stored}});

  m.new_usertype<IntegralEngineDF>(
      "IntegralEngineDF",
      sol::call_constructor,
      sol::constructors<IntegralEngineDF(const std::vector<Atom> &,
                                          const std::vector<Shell> &,
                                          const std::vector<Shell> &)>(),
      "set_integral_policy", &IntegralEngineDF::set_integral_policy,
      "set_range_separated_omega",
      &IntegralEngineDF::set_range_separated_omega,
      "set_precision", &IntegralEngineDF::set_precision,
      sol::meta_function::to_string, [](const IntegralEngineDF &e) {
        const char *p = "?";
        switch (e.integral_policy()) {
        case IntegralEngineDF::Policy::Choose: p = "Choose"; break;
        case IntegralEngineDF::Policy::Direct: p = "Direct"; break;
        case IntegralEngineDF::Policy::Stored: p = "Stored"; break;
        }
        return fmt::format("<IntegralEngineDF policy={} precision={:.2e}>",
                           p, e.precision());
      });

  using occ::gto::GTOValues;
  m.new_usertype<GTOValues>(
      "GTOValues",
      sol::call_constructor,
      sol::factories([]() { return GTOValues{}; }),
      "reserve", &GTOValues::reserve,
      "set_zero", &GTOValues::set_zero,
      "phi",
      sol::readonly_property([](const GTOValues &g, sol::this_state s) {
        return g.phi;
      }),
      "phi_x",
      sol::readonly_property([](const GTOValues &g, sol::this_state s) {
        return g.phi_x;
      }),
      "phi_y",
      sol::readonly_property([](const GTOValues &g, sol::this_state s) {
        return g.phi_y;
      }),
      "phi_z",
      sol::readonly_property([](const GTOValues &g, sol::this_state s) {
        return g.phi_z;
      }));
}

// ---- free vibrational-analysis helpers --------------------------------

void register_vibrational_free_functions(sol::table &m) {
  m.set_function(
      "compute_vibrational_modes",
      [](const sol::table &hessian, const occ::core::Molecule &mol,
         sol::optional<bool> project) {
        const int n = static_cast<int>(hessian.size());
        Mat H(n, n);
        for (int i = 0; i < n; ++i) {
          sol::table row = hessian.get<sol::table>(i + 1);
          for (int j = 0; j < n; ++j) H(i, j) = row.get<double>(j + 1);
        }
        return occ::core::compute_vibrational_modes(H, mol,
                                                     project.value_or(false));
      });

  m.set_function(
      "eigenvalues_to_frequencies_cm",
      [](const sol::table &eigenvalues, sol::this_state s) {
        Vec ev = table_to_vecx(eigenvalues);
        return occ::core::eigenvalues_to_frequencies_cm(ev);
      });

  m.set_function(
      "frequencies_cm_to_hartree",
      [](const sol::table &freqs_cm, sol::this_state s) {
        Vec f = table_to_vecx(freqs_cm);
        return occ::core::frequencies_cm_to_hartree(f);
      });
}

} // namespace

void register_qm_bindings(sol::state_view, sol::table &occ_module) {
  register_enums_and_small_types(occ_module);
  register_basis(occ_module);
  register_mo_and_wfn(occ_module);
  register_hf_and_scf(occ_module);
  register_external_potentials(occ_module);
  register_hessian_and_vibrations(occ_module);
  register_integral_engines(occ_module);
  register_vibrational_free_functions(occ_module);
}

} // namespace occ::lua_bindings
