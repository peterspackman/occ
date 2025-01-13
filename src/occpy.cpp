#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/core/point_charge.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/io/cifparser.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_cg.h>
#include <occ/qm/expectation.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;
using namespace nb::literals;

template <typename K, typename V>
inline nb::dict
convert_map_to_dict(const ankerl::unordered_dense::map<K, V> &map) {
  nb::dict result;
  for (const auto &[key, value] : map) {
    result[nb::cast(key)] = nb::cast(value);
  }
  return result;
}

using occ::IVec;
using occ::Mat;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Atom;
using occ::core::Dimer;
using occ::core::Element;
using occ::core::Molecule;
using occ::core::PointCharge;
using occ::crystal::AsymmetricUnit;
using occ::crystal::Crystal;
using occ::crystal::CrystalAtomRegion;
using occ::crystal::CrystalDimers;
using SymmetryRelatedDimer = occ::crystal::CrystalDimers::SymmetryRelatedDimer;
using occ::crystal::HKL;
using occ::crystal::UnitCell;
using occ::dft::DFT;
using occ::qm::AOBasis;
using occ::qm::HartreeFock;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::Shell;
using occ::qm::Wavefunction;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto G = occ::qm::SpinorbitalKind::General;

NB_MODULE(_occpy, m) {
  nb::class_<Element>(m, "Element")
      .def(nb::init<const std::string &>())
      .def("symbol", &Element::symbol,
           "the symbol of the element e.g. H, He ...")
      .def("mass", &Element::mass,
           "mass number of the element e.g. 12.01 for C")
      .def("name", &Element::name,
           "the name of the element e.g hydrogen, helium...'")
      .def("van_der_waals_radius", &Element::van_der_waals_radius,
           "bondi van der Waals radius for element")
      .def("covalent_radius", &Element::covalent_radius,
           "covalent radius for element")
      .def("atomic_number", &Element::atomic_number,
           "atomic number e.g. 1, 2 ...")
      .def("__repr__",
           [](const Element &a) { return "<Element '" + a.symbol() + "'>"; });

  nb::class_<Atom>(m, "Atom")
      .def(nb::init<int, double, double, double>())
      .def_rw("atomic_number", &Atom::atomic_number,
              "atomic number for corresponding element")
      .def_prop_rw("position", &Atom::position, &Atom::set_position,
                   "Cartesian position of the atom (Bohr)")
      .def("__repr__", [](const Atom &a) {
        return fmt::format("<Atom {} [{:.5f}, {:.5f}, {:.5f}>", a.atomic_number,
                           a.x, a.y, a.z);
      });

  nb::class_<PointCharge>(m, "PointCharge")
      .def(nb::init<double, double, double, double>())
      .def(nb::init<double, std::array<double, 3>>())
      .def(nb::init<double, const occ::Vec3 &>())
      .def_prop_rw("charge", &PointCharge::charge, &PointCharge::set_charge)
      .def_prop_rw("position", &PointCharge::position,
                   &PointCharge::set_position)
      .def("__repr__", [](const PointCharge &pc) {
        const auto &pos = pc.position();
        return fmt::format("<PointCharge q={:.5f} [{:.5f}, {:.5f}, {:.5f}]>",
                           pc.charge(), pos.x(), pos.y(), pos.z());
      });

  nb::class_<Shell>(m, "Shell")
      .def(nb::init<PointCharge, double>())
      .def_ro("origin", &Shell::origin, "shell position/origin (Bohr)")
      .def_ro("exponents", &Shell::exponents,
              "array of exponents for primitives in this shell")
      .def_ro("contraction_coefficients", &Shell::contraction_coefficients,
              "array of contraction coefficients for in this shell")
      .def("num_contractions", &Shell::num_contractions,
           "number of contractions")
      .def("num_primitives", &Shell::num_primitives,
           "number of primitive gaussians")
      .def("norm", &Shell::norm, "norm of the shell")
      .def("__repr__", [](const Shell &s) {
        return fmt::format("<Shell l={} [{:.5f}, {:.5f}, {:.5f}]>", s.l,
                           s.origin(0), s.origin(1), s.origin(2));
      });

  nb::class_<AOBasis>(m, "AOBasis")
      .def_static("load", &AOBasis::load)
      .def("shells", &AOBasis::shells)
      .def("set_pure", &AOBasis::set_pure)
      .def("size", &AOBasis::size)
      .def("nbf", &AOBasis::nbf)
      .def("atoms", &AOBasis::atoms)
      .def("first_bf", &AOBasis::first_bf)
      .def("bf_to_shell", &AOBasis::bf_to_shell)
      .def("bf_to_atom", &AOBasis::bf_to_atom)
      .def("shell_to_atom", &AOBasis::shell_to_atom)
      .def("atom_to_shell", &AOBasis::atom_to_shell)
      .def("l_max", &AOBasis::l_max)
      .def("name", &AOBasis::name)
      .def("__repr__", [](const AOBasis &basis) {
        return fmt::format("<AOBasis ({}) nsh={} nbf={} natoms={}>",
                           basis.name(), basis.nsh(), basis.nbf(),
                           basis.atoms().size());
      });

  nb::class_<MolecularOrbitals>(m, "MolecularOrbitals")
      .def_rw("num_alpha", &MolecularOrbitals::n_alpha)
      .def_rw("num_beta", &MolecularOrbitals::n_beta)
      .def_rw("num_ao", &MolecularOrbitals::n_ao)
      .def_rw("orbital_coeffs", &MolecularOrbitals::C)
      .def_rw("occupied_orbital_coeffs", &MolecularOrbitals::Cocc)
      .def_rw("density_matrix", &MolecularOrbitals::D)
      .def_rw("orbital_energies", &MolecularOrbitals::energies)
      .def("expectation_value", [](const MolecularOrbitals &mo, const Mat &op) {
        return 2 * occ::qm::expectation(mo.kind, mo.D, op);
      });

  nb::class_<Wavefunction>(m, "Wavefunction")
      .def_rw("molecular_orbitals", &Wavefunction::mo)
      .def_ro("atoms", &Wavefunction::atoms)
      .def("mulliken_charges", &Wavefunction::mulliken_charges)
      .def("multiplicity", &Wavefunction::multiplicity)
      .def("rotate", &Wavefunction::apply_rotation)
      .def("translate", &Wavefunction::apply_translation)
      .def("transform", &Wavefunction::apply_transformation)
      .def("charge", &Wavefunction::charge)
      .def_static("load", &Wavefunction::load)
      .def("save", nb::overload_cast<const std::string &>(&Wavefunction::save))
      .def_ro("basis", &Wavefunction::basis)
      .def("to_fchk",
           [](Wavefunction &wfn, const std::string &filename) {
             auto writer = occ::io::FchkWriter(filename);
             wfn.save(writer);
             writer.write();
           })
      .def_static("from_fchk",
                  [](const std::string &filename) {
                    auto reader = occ::io::FchkReader(filename);
                    Wavefunction wfn(reader);
                    return wfn;
                  })
      .def_static("from_molden", [](const std::string &filename) {
        auto reader = occ::io::MoldenReader(filename);
        Wavefunction wfn(reader);
        return wfn;
      });

  using HF = SCF<HartreeFock>;

  nb::class_<HF>(m, "HF")
      .def(nb::init<HartreeFock &>())
      .def("set_charge_multiplicity", &HF::set_charge_multiplicity)
      .def("set_initial_guess", &HF::set_initial_guess_from_wfn)
      .def("scf_kind", &HF::scf_kind)
      .def("run", &HF::compute_scf_energy)
      .def("wavefunction", &HF::wavefunction)
      .def("__repr__", [](const HF &hf) {
        return fmt::format("<SCF(HF) ({}, {} atoms)>",
                           hf.m_procedure.aobasis().name(),
                           hf.m_procedure.atoms().size());
      });

  using KS = SCF<DFT>;

  nb::class_<KS>(m, "KS")
      .def(nb::init<DFT &>())
      .def("set_charge_multiplicity", &KS::set_charge_multiplicity)
      .def("set_initial_guess", &KS::set_initial_guess_from_wfn)
      .def("scf_kind", &KS::scf_kind)
      .def("run", &KS::compute_scf_energy)
      .def("wavefunction", &KS::wavefunction)
      .def("__repr__", [](const KS &ks) {
        return fmt::format("<SCF(KS) ({}, {} atoms)>",
                           ks.m_procedure.aobasis().name(),
                           ks.m_procedure.atoms().size());
      });

  nb::class_<HartreeFock>(m, "HartreeFock")
      .def(nb::init<const AOBasis &>())
      .def("point_charge_interaction_energy",
           &HartreeFock::nuclear_point_charge_interaction_energy)
      .def("wolf_point_charge_interaction_energy",
           &HartreeFock::wolf_point_charge_interaction_energy)
      .def("point_charge_interaction_matrix",
           &HartreeFock::compute_point_charge_interaction_matrix,
           "point_charges"_a, "alpha"_a = 1e16)
      .def("wolf_interaction_matrix",
           &HartreeFock::compute_wolf_interaction_matrix)
      .def("nuclear_attraction_matrix",
           &HartreeFock::compute_nuclear_attraction_matrix)
      .def("nuclear_attraction_matrix",
           &HartreeFock::compute_nuclear_attraction_matrix)
      .def("set_density_fitting_basis", &HartreeFock::set_density_fitting_basis)
      .def("kinetic_matrix", &HartreeFock::compute_kinetic_matrix)
      .def("overlap_matrix", &HartreeFock::compute_overlap_matrix)
      .def("overlap_matrix_for_basis",
           &HartreeFock::compute_overlap_matrix_for_basis)
      .def("nuclear_repulsion", &HartreeFock::nuclear_repulsion_energy)
      .def(
          "scf",
          [](HartreeFock &hf, bool unrestricted = false) {
            if (unrestricted)
              return HF(hf, U);
            else
              return HF(hf, R);
          },
          "unrestricted"_a = false)
      .def("set_precision", &HartreeFock::set_precision)
      .def("coulomb_matrix",
           [](const HartreeFock &hf, const MolecularOrbitals &mo) {
             return hf.compute_J(mo);
           })
      .def("coulomb_and_exchange_matrices",
           [](const HartreeFock &hf, const MolecularOrbitals &mo) {
             return hf.compute_JK(mo);
           })
      .def("fock_matrix",
           [](const HartreeFock &hf, const MolecularOrbitals &mo) {
             return hf.compute_fock(mo);
           })
      .def("__repr__", [](const HartreeFock &hf) {
        return fmt::format("<HartreeFock ({}, {} atoms)>", hf.aobasis().name(),
                           hf.atoms().size());
      });

  using occ::io::BeckeGridSettings;
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

  nb::class_<DFT>(m, "DFT")
      .def(nb::init<const std::string &, const AOBasis &>())
      .def(nb::init<const std::string &, const AOBasis &,
                    const BeckeGridSettings &>())
      .def("nuclear_attraction_matrix", &DFT::compute_nuclear_attraction_matrix)
      .def("kinetic_matrix", &DFT::compute_kinetic_matrix)
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
          [](DFT &dft, bool unrestricted) {
            if (unrestricted)
              return KS(dft, U);
            else
              return KS(dft, R);
          },
          "unrestricted"_a = false)
      .def("__repr__", [](const DFT &dft) {
        return fmt::format("<DFT {} ({}, {} atoms)>", dft.method_string(),
                           dft.aobasis().name(), dft.atoms().size());
      });

  nb::class_<Molecule>(m, "Molecule")
      .def(nb::init<const IVec &, const Mat3N &>())
      .def("__len__", &Molecule::size)
      .def("elements", &Molecule::elements)
      .def("positions", &Molecule::positions)
      .def_prop_rw("name", &Molecule::name, &Molecule::set_name)
      .def("atomic_numbers", &Molecule::atomic_numbers)
      .def("vdw_radii", &Molecule::vdw_radii)
      .def("molar_mass", &Molecule::molar_mass)
      .def("atoms", &Molecule::atoms)
      .def("center_of_mass", &Molecule::center_of_mass)
      .def_static("from_xyz_file",
                  [](const std::string &filename) {
                    return occ::io::molecule_from_xyz_file(filename);
                  })
      .def_static("from_xyz_string",
                  [](const std::string &contents) {
                    return occ::io::molecule_from_xyz_string(contents);
                  })
      .def("__repr__", [](const Molecule &mol) {
        auto com = mol.center_of_mass();
        return fmt::format("<Molecule {} @[{:.5f}, {:.5f}, {:.5f}]>",
                           mol.name(), com.x(), com.y(), com.z());
      });

  nb::class_<Dimer>(m, "Dimer")
      .def(nb::init<const Molecule &, const Molecule &>())
      .def(nb::init<const std::vector<Atom> &, const std::vector<Atom> &>())
      .def_prop_ro("a", &Dimer::a)
      .def_prop_ro("b", &Dimer::b)
      .def_prop_rw("name", &Dimer::name, &Dimer::set_name);

  // occ::crystal
  nb::class_<HKL>(m, "HKL")
      .def(nb::init<int, int, int>())
      .def("d", &HKL::d)
      .def_static("floor", &HKL::floor)
      .def_static("ceil", &HKL::ceil)
      .def_static("maximum", &HKL::maximum)
      .def_static("minimum", &HKL::minimum)
      .def("vector", &HKL::vector)
      .def_rw("h", &HKL::h)
      .def_rw("k", &HKL::k)
      .def_rw("l", &HKL::l)
      .def("__repr__", [](const HKL &hkl) {
        return fmt::format("<HKL [{} {} {}]>", hkl.h, hkl.k, hkl.l);
      });

  nb::class_<Crystal>(m, "Crystal")
      .def("symmetry_unique_molecules", &Crystal::symmetry_unique_molecules)
      .def("symmetry_unique_dimers", &Crystal::symmetry_unique_dimers)
      .def("unit_cell", &Crystal::unit_cell)
      .def("unit_cell_molecules", &Crystal::unit_cell_molecules)
      .def("unit_cell_atoms", &Crystal::unit_cell_atoms)
      .def("unit_cell_dimers", &Crystal::unit_cell_dimers)
      .def("atom_surroundings", &Crystal::atom_surroundings)
      .def("dimer_symmetry_string", &Crystal::dimer_symmetry_string)
      .def("asymmetric_unit_atom_surroundings",
           &Crystal::asymmetric_unit_atom_surroundings)
      .def("num_sites", &Crystal::num_sites)
      .def("labels", &Crystal::labels)
      .def("to_fractional", &Crystal::to_fractional)
      .def("to_cartesian", &Crystal::to_cartesian)
      .def("volume", &Crystal::volume)
      .def("slab", &Crystal::slab)
      .def("asymmetric_unit",
           nb::overload_cast<>(&Crystal::asymmetric_unit, nb::const_))
      .def_static("create_primitive_supercell",
                  &Crystal::create_primitive_supercell)
      .def_static("from_cif_file",
                  [](const std::string &filename) {
                    occ::io::CifParser parser;
                    return parser.parse_crystal_from_file(filename).value();
                  })
      .def_static("from_cif_string",
                  [](const std::string &contents) {
                    occ::io::CifParser parser;
                    return parser.parse_crystal_from_string(contents).value();
                  })

      .def("__repr__", [](const Crystal &c) {
        return fmt::format("<Crystal {} {}>",
                           c.asymmetric_unit().chemical_formula(),
                           c.space_group().symbol());
      });

  nb::class_<CrystalAtomRegion>(m, "CrystalAtomRegion")
      .def_ro("frac_pos", &CrystalAtomRegion::frac_pos)
      .def_ro("cart_pos", &CrystalAtomRegion::cart_pos)
      .def_ro("asym_idx", &CrystalAtomRegion::asym_idx)
      .def_ro("atomic_numbers", &CrystalAtomRegion::atomic_numbers)
      .def_ro("symop", &CrystalAtomRegion::symop)
      .def("size", &CrystalAtomRegion::size)
      .def("__repr__", [](const CrystalAtomRegion &region) {
        return fmt::format("<CrystalAtomRegion (n={})>", region.size());
      });

  nb::class_<SymmetryRelatedDimer>(m, "SymmetryRelatedDimer")
      .def_ro("unique_index", &SymmetryRelatedDimer::unique_index)
      .def_ro("dimer", &SymmetryRelatedDimer::dimer);

  nb::class_<CrystalDimers>(m, "CrystalDimers")
      .def_ro("radius", &CrystalDimers::radius)
      .def_ro("unique_dimers", &CrystalDimers::unique_dimers)
      .def_ro("molecule_neighbors", &CrystalDimers::molecule_neighbors);

  nb::class_<AsymmetricUnit>(m, "AsymmetricUnit")
      .def_rw("positions", &AsymmetricUnit::positions)
      .def_rw("atomic_numbers", &AsymmetricUnit::atomic_numbers)
      .def_rw("occupations", &AsymmetricUnit::occupations)
      .def_rw("charges", &AsymmetricUnit::charges)
      .def_rw("labels", &AsymmetricUnit::labels)
      .def("__len__", &AsymmetricUnit::size)
      .def("__repr__", [](const AsymmetricUnit &asym) {
        return fmt::format("<AsymmetricUnit {}>", asym.chemical_formula());
      });

  nb::class_<UnitCell>(m, "UnitCell")
      .def_prop_rw("a", &UnitCell::a, &UnitCell::set_a)
      .def_prop_rw("b", &UnitCell::b, &UnitCell::set_b)
      .def_prop_rw("c", &UnitCell::c, &UnitCell::set_c)
      .def_prop_rw("alpha", &UnitCell::alpha, &UnitCell::set_alpha)
      .def_prop_rw("beta", &UnitCell::beta, &UnitCell::set_beta)
      .def_prop_rw("gamma", &UnitCell::gamma, &UnitCell::set_gamma)
      .def("lengths", &UnitCell::lengths)
      .def("to_fractional", &UnitCell::to_fractional)
      .def("to_cartesian", &UnitCell::to_cartesian)
      .def("cell_type", &UnitCell::cell_type)
      .def("__repr__", [](const UnitCell &uc) {
        return fmt::format("<UnitCell {} ({:.5f}, {:.5f}, {:.5f})>",
                           uc.cell_type(), uc.a(), uc.b(), uc.c());
      });

  using occ::cg::CrystalGrowthResult;
  using occ::cg::DimerResult;
  using occ::cg::DimerSolventTerm;
  using occ::cg::MoleculeResult;
  using occ::interaction::LatticeConvergenceSettings;
  using occ::main::CGConfig;

  nb::class_<LatticeConvergenceSettings>(m, "LatticeConvergenceSettings")
      .def(nb::init<>())
      .def_rw("min_radius", &LatticeConvergenceSettings::min_radius)
      .def_rw("max_radius", &LatticeConvergenceSettings::max_radius)
      .def_rw("radius_increment", &LatticeConvergenceSettings::radius_increment)
      .def_rw("energy_tolerance", &LatticeConvergenceSettings::energy_tolerance)
      .def_rw("wolf_sum", &LatticeConvergenceSettings::energy_tolerance)
      .def_rw("crystal_field_polarization",
              &LatticeConvergenceSettings::crystal_field_polarization)
      .def_rw("model_name", &LatticeConvergenceSettings::model_name)
      .def_rw("crystal_filename", &LatticeConvergenceSettings::crystal_filename)
      .def_rw("output_json_filename",
              &LatticeConvergenceSettings::output_json_filename);

  nb::class_<CGConfig>(m, "CrystalGrowthConfig")
      .def(nb::init<>())
      .def_rw("lattice_settings", &CGConfig::lattice_settings)
      .def_rw("cg_radius", &CGConfig::cg_radius)
      .def_rw("solvent", &CGConfig::solvent)
      .def_rw("wavefunction_choice", &CGConfig::wavefunction_choice)
      .def_rw("num_surface_energies", &CGConfig::max_facets);

  nb::class_<DimerSolventTerm>(m, "DimerSolventTerm")
      .def_ro("ab", &DimerSolventTerm::ab)
      .def_ro("ba", &DimerSolventTerm::ba)
      .def_ro("total", &DimerSolventTerm::total);

  nb::class_<DimerResult>(m, "DimerResult")
      .def_ro("dimer", &DimerResult::dimer)
      .def_ro("unique_idx", &DimerResult::unique_idx)
      .def("total_energy", &DimerResult::total_energy)
      .def("energy_component", &DimerResult::energy_component)
      .def("energy_components",
           [](const DimerResult &d) {
             return convert_map_to_dict(d.energy_components);
           })
      .def_ro("is_nearest_neighbor", &DimerResult::is_nearest_neighbor);

  nb::class_<MoleculeResult>(m, "MoleculeResult")
      .def_ro("dimer_results", &MoleculeResult::dimer_results)
      .def_ro("total", &MoleculeResult::total)
      .def_ro("has_inversion_symmetry", &MoleculeResult::has_inversion_symmetry)
      .def("total_energy", &MoleculeResult::total_energy)
      .def("energy_components",
           [](const MoleculeResult &m) {
             return convert_map_to_dict(m.energy_components);
           })
      .def("energy_component", &MoleculeResult::energy_component);

  nb::class_<CrystalGrowthResult>(m, "CrystalGrowthResult")
      .def_ro("molecule_results", &CrystalGrowthResult::molecule_results);

  nb::class_<occ::cg::EnergyTotal>(m, "CrystalGrowthEnergyTotal")
      .def_ro("crystal", &occ::cg::EnergyTotal::crystal_energy)
      .def_ro("int", &occ::cg::EnergyTotal::interaction_energy)
      .def_ro("solution", &occ::cg::EnergyTotal::solution_term)
      .def("__repr__", [](const occ::cg::EnergyTotal &tot) {
        return fmt::format("(crys={:.6f}, int={:.6f}, sol={:.6f})",
                           tot.crystal_energy, tot.interaction_energy,
                           tot.solution_term);
      });

  m.def("calculate_crystal_growth_energies",
        [](const CGConfig &config) { return occ::main::run_cg(config); });

  m.def("setup_logging", [](int v) { occ::log::setup_logging(v); });
  m.def("set_num_threads", [](int n) { occ::parallel::set_num_threads(n); });
  m.def("set_data_directory",
        [](const std::string &s) { occ::qm::override_basis_set_directory(s); });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "0.6.11";
#endif
}
