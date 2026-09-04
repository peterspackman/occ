#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/gto/shell.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/hf.h>
#include <occ/qm/initial_guess.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/scf.h>

using occ::Mat;
using occ::Vec;
using occ::gto::AOBasis;
using occ::qm::build_guess;
using occ::qm::Guess;
using occ::qm::GuessKind;
using occ::qm::GuessRequest;
using occ::qm::HartreeFock;
using occ::qm::minimal_basis_covers;
using occ::qm::select_guess;
using occ::qm::SpinorbitalKind;

namespace {

std::vector<occ::core::Atom> water() {
  return {{8, 0.0, 0.0, 0.221668},
          {1, 0.0, 1.431042, -0.886659},
          {1, 0.0, -1.431042, -0.886659}};
}

std::vector<occ::core::Atom> gold_hydride() {
  return {{79, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 2.880226}};
}

AOBasis load(const std::vector<occ::core::Atom> &atoms,
             const std::string &name) {
  auto basis = AOBasis::load(atoms, name);
  basis.set_pure(false);
  return basis;
}

/// Electrons the guess density accounts for, as `2 tr(D S)` -- occ normalises
/// densities to half the electron count.
double guess_electrons(const Guess &guess) {
  occ::qm::IntegralEngine engine(guess.density_basis);
  const Mat overlap =
      engine.one_electron_operator(occ::qm::cint::Operator::overlap);
  return 2.0 * (guess.density * overlap).trace();
}

} // namespace

TEST_CASE("Ground state multiplicities follow the experimental terms",
          "[guess]") {
  using occ::qm::guess::ground_state_multiplicity;

  // The cases plain Hund's rule gets wrong: chromium and copper promote an s
  // electron to half-fill or fill the d shell, and palladium empties its s
  // shell entirely. Getting these wrong sets up the atomic guess with the
  // wrong number of unpaired electrons.
  CHECK(ground_state_multiplicity(24) == 7); // Cr, d5 s1
  CHECK(ground_state_multiplicity(29) == 2); // Cu, d10 s1
  CHECK(ground_state_multiplicity(46) == 1); // Pd, d10
  CHECK(ground_state_multiplicity(26) == 5); // Fe
  CHECK(ground_state_multiplicity(79) == 2); // Au, one unpaired 6s
  CHECK(ground_state_multiplicity(1) == 2);
  CHECK(ground_state_multiplicity(2) == 1);

  // Out of range, rather than a wrong answer.
  CHECK(ground_state_multiplicity(0) == 0);
  CHECK(ground_state_multiplicity(119) == 0);

  // Unpaired electrons and total electrons must share a parity, or the
  // spin-resolved SCF cannot fill the orbitals at all.
  for (int Z = 1; Z <= 118; Z++) {
    const int multiplicity = ground_state_multiplicity(Z);
    INFO("Z = " << Z << " multiplicity " << multiplicity);
    REQUIRE(multiplicity >= 1);
    REQUIRE((multiplicity - 1) % 2 == Z % 2);
  }
}

TEST_CASE("Guess names round-trip", "[guess]") {
  using occ::qm::guess_kind_from_string;
  using occ::qm::guess_kind_name;
  for (const auto kind : {GuessKind::Auto, GuessKind::Core, GuessKind::Soad,
                          GuessKind::Sap, GuessKind::AtomicScf}) {
    INFO("kind " << guess_kind_name(kind));
    REQUIRE(guess_kind_from_string(guess_kind_name(kind)) == kind);
  }
  CHECK(guess_kind_from_string("hcore") == GuessKind::Core);
  CHECK(guess_kind_from_string("sad") == GuessKind::Soad);
  CHECK(!guess_kind_from_string("nonsense").has_value());
}

TEST_CASE("Auto picks SOAD only where the minimal basis reaches", "[guess]") {
  const auto light = load(water(), "def2-svp");
  REQUIRE(minimal_basis_covers(light));
  CHECK(select_guess(GuessKind::Auto, light) == GuessKind::Soad);

  // The shipped minimal basis stops well short of gold, so SOAD would ignore
  // the heaviest atom in the system entirely.
  const auto heavy = load(gold_hydride(), "def2-svp");
  REQUIRE(!minimal_basis_covers(heavy));
  CHECK(select_guess(GuessKind::Auto, heavy) == GuessKind::AtomicScf);

  // An explicit request is honoured either way.
  CHECK(select_guess(GuessKind::Core, light) == GuessKind::Core);
  CHECK(select_guess(GuessKind::Sap, heavy) == GuessKind::Sap);
}

TEST_CASE("Guess densities hold the right number of electrons", "[guess]") {
  SECTION("SOAD, in the minimal basis") {
    const auto basis = load(water(), "def2-svp");
    const auto guess = build_guess(GuessKind::Soad, {basis, 0, 10});
    REQUIRE(guess.kind == GuessKind::Soad);
    REQUIRE(guess.density_is_shell_diagonal);
    CHECK(guess_electrons(guess) == Catch::Approx(10.0).margin(1e-8));
  }

  SECTION("SOAD carries a net charge") {
    const auto basis = load(water(), "def2-svp");
    const auto guess = build_guess(GuessKind::Soad, {basis, 1, 9});
    CHECK(guess_electrons(guess) == Catch::Approx(9.0).margin(1e-8));
  }

  SECTION("Atomic SCF, in the orbital basis") {
    const auto basis = load(water(), "def2-svp");
    const auto guess = build_guess(GuessKind::AtomicScf, {basis, 0, 10});
    REQUIRE(guess.kind == GuessKind::AtomicScf);
    REQUIRE(!guess.density_is_shell_diagonal);
    REQUIRE(guess.density.rows() == static_cast<Eigen::Index>(basis.nbf()));
    CHECK(guess_electrons(guess) == Catch::Approx(10.0).margin(1e-8));
  }

  SECTION("Atomic SCF counts only the electrons an ECP leaves behind") {
    const auto basis = load(gold_hydride(), "def2-svp");
    // def2 replaces gold's inner 60 electrons, leaving 19 valence plus the
    // hydrogen.
    REQUIRE(basis.total_ecp_electrons() == 60);
    const auto guess = build_guess(GuessKind::AtomicScf, {basis, 0, 20});
    CHECK(guess_electrons(guess) == Catch::Approx(20.0).margin(1e-8));
  }
}

TEST_CASE("Atomic guess places each atom in its own block", "[guess]") {
  // `AOBasis::first_bf` is indexed by shell rather than by atom, so an atom
  // block placed at the wrong offset is easy to write and hard to see: the
  // density stays the right size and the SCF still converges, just from a
  // worse starting point. Check every atom's diagonal block carries its own
  // electrons.
  const std::vector<occ::core::Atom> formaldehyde{{8, 0.0, 0.0, 1.207471},
                                                  {6, 0.0, 0.0, -1.058642},
                                                  {1, 0.0, 1.766646, -2.145607},
                                                  {1, 0.0, -1.766646, -2.145607}};
  const auto basis = load(formaldehyde, "def2-svp");
  const auto guess = build_guess(GuessKind::AtomicScf, {basis, 0, 16});
  REQUIRE(guess.density.rows() == static_cast<Eigen::Index>(basis.nbf()));

  occ::qm::IntegralEngine engine(basis);
  const Mat overlap =
      engine.one_electron_operator(occ::qm::cint::Operator::overlap);

  const auto &atom_to_shell = basis.atom_to_shell();
  const auto &shell_first_bf = basis.first_bf();
  const std::vector<double> expected{8.0, 6.0, 1.0, 1.0};

  for (size_t a = 0; a < formaldehyde.size(); a++) {
    const int first = shell_first_bf[atom_to_shell[a].front()];
    int width = 0;
    for (const int s : atom_to_shell[a])
      width += basis[s].size();

    // The atomic densities are block diagonal, so an atom's own block against
    // its own overlap block recovers exactly that atom's electrons.
    const double electrons =
        2.0 * (guess.density.block(first, first, width, width) *
               overlap.block(first, first, width, width))
                  .trace();
    INFO("atom " << a << " (Z = " << formaldehyde[a].atomic_number
                 << ") block at " << first << " width " << width);
    CHECK(electrons == Catch::Approx(expected[a]).margin(1e-6));
  }
}

TEST_CASE("Every guess starts the SCF closer than the core Hamiltonian",
          "[guess]") {
  // A guess is only worth its cost if the first iteration lands nearer the
  // answer than doing nothing. Recorded as a band rather than a number: this
  // guards against a guess silently breaking, not against it changing.
  const auto basis = load(water(), "def2-svp");

  auto first_iteration_error = [&](GuessKind kind) {
    HartreeFock hf(basis);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::Restricted);
    scf.set_guess_kind(kind);
    scf.maxiter = 1;
    // Stopping after one iteration is the point of the test, so its "did not
    // converge" complaint is noise that would look like a real failure.
    const auto previous = spdlog::default_logger()->level();
    occ::log::set_log_level(spdlog::level::critical);
    try {
      scf.compute_scf_energy();
    } catch (const std::exception &) {
      // One iteration never converges; the energy it reached is the point.
    }
    occ::log::set_log_level(previous);
    return std::abs(scf.ctx.energy["total"] - (-75.9621870));
  };

  const double core = first_iteration_error(GuessKind::Core);
  const double soad = first_iteration_error(GuessKind::Soad);
  const double atomic = first_iteration_error(GuessKind::AtomicScf);
  const double sap = first_iteration_error(GuessKind::Sap);

  INFO("core " << core << " soad " << soad << " atomic " << atomic << " sap "
               << sap);
  CHECK(core > 1.0);
  CHECK(soad < 0.1);
  CHECK(atomic < 0.5);
  CHECK(sap < 1.0);
}

TEST_CASE("Every guess leaves behind a usable set of orbitals", "[guess]") {
  // Density-fitted exchange is built from the occupied coefficients, not the
  // density, so a guess that supplies only a density crashes the first Fock
  // build. That is exactly what the minimal-basis SOAD shortcut used to do.
  const auto basis = load(water(), "sto-3g");
  for (const auto kind : {GuessKind::Core, GuessKind::Soad, GuessKind::Sap,
                          GuessKind::AtomicScf}) {
    INFO("guess " << occ::qm::guess_kind_name(kind));
    HartreeFock hf(basis);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::Restricted);
    scf.set_guess_kind(kind);
    scf.compute_initial_guess();
    REQUIRE(scf.ctx.mo.C.rows() == static_cast<Eigen::Index>(basis.nbf()));
    REQUIRE(scf.ctx.mo.Cocc.cols() == scf.ctx.n_occ);
    REQUIRE(scf.ctx.mo.Cocc.cols() > 0);
  }
}

TEST_CASE("The guess does not change where the SCF converges", "[guess]") {
  // Different starting points, same answer: the guess may only affect how
  // many iterations it takes.
  const auto basis = load(water(), "sto-3g");
  double reference = 0.0;
  for (const auto kind :
       {GuessKind::Core, GuessKind::Soad, GuessKind::AtomicScf}) {
    HartreeFock hf(basis);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::Restricted);
    scf.set_guess_kind(kind);
    const double energy = scf.compute_scf_energy();
    INFO("guess " << occ::qm::guess_kind_name(kind) << " gave " << energy);
    REQUIRE(scf.ctx.converged);
    if (reference == 0.0)
      reference = energy;
    else
      CHECK(energy == Catch::Approx(reference).margin(1e-8));
  }
}

TEST_CASE("SOAD fills the shells the basis actually has", "[guess]") {
  // The occupation vector is built in filling order -- 4s before 3d -- and
  // then written onto basis functions by position. That only works if the
  // minimal basis lists its shells in the same order, and for the transition
  // metals it does not: STO-3G gives iron 1s 2s 2p 3s 3p 4s 4p 3d, with the d
  // shell last. Assigning by position therefore pours the 3d electrons into
  // the empty 4p shell and leaves half the 3d shell bare.
  //
  // Checked per shell as a Mulliken population, which sums to the electron
  // count however the density is arranged, so it sees a permutation that a
  // total-electron check cannot.
  struct Case {
    int Z;
    std::vector<double> electrons_per_shell; // in basis order
  };
  const std::vector<Case> cases{
      {6, {2, 2, 2}},                    // C:  1s 2s 2p
      {8, {2, 2, 4}},                    // O:  1s 2s 2p
      {14, {2, 2, 6, 2, 2}},             // Si: 1s 2s 2p 3s 3p
      {26, {2, 2, 6, 2, 6, 2, 0, 6}},    // Fe: 1s 2s 2p 3s 3p 4s 4p 3d
      {29, {2, 2, 6, 2, 6, 2, 0, 9}},    // Cu: aufbau 4s2 3d9
      {19, {2, 2, 6, 2, 6, 1, 0}},       // K:  no 3d shell in the minimal basis
      {30, {2, 2, 6, 2, 6, 2, 0, 10}},   // Zn
  };

  for (const auto &c : cases) {
    const std::vector<occ::core::Atom> atom{{c.Z, 0.0, 0.0, 0.0}};
    const auto basis = load(atom, "def2-svp");
    int electrons = 0;
    for (const double n : c.electrons_per_shell)
      electrons += static_cast<int>(n);
    const auto guess = build_guess(GuessKind::Soad, {basis, 0, electrons});

    occ::qm::IntegralEngine engine(guess.density_basis);
    const Mat overlap =
        engine.one_electron_operator(occ::qm::cint::Operator::overlap);
    const Vec population = 2.0 * (guess.density * overlap).diagonal();

    const auto &shells = guess.density_basis.shells();
    const auto &first_bf = guess.density_basis.first_bf();
    INFO("Z = " << c.Z << ", " << shells.size() << " minimal-basis shells");
    REQUIRE(shells.size() == c.electrons_per_shell.size());

    for (size_t sh = 0; sh < shells.size(); sh++) {
      const double got =
          population.segment(first_bf[sh], shells[sh].size()).sum();
      INFO("shell " << sh << " (l = " << shells[sh].l << ") holds " << got
                    << ", expected " << c.electrons_per_shell[sh]);
      CHECK(got == Catch::Approx(c.electrons_per_shell[sh]).margin(1e-6));
    }
  }
}

TEST_CASE("The SOAD guess does not depend on how a molecule is oriented",
          "[guess]") {
  // A superposition of atomic densities is spherically symmetric per atom, so
  // the guess energy can only depend on the geometry, never on the frame it
  // is written in. That holds automatically for s and p shells, whose
  // functions are mutually orthogonal, and fails for cartesian d and f: the
  // six cartesian d functions span the five real d functions plus a spurious
  // s-type r^2 function, so spreading a subshell evenly over them gives a
  // cubically symmetric density rather than a spherical one.
  //
  // Iron is the cheapest element that exercises it. On a real complex the
  // defect was worth 0.17 Hartree of pure orientation dependence.
  const std::vector<occ::core::Atom> upright{{26, 0.0, 0.0, 0.0},
                                             {1, 0.0, 0.0, 3.0},
                                             {1, 2.6, 0.0, -1.5}};

  // An arbitrary rotation, deliberately not a multiple of a right angle --
  // the cubic group is exactly what a flat cartesian diagonal is invariant
  // under, so a 90 degree rotation would hide the problem.
  const Eigen::AngleAxisd rotation(0.37,
                                   occ::Vec3(1.0, 2.0, 3.0).normalized());
  std::vector<occ::core::Atom> rotated = upright;
  for (auto &atom : rotated) {
    const occ::Vec3 position =
        rotation * occ::Vec3(atom.x, atom.y, atom.z);
    atom.x = position.x();
    atom.y = position.y();
    atom.z = position.z();
  }

  auto guess_energy = [](const std::vector<occ::core::Atom> &atoms) {
    HartreeFock hf(load(atoms, "def2-svp"));
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::Restricted);
    scf.set_guess_kind(GuessKind::Soad);
    scf.compute_initial_guess();
    scf.update_scf_energy(false);
    return scf.ctx.energy["total"];
  };

  const double upright_energy = guess_energy(upright);
  const double rotated_energy = guess_energy(rotated);
  INFO("upright " << upright_energy << " rotated " << rotated_energy
                  << " difference " << (rotated_energy - upright_energy));
  CHECK(rotated_energy == Catch::Approx(upright_energy).margin(1e-8));
}
