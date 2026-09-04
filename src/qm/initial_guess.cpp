#include <ankerl/unordered_dense.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/gto/gto.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/hf.h>
#include <occ/qm/initial_guess.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/scf.h>
#include <algorithm>
#include <set>

namespace occ::qm {

namespace {

using gto::AOBasis;
using gto::OCC_MINIMAL_BASIS;

/// Per-atom offset and width in the orbital basis.
///
/// `AOBasis::first_bf` is indexed by shell rather than by atom, so an atom's
/// block starts at the first basis function of its first shell.
struct AtomBlocks {
  std::vector<int> first_bf;
  std::vector<int> nbf;
};

AtomBlocks atom_blocks(const AOBasis &basis) {
  const auto &atom_to_shell = basis.atom_to_shell();
  const auto &shell_first_bf = basis.first_bf();
  AtomBlocks blocks;
  blocks.first_bf.assign(atom_to_shell.size(), 0);
  blocks.nbf.assign(atom_to_shell.size(), 0);
  for (size_t a = 0; a < atom_to_shell.size(); a++) {
    const auto &shells = atom_to_shell[a];
    if (shells.empty())
      continue;
    blocks.first_bf[a] = shell_first_bf[shells.front()];
    for (const int s : shells)
      blocks.nbf[a] += basis[s].size();
  }
  return blocks;
}

/// Quieten the log for the duration of a scope, without ever making it
/// louder than the user asked for.
///
/// The nested atomic calculations are scaffolding, not results: their
/// iteration tables would bury the calculation actually asked for. Restores
/// on the way out however the scope is left.
class QuietLog {
public:
  explicit QuietLog(spdlog::level::level_enum floor)
      : m_previous(spdlog::default_logger()->level()) {
    if (m_previous < floor)
      log::set_log_level(floor);
  }
  ~QuietLog() { log::set_log_level(m_previous); }
  QuietLog(const QuietLog &) = delete;
  QuietLog &operator=(const QuietLog &) = delete;

private:
  spdlog::level::level_enum m_previous;
};

/// Electrons an effective core potential removes from each atom, or an empty
/// vector when the basis carries no ECP.
std::vector<int> ecp_electrons_per_atom(const AOBasis &basis) {
  if (basis.total_ecp_electrons() == 0)
    return {};
  return basis.ecp_electrons();
}

/// Tabulated atomic occupations, spread over the minimal basis.
///
/// Each subshell's electrons are smeared evenly over its functions, which is a
/// spherical average rather than a real atomic state, and leaves the result
/// diagonal. Returned normalised to half the electron count.
Mat minimal_basis_soad_density(const GuessRequest &request,
                               const AOBasis &minimal, const Mat &overlap) {
  const auto &atoms = request.basis.atoms();
  const auto frozen = ecp_electrons_per_atom(request.basis);
  const auto &shells = minimal.shells();
  const auto &shell_to_atom = minimal.shell_to_atom();
  const auto &first_bf = minimal.first_bf();

  Mat D = Mat::Zero(minimal.nbf(), minimal.nbf());

  for (size_t a = 0; a < atoms.size(); a++) {
    const int Z = atoms[a].atomic_number;
    auto occupations = guess::minimal_basis_subshell_occupations(Z);

    // An effective core potential replaces the innermost closed shells, so
    // take its electrons off the front of the filling order.
    double remaining_frozen = a < frozen.size() ? frozen[a] : 0.0;
    for (auto &subshell : occupations) {
      if (remaining_frozen <= 0.0)
        break;
      const double removed = std::min(subshell.electrons, remaining_frozen);
      subshell.electrons -= removed;
      remaining_frozen -= removed;
    }

    // The minimal basis does not list its shells in filling order, so match
    // each one to its subshell by (n, l). Within an element the shells of a
    // given angular momentum appear in increasing n and none is missing, so
    // the k-th shell with angular momentum l is principal shell k + l + 1.
    std::vector<int> seen_with_l;
    for (size_t sh = 0; sh < shells.size(); sh++) {
      if (shell_to_atom[sh] != static_cast<int>(a))
        continue;
      const int l = shells[sh].l;
      if (static_cast<size_t>(l) >= seen_with_l.size())
        seen_with_l.resize(l + 1, 0);
      const int n = seen_with_l[l]++ + l + 1;

      const auto found = std::find_if(
          occupations.begin(), occupations.end(),
          [n, l](const auto &o) { return o.n == n && o.l == l; });
      if (found == occupations.end())
        continue;

      const int size = shells[sh].size();
      const double per_function = found->electrons / size;
      for (int bf = 0; bf < size; bf++)
        D(first_bf[sh] + bf, first_bf[sh] + bf) = per_function;
      found->electrons = 0.0;
    }

    // Anything left over had no shell to go in, which means the assumption
    // above about the basis is wrong for this element -- worth saying, since
    // the total-electron check downstream cannot see a misplacement.
    for (const auto &subshell : occupations)
      if (subshell.electrons > 1e-12)
        log::warn("{} has no {}{} shell for Z = {}, so {} electrons are "
                  "missing from the SOAD guess",
                  OCC_MINIMAL_BASIS, subshell.n,
                  "spdfghi"[std::min(subshell.l, 6)], Z, subshell.electrons);
  }

  // Smear the net charge evenly rather than guessing which atom carries it.
  if (request.charge != 0) {
    const double per_function =
        static_cast<double>(request.charge) / D.rows();
    D.diagonal().array() -= per_function;
  }

  // The tabulated occupations count electrons, so divide out the diagonal of
  // the overlap to turn them into density-matrix elements. A minimal-basis
  // function that is not normalised is worth knowing about -- it means the
  // shipped basis is off, not the guess.
  for (int bf = 0; bf < D.rows(); bf++) {
    if (std::abs(overlap(bf, bf) - 1.0) > 1e-6)
      log::debug("{} function {} is not normalised: <{}|{}> = {}",
                 OCC_MINIMAL_BASIS, bf, bf, bf, overlap(bf, bf));
    D(bf, bf) /= overlap(bf, bf);
  }

  const double electrons = (D * overlap).trace();
  const double difference = electrons - request.num_electrons;
  if (std::abs(difference) > 1e-6)
    log::warn("SOAD guess holds {} electrons, {} away from the {} requested",
              electrons, difference, request.num_electrons);

  return 0.5 * D;
}

Guess build_soad(const GuessRequest &request) {
  // Built in a spherical minimal basis whatever the orbital basis is.
  // Spreading a subshell's electrons evenly over its 2l+1 spherical functions
  // is a genuine spherical average; doing the same over cartesian functions is
  // only cubically symmetric, because the six cartesian d functions span the
  // five real d functions plus a spurious s-type r^2 function. A flat diagonal
  // there both leaks density into that spurious function and gives the shell
  // an l=4 shape, which on iron came to 0.17 Hartree of dependence on how the
  // molecule happened to be oriented in the input.
  auto minimal = AOBasis::load_minimal_basis(request.basis.atoms());
  minimal.set_pure(true);

  IntegralEngine engine(minimal);
  const Mat overlap = engine.one_electron_operator(cint::Operator::overlap);
  Mat density = minimal_basis_soad_density(request, minimal, overlap);

  Guess guess;
  guess.kind = GuessKind::Soad;
  // Still block diagonal over shells after the transformation, since the
  // spherical density has no cross-shell blocks to spread -- the mixed-basis
  // Fock build uses that to skip integrals.
  guess.density_is_shell_diagonal = true;

  if (request.basis.is_pure()) {
    guess.density = std::move(density);
  } else {
    // The mixed-basis Fock build applies one shell kind to all four indices,
    // so a cartesian orbital basis needs a cartesian density.
    guess.density =
        gto::transform_density_matrix_spherical_to_cartesian(minimal, density);
    minimal.set_pure(false);
  }
  guess.density_basis = std::move(minimal);
  return guess;
}

/// One element's contribution to the atomic guess, with enough of the
/// calculation's outcome to say what happened without replaying its whole
/// iteration table.
struct AtomicDensity {
  Mat density;
  double energy{0.0};
  int iterations{0};
  bool converged{false};
};

/// Converge one neutral atom in its own slice of the orbital basis.
///
/// Hartree-Fock regardless of the molecular method: this is a starting
/// density, and it saves building an exchange-correlation functional for
/// every element. The density comes back spin-summed and normalised to half
/// the atom's electron count, or empty if the atom could not be set up.
AtomicDensity converge_atom(const core::Atom &atom, const AOBasis &basis,
                            size_t atom_index) try {
  std::vector<gto::Shell> shells;
  for (const int s : basis.atom_to_shell()[atom_index])
    shells.push_back(basis[s]);
  if (shells.empty())
    return {};

  std::vector<gto::Shell> ecp_shells;
  const auto &ecp_shell_to_atom = basis.ecp_shell_to_atom();
  for (size_t s = 0; s < ecp_shell_to_atom.size(); s++)
    if (ecp_shell_to_atom[s] == static_cast<int>(atom_index))
      ecp_shells.push_back(basis.ecp_shells()[s]);

  AOBasis atom_basis({atom}, shells, basis.name(), ecp_shells);
  atom_basis.set_pure(basis.is_pure());
  const auto &ecp_electrons = basis.ecp_electrons();
  if (atom_index < ecp_electrons.size())
    atom_basis.set_ecp_electrons({ecp_electrons[atom_index]});

  // Conventional four-centre integrals: a lone atom is small enough that
  // building a fitting basis and its metric would cost more than the SCF it
  // accelerates, and the auxiliary basis need not even cover the element.
  // `HartreeFock` installs no density fitting unless asked, so this is simply
  // a matter of not asking.
  HartreeFock atom_hf(atom_basis);
  const int electrons = atom_hf.active_electrons();

  // Most elements have an odd electron count, so the spin-resolved kind is
  // the rule here rather than the exception. An ECP removes closed shells
  // only, so the tabulated neutral-atom multiplicity still applies to the
  // valence-only atom -- gold keeps its one unpaired 6s electron. Fall back
  // on parity for any Z outside the table, and whenever the tabulated value
  // disagrees with the electron count the spin-resolved SCF cannot fill.
  int multiplicity = guess::ground_state_multiplicity(atom.atomic_number);
  if (multiplicity < 1 || ((multiplicity - 1) % 2) != (electrons % 2))
    multiplicity = (electrons % 2 == 0) ? 1 : 2;

  SCF<HartreeFock> atom_scf(atom_hf, multiplicity == 1
                                         ? SpinorbitalKind::Restricted
                                         : SpinorbitalKind::Unrestricted);
  atom_scf.set_charge_multiplicity(0, multiplicity);
  // A single atom starts well enough from the core Hamiltonian, and this is
  // also what stops the atomic guess recursing into itself.
  atom_scf.set_guess_kind(GuessKind::Core);
  atom_scf.maxiter = 50;

  {
    // Quiet only for the nested SCF itself; what came of it is reported by
    // the caller, at the level the user actually asked for.
    const QuietLog quiet(spdlog::level::err);
    try {
      atom_scf.compute_scf_energy();
    } catch (const std::exception &e) {
      // A tightly converged atom is not required. Several neutral atoms have
      // open-shell ground states a spherically averaged solution cannot
      // represent, so the commutator stalls long after the energy has settled
      // -- carbon is the common case. The density from that point is still
      // far better than none, so keep it and let the molecular SCF finish the
      // job.
      log::debug("atomic guess for Z = {} stopped early ({})",
                 atom.atomic_number, e.what());
    }
  }

  AtomicDensity result;
  result.energy = atom_scf.ctx.energy["total"];
  result.iterations = atom_scf.iter;
  result.converged = atom_scf.ctx.converged;

  const auto &mo = atom_scf.ctx.mo;
  if (mo.kind == SpinorbitalKind::Restricted) {
    result.density = mo.D;
  } else {
    // Spin blocks stack vertically, and `density_matrix_unrestricted` already
    // halves them, so their sum is on the same "half the electron count"
    // convention a restricted density uses.
    result.density = block::a(mo.D) + block::b(mo.D);
  }
  return result;
} catch (const std::exception &e) {
  // Setting the atom up can throw as readily as converging it: building the
  // one-atom basis, constructing the method, and settling the charge and
  // multiplicity all do, and all of them sit outside the inner try. The
  // header promises `build_guess` always hands back a usable starting point,
  // so an atom that cannot be set up drops out with an empty density and the
  // caller warns and carries on with the rest.
  log::debug("atomic guess for Z = {} could not be set up ({})",
             atom.atomic_number, e.what());
  return {};
}

Guess build_atomic_scf(const GuessRequest &request) {
  const auto &basis = request.basis;
  const auto &atoms = basis.atoms();
  const auto blocks = atom_blocks(basis);

  std::set<int> elements;
  for (const auto &atom : atoms)
    elements.insert(atom.atomic_number);
  log::info("Converging {} isolated atom{} in {} for the initial guess",
            elements.size(), elements.size() == 1 ? "" : "s", basis.name());

  Mat D = Mat::Zero(basis.nbf(), basis.nbf());
  // Atoms of the same element differ only in where they sit, and the density
  // is expressed in that atom's own shells, so one calculation per element is
  // enough.
  //
  // Deliberately per call rather than per process. Each element costs on the
  // order of ten milliseconds, and the driver caches monomer wavefunctions,
  // so a longer-lived cache would save a fraction of a percent of a run that
  // is measured in seconds. It would cost a mutex and a key -- (Z, basis
  // name, ECP electrons, spherical) -- that is not quite sound, since a
  // custom basis reusing a shipped name would collide, and the size check
  // below only catches that when the sizes happen to differ.
  ankerl::unordered_dense::map<int, AtomicDensity> converged;
  size_t placed = 0;

  for (size_t a = 0; a < atoms.size(); a++) {
    const int Z = atoms[a].atomic_number;
    auto it = converged.find(Z);
    if (it == converged.end()) {
      it = converged.emplace(Z, converge_atom(atoms[a], basis, a)).first;
      // The nested SCFs run silent -- their iteration tables would bury the
      // calculation actually asked for -- so this is the only trace they
      // leave. Enough to see that each element was solved, and which one
      // stalled when the molecular SCF then starts badly.
      const auto &atom = it->second;
      if (atom.density.rows() == 0)
        log::warn("  {:>2s}  could not be set up for the guess",
                  core::Element(Z).symbol());
      else if (atom.converged)
        log::info("  {:>2s}  E = {:>18.9f}  ({} iterations)",
                  core::Element(Z).symbol(), atom.energy, atom.iterations);
      else
        log::warn("  {:>2s}  E = {:>18.9f}  did not converge in {} iterations; "
                  "using the density it reached",
                  core::Element(Z).symbol(), atom.energy, atom.iterations);
    }

    const Mat &density = it->second.density;
    if (density.rows() == 0)
      continue;
    if (density.rows() != blocks.nbf[a]) {
      log::warn("atomic guess for atom {} (Z = {}) has {} functions but the "
                "molecular basis gives it {}; leaving its block empty",
                a, Z, density.rows(), blocks.nbf[a]);
      continue;
    }
    D.block(blocks.first_bf[a], blocks.first_bf[a], density.rows(),
            density.cols()) = density;
    placed++;
  }

  if (placed == 0) {
    log::warn("no atomic densities could be converged for the guess");
    return Guess{};
  }
  if (placed != atoms.size())
    log::warn("atomic guess covers {} of {} atoms; the rest start bare",
              placed, atoms.size());

  Guess guess;
  guess.kind = GuessKind::AtomicScf;
  guess.density = std::move(D);
  guess.density_basis = basis;
  return guess;
}

Guess build_sap(const GuessRequest &request) {
  // The shipped fits are all-electron atomic potentials, and on an ECP centre
  // the core Hamiltonian already carries the ECP together with a nuclear
  // attraction built from the effective charge -- which is that atom's
  // valence effective potential already. Adding an all-electron potential on
  // top counts the core twice, so `compute_sap_matrix` drops those centres,
  // which leaves them with no screening at all. Neither is right: no
  // published prescription reconciles SAP with an ECP, and the elements that
  // carry one are precisely where SAP would otherwise be most useful.
  if (request.basis.total_ecp_electrons() > 0)
    log::warn("the SAP guess has no sound treatment of effective core "
              "potentials; those centres are left unscreened. Prefer --guess "
              "atomic for this system");

  Guess guess;
  guess.kind = GuessKind::Sap;
  guess.potential =
      guess::compute_sap_matrix(request.basis.atoms(), request.basis);
  return guess;
}

} // namespace

bool minimal_basis_covers(const AOBasis &basis) {
  const auto &atoms = basis.atoms();
  const auto minimal = AOBasis::load_minimal_basis(atoms);
  std::vector<bool> has_shell(atoms.size(), false);
  for (const int a : minimal.shell_to_atom())
    if (a >= 0 && static_cast<size_t>(a) < has_shell.size())
      has_shell[a] = true;

  for (size_t a = 0; a < atoms.size(); a++) {
    if (!has_shell[a]) {
      log::debug("{} has no shells for atom {} (Z = {})", OCC_MINIMAL_BASIS, a,
                 atoms[a].atomic_number);
      return false;
    }
  }
  return true;
}

GuessKind select_guess(GuessKind requested, const AOBasis &basis) {
  if (requested != GuessKind::Auto)
    return requested;

  // SOAD is the default: it costs one mixed-basis Fock build and starts the
  // SCF close enough that DIIS takes over immediately. Its only limit is the
  // shipped minimal basis, which stops well short of the periodic table --
  // an uncovered atom would silently contribute nothing at all to the guess.
  if (minimal_basis_covers(basis))
    return GuessKind::Soad;

  // Converging the isolated atoms needs no minimal basis and carries whatever
  // ECP they have, so it reaches every element the orbital basis does. SAP
  // would be the cheaper fallback, but it has no sound ECP treatment and the
  // elements that get here are exactly the ones that tend to carry an ECP.
  log::info("{} does not have shells for every element present, so a SOAD "
            "guess would ignore some atoms entirely",
            OCC_MINIMAL_BASIS);
  return GuessKind::AtomicScf;
}

namespace {

/// Report how many electrons a guess density actually holds.
///
/// A guess that has lost or gained electrons still converges, just slowly and
/// from somewhere odd, so this is worth saying out loud rather than leaving to
/// be inferred from a bad first iteration.
void check_electron_count(const Guess &guess, const GuessRequest &request) {
  if (guess.density.size() == 0)
    return;
  IntegralEngine engine(guess.density_basis);
  const Mat overlap = engine.one_electron_operator(cint::Operator::overlap);
  const double electrons = 2.0 * (guess.density * overlap).trace();
  log::debug("{} guess holds {:.6f} electrons ({} expected)",
             guess_kind_name(guess.kind), electrons, request.num_electrons);
  if (std::abs(electrons - request.num_electrons) > 1e-6)
    log::warn("{} guess holds {:.6f} electrons, {} expected",
              guess_kind_name(guess.kind), electrons, request.num_electrons);
}

Guess build_guess_impl(GuessKind kind, const GuessRequest &request) {
  switch (kind) {
  case GuessKind::Core:
    return Guess{};
  case GuessKind::Soad:
    if (!minimal_basis_covers(request.basis)) {
      log::warn("SOAD guess requested but {} does not cover every element "
                "present; falling back to the core Hamiltonian",
                OCC_MINIMAL_BASIS);
      return Guess{};
    }
    return build_soad(request);
  case GuessKind::Sap:
    return build_sap(request);
  case GuessKind::AtomicScf:
    return build_atomic_scf(request);
  case GuessKind::Auto:
    return build_guess_impl(select_guess(kind, request.basis), request);
  }
  return Guess{};
}

} // namespace

Guess build_guess(GuessKind kind, const GuessRequest &request) {
  Guess guess = build_guess_impl(kind, request);
  check_electron_count(guess, request);
  return guess;
}

} // namespace occ::qm
