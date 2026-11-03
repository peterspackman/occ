// Per-term timing of the SCF nuclear gradient (or, with --hessian, of the
// analytic Hessian).
//
//   gradient_bench <xyz> <method> <basis> [--threads N] [--cart] [--df]
//       [--df-basis NAME] [--cosx] [--cosx-grid 1|2|3] [--hessian]
//       [--no-grid-response] [--no-reference] [--print-gradient]
//
// Not a test: a fixed harness so gradient changes can be compared term by
// term against the same molecule, basis, and thread count. Prints one line per
// term and the final gradient norm (so a speedup that changes the answer is
// visible immediately). --no-reference skips the derivative-Fock reference
// for the two-electron term, which stores 3 x natom Fock matrices per thread.
#include "bench_common.h"
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/dft/dft.h>
#include <occ/io/xyz.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hessians.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <string>

namespace {

using occ::bench::seconds;

// Per-term timing of the analytic Hessian.
template <typename Proc>
int run_hessian(Proc &proc, const std::string &label) {
  using occ::Mat;
  occ::qm::SCF<Proc> scf(proc);
  scf.convergence_settings.energy_threshold = 1e-10;
  scf.convergence_settings.commutator_threshold = 1e-8;
  double e = 0.0;
  double t_scf = seconds([&] { e = scf.compute_scf_energy(); });
  auto wfn = scf.wavefunction();
  const auto &mo = wfn.mo;
  const auto natom = proc.atoms().size();
  fmt::print("{}  natom={} nbf={}  E={:.10f}\n", label, natom,
             proc.aobasis().nbf(), e);
  fmt::print("  {:<26s} {:9.3f} s\n", "scf", t_scf);
  const Mat schwarz = proc.compute_schwarz_ints();
  const Eigen::Index nocc = mo.n_alpha;
  const Mat Co = mo.C.leftCols(nocc);
  const Mat P = 2.0 * Co * Co.transpose();
  const Mat Q = 2.0 * Co * mo.energies.head(nocc).asDiagonal() * Co.transpose();
  // Response timings first: they apply even where the full Hessian is
  // refused (e.g. density-fitted exchange).
  fmt::print("  {:<26s} {:9.3f} s\n", "one response Fock (1 dens)",
             seconds([&] {
               proc.compute_response_fock(mo, {mo.D}, schwarz);
             }));
  {
    // A CPHF iteration's worth of response densities (3N), shaped like CPHF
    // densities, C_v U C_o^T + transpose (density-fitted exchange needs that
    // form); for HF also split into the Coulomb and exchange parts.
    const Mat Cv = mo.C.rightCols(mo.C.cols() - nocc);
    std::vector<Mat> Cx, D1;
    for (size_t k = 0; k < 3 * natom; k++) {
      Cx.push_back(1e-2 * Cv * Mat::Random(Cv.cols(), nocc));
      const Mat T = Cx.back() * Co.transpose();
      D1.push_back(T + T.transpose());
    }
    fmt::print("  {:<26s} {:9.3f} s\n",
               fmt::format("response Fock ({} dens)", D1.size()),
               seconds([&] { proc.compute_response_fock(mo, D1, schwarz); }));
    fmt::print("  {:<26s} {:9.3f} s\n",
               "  as G C_o (CPHF)", seconds([&] {
                 proc.compute_response_fock_occ(mo, Cx, schwarz);
               }));
    fmt::print("  {:<26s} {:9.3f} s\n", "  approximate", seconds([&] {
                 proc.compute_response_fock_occ_approx(mo, Cx, schwarz);
               }));
    if constexpr (std::is_same_v<Proc, occ::qm::HartreeFock>) {
      fmt::print("  {:<26s} {:9.3f} s\n", "  of which J", seconds([&] {
                   proc.response_occ(mo, Cx, 1.0, {}, schwarz);
                 }));
      fmt::print("  {:<26s} {:9.3f} s\n", "  of which K", seconds([&] {
                   proc.response_occ(mo, Cx, 0.0, {{1.0, 0.0}}, schwarz);
                 }));
    }
  }
  if (const auto why = proc.analytic_hessian_unsupported(mo.kind);
      !why.empty()) {
    fmt::print("  analytic Hessian not supported: {}\n", why);
    return 0;
  }
  fmt::print("  {:<26s} {:9.3f} s\n", "one-electron skeleton",
             seconds([&] { proc.compute_one_electron_hessian(P, Q); }));
  fmt::print("  {:<26s} {:9.3f} s\n", "two-electron skeleton",
             seconds([&] { proc.two_electron_hessian(mo, schwarz); }));
  std::vector<Mat> S1, H1;
  fmt::print("  {:<26s} {:9.3f} s\n", "S1/h1 matrices", seconds([&] {
               proc.compute_one_electron_derivative_matrices(S1, H1);
             }));
  // COSX Fock derivatives are only formed contracted with C_o, in the same
  // sweep as the skeleton Hessian (so the F C_o line below reuses it).
  if (!proc.using_cosx())
    fmt::print("  {:<26s} {:9.3f} s\n", "skeleton Fock derivs", seconds([&] {
                 proc.two_electron_derivative_fock(mo, schwarz);
               }));
  fmt::print("  {:<26s} {:9.3f} s\n", "  as F C_o (Hessian)", seconds([&] {
               proc.two_electron_derivative_fock_occ(mo, schwarz);
             }));
  if constexpr (std::is_same_v<Proc, occ::dft::DFT>) {
    fmt::print("  {:<26s} {:9.3f} s\n", "  of which XC (full)",
               seconds([&] { proc.compute_xc_derivative_fock(mo); }));
    fmt::print("  {:<26s} {:9.3f} s\n", "  of which XC (F C_o)",
               seconds([&] { proc.compute_xc_derivative_fock_occ(mo); }));
    fmt::print("  {:<26s} {:9.3f} s\n", "XC skeleton Hessian",
               seconds([&] { proc.compute_xc_hessian(mo); }));
  }
  // On a fresh copy of the procedure, so nothing computed above is reused,
  // with the evaluator's own breakdown.
  occ::Mat H;
  Proc fresh = proc.with_new_basis(proc.aobasis());
  occ::log::set_log_level(spdlog::level::info);
  const double t_total = seconds([&] {
    occ::qm::HessianEvaluator<Proc> ev(fresh);
    ev.set_method(occ::qm::HessianEvaluator<Proc>::Method::Analytical);
    H = ev(wfn);
  });
  occ::log::set_log_level(spdlog::level::warn);
  fmt::print("  {:<26s} {:9.3f} s\n", "analytic Hessian (total)", t_total);
  return 0;
}

template <typename Proc>
int run(Proc &proc, const std::string &label, bool reference,
        bool print_gradient) {
  occ::qm::SCF<Proc> scf(proc);
  double e = 0.0;
  double t_scf = seconds([&] { e = scf.compute_scf_energy(); });
  auto mo = scf.molecular_orbitals();
  const auto natom = proc.atoms().size();

  fmt::print("{}  natom={} nbf={} nsh={}  E={:.10f}\n", label, natom,
             proc.aobasis().nbf(), proc.aobasis().size(), e);
  fmt::print("  {:<22s} {:9.3f} s\n", "scf", t_scf);

  occ::Mat schwarz;
  fmt::print("  {:<22s} {:9.3f} s\n", "schwarz",
             seconds([&] { schwarz = proc.compute_schwarz_ints(); }));
  fmt::print("  {:<22s} {:9.3f} s\n", "overlap grad",
             seconds([&] { proc.compute_overlap_gradient(); }));
  fmt::print("  {:<22s} {:9.3f} s\n", "kinetic grad",
             seconds([&] { proc.compute_kinetic_gradient(); }));
  fmt::print("  {:<22s} {:9.3f} s\n", "nuclear grad",
             seconds([&] { proc.compute_nuclear_attraction_gradient(); }));
  fmt::print("  {:<22s} {:9.3f} s\n", "rinv (all atoms)", seconds([&] {
               for (size_t a = 0; a < natom; a++)
                 proc.compute_rinv_gradient_for_atom(a);
             }));
  // Reference cost: one full (non-incremental) Fock build at the converged
  // density, the unit the two-electron gradient is best compared against.
  fmt::print("  {:<22s} {:9.3f} s\n", "full Fock build",
             seconds([&] { proc.compute_fock(mo, schwarz); }));
  // Reference for the two-electron term: the derivative Fock matrices,
  // contracted with the density.
  occ::Mat3N ref2e = occ::Mat3N::Zero(3, natom);
  if (reference) {
    occ::MatTriple f;
    fmt::print("  {:<22s} {:9.3f} s\n", "fock grad (J/K) ref",
               seconds([&] { f = proc.compute_fock_gradient(mo, schwarz); }));
    const auto &basis = proc.aobasis();
    for (size_t atom = 0; atom < natom; atom++)
      for (int sh : basis.atom_to_shell()[atom])
        for (int bf = basis.first_bf()[sh];
             bf < basis.first_bf()[sh] + int(basis[sh].size()); bf++)
          for (int x = 0; x < 3; x++)
            ref2e(x, atom) +=
                occ::qm::impl::accumulate2(mo.kind, bf, f[x], mo.D);
  }
  occ::Mat3N g2e;
  const double t2e =
      seconds([&] { g2e = proc.two_electron_gradient(mo, schwarz); });
  fmt::print("  {:<22s} {:9.3f} s\n", "two_electron_gradient", t2e);
  if constexpr (std::is_same_v<Proc, occ::qm::HartreeFock>) {
    if (proc.using_cosx()) {
      const double tk =
          seconds([&] { proc.compute_exchange_gradient(mo, 1.0, 0.0); });
      fmt::print("  {:<22s} {:9.3f} s\n", "  COSX K", tk);
      fmt::print("  {:<22s} {:9.3f} s\n", "  J (difference)", t2e - tk);
      fmt::print("  {:<22s} {:9.3f} s\n", "  J (direct)", seconds([&] {
                   proc.compute_coulomb_gradient(mo, schwarz);
                 }));
    }
  }
  if (reference)
    fmt::print("  {:<22s} {:9.2e}  (|ref| = {:.6f}, sum_atoms 2e = {:.1e})\n",
               "max |2e - ref|", (g2e - ref2e).cwiseAbs().maxCoeff(),
               ref2e.norm(), g2e.rowwise().sum().norm());
  fmt::print("  {:<22s} {:9.3f} s\n", "additional (XC)",
             seconds([&] { proc.additional_atomic_gradients(mo); }));

  occ::Mat3N g;
  double t_total = seconds([&] {
    occ::qm::GradientEvaluator<Proc> eval(proc);
    g = eval(mo);
  });
  fmt::print("  {:<22s} {:9.3f} s\n", "GradientEvaluator", t_total);
  fmt::print("  |g| = {:.12f}  max|g_i| = {:.12f}  sum_atoms = {:.2e}\n",
             g.norm(), g.cwiseAbs().maxCoeff(),
             g.rowwise().sum().norm());
  if (print_gradient)
    for (Eigen::Index a = 0; a < g.cols(); a++)
      fmt::print("  grad {:3d} {:16.10f} {:16.10f} {:16.10f}\n", a, g(0, a),
                 g(1, a), g(2, a));
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const occ::bench::Args args(
      argc, argv,
      "<xyz> <method> <basis> [--threads N] [--cart] [--df] [--df-basis NAME] "
      "[--cosx] [--cosx-grid 1|2|3] [--hessian] [--no-grid-response] "
      "[--no-reference] [--print-gradient]",
      {"--cart", "--df", "--cosx", "--hessian", "--no-grid-response",
       "--no-reference", "--print-gradient"},
      {"--df-basis", "--cosx-grid"});
  if (!args.valid(3, 3))
    return 1;
  const std::string xyz = args.positional(0), method = args.positional(1),
                    basis_name = args.positional(2);
  const int threads = args.threads();
  const bool df = args.has("--df"), cosx = args.has("--cosx"),
             hessian = args.has("--hessian"),
             reference = !args.has("--no-reference"),
             print_gradient = args.has("--print-gradient");
  const auto cosx_grid = static_cast<occ::numint::COSXGridLevel>(
      args.get_int("--cosx-grid", 1) - 1);
  const std::string df_basis = args.get("--df-basis", "def2-universal-jkfit");
  occ::parallel::set_num_threads(threads);
  occ::log::set_log_level(spdlog::level::warn);

  auto mol = occ::io::molecule_from_xyz_file(xyz);
  auto basis = occ::gto::AOBasis::load(mol.atoms(), basis_name);
  basis.set_pure(!args.has("--cart"));
  const std::string label =
      fmt::format("{}/{} {} threads={}{}", method, basis_name, xyz, threads,
                  df ? " df" : "");

  if (method == "hf" || method == "rhf") {
    occ::qm::HartreeFock hf(basis);
    if (df)
      hf.set_density_fitting_basis(df_basis);
    if (cosx)
      hf.set_cosx_exchange(cosx_grid);
    if (hessian)
      return run_hessian(hf, label);
    return run(hf, label, reference, print_gradient);
  }
  occ::dft::DFT dft(method, basis);
  dft.set_grid_response(!args.has("--no-grid-response"));
  if (cosx)
    dft.set_cosx_exchange(cosx_grid);
  if (df)
    dft.set_density_fitting_basis(df_basis);
  if (hessian)
    return run_hessian(dft, label);
  return run(dft, label, reference, print_gradient);
}
