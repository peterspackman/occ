// Split RI-J and RI-J gradient timing harness.
//
//   rij_bench <wavefunction.owf.json> <aux basis> [--threads N] [--reps N]
//             [--write <prefix>] [--ref <prefix>] [--no-grad] [--no-j]
//             [--debug]
//   rij_bench --micro <max L_ab> [<max lc>]
//
// Not a test: loads a converged wavefunction (no SCF), builds the split RI-J
// engine once and times SplitRIJ::coulomb_from_density and
// IntegralEngineDF::coulomb_gradient, best of N. --write saves J and the
// gradient as raw doubles; --ref compares against a saved pair. --micro
// times the bare kernels per (L_ab, lc) instead.
#include "bench_common.h"
#include <fmt/core.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/split_ri_j.h>
#include <occ/qm/wavefunction.h>
#include <occ/ints/boys.h>
#include <string>
#include <random>
#include "../src/qm/split_ri_j_kernels.h"

namespace {

using occ::bench::seconds;

void write_raw(const std::string &fn, const double *p, size_t n) {
  std::ofstream f(fn, std::ios::binary);
  f.write(reinterpret_cast<const char *>(p), n * sizeof(double));
}

bool read_raw(const std::string &fn, double *p, size_t n) {
  std::ifstream f(fn, std::ios::binary);
  if (!f)
    return false;
  f.read(reinterpret_cast<char *>(p), n * sizeof(double));
  return static_cast<size_t>(f.gcount()) == n * sizeof(double);
}

// Kernel microbenchmark: ns per (primitive pair, aux primitive) quartet for
// each (L_ab, lc), forward and backward, single thread, synthetic geometry.
int micro(int maxla, int maxlc) {
  using namespace occ::qm::srij;
  occ::ints::Boys<> boys;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> pos(-8.0, 8.0), lexp(-1.5, 3.0);
  const int npair = 64, naux = 256;
  std::vector<PrimPair> pp(npair);
  for (auto &q : pp)
    q = {std::pow(10.0, lexp(rng)), pos(rng), pos(rng), pos(rng), 1.0};
  std::vector<double> g(naux), cp(naux, 1.0), cx(naux), cy(naux), cz(naux);
  for (int k = 0; k < naux; k++) {
    g[k] = std::pow(10.0, lexp(rng));
    cx[k] = pos(rng);
    cy[k] = pos(rng);
    cz[k] = pos(rng);
  }
  AuxView a{g.data(), cp.data(), cx.data(), cy.data(), cz.data(), naux};
  std::vector<double> X(npair * 400, 0.01), Y(400 * naux, 0.0);
  fmt::print("  ns/quartet   (la, lc): fwd bwd\n");
  for (int la = 0; la <= maxla; la++) {
    fmt::print("  L_ab={}:", la);
    for (int lc = 0; lc <= maxlc; lc++) {
      auto f = forward_kernel(la, lc);
      auto b = backward_kernel(la, lc);
      const int nh = occ::ints::nhermsum(la);
      double tf = 1e30, tb = 1e30;
      for (int r = 0; r < 3; r++) {
        tf = std::min(tf, seconds([&] {
                        f(boys.table(), pp.data(), npair, X.data(), nh, a,
                          naux, Y.data());
                      }));
        tb = std::min(tb, seconds([&] {
                        b(boys.table(), pp.data(), npair, a, naux, Y.data(),
                          X.data(), nh);
                      }));
      }
      const double nq = double(npair) * naux;
      fmt::print("  {}:{:5.1f} {:5.1f}", lc, 1e9 * tf / nq, 1e9 * tb / nq);
    }
    fmt::print("\n");
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const occ::bench::Args args(
      argc, argv,
      "<owf.json> <aux basis> [--threads N] [--reps N] [--write <prefix>] "
      "[--ref <prefix>] [--no-grad] [--no-j] [--debug]\n"
      "       rij_bench --micro <max L_ab> [<max lc>]",
      {"--micro", "--no-grad", "--no-j", "--debug"}, {"--reps", "--write", "--ref"});
  if (args.has("--micro")) {
    if (!args.valid(1, 2))
      return 1;
    return micro(std::stoi(args.positional(0)),
                 args.num_positional() > 1 ? std::stoi(args.positional(1)) : 4);
  }
  if (!args.valid(2, 2))
    return 1;
  const std::string wfn_file = args.positional(0), aux_name = args.positional(1);
  const int threads = args.threads(), reps = args.reps();
  const std::string write_prefix = args.get("--write", ""),
                    ref_prefix = args.get("--ref", "");
  const bool do_grad = !args.has("--no-grad"), do_j = !args.has("--no-j"),
             debug = args.has("--debug");
  occ::parallel::set_num_threads(threads);
  occ::log::set_log_level(debug ? spdlog::level::debug : spdlog::level::warn);

  auto wfn = occ::qm::Wavefunction::load(wfn_file);
  const auto &basis = wfn.basis;
  auto aux = occ::gto::AOBasis::load(basis.atoms(), aux_name);
  aux.set_pure(basis.is_pure());
  const auto &mo = wfn.mo;
  fmt::print("{}  nbf={} naux={} natom={} threads={}\n", wfn_file, basis.nbf(),
             aux.nbf(), basis.atoms().size(), threads);

  occ::qm::IntegralEngineDF df(basis.atoms(), basis.shells(), aux.shells());
  const auto &ao = df.ao_engine();

  occ::Mat J;
  if (do_j) {
    std::unique_ptr<occ::qm::SplitRIJ> srij;
    const double t_init = seconds([&] {
      srij = std::make_unique<occ::qm::SplitRIJ>(basis, aux, ao.shellpairs(),
                                                 ao.schwarz());
    });
    // The same with the metric factorization shared, as IntegralEngineDF
    // builds it (it has V's diagonal already).
    const occ::Vec vd = df.aux_engine()
                            .one_electron_operator(occ::qm::cint::Operator::coulomb, false)
                            .diagonal()
                            .array()
                            .sqrt();
    const double t_init_shared = seconds([&] {
      occ::qm::SplitRIJ s2(basis, aux, ao.shellpairs(), ao.schwarz(),
                           df.coulomb_metric(), vd);
    });
    fmt::print("  split RI-J  init with the metric shared {:7.3f} s\n",
               t_init_shared);
    double best = 1e30;
    for (int r = 0; r < reps; r++)
      best = std::min(best, seconds([&] { J = srij->coulomb_from_density(mo.D); }));
    const double eJ = mo.D.cwiseProduct(J).sum();
    fmt::print("  split RI-J  init {:7.3f} s   J best-of-{} {:7.3f} s   "
               "tr(DJ) = {:.12f}\n",
               t_init, reps, best, eJ);
  }

  occ::Mat3N G;
  if (do_grad) {
    double best = 1e30;
    for (int r = 0; r < reps; r++)
      best = std::min(best, seconds([&] { G = df.coulomb_gradient(mo); }));
    fmt::print("  RI-J grad   best-of-{} {:7.3f} s   |G| = {:.12f}\n", reps,
               best, G.norm());
  }

  if (!write_prefix.empty()) {
    if (do_j)
      write_raw(write_prefix + ".J.bin", J.data(), J.size());
    if (do_grad)
      write_raw(write_prefix + ".G.bin", G.data(), G.size());
  }
  if (!ref_prefix.empty()) {
    if (do_j) {
      occ::Mat Jr(J.rows(), J.cols());
      if (read_raw(ref_prefix + ".J.bin", Jr.data(), Jr.size())) {
        const double d = (J - Jr).cwiseAbs().maxCoeff();
        fmt::print("  J  max|dJ| = {:.3e}   rel {:.3e}   d tr(DJ) = {:.3e}\n",
                   d, d / Jr.cwiseAbs().maxCoeff(),
                   mo.D.cwiseProduct(J - Jr).sum());
      }
    }
    if (do_grad) {
      occ::Mat3N Gr(G.rows(), G.cols());
      if (read_raw(ref_prefix + ".G.bin", Gr.data(), Gr.size())) {
        const double d = (G - Gr).cwiseAbs().maxCoeff();
        fmt::print("  G  max|dG| = {:.3e}   rel {:.3e}\n", d,
                   d / Gr.cwiseAbs().maxCoeff());
      }
    }
  }
  return 0;
}
