// Timing of the COSX exchange gradient alone, at a converged wavefunction,
// on the GRIDX 3 grid the SCF finishes on, without running an SCF.
//
//   cosx_grad_bench <wfn.owf.json> [--threads N] [--reps N] [--kf X]
//                   [--threshold T] [--k] [--hessian] [--save F] [--ref F]
//
// Not a test: a fixed harness for performance work on the COSX gradient
// kernels. --kf sets the exchange fraction the screening is scaled by (as
// HartreeFock::compute_exchange_gradient does; 0.25 for PBE0). --save writes
// the gradient (or with --hessian the exchange Hessian) to F; --ref compares
// against such a file. --k times an SCF exchange build instead (and prints
// tr(D K)); --hessian the exchange second derivatives (skeleton Hessian and
// Fock derivatives, one sweep).
#include "bench_common.h"
#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/numint/grid_settings.h>
#include <occ/qm/seminumerical_exchange.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace {

using occ::bench::median;
using occ::bench::minimum;
using occ::bench::seconds;

void print_timing(const char *label, const std::vector<double> &t) {
  fmt::print("  {:<22s} {:9.3f} s   (min {:.3f}, first {:.3f})\n", label,
             median(t), minimum(t), t.front());
}

// Writes m to `save` and compares it against `ref`, raw doubles, if given.
void save_and_compare(const occ::Mat &m, const std::string &save,
                      const std::string &ref) {
  if (!save.empty()) {
    std::ofstream os(save, std::ios::binary);
    os.write(reinterpret_cast<const char *>(m.data()),
             sizeof(double) * m.size());
  }
  if (!ref.empty()) {
    std::ifstream is(ref, std::ios::binary);
    occ::Mat r(m.rows(), m.cols());
    is.read(reinterpret_cast<char *>(r.data()), sizeof(double) * r.size());
    fmt::print("  vs ref: max |diff| = {:.3e}\n", (m - r).cwiseAbs().maxCoeff());
  }
}

} // namespace

int main(int argc, char *argv[]) {
  const occ::bench::Args args(
      argc, argv,
      "<wfn.owf.json> [--threads N] [--reps N] [--kf X] [--threshold T] [--k] "
      "[--hessian] [--save F] [--ref F]",
      {"--k", "--hessian"}, {"--reps", "--kf", "--threshold", "--save", "--ref"});
  if (!args.valid(1, 1))
    return 1;
  const std::string wfn_file = args.positional(0);
  const int threads = args.threads(), reps = args.reps();
  const double kf = std::clamp(std::abs(args.get_double("--kf", 1.0)), 1e-3, 1.0);
  const double threshold = args.get_double("--threshold", 1e-10 / kf);
  const std::string save_file = args.get("--save", ""),
                    ref_file = args.get("--ref", "");
  occ::log::set_log_level(spdlog::level::warn);
  occ::parallel::set_num_threads(threads);

  auto wfn = occ::qm::Wavefunction::load(wfn_file);
  occ::qm::cosx::SemiNumericalExchange cosx(
      wfn.basis,
      occ::numint::GridSettings::for_cosx(occ::numint::COSXGridLevel::Grid3));
  cosx.set_use_esp(true);
  cosx.set_use_spatial_hierarchy(true);
  fmt::print("{}  natom={} nbf={} threads={} threshold={:.1e}\n", wfn_file,
             wfn.atoms.size(), wfn.basis.nbf(), threads, threshold);

  if (args.has("--k")) {
    // one SCF exchange build at the settings HartreeFock uses for a hybrid
    // with exchange fraction kf: f_threshold / kf
    auto settings = cosx.settings();
    settings.f_threshold /= kf;
    cosx.set_settings(settings);
    occ::Mat K;
    std::vector<double> t;
    for (int r = 0; r < reps; r++)
      t.push_back(seconds([&] { K = cosx.compute_K(wfn.mo); }));
    print_timing("K build", t);
    fmt::print("  E_K = tr(D K) = {:.12f}\n", wfn.mo.D.cwiseProduct(K).sum());
    return 0;
  }

  if (args.has("--hessian")) {
    occ::qm::cosx::SemiNumericalExchange::SecondDerivatives d;
    std::vector<double> t;
    for (int r = 0; r < reps; r++)
      t.push_back(seconds([&] {
        d = cosx.compute_K_second_derivatives(wfn.mo, 0.0, threshold);
      }));
    print_timing("K second derivatives", t);
    fmt::print("  |H_K| = {:.12f}\n", d.hessian.norm());
    save_and_compare(d.hessian, save_file, ref_file);
    return 0;
  }

  occ::Mat3N g;
  std::vector<double> t;
  for (int r = 0; r < reps; r++)
    t.push_back(
        seconds([&] { g = cosx.compute_K_gradient(wfn.mo, 0.0, threshold); }));
  print_timing("K gradient", t);
  fmt::print("  |g_K| = {:.12f}\n", g.norm());
  save_and_compare(g, save_file, ref_file);
  return 0;
}
