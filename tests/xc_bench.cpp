// Timing of the DFT exchange-correlation build alone: the XC potential matrix
// (one SCF iteration's worth) and the XC nuclear gradient, at a converged
// wavefunction, without running an SCF.
//
//   xc_bench <wfn.owf.json> <method> [--threads N] [--reps N] [--save F]
//            [--ref F] [--no-vxc] [--no-grad] [--split]
//
// Not a test: a fixed harness for performance work on the XC code. --save
// writes Vxc, E_xc and the XC gradient to F; --ref compares against such a
// file, so a speedup that changes the answer is visible immediately. --split
// also times the gradient without the grid response.
#include "bench_common.h"
#include <fmt/core.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/dft/dft.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace {

using occ::bench::median;
using occ::bench::minimum;
using occ::bench::seconds;

void write_mat(std::ofstream &os, const occ::Mat &m) {
  int64_t r = m.rows(), c = m.cols();
  os.write(reinterpret_cast<const char *>(&r), sizeof r);
  os.write(reinterpret_cast<const char *>(&c), sizeof c);
  os.write(reinterpret_cast<const char *>(m.data()), sizeof(double) * r * c);
}

occ::Mat read_mat(std::ifstream &is) {
  int64_t r = 0, c = 0;
  is.read(reinterpret_cast<char *>(&r), sizeof r);
  is.read(reinterpret_cast<char *>(&c), sizeof c);
  occ::Mat m(r, c);
  is.read(reinterpret_cast<char *>(m.data()), sizeof(double) * r * c);
  return m;
}

} // namespace

int main(int argc, char *argv[]) {
  const occ::bench::Args args(
      argc, argv,
      "<wfn.owf.json> <method> [--threads N] [--reps N] [--save F] [--ref F] "
      "[--no-vxc] [--no-grad] [--split]",
      {"--no-vxc", "--no-grad", "--split"}, {"--reps", "--save", "--ref"});
  if (!args.valid(2, 2))
    return 1;
  const std::string wfn_file = args.positional(0), method = args.positional(1);
  const int threads = args.threads(), reps = args.reps();
  const std::string save_file = args.get("--save", ""),
                    ref_file = args.get("--ref", "");
  const bool skip_grad = args.has("--no-grad"), skip_vxc = args.has("--no-vxc"),
             split = args.has("--split");
  occ::log::set_log_level(spdlog::level::warn);
  occ::parallel::set_num_threads(threads);

  auto wfn = occ::qm::Wavefunction::load(wfn_file);
  const auto &mo = wfn.mo;
  occ::dft::DFT dft(method, wfn.basis, occ::numint::GridSettings{});
  dft.set_xc_screening_threshold(1e-10);

  fmt::print("{}  {}  natom={} nbf={} threads={}\n", wfn_file, method,
             wfn.atoms.size(), wfn.basis.nbf(), threads);

  occ::Mat V;
  double t_first = seconds([&] { V = dft.compute_vxc(mo); });
  const double exc = dft.exchange_correlation_energy();
  fmt::print("  {:<22s} {:9.3f} s   (includes grid build)\n", "first vxc",
             t_first);

  std::vector<double> tv, tg;
  if (!skip_vxc)
    for (int r = 0; r < reps; r++)
      tv.push_back(seconds([&] { V = dft.compute_vxc(mo); }));
  occ::Mat3N g;
  if (!skip_grad)
    for (int r = 0; r < reps; r++)
      tg.push_back(seconds([&] { g = dft.compute_xc_gradient(mo); }));

  // The same without the grid response: the difference is the partition
  // weight derivative (the orbital term is computed either way).
  std::vector<double> tg0;
  if (!skip_grad && split) {
    dft.set_grid_response(false);
    for (int r = 0; r < reps; r++)
      tg0.push_back(seconds([&] { dft.compute_xc_gradient(mo); }));
    dft.set_grid_response(true);
  }

  if (!tg0.empty())
    fmt::print("  {:<22s} {:9.3f} s\n", "xc grad (no grid resp)",
               median(tg0));
  if (!tv.empty())
    fmt::print("  {:<22s} {:9.3f} s   (min {:.3f})\n", "vxc", median(tv),
               minimum(tv));
  if (!tg.empty())
    fmt::print("  {:<22s} {:9.3f} s   (min {:.3f})\n", "xc gradient",
               median(tg), minimum(tg));
  fmt::print("  E_xc = {:.12f}", exc);
  if (!skip_grad)
    fmt::print("   |g_xc| = {:.12f}", g.norm());
  fmt::print("\n");

  if (!save_file.empty()) {
    std::ofstream os(save_file, std::ios::binary);
    os.write(reinterpret_cast<const char *>(&exc), sizeof exc);
    write_mat(os, V);
    write_mat(os, g);
  }
  if (!ref_file.empty()) {
    std::ifstream is(ref_file, std::ios::binary);
    double exc_ref = 0.0;
    is.read(reinterpret_cast<char *>(&exc_ref), sizeof exc_ref);
    occ::Mat Vref = read_mat(is);
    occ::Mat gref = read_mat(is);
    fmt::print("  vs ref: dE_xc = {:.3e}", exc - exc_ref);
    if (!skip_vxc && Vref.size() == V.size())
      fmt::print("  max|dV| = {:.3e}  dTr(DV) = {:.3e}",
                 (V - Vref).cwiseAbs().maxCoeff(),
                 (mo.D.array() * (V - Vref).array()).sum());
    if (!skip_grad && gref.size() == g.size())
      fmt::print("  max|dg| = {:.3e}", (g - gref).cwiseAbs().maxCoeff());
    fmt::print("\n");
  }
  return 0;
}
