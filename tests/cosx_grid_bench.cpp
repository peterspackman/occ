// COSX grid accuracy and cost harness.
//
//   cosx_grid_bench <xyz> <basis> [--threads N] [--reps N] [--no-scf]
//                   [--dump <prefix>]
//
// Not a test: converges exact HF (four-centre J and K), then for each COSX
// grid level reports the grid size (total, and per element for its last
// atom), the time of one COSX K build, the exchange-energy error at the exact
// density, and (unless --no-scf) the total-energy error of a COSX SCF (exact
// J, COSX K) against the exact SCF. --dump writes each level's partitioned grid as
// "<prefix>.gridxN.txt" (x y z w atom, bohr) for comparison with other codes.
#include "bench_common.h"
#include <fmt/core.h>
#include <fmt/os.h>
#include <map>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/io/xyz.h>
#include <occ/numint/grid_settings.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/seminumerical_exchange.h>
#include <string>

namespace {

using occ::bench::seconds;

template <typename Proc> void tighten(occ::qm::SCF<Proc> &scf) {
  scf.convergence_settings.energy_threshold = 1e-10;
  scf.convergence_settings.commutator_threshold = 1e-8;
}

} // namespace

int main(int argc, char **argv) {
  const occ::bench::Args args(
      argc, argv,
      "<xyz> <basis> [--threads N] [--reps N] [--no-scf] [--dump <prefix>]",
      {"--no-scf"}, {"--reps", "--dump"});
  if (!args.valid(2, 2))
    return 1;
  const std::string xyz = args.positional(0), basis_name = args.positional(1);
  const int threads = args.threads(), reps = args.reps();
  const bool do_scf = !args.has("--no-scf");
  const std::string dump = args.get("--dump", "");
  occ::parallel::set_num_threads(threads);
  occ::log::set_log_level(spdlog::level::warn);

  auto mol = occ::io::molecule_from_xyz_file(xyz);
  auto basis = occ::gto::AOBasis::load(mol.atoms(), basis_name);
  basis.set_pure(true);

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  tighten(scf);
  double e_exact = 0.0;
  const double t_scf = seconds([&] { e_exact = scf.compute_scf_energy(); });
  const auto mo = scf.molecular_orbitals();
  // Restricted: E_K = -1/2 tr(D K) with occ's D = C_occ C_occ^T, K = K[D]
  // (the SCF two-electron energy uses the same contraction).
  const occ::Mat K_exact = hf.compute_K(mo);
  const double ex_exact = -mo.D.cwiseProduct(K_exact).sum();
  fmt::print("{} {} nbf={} natom={}  E(exact)={:.10f}  E_x={:.10f}  "
             "scf {:.2f} s\n",
             xyz, basis_name, basis.nbf(), mol.size(), e_exact, ex_exact,
             t_scf);

  for (auto level : {occ::numint::COSXGridLevel::Grid1,
                     occ::numint::COSXGridLevel::Grid2,
                     occ::numint::COSXGridLevel::Grid3}) {
    const auto settings = occ::numint::GridSettings::for_cosx(level);
    occ::qm::cosx::SemiNumericalExchange sgx(basis, settings);
    sgx.set_use_esp(true);
    sgx.set_use_spatial_hierarchy(true);

    std::map<int, size_t> per_element;
    size_t total = 0;
    std::string dumped;
    for (size_t i = 0; i < mol.size(); i++) {
      const auto ag = sgx.grid().get_partitioned_atom_grid(i);
      const size_t n = ag.num_points();
      per_element[mol.atoms()[i].atomic_number] = n;
      total += n;
      if (dump.empty())
        continue;
      for (Eigen::Index g = 0; g < ag.points.cols(); g++)
        dumped += fmt::format("{:.17g} {:.17g} {:.17g} {:.17g} {}\n",
                              ag.points(0, g), ag.points(1, g),
                              ag.points(2, g), ag.weights(g), i);
    }
    if (!dump.empty()) {
      auto out = fmt::output_file(
          fmt::format("{}.gridx{}.txt", dump, static_cast<int>(level) + 1));
      out.print("{}", dumped);
    }
    occ::Mat K;
    sgx.compute_K(mo); // warm up (screening data, ESP tables)
    double t_k = 1e30;
    for (int rep = 0; rep < reps; rep++)
      t_k = std::min(t_k, seconds([&] { K = sgx.compute_K(mo); }));
    const double ex = -mo.D.cwiseProduct(K).sum();

    std::string elems;
    for (const auto &[z, n] : per_element)
      elems += fmt::format(" {}:{}", occ::core::Element(z).symbol(), n);
    fmt::print("  {}  npts={:7d} [{} ]  K {:.3f} s  dE_x(fixed D)={:+.3e}",
               occ::numint::cosx_grid_level_to_string(level), total, elems,
               t_k, ex - ex_exact);

    if (do_scf) {
      occ::qm::HartreeFock hf_cosx(basis);
      hf_cosx.set_cosx_exchange(level);
      occ::qm::SCF<occ::qm::HartreeFock> scf_cosx(hf_cosx);
      tighten(scf_cosx);
      double e = 0.0;
      const double t = seconds([&] { e = scf_cosx.compute_scf_energy(); });
      fmt::print("  E={:.10f} dE={:+.3e}  scf {:.2f} s", e, e - e_exact, t);
    }
    fmt::print("\n");
  }
  return 0;
}
