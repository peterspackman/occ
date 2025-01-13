#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/atom.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

namespace occ::dft {

using occ::qm::AOBasis;
using occ::qm::SpinorbitalKind;

int DFT::density_derivative() const {
  int deriv = 0;
  for (const auto &func : m_funcs.unpolarized) {
    deriv = std::max(deriv, func.derivative_order());
  }
  return deriv;
}

DFT::DFT(const std::string &method, const AOBasis &basis,
         const BeckeGridSettings &grid_settings)
    : SCFMethodBase(basis.atoms()), m_hf(basis), m_grid(basis, grid_settings) {
  update_electron_count();
  set_method(method);
  set_integration_grid(grid_settings);
}

void DFT::set_method(const std::string &method_string) {
  if (m_method_string == method_string)
    return;

  if (m_method_string != method_string) {
    occ::log::info("DFT method string: {}", method_string);
  }
  m_method_string = method_string;
  m_funcs = parse_dft_method(method_string);

  for (const auto &func : m_funcs.unpolarized) {
    occ::log::debug("Functional: {} {} {}, exact exchange = {}", func.name(),
                    func.kind_string(), func.family_string(),
                    func.exact_exchange_factor(), func.polarized());
  }

  m_rs_params = {};
  for (const auto &func : m_funcs.unpolarized) {
    auto rs = func.range_separated_parameters();
    if (rs.omega != 0.0) {
      m_rs_params = rs;
    }
  }
  if (m_rs_params.omega != 0.0) {
    occ::log::info("    RS omega = {}", m_rs_params.omega);
    occ::log::info("    RS alpha = {}", m_rs_params.alpha);
    occ::log::info("    RS beta  = {}", m_rs_params.beta);
  }

  double hfx = exact_exchange_factor();
  if (hfx > 0.0) {
    occ::log::debug("    {} x HF exchange", hfx);
  }
}

void DFT::set_integration_grid(const BeckeGridSettings &settings) {
  const auto &atoms = m_hf.aobasis().atoms();
  if (settings != m_grid.settings()) {
    m_grid = MolecularGrid(m_hf.aobasis(), settings);
  }
  occ::log::debug("start calculating atom grids... ");
  m_atom_grids.clear();
  for (size_t i = 0; i < atoms.size(); i++) {
    m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
  }
  size_t num_grid_points = std::accumulate(
      m_atom_grids.begin(), m_atom_grids.end(), 0.0,
      [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
  occ::log::info("finished calculating atom grids ({} points)",
                 num_grid_points);
  occ::log::debug("Grid initialization took {} seconds",
                  occ::timing::total(occ::timing::grid_init));
  occ::log::debug("Grid point creation took {} seconds",
                  occ::timing::total(occ::timing::grid_points));

  if (!settings.filename.empty()) {
    occ::log::info("Writing DFT grids to {}", settings.filename);
    auto grid_file = fmt::output_file(settings.filename);
    int atom_idx = 0;
    for (const auto &grid : m_atom_grids) {
      grid_file.print("Atom grid {} Z = {}\n", atom_idx, grid.atomic_number);
      grid_file.print("{:>20s} {:>20s} {:>20s} {:>20s}\n", "weight", "x", "y",
                      "z");
      for (int i = 0; i < grid.num_points(); i++) {
        double w = grid.weights(i);
        double x = grid.points(0, i);
        double y = grid.points(1, i);
        double z = grid.points(2, i);
        grid_file.print("{: 20.12e} {: 20.12e} {: 20.12e} {: 20.12e}\n", w, x,
                        y, z);
      }
      atom_idx++;
    }
  }

  for (const auto &func : m_funcs.unpolarized) {
    if (func.needs_nlc_correction()) {
      m_nlc.set_integration_grid(m_hf.aobasis());
      break;
    }
  }
}

RangeSeparatedParameters DFT::range_separated_parameters() const {
  return m_rs_params;
}
} // namespace occ::dft
