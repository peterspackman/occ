#include <algorithm>
#include <fmt/core.h>
#include <memory>
#include <vector>
#include <occ/cg/solvent_surface.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/cg_pipeline.h>
#include <occ/driver/monomer_wavefunctions.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/mults/dimer_interaction.h>
#include <occ/mults/dma_cg.h>
#include <occ/scrf/reaction_field.h>

namespace occ::mults {

using occ::driver::CrystalGrowthCalculatorOptions;
using occ::interaction::LatticeConvergenceSettings;
using occ::interaction::LatticeEnergyCalculator;

namespace {

// Electrostatic-only continuum solvation of one monomer with the solute fixed
// (no polarization): the gas-phase DMA multipoles set the potential on the
// cavity and E_solv = 1/2 sigma.phi. Packaged as SMDSolventSurfaces so the cg
// partitioner consumes it unchanged.
cg::SMDSolventSurfaces compute_monomer_esp_solvation(const DMAMonomer &mon,
                                                     const std::string &solvent) {
  const int n = static_cast<int>(mon.reference.size());
  IVec Z(n);
  for (int i = 0; i < n; ++i)
    Z(i) = mon.reference.atomic_numbers[i];

  scrf::Options opts;
  opts.solvent = solvent;
  opts.backend = scrf::Options::Backend::CPCM;
  opts.radii = scrf::Options::Radii::CosmoVdW;
  scrf::ReactionFieldEngine engine(opts);
  engine.initialize(mon.reference.positions, Z); // Bohr

  // Fixed solute potential on the cavity from the gas-phase DMA multipoles.
  const Mat3N &points = engine.es_cavity().vertices; // Bohr
  MultipoleInteractions mi;
  Vec phi = Vec::Zero(points.cols());
  for (int s = 0; s < n; ++s) {
    auto esp = mi.compute_esp_grid(mon.reference.multipoles[s],
                                   Vec3(mon.reference.positions.col(s)), points);
    for (int i = 0; i < points.cols(); ++i)
      phi(i) += esp[i];
  }

  engine.solve_asc(phi);
  return cg::from_scrf_surfaces(engine.surfaces());
}

// Warn about element pairs the force field has no exp-6 parameters for;
// dimer_interaction_energy skips them, so they contribute electrostatics only.
void warn_missing_exp6_coverage(const std::vector<DMAMonomer> &monomers,
                                const ForceFieldParams &ff) {
  std::vector<int> elements;
  for (const auto &mon : monomers)
    for (int z : mon.reference.atomic_numbers)
      elements.push_back(z);

  for (const auto &[za, zb] : missing_exp6_pairs(elements, ff))
    occ::log::warn("DMA+exp-6: no exp-6 parameters for element pair Z{}-Z{}; "
                   "those atom pairs contribute electrostatics only "
                   "(repulsion/dispersion omitted)",
                   za, zb);
}

// Pick the short-range exp-6 set from the model name. The name -> parameter
// set table is shared with `occ dma --csp-force-field`; see
// short_range_model_registry().
ForceFieldParams short_range_ff_for_model(const std::string &model_name) {
  const auto &model = short_range_model_for_model_name(model_name);
  occ::log::info("DMA+exp-6: using the {} potential: {}", model.name,
                 model.description);
  return make_force_field(model);
}

} // namespace

DMACrystalGrowthCalculator::DMACrystalGrowthCalculator(
    const occ::crystal::Crystal &crystal,
    const CrystalGrowthCalculatorOptions &options)
    : occ::driver::CEModelCrystalGrowthCalculator(crystal, options) {}

DMACrystalGrowthCalculator::ResolvedLevel
DMACrystalGrowthCalculator::resolved_reference_level() const {
  const std::string model =
      m_reference.model.empty() ? "ce-b3lyp" : m_reference.model;
  const auto pm = occ::interaction::ce_model_from_string(model);
  const bool overridden =
      !m_reference.method.empty() || !m_reference.basis.empty();
  std::string method =
      m_reference.method.empty() ? pm.method : m_reference.method;
  std::string basis = m_reference.basis.empty() ? pm.basis : m_reference.basis;
  std::string label = overridden ? fmt::format("{}/{}", method, basis) : model;
  return {std::move(method), std::move(basis), std::move(label)};
}

void DMACrystalGrowthCalculator::init_monomer_energies() {
  const auto &opts = options();
  const auto ref = resolved_reference_level();
  occ::log::info("DMA+exp-6 model: computing monomer wavefunctions at {}",
                 ref.label);
  gas_phase_wavefunctions() = occ::driver::calculate_wavefunctions(
      opts.basename, molecules(), ref.method, ref.basis, /*spherical=*/false);
  solvated_wavefunctions() = gas_phase_wavefunctions();

  m_monomers.clear();
  m_monomers.reserve(gas_phase_wavefunctions().size());
  for (auto &wfn : gas_phase_wavefunctions()) {
    auto mon = DMAMonomer::from_wavefunction(wfn);
    int max_rank = 0;
    for (const auto &mp : mon.reference.multipoles)
      max_rank = std::max(max_rank, mp.max_rank);
    occ::log::info("DMA: molecule {} -> {} sites, max rank {}", m_monomers.size(),
                   mon.reference.size(), max_rank);
    m_monomers.push_back(std::move(mon));
  }

  auto &surfs = solvated_surface_properties();
  surfs.clear();
  surfs.reserve(m_monomers.size());
  for (const auto &mon : m_monomers) {
    auto s = compute_monomer_esp_solvation(mon, opts.solvent);
    occ::log::info("DMA solvation: molecule {} dG_es = {:.2f} kJ/mol", surfs.size(),
                   s.total_solvation_energy * occ::units::AU_TO_KJ_PER_MOL);
    surfs.push_back(std::move(s));
  }
}

void DMACrystalGrowthCalculator::converge_lattice_energy() {
  const auto &opts = options();
  occ::log::info("DMA+exp-6 model: converging lattice energy");

  LatticeConvergenceSettings settings;
  // Fold the reference level into the model name: it keys the on-disk dimer
  // cache, so different reference levels must not share cached energies.
  settings.model_name =
      fmt::format("{}_{}", opts.energy_model, resolved_reference_level().label);
  settings.max_radius = opts.outer_radius;
  // Wolf summation assumes CE-model Coulomb semantics and diverges with
  // distributed multipoles; a charged-multipole sum needs a multipole Ewald.
  settings.wolf_sum = false;
  // No induction in this model (electrostatics + exp-6 only).
  settings.crystal_field_polarization = false;

  if (opts.use_wolf_sum || opts.use_crystal_polarization)
    occ::log::warn("DMA+exp-6: charged-system options (--charges) are not "
                   "supported by this model: it has neither Ewald/Wolf summation "
                   "for distributed multipoles nor polarization. The lattice "
                   "energy for a charged/ionic cell is conditionally convergent "
                   "and unreliable here — treat it with caution.");

  auto ff = short_range_ff_for_model(opts.energy_model);
  // The typed sets (FIT / W99) carry their own halogen/sulfur parameters.
  if (!ff.use_short_range_typing())
    warn_missing_exp6_coverage(m_monomers, ff);
  auto model =
      std::make_unique<DMAExp6EnergyModel>(crystal(), m_monomers, std::move(ff));

  LatticeEnergyCalculator calc(std::move(model), crystal(), opts.basename,
                               settings);
  auto result = calc.compute();
  full_dimers() = result.dimers;
  dimer_energies() = result.energy_components;
  nearest_dimers() = crystal().symmetry_unique_dimers(opts.inner_radius);

  if (full_dimers().unique_dimers.empty())
    occ::log::error("DMA+exp-6: no dimers within radius {:.3f}",
                    opts.outer_radius);
}

bool model_name_is_dma(const std::string &model_name) {
  std::string lower(model_name.size(), '\0');
  std::transform(model_name.begin(), model_name.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower == "dma" || lower == "williams" || lower == "fit" ||
         lower == "w99" || lower.rfind("dma-", 0) == 0 ||
         lower.rfind("williams-", 0) == 0;
}

occ::cg::CrystalGrowthResult run_cg_dma(const occ::driver::CGConfig &config) {
  auto prep = occ::driver::prepare_cg(config);
  DMACrystalGrowthCalculator calc(prep.crystal, prep.opts);
  calc.set_reference_level(config.dma_reference);
  if (!prep.charges.empty())
    calc.set_molecule_charges(prep.charges);
  return occ::driver::run_cg_pipeline(calc, prep.opts, config);
}

occ::cg::CrystalGrowthResult
run_crystal_growth(const occ::driver::CGConfig &config) {
  if (model_name_is_dma(config.lattice_settings.model_name))
    return run_cg_dma(config);
  return occ::driver::run_cg(config);
}

} // namespace occ::mults
