#include <fmt/core.h>
#include <occ/xtb/cpcmx.h>

namespace occ::xtb {

namespace {

occ::scrf::Options to_engine_options(const CpcmXOptions &o) {
  occ::scrf::Options out;
  out.backend = occ::scrf::Options::Backend::CPCM;
  out.radii = occ::scrf::Options::Radii::CosmoVdW;
  out.solvent = o.solvent;
  out.dielectric_override = o.dielectric_override;
  out.f_eps_x = o.x;
  out.probe_radius_angs = o.probe_radius_angs;
  out.smoothing_width_bohr = o.smoothing_width_bohr;
  out.include_cds = false;
  return out;
}

} // namespace

CpcmXSolvationModel::CpcmXSolvationModel(CpcmXOptions opts)
    : m_opts(std::move(opts)), m_engine(to_engine_options(m_opts)) {}

void CpcmXSolvationModel::initialize(const Mat3N &positions_bohr,
                                     const IVec &atomic_numbers) {
  m_engine.initialize(positions_bohr, atomic_numbers);
}

void CpcmXSolvationModel::update(const Vec &atomic_charges) {
  m_engine.update_from_atom_charges(atomic_charges);
}

std::string CpcmXSolvationModel::name() const {
  return fmt::format("CPCM-X(solvent='{}', eps={:.3f})", m_opts.solvent,
                     m_engine.dielectric());
}

std::optional<SolvationSurfaces> CpcmXSolvationModel::surfaces() const {
  auto s = m_engine.surfaces();
  if (!s.coulomb && !s.cds)
    return std::nullopt;
  return s;
}

} // namespace occ::xtb
