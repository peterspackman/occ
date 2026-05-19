#include <fmt/core.h>
#include <occ/xtb/smd_xtb.h>

namespace occ::xtb {

namespace {

occ::scrf::Options smd_engine_options(const std::string &solvent) {
  occ::scrf::Options o;
  o.backend = occ::scrf::Options::Backend::CPCM;
  o.radii = occ::scrf::Options::Radii::SmdIntrinsicCoulomb;
  o.solvent = solvent;
  o.f_eps_x = 0.0; // CPCM ideal-conductor; SMD ES shares this convention
  o.probe_radius_angs = 0.4;
  o.smoothing_width_bohr = 0.1;
  o.include_cds = true;
  return o;
}

} // namespace

SmdSolvationModel::SmdSolvationModel(std::string solvent)
    : m_solvent(std::move(solvent)), m_engine(smd_engine_options(m_solvent)) {}

void SmdSolvationModel::initialize(const Mat3N &positions_bohr,
                                   const IVec &atomic_numbers) {
  m_engine.initialize(positions_bohr, atomic_numbers);
}

void SmdSolvationModel::update(const Vec &atomic_charges) {
  m_engine.update_from_atom_charges(atomic_charges);
}

std::string SmdSolvationModel::name() const {
  return fmt::format("SMD-xtb(solvent='{}', eps={:.3f})", m_solvent,
                     m_engine.dielectric());
}

std::optional<SolvationSurfaces> SmdSolvationModel::surfaces() const {
  auto s = m_engine.surfaces();
  if (!s.coulomb && !s.cds)
    return std::nullopt;
  return s;
}

} // namespace occ::xtb
