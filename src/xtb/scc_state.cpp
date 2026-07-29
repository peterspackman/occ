#include <algorithm>
#include <occ/xtb/scc_state.h>

namespace occ::xtb {

namespace {

template <typename Block>
double max_abs_diff(const Block &a, const Block &b) {
  return a.size() == 0 ? 0.0 : (a - b).cwiseAbs().maxCoeff();
}

} // namespace

SccMixerState SccMixerState::zero(int n_shells, int n_atoms,
                                  bool magnetization, bool multipoles) {
  SccMixerState s;
  s.shell_charges = Vec::Zero(n_shells);
  if (magnetization)
    s.magnetization = Vec::Zero(n_shells);
  if (multipoles) {
    s.multipoles.dipm = Mat3N::Zero(3, n_atoms);
    s.multipoles.qp = Mat::Zero(6, n_atoms);
  }
  return s;
}

Eigen::Index SccMixerState::size() const {
  return shell_charges.size() + magnetization.size() + multipoles.dipm.size() +
         multipoles.qp.size();
}

Vec SccMixerState::pack() const {
  Vec out(size());
  Eigen::Index at = 0;
  auto append = [&](const auto &block) {
    Eigen::Map<const Vec> flat(block.data(), block.size());
    out.segment(at, flat.size()) = flat;
    at += flat.size();
  };
  append(shell_charges);
  append(magnetization);
  append(multipoles.dipm);
  append(multipoles.qp);
  return out;
}

void SccMixerState::unpack(const Vec &packed) {
  Eigen::Index at = 0;
  auto take = [&](auto &block) {
    Eigen::Map<Vec>(block.data(), block.size()) =
        packed.segment(at, block.size());
    at += block.size();
  };
  take(shell_charges);
  take(magnetization);
  take(multipoles.dipm);
  take(multipoles.qp);
}

void SccMixerState::damp_toward(const SccMixerState &fresh, double factor) {
  auto mix = [factor](auto &prev, const auto &next) {
    prev = (1.0 - factor) * next + factor * prev;
  };
  mix(shell_charges, fresh.shell_charges);
  mix(magnetization, fresh.magnetization);
  mix(multipoles.dipm, fresh.multipoles.dipm);
  mix(multipoles.qp, fresh.multipoles.qp);
}

double SccMixerState::max_change(const SccMixerState &other) const {
  return std::max(
      {max_abs_diff(shell_charges, other.shell_charges),
       max_abs_diff(magnetization, other.magnetization),
       max_abs_diff(multipoles.dipm, other.multipoles.dipm),
       max_abs_diff(multipoles.qp, other.multipoles.qp)});
}

} // namespace occ::xtb
