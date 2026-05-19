#include <occ/xtb/gfn2_engine.h>
#include <occ/xtb/scc.h>

namespace occ::xtb {

SccResult run_charge_only_scc(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params,
                              const SccOptions &opts) {
  Gfn2Engine calc(atoms, params);
  return calc.run_charge_only(opts);
}

SccResult run_gfn2_scc(const std::vector<core::Atom> &atoms,
                       const Gfn2Parameters &params,
                       const SccOptions &opts) {
  Gfn2Engine calc(atoms, params);
  return calc.run_full(opts);
}

} // namespace occ::xtb
