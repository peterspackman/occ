#pragma once
#include <fmt/core.h>
#include <occ/dft/dft.h>
#include <occ/driver/method_parser.h>
#include <occ/gto/gto.h>
#include <occ/qm/hf.h>
#include <occ/solvent/solvation_correction.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace occ::driver {

/// Build the SCF procedure that `method` names -- Hartree-Fock or a DFT
/// functional -- on `basis`, wrap it in a solvation correction constructed
/// from `solvation_args`, and hand the wrapped procedure to `run`.
///
/// The solvated calculations (SMD surfaces, the COSMO-RS conductor) take their
/// method from the energy model, and a CE model can name either kind, so they
/// cannot assume DFT. `run` is called with one of two procedure types and must
/// return the same type for both, which a generic lambda does naturally.
template <typename Run, typename... SolvationArgs>
decltype(auto) with_solvated_procedure(const std::string &method,
                                       const occ::gto::AOBasis &basis,
                                       Run &&run,
                                       SolvationArgs &&...solvation_args) {
  switch (method_kind_from_string(method)) {
  case MethodKind::HF: {
    occ::qm::HartreeFock hf(basis);
    occ::solvent::SolvationCorrectedProcedure<occ::qm::HartreeFock> proc(
        hf, std::forward<SolvationArgs>(solvation_args)...);
    return run(proc);
  }
  case MethodKind::DFT: {
    occ::dft::DFT ks(method, basis);
    occ::solvent::SolvationCorrectedProcedure<occ::dft::DFT> proc(
        ks, std::forward<SolvationArgs>(solvation_args)...);
    return run(proc);
  }
  default:
    throw std::invalid_argument(fmt::format(
        "Solvated SCF needs Hartree-Fock or a DFT functional, not '{}'",
        method));
  }
}

} // namespace occ::driver
