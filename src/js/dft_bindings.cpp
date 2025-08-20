#include "dft_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/em_js.h>
#include <emscripten/val.h>
#include <occ/dft/dft.h>
#include <occ/io/grid_settings.h>
#include <occ/qm/scf.h>
#include <occ/qm/hessians.h>
#include <occ/qm/gradients.h>

using namespace emscripten;
using namespace occ::dft;
using namespace occ::qm;
using namespace occ::io;

EM_JS(void, debug_log_dft, (const char *msg),
      { console.log('DEBUG DFT C++:', UTF8ToString(msg)); });

void register_dft_bindings() {
  // GridSettings class binding
  class_<GridSettings>("GridSettings")
      .constructor<>()
      .property("maxAngularPoints", &GridSettings::max_angular_points)
      .property("minAngularPoints", &GridSettings::min_angular_points)
      .property("radialPoints", &GridSettings::radial_points)
      .property("radialPrecision", &GridSettings::radial_precision)
      .function("toString", optional_override([](const GridSettings &settings) {
                  return std::string("<GridSettings ang=(") +
                         std::to_string(settings.min_angular_points) + "," +
                         std::to_string(settings.max_angular_points) +
                         ") radial=" + std::to_string(settings.radial_points) +
                         ", prec=" + std::to_string(settings.radial_precision) +
                         ">";
                }));

  // DFT class binding
  class_<DFT>("DFT")
      .constructor<const std::string &, const AOBasis &>()
      .constructor<const std::string &, const AOBasis &, const GridSettings &>()
      .function("nuclearAttractionMatrix",
                &DFT::compute_nuclear_attraction_matrix)
      .function("kineticMatrix", &DFT::compute_kinetic_matrix)
      .function("setDensityFittingBasis", &DFT::set_density_fitting_basis)
      .function("overlapMatrix", &DFT::compute_overlap_matrix)
      .function("nuclearRepulsion", optional_override([](const DFT &dft) {
                  return dft.nuclear_repulsion_energy();
                }))
      .function("setPrecision", &DFT::set_precision)
      .function("setMethod", &DFT::set_method)
      .function("fockMatrix",
                optional_override([](DFT &dft, const MolecularOrbitals &mo) {
                  return dft.compute_fock(mo);
                }))
      .function("scf", optional_override([](DFT &dft, SpinorbitalKind kind) {
                  return SCF<DFT>(dft, kind);
                }),
                allow_raw_pointers())
      .function("computeGradient",
                optional_override([](DFT &dft, const MolecularOrbitals &mo) {
                  occ::qm::GradientEvaluator<DFT> grad(dft);
                  return grad(mo);
                }))
      .function("hessianEvaluator",
                optional_override([](DFT &dft) {
                  return occ::qm::HessianEvaluator<DFT>(dft);
                }))
      .function("toString", optional_override([](const DFT &dft) {
                  return std::string("<DFT ") + dft.method_string() + " (" +
                         dft.aobasis().name() + ", " +
                         std::to_string(dft.atoms().size()) + " atoms)>";
                }));

  // Kohn-Sham SCF class binding - manual wrapper for SCF<DFT>
  class_<SCF<DFT>>("KohnShamSCF")
      .constructor<DFT &>()
      .constructor<DFT &, SpinorbitalKind>()
      .property("convergenceSettings", &SCF<DFT>::convergence_settings)
      .function("setChargeMultiplicity", &SCF<DFT>::set_charge_multiplicity)
      .function("setInitialGuess", &SCF<DFT>::set_initial_guess_from_wfn)
      .function("getScfKind", optional_override([](const SCF<DFT> &scf) {
                  return std::string(scf.scf_kind());
                }))
      .function("run", optional_override([](SCF<DFT> &scf) {
                  try {
                    debug_log_dft("Starting DFT SCF calculation");
                    auto result = scf.compute_scf_energy();
                    debug_log_dft("DFT SCF calculation completed successfully");
                    return result;
                  } catch (const std::exception &e) {
                    debug_log_dft("DFT SCF calculation threw std::exception");
                    debug_log_dft(e.what());
                    throw;
                  } catch (...) {
                    debug_log_dft(
                        "DFT SCF calculation threw unknown exception");
                    throw;
                  }
                }))
      .function("computeScfEnergy", optional_override([](SCF<DFT> &scf) {
                  return scf.compute_scf_energy();
                }))
      .function("wavefunction", &SCF<DFT>::wavefunction)
      .function("toString", optional_override([](const SCF<DFT> &scf) {
                  return std::string("<SCF(KS) (") +
                         scf.m_procedure.aobasis().name() + ", " +
                         std::to_string(scf.m_procedure.atoms().size()) +
                         " atoms)>";
                }));
}