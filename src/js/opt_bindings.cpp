#include "opt_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/molecule.h>
#include <occ/core/vibration.h>
#include <occ/opt/geometry_optimizer.h>
#include <occ/qm/hessians.h>
#include <occ/qm/hf.h>
#include <occ/dft/dft.h>
#include <occ/io/xyz.h>

using namespace emscripten;
using namespace occ::opt;
using namespace occ::core;
using namespace occ::qm;
using namespace occ::dft;
using occ::Mat;
using occ::Vec;
using occ::Mat3N;

void register_opt_bindings() {
  class_<ConvergenceCriteria>("ConvergenceCriteria")
      .constructor<>()
      .property("gradientMax", &ConvergenceCriteria::gradient_max)
      .property("gradientRms", &ConvergenceCriteria::gradient_rms)
      .property("stepMax", &ConvergenceCriteria::step_max)
      .property("stepRms", &ConvergenceCriteria::step_rms)
      .property("energyChange", &ConvergenceCriteria::energy_change)
      .property("useEnergyCriterion", &ConvergenceCriteria::use_energy_criterion)
      .property("maxIterations", &ConvergenceCriteria::max_iterations);

  // Trust-region optimizer in TRIC coordinates: update(energy, gradient) at
  // getNextGeometry(), then step() until it returns true.
  class_<GeometryOptimizer>("GeometryOptimizer")
      .constructor<const Molecule &>()
      .constructor<const Molecule &, const ConvergenceCriteria &>()
      .function("step", &GeometryOptimizer::step)
      .function("update", &GeometryOptimizer::update)
      .function("getNextGeometry", &GeometryOptimizer::get_next_geometry)
      .function("setInitialHessian", &GeometryOptimizer::set_initial_hessian)
      .function("isConverged", &GeometryOptimizer::is_converged)
      .function("currentStep", &GeometryOptimizer::current_step)
      .function("currentEnergy", &GeometryOptimizer::current_energy)
      .function("trustRadius", &GeometryOptimizer::trust_radius)
      .function("toString", optional_override([](const GeometryOptimizer &o) {
                  return std::string("<GeometryOptimizer: step=") +
                         std::to_string(o.current_step()) + ", converged=" +
                         (o.is_converged() ? "true" : "false") + ">";
                }));

  // Hessian evaluator for HartreeFock  
  class_<HessianEvaluator<HartreeFock>>("HessianEvaluatorHF")
      .constructor<HartreeFock &>()
      .function("setStepSize", &HessianEvaluator<HartreeFock>::set_step_size)
      .function("setUseAcousticSumRule", &HessianEvaluator<HartreeFock>::set_use_acoustic_sum_rule)
      .function("stepSize", &HessianEvaluator<HartreeFock>::step_size)
      .function("useAcousticSumRule", &HessianEvaluator<HartreeFock>::use_acoustic_sum_rule)
      .function("nuclearRepulsion", &HessianEvaluator<HartreeFock>::nuclear_repulsion)
      .function("compute", &HessianEvaluator<HartreeFock>::operator())
      .function("toString", optional_override([](const HessianEvaluator<HartreeFock> &hess) {
                  return std::string("<HessianEvaluatorHF step_size=") + std::to_string(hess.step_size()) + 
                         " acoustic_sum_rule=" + (hess.use_acoustic_sum_rule() ? "true" : "false") + ">";
                }));

  // Hessian evaluator for DFT
  class_<HessianEvaluator<DFT>>("HessianEvaluatorDFT")
      .constructor<DFT &>()
      .function("setStepSize", &HessianEvaluator<DFT>::set_step_size)
      .function("setUseAcousticSumRule", &HessianEvaluator<DFT>::set_use_acoustic_sum_rule)
      .function("stepSize", &HessianEvaluator<DFT>::step_size)
      .function("useAcousticSumRule", &HessianEvaluator<DFT>::use_acoustic_sum_rule)
      .function("nuclearRepulsion", &HessianEvaluator<DFT>::nuclear_repulsion)
      .function("compute", &HessianEvaluator<DFT>::operator())
      .function("toString", optional_override([](const HessianEvaluator<DFT> &hess) {
                  return std::string("<HessianEvaluatorDFT step_size=") + std::to_string(hess.step_size()) + 
                         " acoustic_sum_rule=" + (hess.use_acoustic_sum_rule() ? "true" : "false") + ">";
                }));

  // Vibrational modes structure
  class_<VibrationalModes>("VibrationalModes")
      .property("frequenciesCm", &VibrationalModes::frequencies_cm)
      .property("frequenciesHartree", &VibrationalModes::frequencies_hartree)
      .property("normalModes", &VibrationalModes::normal_modes)
      .property("massWeightedHessian", &VibrationalModes::mass_weighted_hessian)
      .property("hessian", &VibrationalModes::hessian)
      .function("nModes", &VibrationalModes::n_modes)
      .function("nAtoms", &VibrationalModes::n_atoms)
      .function("summaryString", &VibrationalModes::summary_string)
      .function("frequenciesString", &VibrationalModes::frequencies_string)
      .function("normalModesString", &VibrationalModes::normal_modes_string)
      .function("getAllFrequencies", &VibrationalModes::get_all_frequencies)
      .function("toString", optional_override([](const VibrationalModes &modes) {
                  return std::string("<VibrationalModes n_modes=") + std::to_string(modes.n_modes()) + 
                         " n_atoms=" + std::to_string(modes.n_atoms()) + ">";
                }));

  // Vibrational analysis functions
  function("computeVibrationalModes", 
           optional_override([](const Mat &hessian, const Vec &masses, const Mat3N &positions, bool project_tr_rot) {
             return occ::core::compute_vibrational_modes(hessian, masses, positions, project_tr_rot);
           }));

  function("computeVibrationalModesFromMolecule",
           optional_override([](const Mat &hessian, const Molecule &molecule, bool project_tr_rot) {
             return occ::core::compute_vibrational_modes(hessian, molecule, project_tr_rot);
           }));

  function("massWeightedHessian",
           optional_override([](const Mat &hessian, const Vec &masses) {
             return occ::core::mass_weighted_hessian(hessian, masses);
           }));

  function("massWeightedHessianFromMolecule",
           optional_override([](const Mat &hessian, const Molecule &molecule) {
             return occ::core::mass_weighted_hessian(hessian, molecule);
           }));

  function("eigenvaluesToFrequenciesCm", &occ::core::eigenvalues_to_frequencies_cm);
  function("frequenciesCmToHartree", &occ::core::frequencies_cm_to_hartree);

  // XYZ file export functionality
  function("moleculeToXYZ", optional_override([](const Molecule &mol) {
             return occ::io::to_xyz_string(mol);
           }));

  function("moleculeToXYZWithComment", optional_override([](const Molecule &mol, const std::string &comment) {
             return occ::io::to_xyz_string(mol, comment);
           }));
}