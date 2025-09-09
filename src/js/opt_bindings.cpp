#include "opt_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/molecule.h>
#include <occ/core/vibration.h>
#include <occ/opt/angle_coordinate.h>
#include <occ/opt/berny_optimizer.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/opt/dihedral_coordinate.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/optimization_state.h>
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
  // Bond coordinate type enum
  enum_<BondCoordinate::Type>("BondCoordinateType")
      .value("COVALENT", BondCoordinate::Type::COVALENT)
      .value("VDW", BondCoordinate::Type::VDW);

  // Bond coordinate class
  class_<BondCoordinate>("BondCoordinate")
      .constructor<int, int, BondCoordinate::Type>()
      .property("i", &BondCoordinate::i)
      .property("j", &BondCoordinate::j)
      .property("bondType", &BondCoordinate::bond_type)
      .function("evaluate", optional_override([](const BondCoordinate &b, const Mat3N &coords) {
                  return b(coords);
                }))
      .function("gradient", optional_override([](const BondCoordinate &b, const Mat3N &coords) {
                  return b.gradient(coords);
                }))
      .function("toString", optional_override([](const BondCoordinate &b) {
                  std::string type = (b.bond_type == BondCoordinate::Type::COVALENT) ? "COVALENT" : "VDW";
                  return std::string("<BondCoordinate(") + std::to_string(b.i) + ", " + 
                         std::to_string(b.j) + ", " + type + ")>";
                }));

  // Angle coordinate class
  class_<AngleCoordinate>("AngleCoordinate")
      .constructor<int, int, int>()
      .property("i", &AngleCoordinate::i)
      .property("j", &AngleCoordinate::j)
      .property("k", &AngleCoordinate::k)
      .function("evaluate", optional_override([](const AngleCoordinate &a, const Mat3N &coords) {
                  return a(coords);
                }))
      .function("gradient", optional_override([](const AngleCoordinate &a, const Mat3N &coords) {
                  return a.gradient(coords);
                }))
      .function("toString", optional_override([](const AngleCoordinate &a) {
                  return std::string("<AngleCoordinate(") + std::to_string(a.i) + ", " + 
                         std::to_string(a.j) + ", " + std::to_string(a.k) + ")>";
                }));

  // Dihedral coordinate class
  class_<DihedralCoordinate>("DihedralCoordinate")
      .constructor<int, int, int, int>()
      .property("i", &DihedralCoordinate::i)
      .property("j", &DihedralCoordinate::j)
      .property("k", &DihedralCoordinate::k)
      .property("l", &DihedralCoordinate::l)
      .function("evaluate", optional_override([](const DihedralCoordinate &d, const Mat3N &coords) {
                  return d(coords);
                }))
      .function("gradient", optional_override([](const DihedralCoordinate &d, const Mat3N &coords) {
                  return d.gradient(coords);
                }))
      .function("toString", optional_override([](const DihedralCoordinate &d) {
                  return std::string("<DihedralCoordinate(") + std::to_string(d.i) + ", " + 
                         std::to_string(d.j) + ", " + std::to_string(d.k) + ", " + 
                         std::to_string(d.l) + ")>";
                }));

  // Internal coordinates options
  class_<InternalCoordinates::Options>("InternalCoordinatesOptions")
      .constructor<>()
      .property("includeDihedrals", &InternalCoordinates::Options::include_dihedrals)
      .property("superweakDihedrals", &InternalCoordinates::Options::superweak_dihedrals);

  // Internal coordinates structure
  class_<InternalCoordinates>("InternalCoordinates")
      .constructor<const Molecule &, const InternalCoordinates::Options &>()
      .function("bonds", &InternalCoordinates::bonds)
      .function("angles", &InternalCoordinates::angles) 
      .function("dihedrals", &InternalCoordinates::dihedrals)
      .function("weights", &InternalCoordinates::weights)
      .function("size", &InternalCoordinates::size)
      .function("toVector", &InternalCoordinates::to_vector)
      .function("toVectorWithTemplate", &InternalCoordinates::to_vector_with_template)
      .function("wilsonBMatrix", &InternalCoordinates::wilson_b_matrix)
      .function("toString", optional_override([](const InternalCoordinates &ic) {
                  return std::string("<InternalCoordinates: ") + std::to_string(ic.bonds().size()) + 
                         " bonds, " + std::to_string(ic.angles().size()) + " angles, " + 
                         std::to_string(ic.dihedrals().size()) + " dihedrals>";
                }));

  // OptPoint structure
  class_<OptPoint>("OptPoint")
      .constructor<>()
      .constructor<const Vec &, double, const Vec &>()
      .property("q", &OptPoint::q)
      .property("E", &OptPoint::E)
      .property("g", &OptPoint::g)
      .function("toString", optional_override([](const OptPoint &opt) {
                  double g_norm = opt.g.size() > 0 ? opt.g.norm() : 0.0;
                  return std::string("<OptPoint: E=") + std::to_string(opt.E) + ", |q|=" + 
                         std::to_string(opt.q.size()) + ", |g|=" + std::to_string(g_norm) + ">";
                }));

  // Convergence criteria structure
  class_<ConvergenceCriteria>("ConvergenceCriteria")
      .constructor<>()
      .property("gradientMax", &ConvergenceCriteria::gradient_max)
      .property("gradientRms", &ConvergenceCriteria::gradient_rms)
      .property("stepMax", &ConvergenceCriteria::step_max)
      .property("stepRms", &ConvergenceCriteria::step_rms);

  // Optimization state structure
  class_<OptimizationState>("OptimizationState")
      .property("positions", &OptimizationState::positions)
      .property("energy", &OptimizationState::energy)
      .property("gradientCartesian", &OptimizationState::gradient_cartesian)
      .property("current", &OptimizationState::current)
      .property("best", &OptimizationState::best)
      .property("previous", &OptimizationState::previous)
      .property("interpolated", &OptimizationState::interpolated)
      .property("predicted", &OptimizationState::predicted)
      .property("future", &OptimizationState::future)
      .property("hessian", &OptimizationState::hessian)
      .property("trustRadius", &OptimizationState::trust_radius)
      .property("firstStep", &OptimizationState::first_step)
      .property("stepNumber", &OptimizationState::step_number)
      .property("converged", &OptimizationState::converged);

  // Berny optimizer class
  class_<BernyOptimizer>("BernyOptimizer")
      .constructor<const Molecule &>()
      .constructor<const Molecule &, const ConvergenceCriteria &>()
      .function("step", &BernyOptimizer::step)
      .function("update", &BernyOptimizer::update)
      .function("getNextGeometry", &BernyOptimizer::get_next_geometry)
      .function("isConverged", &BernyOptimizer::is_converged)
      .function("currentStep", &BernyOptimizer::current_step)
      .function("currentEnergy", &BernyOptimizer::current_energy)
      .function("currentTrustRadius", &BernyOptimizer::current_trust_radius)
      .function("toString", optional_override([](const BernyOptimizer &opt) {
                  return std::string("<BernyOptimizer: step=") + std::to_string(opt.current_step()) + 
                         ", converged=" + (opt.is_converged() ? "true" : "false") + ">";
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

  // Utility function to transform step from internal to Cartesian coordinates
  function("transformStepToCartesian", &occ::opt::transform_step_to_cartesian);
}