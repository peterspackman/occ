#pragma once
// Single source of truth for every C++ enum exposed to Lua.
//
// For each enum we declare:
//   * an alias  OCC_E_<Name>          - the fully-qualified C++ type
//   * a list    OCC_ENUM_<Name>(X)    - one X("LuaName", OCC_E_<Name>::Value)
//                                       per enumerator
//
// Two consumers expand these lists, so the value set lives in exactly one
// place and can never drift between them:
//   * enum_stacks.h          -> luabridge::Stack<E> specialization, giving
//                               the enum compile-time round-trip + runtime
//                               value validation (luabridge::Enum<E, Vals...>)
//   * the *_bindings.cpp TUs -> the `occ.<Name>` Lua constants, via
//                               OCC_LUA_ENUM_NAMESPACE() below
//
// Add, remove, or rename a value in the list here and BOTH the validation
// list and the Lua-visible constant update together. (Completeness against
// the C++ enum is still manual — there is no Count sentinel to assert on —
// but there is now a single list to keep in sync, not two.)

#include <occ/core/dimer.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/point_group.h>
#include <occ/dft/dft.h>
#include <occ/isosurface/orbital_index.h>
#include <occ/isosurface/surface_types.h>
#include <occ/isosurface/volume_data.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/force_field_params.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/qm/hessians.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/orbital_smearing.h>
#include <occ/qm/spinorbital.h>
#include <occ/xtb/xtb_calculator.h>

// ---------- qm ----------------------------------------------------------

#define OCC_E_SpinorbitalKind ::occ::qm::SpinorbitalKind
#define OCC_ENUM_SpinorbitalKind(X)                                            \
  X("Restricted", OCC_E_SpinorbitalKind::Restricted)                           \
  X("Unrestricted", OCC_E_SpinorbitalKind::Unrestricted)                       \
  X("General", OCC_E_SpinorbitalKind::General)

#define OCC_E_OrbitalSmearingKind ::occ::qm::OrbitalSmearing::Kind
#define OCC_ENUM_OrbitalSmearingKind(X)                                        \
  X("None_", OCC_E_OrbitalSmearingKind::None)                                  \
  X("Fermi", OCC_E_OrbitalSmearingKind::Fermi)                                 \
  X("Gaussian", OCC_E_OrbitalSmearingKind::Gaussian)                           \
  X("Linear", OCC_E_OrbitalSmearingKind::Linear)

#define OCC_E_IntegralEngineDFPolicy ::occ::qm::IntegralEngineDF::Policy
#define OCC_ENUM_IntegralEngineDFPolicy(X)                                     \
  X("Choose", OCC_E_IntegralEngineDFPolicy::Choose)                            \
  X("Direct", OCC_E_IntegralEngineDFPolicy::Direct)                            \
  X("Stored", OCC_E_IntegralEngineDFPolicy::Stored)

// HessianEvaluator<Proc>::Method is a distinct type per instantiation and is
// not exposed as a Lua namespace; only the Stack specialization is generated.
#define OCC_E_HessianMethodHF                                                  \
  ::occ::qm::HessianEvaluator<::occ::qm::HartreeFock>::Method
#define OCC_ENUM_HessianMethodHF(X)                                            \
  X("FiniteDifferences", OCC_E_HessianMethodHF::FiniteDifferences)             \
  X("Analytical", OCC_E_HessianMethodHF::Analytical)

#define OCC_E_HessianMethodDFT                                                 \
  ::occ::qm::HessianEvaluator<::occ::dft::DFT>::Method
#define OCC_ENUM_HessianMethodDFT(X)                                           \
  X("FiniteDifferences", OCC_E_HessianMethodDFT::FiniteDifferences)            \
  X("Analytical", OCC_E_HessianMethodDFT::Analytical)

// ---------- xtb ---------------------------------------------------------

#define OCC_E_XtbMethod ::occ::xtb::XtbCalculator::Method
#define OCC_ENUM_XtbMethod(X) X("GFN2", OCC_E_XtbMethod::GFN2)

// ---------- mults -------------------------------------------------------

#define OCC_E_ForceFieldType ::occ::mults::ForceFieldType
#define OCC_ENUM_ForceFieldType(X)                                             \
  X("None_", OCC_E_ForceFieldType::None)                                       \
  X("LennardJones", OCC_E_ForceFieldType::LennardJones)                        \
  X("BuckinghamDE", OCC_E_ForceFieldType::BuckinghamDE)                        \
  X("Custom", OCC_E_ForceFieldType::Custom)

#define OCC_E_OptimizationMethod ::occ::mults::OptimizationMethod
#define OCC_ENUM_OptimizationMethod(X)                                         \
  X("MSTMIN", OCC_E_OptimizationMethod::MSTMIN)                                \
  X("LBFGS", OCC_E_OptimizationMethod::LBFGS)                                  \
  X("TrustRegion", OCC_E_OptimizationMethod::TrustRegion)                      \
  X("TrustRegionBFGS", OCC_E_OptimizationMethod::TrustRegionBFGS)

// ---------- core --------------------------------------------------------

#define OCC_E_MirrorType ::occ::core::MirrorType
#define OCC_ENUM_MirrorType(X)                                                 \
  X("None_", OCC_E_MirrorType::None)                                           \
  X("H", OCC_E_MirrorType::H)                                                  \
  X("D", OCC_E_MirrorType::D)                                                  \
  X("V", OCC_E_MirrorType::V)

// MoleculeOrder is not exposed as a Lua namespace; only the Stack spec.
#define OCC_E_MoleculeOrder ::occ::core::Dimer::MoleculeOrder
#define OCC_ENUM_MoleculeOrder(X)                                              \
  X("AB", OCC_E_MoleculeOrder::AB)                                             \
  X("BA", OCC_E_MoleculeOrder::BA)

#define OCC_E_AveragingScheme ::occ::core::ElasticTensor::AveragingScheme
#define OCC_ENUM_AveragingScheme(X)                                            \
  X("VOIGT", OCC_E_AveragingScheme::Voigt)                                     \
  X("REUSS", OCC_E_AveragingScheme::Reuss)                                     \
  X("HILL", OCC_E_AveragingScheme::Hill)                                       \
  X("NUMERICAL", OCC_E_AveragingScheme::Numerical)

#define OCC_E_PointGroup ::occ::core::PointGroup
#define OCC_ENUM_PointGroup(X)                                                 \
  X("C1", OCC_E_PointGroup::C1) X("Ci", OCC_E_PointGroup::Ci)                  \
  X("Cs", OCC_E_PointGroup::Cs) X("C2", OCC_E_PointGroup::C2)                  \
  X("C3", OCC_E_PointGroup::C3) X("C4", OCC_E_PointGroup::C4)                  \
  X("C5", OCC_E_PointGroup::C5) X("C6", OCC_E_PointGroup::C6)                  \
  X("C8", OCC_E_PointGroup::C8) X("Coov", OCC_E_PointGroup::Coov)              \
  X("Dooh", OCC_E_PointGroup::Dooh) X("C2v", OCC_E_PointGroup::C2v)            \
  X("C3v", OCC_E_PointGroup::C3v) X("C4v", OCC_E_PointGroup::C4v)              \
  X("C5v", OCC_E_PointGroup::C5v) X("C6v", OCC_E_PointGroup::C6v)              \
  X("C2h", OCC_E_PointGroup::C2h) X("C3h", OCC_E_PointGroup::C3h)              \
  X("C4h", OCC_E_PointGroup::C4h) X("C5h", OCC_E_PointGroup::C5h)              \
  X("C6h", OCC_E_PointGroup::C6h) X("D2", OCC_E_PointGroup::D2)                \
  X("D3", OCC_E_PointGroup::D3) X("D4", OCC_E_PointGroup::D4)                  \
  X("D5", OCC_E_PointGroup::D5) X("D6", OCC_E_PointGroup::D6)                  \
  X("D7", OCC_E_PointGroup::D7) X("D8", OCC_E_PointGroup::D8)                  \
  X("D2h", OCC_E_PointGroup::D2h) X("D3h", OCC_E_PointGroup::D3h)              \
  X("D4h", OCC_E_PointGroup::D4h) X("D5h", OCC_E_PointGroup::D5h)              \
  X("D6h", OCC_E_PointGroup::D6h) X("D7h", OCC_E_PointGroup::D7h)              \
  X("D8h", OCC_E_PointGroup::D8h) X("D2d", OCC_E_PointGroup::D2d)              \
  X("D3d", OCC_E_PointGroup::D3d) X("D4d", OCC_E_PointGroup::D4d)              \
  X("D5d", OCC_E_PointGroup::D5d) X("D6d", OCC_E_PointGroup::D6d)              \
  X("D7d", OCC_E_PointGroup::D7d) X("D8d", OCC_E_PointGroup::D8d)              \
  X("S4", OCC_E_PointGroup::S4) X("S6", OCC_E_PointGroup::S6)                  \
  X("S8", OCC_E_PointGroup::S8) X("T", OCC_E_PointGroup::T)                    \
  X("Td", OCC_E_PointGroup::Td) X("Th", OCC_E_PointGroup::Th)                  \
  X("O", OCC_E_PointGroup::O) X("Oh", OCC_E_PointGroup::Oh)                    \
  X("I", OCC_E_PointGroup::I) X("Ih", OCC_E_PointGroup::Ih)

// ---------- opt ---------------------------------------------------------

#define OCC_E_BondCoordinateType ::occ::opt::BondCoordinate::Type
#define OCC_ENUM_BondCoordinateType(X)                                         \
  X("COVALENT", OCC_E_BondCoordinateType::COVALENT)                            \
  X("VDW", OCC_E_BondCoordinateType::VDW)

// ---------- isosurface --------------------------------------------------

#define OCC_E_SurfaceKind ::occ::isosurface::SurfaceKind
#define OCC_ENUM_SurfaceKind(X)                                                \
  X("PromoleculeDensity", OCC_E_SurfaceKind::PromoleculeDensity)               \
  X("Hirshfeld", OCC_E_SurfaceKind::Hirshfeld)                                 \
  X("EEQ_ESP", OCC_E_SurfaceKind::EEQ_ESP)                                     \
  X("ElectronDensity", OCC_E_SurfaceKind::ElectronDensity)                     \
  X("ESP", OCC_E_SurfaceKind::ESP)                                             \
  X("SpinDensity", OCC_E_SurfaceKind::SpinDensity)                             \
  X("DeformationDensity", OCC_E_SurfaceKind::DeformationDensity)               \
  X("Orbital", OCC_E_SurfaceKind::Orbital)                                     \
  X("CrystalVoid", OCC_E_SurfaceKind::CrystalVoid)                             \
  X("VolumeGrid", OCC_E_SurfaceKind::VolumeGrid)                               \
  X("SoftVoronoi", OCC_E_SurfaceKind::SoftVoronoi)                             \
  X("VDWLogSumExp", OCC_E_SurfaceKind::VDWLogSumExp)                           \
  X("HSRinv", OCC_E_SurfaceKind::HSRinv)                                       \
  X("HSExp", OCC_E_SurfaceKind::HSExp)

#define OCC_E_PropertyKind ::occ::isosurface::PropertyKind
#define OCC_ENUM_PropertyKind(X)                                               \
  X("Dnorm", OCC_E_PropertyKind::Dnorm)                                        \
  X("Dint_norm", OCC_E_PropertyKind::Dint_norm)                                \
  X("Dext_norm", OCC_E_PropertyKind::Dext_norm)                                \
  X("Dint", OCC_E_PropertyKind::Dint)                                          \
  X("Dext", OCC_E_PropertyKind::Dext)                                          \
  X("FragmentPatch", OCC_E_PropertyKind::FragmentPatch)                        \
  X("ShapeIndex", OCC_E_PropertyKind::ShapeIndex)                              \
  X("Curvedness", OCC_E_PropertyKind::Curvedness)                              \
  X("EEQ_ESP", OCC_E_PropertyKind::EEQ_ESP)                                    \
  X("PromoleculeDensity", OCC_E_PropertyKind::PromoleculeDensity)              \
  X("ESP", OCC_E_PropertyKind::ESP)                                            \
  X("ElectronDensity", OCC_E_PropertyKind::ElectronDensity)                    \
  X("SpinDensity", OCC_E_PropertyKind::SpinDensity)                            \
  X("DeformationDensity", OCC_E_PropertyKind::DeformationDensity)              \
  X("Orbital", OCC_E_PropertyKind::Orbital)                                    \
  X("GaussianCurvature", OCC_E_PropertyKind::GaussianCurvature)                \
  X("MeanCurvature", OCC_E_PropertyKind::MeanCurvature)                        \
  X("CurvatureK1", OCC_E_PropertyKind::CurvatureK1)                            \
  X("CurvatureK2", OCC_E_PropertyKind::CurvatureK2)

#define OCC_E_OrbitalReference ::occ::isosurface::OrbitalIndex::Reference
#define OCC_ENUM_OrbitalReference(X)                                           \
  X("Absolute", OCC_E_OrbitalReference::Absolute)                              \
  X("HOMO", OCC_E_OrbitalReference::HOMO)                                      \
  X("LUMO", OCC_E_OrbitalReference::LUMO)

#define OCC_E_VolumePropertyKind ::occ::isosurface::VolumePropertyKind
#define OCC_ENUM_VolumePropertyKind(X)                                         \
  X("ElectronDensity", OCC_E_VolumePropertyKind::ElectronDensity)              \
  X("ElectronDensityAlpha", OCC_E_VolumePropertyKind::ElectronDensityAlpha)    \
  X("ElectronDensityBeta", OCC_E_VolumePropertyKind::ElectronDensityBeta)      \
  X("ElectricPotential", OCC_E_VolumePropertyKind::ElectricPotential)          \
  X("EEQ_ESP", OCC_E_VolumePropertyKind::EEQ_ESP)                              \
  X("PromoleculeDensity", OCC_E_VolumePropertyKind::PromoleculeDensity)        \
  X("DeformationDensity", OCC_E_VolumePropertyKind::DeformationDensity)        \
  X("XCDensity", OCC_E_VolumePropertyKind::XCDensity)                          \
  X("CrystalVoid", OCC_E_VolumePropertyKind::CrystalVoid)

#define OCC_E_SpinComponent ::occ::isosurface::SpinComponent
#define OCC_ENUM_SpinComponent(X)                                             \
  X("Total", OCC_E_SpinComponent::Total)                                      \
  X("Alpha", OCC_E_SpinComponent::Alpha)                                      \
  X("Beta", OCC_E_SpinComponent::Beta)

// ---------- consumer: register a value list as a Lua sub-namespace ------
// Expands inside a LuaBridge builder chain to
//   .beginNamespace("Name") .addProperty(...) ... .endNamespace()
// Each constant is a zero-arg getter returning the enum value, which
// luabridge::Stack<E> (see enum_stacks.h) pushes as a lua_Integer.

#define OCC_LUA_ENUM_REG_VAL(lua_name, qual_value)                             \
  .addProperty(lua_name, +[]() { return qual_value; })
#define OCC_LUA_ENUM_NAMESPACE(ns_name, LIST)                                  \
  .beginNamespace(ns_name) LIST(OCC_LUA_ENUM_REG_VAL).endNamespace()
