#pragma once
// LuaBridge3 `Stack` specializations for every C++ enum class we expose
// to Lua. Including this header makes the enum type round-trip through
// `lua_Integer` automatically — you can use it directly in
// `addProperty`/`addPropertyReadWrite` on enum-typed struct fields, take
// it as an argument in `addFunction` lambdas, and return it from
// getters. The variadic value list on each `Enum<E, Vals...>` performs
// runtime validation when Lua passes an integer back.
//
// Without these specializations, every binding that touches an enum
// type errors with `"The class is not registered in LuaBridge"`. See
// LuaBridge3 Manual §2.8.2 and `Source/LuaBridge/detail/Enum.h`.
//
// The enumerator lists live in enum_defs.h (the single source of truth,
// shared with the namespace-registration sites in the *_bindings.cpp
// files). Each binding TU that uses an enum-typed property/arg/return
// must include this header so the specialization is visible at the point
// of Stack<E>::push / Stack<E>::get instantiation.

#include <LuaBridge/LuaBridge.h>

#include "enum_defs.h"

// Expand one OCC_ENUM_<Name> list into the value pack of luabridge::Enum.
// The leading comma is intentional: the pack follows `QualType` in the
// Enum<> argument list, so `Enum<QualType , QualType::A , QualType::B>`.
#define OCC_LUA_ENUM_STACK_VAL(lua_name, qual_value) , qual_value
#define OCC_LUA_DEFINE_ENUM_STACK(QualType, LIST)                              \
  template <>                                                                  \
  struct luabridge::Stack<QualType>                                            \
      : luabridge::Enum<QualType LIST(OCC_LUA_ENUM_STACK_VAL)> {};

// ---------- qm ----------------------------------------------------------
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_SpinorbitalKind, OCC_ENUM_SpinorbitalKind)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_OrbitalSmearingKind,
                          OCC_ENUM_OrbitalSmearingKind)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_IntegralEngineDFPolicy,
                          OCC_ENUM_IntegralEngineDFPolicy)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_HessianMethodHF, OCC_ENUM_HessianMethodHF)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_HessianMethodDFT, OCC_ENUM_HessianMethodDFT)

// ---------- xtb ---------------------------------------------------------
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_XtbMethod, OCC_ENUM_XtbMethod)

// ---------- mults -------------------------------------------------------
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_ForceFieldType, OCC_ENUM_ForceFieldType)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_OptimizationMethod, OCC_ENUM_OptimizationMethod)

// ---------- core --------------------------------------------------------
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_MirrorType, OCC_ENUM_MirrorType)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_MoleculeOrder, OCC_ENUM_MoleculeOrder)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_AveragingScheme, OCC_ENUM_AveragingScheme)
// PointGroup is now validated like every other enum (previously it skipped
// the value list, silently accepting any integer — see enum_defs.h).
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_PointGroup, OCC_ENUM_PointGroup)

// ---------- opt ---------------------------------------------------------
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_BondCoordinateType, OCC_ENUM_BondCoordinateType)

// ---------- isosurface --------------------------------------------------
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_SurfaceKind, OCC_ENUM_SurfaceKind)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_PropertyKind, OCC_ENUM_PropertyKind)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_OrbitalReference, OCC_ENUM_OrbitalReference)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_VolumePropertyKind, OCC_ENUM_VolumePropertyKind)
OCC_LUA_DEFINE_ENUM_STACK(OCC_E_SpinComponent, OCC_ENUM_SpinComponent)
