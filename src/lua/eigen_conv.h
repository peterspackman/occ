#pragma once
// Lua ↔ Eigen glue for LuaBridge3 bindings. Eigen matrices and vectors
// cross the boundary as 1-indexed Lua tables (we don't expose them as
// LuaBridge userdata — see [[feedback-luabridge-priorities]]).
//
// Convention:
//   - Vec is a flat 1-indexed table {a, b, c, ...}
//   - Mat is row-major nested tables (outer = rows, inner = cols)
//   - Mat3N is 3×N (3 outer rows = x/y/z, N inner = atoms) — same
//     orientation numpy gets from occpy
//
// Lua 5.4 headers ship without an `extern "C"` wrapper, and LuaBridge3
// enforces "Lua headers must be included prior to LuaBridge ones".
// Pull them in here so every binding TU that includes this header gets
// the order right automatically.

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}
#include <LuaBridge/LuaBridge.h>
// Vector.h provides Stack<std::vector<T>> push/get so bindings can
// return STL vectors directly (e.g. Molecule::atoms() returning
// `std::vector<Atom>`). Pull it in centrally so every binding TU gets
// the support without per-file include.
#include <LuaBridge/Vector.h>
#include <occ/core/linear_algebra.h>
#include <stdexcept>
#include <string>

namespace occ::lua_bindings {

// ---- table → Eigen ---------------------------------------------------------

inline double lua_get_num(const luabridge::LuaRef &t, int i) {
  auto v = t[i].template cast<double>();
  if (!v) {
    throw std::runtime_error("expected number at table index " +
                             std::to_string(i));
  }
  return *v;
}

inline luabridge::LuaRef lua_get_table(const luabridge::LuaRef &t, int i) {
  auto v = t[i].template cast<luabridge::LuaRef>();
  if (!v) {
    throw std::runtime_error("expected nested table at index " +
                             std::to_string(i));
  }
  return *v;
}

inline Vec table_to_vecx(const luabridge::LuaRef &t) {
  const int n = t.length();
  Vec v(n);
  for (int i = 0; i < n; ++i)
    v(i) = lua_get_num(t, i + 1);
  return v;
}

template <int N>
inline Eigen::Matrix<double, N, 1> table_to_vec(const luabridge::LuaRef &t) {
  Eigen::Matrix<double, N, 1> v;
  for (int i = 0; i < N; ++i)
    v(i) = lua_get_num(t, i + 1);
  return v;
}

inline Vec3 table_to_vec3(const luabridge::LuaRef &t) {
  return Vec3(lua_get_num(t, 1), lua_get_num(t, 2), lua_get_num(t, 3));
}

inline IVec3 table_to_ivec3(const luabridge::LuaRef &t) {
  return IVec3(static_cast<int>(lua_get_num(t, 1)),
               static_cast<int>(lua_get_num(t, 2)),
               static_cast<int>(lua_get_num(t, 3)));
}

inline Mat3 table_to_mat3(const luabridge::LuaRef &t) {
  Mat3 m;
  for (int i = 0; i < 3; ++i) {
    luabridge::LuaRef row = lua_get_table(t, i + 1);
    for (int j = 0; j < 3; ++j)
      m(i, j) = lua_get_num(row, j + 1);
  }
  return m;
}

inline Mat4 table_to_mat4(const luabridge::LuaRef &t) {
  Mat4 m;
  for (int i = 0; i < 4; ++i) {
    luabridge::LuaRef row = lua_get_table(t, i + 1);
    for (int j = 0; j < 4; ++j)
      m(i, j) = lua_get_num(row, j + 1);
  }
  return m;
}

inline Mat3N table_to_mat3n(const luabridge::LuaRef &t) {
  const int rows = t.length();
  if (rows == 0)
    return Mat3N(3, 0);
  if (rows != 3) {
    throw std::runtime_error(
        "expected a 3×N table (rows = x/y/z, columns = atoms); got " +
        std::to_string(rows) + " outer rows");
  }
  luabridge::LuaRef first = lua_get_table(t, 1);
  const int n = first.length();
  Mat3N m(3, n);
  for (int i = 0; i < 3; ++i) {
    luabridge::LuaRef row = lua_get_table(t, i + 1);
    for (int j = 0; j < n; ++j)
      m(i, j) = lua_get_num(row, j + 1);
  }
  return m;
}

// Generic nested-table → fixed/dynamic Eigen matrix. Outer = rows.
template <typename Mat>
inline Mat table_to_eigen_matrix(const luabridge::LuaRef &t) {
  using Scalar = typename Mat::Scalar;
  const int rows = t.length();
  if (rows == 0)
    return Mat();
  luabridge::LuaRef first = lua_get_table(t, 1);
  const int cols = first.length();

  Mat m;
  if constexpr (Mat::RowsAtCompileTime == Eigen::Dynamic ||
                Mat::ColsAtCompileTime == Eigen::Dynamic) {
    m.resize(rows, cols);
  } else if (rows != Mat::RowsAtCompileTime || cols != Mat::ColsAtCompileTime) {
    throw std::runtime_error("table shape (" + std::to_string(rows) + "x" +
                             std::to_string(cols) +
                             ") does not match fixed Eigen size");
  }
  for (int i = 0; i < rows; ++i) {
    luabridge::LuaRef row = lua_get_table(t, i + 1);
    for (int j = 0; j < cols; ++j) {
      m(i, j) = static_cast<Scalar>(lua_get_num(row, j + 1));
    }
  }
  return m;
}

// ---- Eigen → table ---------------------------------------------------------

template <typename Derived>
inline luabridge::LuaRef vec_to_table(lua_State *L,
                                      const Eigen::MatrixBase<Derived> &v) {
  luabridge::LuaRef t = luabridge::newTable(L);
  for (Eigen::Index i = 0; i < v.size(); ++i)
    t[i + 1] = v(i);
  return t;
}

template <typename Derived>
inline luabridge::LuaRef mat_to_table(lua_State *L,
                                      const Eigen::MatrixBase<Derived> &m) {
  luabridge::LuaRef t = luabridge::newTable(L);
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    luabridge::LuaRef row = luabridge::newTable(L);
    for (Eigen::Index j = 0; j < m.cols(); ++j)
      row[j + 1] = m(i, j);
    t[i + 1] = row;
  }
  return t;
}

} // namespace occ::lua_bindings
