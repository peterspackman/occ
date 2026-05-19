#pragma once
// LuaBridge3 wrappers for the Eigen types exposed to Lua, with enough
// ergonomics to make the existing examples (examples/lua/*.lua) work
// without rewriting:
//
//   - `m[i][j]` reads / writes through a thin row-proxy userdata
//   - `m[i] = {a, b, c}` whole-row table assignment
//   - `v[i]` integer indexing on vectors (read + write)
//   - `#m` returns rows, `#v` returns size
//   - constructors from nested Lua tables (matrix) / flat tables (vector)
//   - `mat:to_table()` / `vec:to_table()` opt-in copy to Lua tables
//
// This is intentionally leaner than the sol2-era version: no sol2-style
// `is_container` / `is_automagical` opt-outs, no ADL push hooks for
// Eigen expression templates. Where a binding returns an Eigen
// expression template (Block, Product, …) the binding-site lambda must
// materialize to a concrete `Mat3N` / `Vec3` / etc. before returning.

#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/format_matrix.h>
#include <sstream>
#include <string>
#include <type_traits>

namespace occ::lua_bindings {

// Thin handle into a parent matrix. The pointer is only safe while the
// parent's Lua userdata is alive — Lua tables holding both the parent
// and a row are fine; outliving the parent is undefined.
template <typename Mat> struct MatrixRow {
  Mat *mat;
  int row; // 0-indexed
};

namespace eigen_matrix_detail {

template <typename Mat>
inline std::string format_matrix_str(const Mat &m, const std::string &kind) {
  std::ostringstream ss;
  ss << kind << "[" << m.rows() << "x" << m.cols() << "]";
  if (m.size() == 0)
    return ss.str();
  ss << "\n";
  for (int i = 0; i < m.rows(); ++i) {
    ss << "  [" << (i + 1) << "]";
    for (int j = 0; j < m.cols(); ++j) {
      if constexpr (std::is_integral_v<typename Mat::Scalar>) {
        ss << fmt::format(fmt::runtime(" {:>10d}"),
                          static_cast<long long>(m(i, j)));
      } else {
        ss << fmt::format(fmt::runtime(" {: 12.6f}"),
                          static_cast<double>(m(i, j)));
      }
    }
    ss << "\n";
  }
  return ss.str();
}

template <typename Vec>
inline std::string format_vector_str(const Vec &v, const std::string &kind) {
  std::ostringstream ss;
  ss << kind << "[" << v.size() << "] [";
  for (int i = 0; i < v.size(); ++i) {
    if constexpr (std::is_integral_v<typename Vec::Scalar>) {
      ss << fmt::format(fmt::runtime(" {:>10d}"), static_cast<long long>(v(i)));
    } else {
      ss << fmt::format(fmt::runtime(" {: 12.6f}"), static_cast<double>(v(i)));
    }
  }
  ss << " ]";
  return ss.str();
}

// Build a fixed/dynamic Eigen matrix from a nested 1-indexed Lua table.
template <typename Mat>
inline Mat *build_matrix_from_table(const luabridge::LuaRef &t) {
  using Scalar = typename Mat::Scalar;
  Mat *m = new Mat();
  const int rows = t.length();
  if (rows == 0)
    return m;
  auto first = t[1].template cast<luabridge::LuaRef>();
  if (!first) {
    delete m;
    throw std::runtime_error("matrix constructor: nested table expected");
  }
  const int cols = first->length();
  if constexpr (Mat::RowsAtCompileTime == Eigen::Dynamic ||
                Mat::ColsAtCompileTime == Eigen::Dynamic) {
    m->resize(rows, cols);
  } else if (rows != Mat::RowsAtCompileTime || cols != Mat::ColsAtCompileTime) {
    delete m;
    throw std::runtime_error("matrix constructor: shape " +
                             std::to_string(rows) + "x" + std::to_string(cols) +
                             " does not match fixed type");
  }
  for (int i = 0; i < rows; ++i) {
    auto row = t[i + 1].template cast<luabridge::LuaRef>();
    if (!row) {
      delete m;
      throw std::runtime_error("matrix constructor: nested table expected");
    }
    for (int j = 0; j < cols; ++j) {
      (*m)(i, j) = static_cast<Scalar>(
          row->operator[](j + 1).template cast<double>().valueOr(0.0));
    }
  }
  return m;
}

template <typename Vec>
inline Vec *build_vector_from_table(const luabridge::LuaRef &t) {
  using Scalar = typename Vec::Scalar;
  Vec *v = new Vec();
  const int n = t.length();
  if constexpr (Vec::SizeAtCompileTime == Eigen::Dynamic) {
    v->resize(n);
  } else if (n != Vec::SizeAtCompileTime) {
    delete v;
    throw std::runtime_error("vector constructor: length " + std::to_string(n) +
                             " does not match fixed type");
  }
  for (int i = 0; i < n; ++i) {
    (*v)(i) =
        static_cast<Scalar>(t[i + 1].template cast<double>().valueOr(0.0));
  }
  return v;
}

} // namespace eigen_matrix_detail

// Register a matrix type T as a LuaBridge3 class. `name` becomes the
// script-side identifier (e.g. "Mat3N", "Matrix").
template <typename Mat>
void register_matrix_userdata(lua_State *L, const std::string &name) {
  using Scalar = typename Mat::Scalar;
  using Row = MatrixRow<Mat>;
  namespace lb = luabridge;

  // Row proxy: m[i] returns one of these; m[i][j] reads / writes the
  // underlying matrix in place. We register MatrixRow<Mat> once per Mat
  // type — its name is `<MatName>Row`.
  const std::string row_name = name + "Row";
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .template beginClass<Row>(row_name.c_str())
      .addFunction(
          "__len",
          +[](const Row *r) { return static_cast<int>(r->mat->cols()); })
      .addIndexMetaMethod(
          +[](Row &r, const lb::LuaRef &key, lua_State *S) -> lb::LuaRef {
            int j = -1;
            if (key.isNumber()) {
              j = key.unsafe_cast<int>();
            } else if (key.isString()) {
              try {
                j = std::stoi(key.unsafe_cast<std::string>());
              } catch (...) {
                return lb::LuaRef(S);
              }
            } else {
              return lb::LuaRef(S);
            }
            if (j < 1 || j > r.mat->cols())
              return lb::LuaRef(S);
            return lb::LuaRef(S, (*r.mat)(r.row, j - 1));
          })
      .addNewIndexMetaMethod(+[](Row &r, const lb::LuaRef &key,
                                 const lb::LuaRef &value,
                                 lua_State *S) -> lb::LuaRef {
        int j = -1;
        if (key.isNumber()) {
          j = key.unsafe_cast<int>();
        } else if (key.isString()) {
          try {
            j = std::stoi(key.unsafe_cast<std::string>());
          } catch (...) {
            luaL_error(S, "row index must be integer");
            return lb::LuaRef(S);
          }
        } else {
          luaL_error(S, "row index must be integer");
          return lb::LuaRef(S);
        }
        if (j < 1 || j > r.mat->cols()) {
          luaL_error(S, "row index out of range");
          return lb::LuaRef(S);
        }
        (*r.mat)(r.row, j - 1) =
            static_cast<Scalar>(value.unsafe_cast<double>());
        return lb::LuaRef(S);
      })
      .addFunction(
          "__tostring",
          [row_name](const Row *r) {
            std::ostringstream ss;
            ss << row_name << "[" << (r->row + 1) << "] [";
            for (int j = 0; j < r->mat->cols(); ++j) {
              if constexpr (std::is_integral_v<Scalar>) {
                ss << fmt::format(" {:>10d}",
                                  static_cast<long long>((*r->mat)(r->row, j)));
              } else {
                ss << fmt::format(" {: 12.6f}",
                                  static_cast<double>((*r->mat)(r->row, j)));
              }
            }
            ss << " ]";
            return ss.str();
          })
      .endClass()

      .template beginClass<Mat>(name.c_str())
      .template addConstructor<void (*)()>()
      .addStaticFunction(
          "from_size",
          +[](int rows, int cols) {
            if constexpr (Mat::RowsAtCompileTime == Eigen::Dynamic ||
                          Mat::ColsAtCompileTime == Eigen::Dynamic) {
              return new Mat(rows, cols);
            } else {
              if (rows != Mat::RowsAtCompileTime ||
                  cols != Mat::ColsAtCompileTime) {
                throw std::runtime_error("fixed-size matrix: shape mismatch");
              }
              return new Mat();
            }
          })
      .addStaticFunction(
          "from_table",
          +[](const lb::LuaRef &t) {
            return eigen_matrix_detail::build_matrix_from_table<Mat>(t);
          })
      .addFunction(
          "rows", +[](const Mat *m) { return static_cast<int>(m->rows()); })
      .addFunction(
          "cols", +[](const Mat *m) { return static_cast<int>(m->cols()); })
      .addFunction(
          "size", +[](const Mat *m) { return static_cast<int>(m->size()); })
      .addFunction(
          "__len", +[](const Mat *m) { return static_cast<int>(m->rows()); })
      .addFunction(
          "get", +[](const Mat *m, int i, int j) { return (*m)(i - 1, j - 1); })
      .addFunction(
          "set",
          +[](Mat *m, int i, int j, Scalar v) { (*m)(i - 1, j - 1) = v; })
      .addFunction(
          "fill", +[](Mat *m, Scalar v) { m->setConstant(v); })
      .addFunction(
          "zero", +[](Mat *m) { m->setZero(); })
      .addFunction(
          "to_table",
          +[](const Mat *m, lua_State *S) { return mat_to_table(S, *m); })
      // LuaBridge3 normalizes __index keys to strings before
      // dispatching to the fallback (verified empirically — integer
      // 1 arrives here as LUA_TSTRING "1"). We detect both shapes.
      .addIndexMetaMethod(
          +[](Mat &m, const lb::LuaRef &key, lua_State *S) -> lb::LuaRef {
            int i = -1;
            if (key.isNumber()) {
              i = key.unsafe_cast<int>();
            } else if (key.isString()) {
              try {
                i = std::stoi(key.unsafe_cast<std::string>());
              } catch (...) {
                return lb::LuaRef(S);
              }
            } else {
              return lb::LuaRef(S);
            }
            if (i < 1 || i > m.rows())
              return lb::LuaRef(S);
            return lb::LuaRef(S, Row{&m, i - 1});
          })
      .addNewIndexMetaMethod(+[](Mat &m, const lb::LuaRef &key,
                                 const lb::LuaRef &value,
                                 lua_State *S) -> lb::LuaRef {
        int i = -1;
        if (key.isNumber()) {
          i = key.unsafe_cast<int>();
        } else if (key.isString()) {
          try {
            i = std::stoi(key.unsafe_cast<std::string>());
          } catch (...) {
            luaL_error(S, "matrix row index must be integer");
            return lb::LuaRef(S);
          }
        } else {
          luaL_error(S, "matrix row index must be integer");
          return lb::LuaRef(S);
        }
        if (i < 1 || i > m.rows()) {
          luaL_error(S, "matrix row index out of range");
          return lb::LuaRef(S);
        }
        if (!value.isTable()) {
          luaL_error(S, "matrix row assignment expects a numeric table");
          return lb::LuaRef(S);
        }
        for (int j = 0; j < m.cols(); ++j) {
          m(i - 1, j) = static_cast<Scalar>(
              value[j + 1].template cast<double>().valueOr(0.0));
        }
        return lb::LuaRef(S);
      })
      .addFunction("__tostring",
                   [name](const Mat *m) {
                     return eigen_matrix_detail::format_matrix_str(*m, name);
                   })
      .endClass()
      .endNamespace();
}

template <typename Vec>
void register_vector_userdata(lua_State *L, const std::string &name) {
  using Scalar = typename Vec::Scalar;
  namespace lb = luabridge;

  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .template beginClass<Vec>(name.c_str())
      .template addConstructor<void (*)()>()
      .addStaticFunction(
          "from_size",
          +[](int n) {
            if constexpr (Vec::SizeAtCompileTime == Eigen::Dynamic) {
              return new Vec(n);
            } else {
              if (n != Vec::SizeAtCompileTime) {
                throw std::runtime_error("fixed-size vector: length mismatch");
              }
              return new Vec();
            }
          })
      .addStaticFunction(
          "from_table",
          +[](const lb::LuaRef &t) {
            return eigen_matrix_detail::build_vector_from_table<Vec>(t);
          })
      .addFunction(
          "size", +[](const Vec *v) { return static_cast<int>(v->size()); })
      .addFunction(
          "__len", +[](const Vec *v) { return static_cast<int>(v->size()); })
      .addFunction(
          "get", +[](const Vec *v, int i) { return (*v)(i - 1); })
      .addFunction(
          "set", +[](Vec *v, int i, Scalar x) { (*v)(i - 1) = x; })
      .addFunction(
          "fill", +[](Vec *v, Scalar x) { v->setConstant(x); })
      .addFunction(
          "zero", +[](Vec *v) { v->setZero(); })
      .addFunction(
          "to_table",
          +[](const Vec *v, lua_State *S) { return vec_to_table(S, *v); })
      .addFunction(
          "sum", +[](const Vec *v) { return v->sum(); })
      .addIndexMetaMethod(
          +[](Vec &v, const lb::LuaRef &key, lua_State *S) -> lb::LuaRef {
            int i = -1;
            if (key.isNumber()) {
              i = key.unsafe_cast<int>();
            } else if (key.isString()) {
              try {
                i = std::stoi(key.unsafe_cast<std::string>());
              } catch (...) {
                return lb::LuaRef(S);
              }
            } else {
              return lb::LuaRef(S);
            }
            if (i < 1 || i > v.size())
              return lb::LuaRef(S);
            return lb::LuaRef(S, v(i - 1));
          })
      .addNewIndexMetaMethod(+[](Vec &v, const lb::LuaRef &key,
                                 const lb::LuaRef &value,
                                 lua_State *S) -> lb::LuaRef {
        int i = -1;
        if (key.isNumber()) {
          i = key.unsafe_cast<int>();
        } else if (key.isString()) {
          try {
            i = std::stoi(key.unsafe_cast<std::string>());
          } catch (...) {
            luaL_error(S, "vector index must be integer");
            return lb::LuaRef(S);
          }
        } else {
          luaL_error(S, "vector index must be integer");
          return lb::LuaRef(S);
        }
        if (i < 1 || i > v.size()) {
          luaL_error(S, "vector index out of range");
          return lb::LuaRef(S);
        }
        v(i - 1) = static_cast<Scalar>(value.unsafe_cast<double>());
        return lb::LuaRef(S);
      })
      .addFunction("__tostring",
                   [name](const Vec *v) {
                     return eigen_matrix_detail::format_vector_str(*v, name);
                   })
      .endClass()
      .endNamespace();

  // Floating-point only conveniences. Re-open the same class to add
  // norm/mean — LuaBridge3 merges the registrations.
  if constexpr (std::is_floating_point_v<Scalar>) {
    lb::getGlobalNamespace(L)
        .beginNamespace("occ")
        .template beginClass<Vec>(name.c_str())
        .addFunction(
            "norm", +[](const Vec *v) { return v->norm(); })
        .addFunction(
            "mean", +[](const Vec *v) { return v->mean(); })
        .endClass()
        .endNamespace();
  }
}

void register_eigen_matrix_types(lua_State *L);

} // namespace occ::lua_bindings
