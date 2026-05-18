#pragma once
// Generic Eigen matrix / vector ↔ Lua userdata layer.
//
// Every `Mat`, `Mat3N`, `Mat3`, ..., `Vec`, `IVec`, ... we want exposed to
// Lua gets registered once via `register_matrix_userdata<T>` /
// `register_vector_userdata<T>`. The result:
//
//   * binding lambdas can return Eigen values directly — sol2 pushes them
//     as opaque userdata, no copy through a Lua table;
//   * binding lambdas can take `const Mat &` / `Mat3N &` / etc. as
//     parameters — the stack getter below auto-converts from either a
//     matching userdata or a nested Lua table fallback;
//   * `pos[i]` returns a row proxy that aliases back into the matrix
//     storage; `pos[i][j]` reads, `pos[i][j] = x` writes in place;
//   * `#pos`, `pos:rows()`, `pos:cols()`, `pos:to_table()`, `pos:get(i,j)`,
//     `pos:set(i,j,x)`, `tostring(pos)` all work;
//   * Lua 1-indexed throughout (translated to Eigen's 0-indexed at the
//     boundary).
//
// The matrix shape convention is Eigen's: rows are the outer dimension
// (`pos[1]` = first row), columns are the inner. For Mat3N that's
// xyz-as-rows, atoms-as-columns — same as numpy when nanobind hands a
// Mat3N to Python.

#include "eigen_conv.h"
#include <fmt/core.h>
#include <fmt/format.h>
#include <sstream>

#include <occ/core/format_matrix.h>
namespace occ::lua_bindings {

// Thin handle into a parent matrix. The pointer is only safe while the
// parent matrix's Lua userdata is alive — capturing a row to a variable
// and then letting the parent get GC'd is undefined. For typical use
// (`pos[1][2]`, `for j=1,#pos[1] do ... end`) the parent is anchored on
// the Lua stack so this isn't a footgun.
template <typename Mat> struct MatrixRow {
  Mat *mat;
  int row; // 0-indexed
};

// Build an Eigen matrix from a nested 1-indexed Lua table (outer = rows,
// inner = cols). For fixed-size types, the table dimensions must match.
template <typename Mat>
inline Mat table_to_eigen_matrix(const sol::table &t) {
  using Scalar = typename Mat::Scalar;
  const int rows = static_cast<int>(t.size());
  if (rows == 0) return Mat();
  sol::table first = t.get<sol::table>(1);
  const int cols = static_cast<int>(first.size());

  Mat m;
  if constexpr (Mat::RowsAtCompileTime == Eigen::Dynamic ||
                Mat::ColsAtCompileTime == Eigen::Dynamic) {
    m.resize(rows, cols);
  } else {
    if (rows != Mat::RowsAtCompileTime || cols != Mat::ColsAtCompileTime) {
      // Cast the Eigen compile-time-traits enums to plain ints before
      // handing them to fmt — the enum type is `CompileTimeTraits`
      // which fmt won't format directly.
      throw std::runtime_error(fmt::format(
          "table shape ({}×{}) does not match fixed Eigen size ({}×{})",
          rows, cols, static_cast<int>(Mat::RowsAtCompileTime),
          static_cast<int>(Mat::ColsAtCompileTime)));
    }
  }
  for (int i = 0; i < rows; ++i) {
    sol::table row = t.get<sol::table>(i + 1);
    for (int j = 0; j < cols; ++j) m(i, j) = row.get<Scalar>(j + 1);
  }
  return m;
}

// Build an Eigen vector from a flat 1-indexed Lua table.
template <typename Vec>
inline Vec table_to_eigen_vector(const sol::table &t) {
  using Scalar = typename Vec::Scalar;
  const int n = static_cast<int>(t.size());
  Vec v;
  if constexpr (Vec::SizeAtCompileTime == Eigen::Dynamic) {
    v.resize(n);
  } else if (n != Vec::SizeAtCompileTime) {
    throw std::runtime_error(fmt::format(
        "table length {} does not match fixed Eigen size {}", n,
        static_cast<int>(Vec::SizeAtCompileTime)));
  }
  for (int i = 0; i < n; ++i) v(i) = t.get<Scalar>(i + 1);
  return v;
}

// Pretty-format an Eigen matrix for `tostring(m)` / `print(m)` output.
// We go through `fmt::runtime` so fmt's constexpr format-string check
// doesn't try to introspect the Eigen scalar expression type — it gets
// a plain `double` / `long long` value after the cast.
template <typename Mat>
inline std::string format_matrix(const Mat &m, const std::string &kind) {
  std::ostringstream ss;
  ss << kind << "[" << m.rows() << "x" << m.cols() << "]";
  if (m.size() == 0) return ss.str();
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
inline std::string format_vector(const Vec &v, const std::string &kind) {
  std::ostringstream ss;
  ss << kind << "[" << v.size() << "] [";
  for (int i = 0; i < v.size(); ++i) {
    if constexpr (std::is_integral_v<typename Vec::Scalar>) {
      ss << fmt::format(fmt::runtime(" {:>10d}"),
                        static_cast<long long>(v(i)));
    } else {
      ss << fmt::format(fmt::runtime(" {: 12.6f}"),
                        static_cast<double>(v(i)));
    }
  }
  ss << " ]";
  return ss.str();
}

// Register a matrix usertype and its row-proxy companion. Pass a unique
// `name` per Eigen instantiation (e.g. "Matrix" for Mat, "Mat3N" for Mat3N).
template <typename Mat>
void register_matrix_userdata(sol::table &occ_module,
                               const std::string &name) {
  using Scalar = typename Mat::Scalar;
  using Row = MatrixRow<Mat>;

  // Row proxy: `pos[i][j]` reads/writes the underlying matrix in place.
  // 1-indexed in Lua; translated to Eigen's 0-indexed on the C++ side.
  occ_module.new_usertype<Row>(
      name + "Row", sol::no_constructor,
      sol::meta_function::index,
      [](const Row &r, int j) {
        if (j < 1 || j > r.mat->cols()) return Scalar{};
        return (*r.mat)(r.row, j - 1);
      },
      sol::meta_function::new_index,
      [](Row &r, int j, Scalar v) {
        if (j < 1 || j > r.mat->cols()) {
          throw std::runtime_error("row index out of range");
        }
        (*r.mat)(r.row, j - 1) = v;
      },
      sol::meta_function::length,
      [](const Row &r) { return static_cast<int>(r.mat->cols()); },
      sol::meta_function::to_string, [name](const Row &r) {
        std::ostringstream ss;
        ss << name << "Row[" << (r.row + 1) << "] [";
        for (int j = 0; j < r.mat->cols(); ++j) {
          if constexpr (std::is_integral_v<Scalar>) {
            ss << fmt::format(" {:>10d}", (*r.mat)(r.row, j));
          } else {
            ss << fmt::format(" {: 12.6f}",
                              static_cast<double>((*r.mat)(r.row, j)));
          }
        }
        ss << " ]";
        return ss.str();
      });

  // Methods are stored in a separate Lua table that the index lambda
  // falls through to for string keys (integer keys → row proxy).
  sol::state_view lua(occ_module.lua_state());
  sol::table methods = lua.create_table();
  methods["rows"] = [](const Mat &m) { return static_cast<int>(m.rows()); };
  methods["cols"] = [](const Mat &m) { return static_cast<int>(m.cols()); };
  methods["size"] = [](const Mat &m) { return static_cast<int>(m.size()); };
  methods["get"] = [](const Mat &m, int i, int j) {
    return m(i - 1, j - 1);
  };
  methods["set"] = [](Mat &m, int i, int j, Scalar v) {
    m(i - 1, j - 1) = v;
  };
  methods["fill"] = [](Mat &m, Scalar v) { m.setConstant(v); };
  methods["zero"] = [](Mat &m) { m.setZero(); };
  methods["to_table"] = [](const Mat &m, sol::this_state s) {
    return mat_to_table(s, m);
  };

  occ_module.new_usertype<Mat>(
      name,
      sol::call_constructor,
      sol::factories(
          []() { return Mat(); },
          [](int rows, int cols) { return Mat(rows, cols); },
          [](const sol::table &t) { return table_to_eigen_matrix<Mat>(t); }),
      sol::meta_function::index,
      [methods](Mat &m, sol::object key, sol::this_state s) -> sol::object {
        sol::state_view lua(s);
        if (key.is<int>()) {
          int i = key.as<int>();
          if (i < 1 || i > m.rows()) return sol::lua_nil;
          return sol::make_object(lua, Row{&m, i - 1});
        }
        return methods[key];
      },
      sol::meta_function::new_index,
      [](Mat &m, sol::object key, sol::object value) {
        if (!key.is<int>()) {
          throw std::runtime_error(
              "matrix assignment expects an integer row index");
        }
        const int i = key.as<int>();
        if (i < 1 || i > m.rows()) {
          throw std::runtime_error("matrix row index out of range");
        }
        // Whole-row assignment: m[i] = {a, b, c}
        if (!value.is<sol::table>()) {
          throw std::runtime_error(
              "matrix row assignment expects a numeric table");
        }
        sol::table row = value.as<sol::table>();
        for (int j = 0; j < m.cols(); ++j) {
          m(i - 1, j) = row.get<Scalar>(j + 1);
        }
      },
      sol::meta_function::length,
      [](const Mat &m) { return static_cast<int>(m.rows()); },
      sol::meta_function::to_string,
      [name](const Mat &m) { return format_matrix(m, name); });
}

// Vector counterpart — scalar at integer keys, write via `v[i] = x`.
template <typename Vec>
void register_vector_userdata(sol::table &occ_module,
                               const std::string &name) {
  using Scalar = typename Vec::Scalar;

  sol::state_view lua(occ_module.lua_state());
  sol::table methods = lua.create_table();
  methods["size"] = [](const Vec &v) { return static_cast<int>(v.size()); };
  methods["get"] = [](const Vec &v, int i) { return v(i - 1); };
  methods["set"] = [](Vec &v, int i, Scalar x) { v(i - 1) = x; };
  methods["fill"] = [](Vec &v, Scalar x) { v.setConstant(x); };
  methods["zero"] = [](Vec &v) { v.setZero(); };
  methods["to_table"] = [](const Vec &v, sol::this_state s) {
    return vec_to_table(s, v);
  };
  methods["sum"] = [](const Vec &v) { return v.sum(); };
  if constexpr (std::is_floating_point_v<Scalar>) {
    methods["norm"] = [](const Vec &v) { return v.norm(); };
    methods["mean"] = [](const Vec &v) { return v.mean(); };
  }

  occ_module.new_usertype<Vec>(
      name,
      sol::call_constructor,
      sol::factories(
          []() { return Vec(); },
          [](int n) { return Vec(n); },
          [](const sol::table &t) { return table_to_eigen_vector<Vec>(t); }),
      sol::meta_function::index,
      [methods](Vec &v, sol::object key, sol::this_state s) -> sol::object {
        sol::state_view lua(s);
        if (key.is<int>()) {
          int i = key.as<int>();
          if (i < 1 || i > v.size()) return sol::lua_nil;
          return sol::make_object(lua, v(i - 1));
        }
        return methods[key];
      },
      sol::meta_function::new_index,
      [](Vec &v, int i, Scalar x) {
        if (i < 1 || i > v.size()) {
          throw std::runtime_error("vector index out of range");
        }
        v(i - 1) = x;
      },
      sol::meta_function::length,
      [](const Vec &v) { return static_cast<int>(v.size()); },
      sol::meta_function::to_string,
      [name](const Vec &v) { return format_vector(v, name); });
}

// Register every Eigen matrix / vector type we expose to Lua. Called
// once during `open_occ_module`. Adding a new occ type means adding one
// line here.
void register_eigen_matrix_types(sol::table &occ_module);

} // namespace occ::lua_bindings

// The `is_automagical` / `is_container` opt-outs for `Eigen::Matrix` live
// in eigen_conv.h (included above) so every binding file sees them.
