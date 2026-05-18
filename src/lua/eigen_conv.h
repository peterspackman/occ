#pragma once
#include <occ/core/linear_algebra.h>
#include <sol/sol.hpp>

// Eigen::Matrix (since 3.4) provides begin()/end(), which sol2's automatic
// container detection latches onto via `meta::has_begin_end_v`. The
// container path then tries to build iterator machinery that fails to
// compile because Eigen iterators dereference to a *reference type* that
// sol2's `decltype(*it)` plumbing can't decompose ("cannot form a reference
// to void"). We never want sol2 to expose Eigen matrices as Lua containers
// — `vec_to_table` / `mat_to_table` already do explicit conversion — so
// opt every Eigen::Matrix specialization out.
//
// We also have to opt `as_container_t<Eigen::Matrix<...>>` out: sol2's
// automagic enrollment unconditionally instantiates a `pairs` registration
// of the form `u_c_launch<as_container_t<T>>` for *every* usertype T (see
// usertype_core.hpp:152), and when ADL hunts for `sol_lua_get` on that
// type it tries to build the `as_container_t<Eigen::Matrix>::iter`
// machinery — same void cascade. The opt-out tells sol2 "treat this as an
// opaque value, never as a container".
namespace sol {
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct is_container<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::false_type {};
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct is_container<
    const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::false_type {};
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct is_container<sol::as_container_t<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>>
    : std::false_type {};
// Also opt every Eigen::Matrix specialization out of sol2's "automagic"
// usertype enrollment — when binding code RETURNS an Eigen matrix sol2
// would otherwise try to set up pairs/call-operator metamethods on it,
// triggering the same Eigen-iterator void cascade. We register the
// matrix types as usertypes manually via `register_matrix_userdata` in
// eigen_matrix.h.
template <typename Scalar, int Rows, int Cols, int O, int MR, int MC>
struct is_automagical<Eigen::Matrix<Scalar, Rows, Cols, O, MR, MC>>
    : std::false_type {};
} // namespace sol

// ----- Eigen expression-template pusher -----------------------------------
//
// Most occ methods that return matrices materialize them into a concrete
// `Mat`/`Mat3N`/etc. type (which sol2 then pushes via the registered
// usertype). But several accessor-style methods in occ are declared as
// `inline auto rotation() const { return m_seitz.block<3,3>(0,0); }` —
// their return type is an Eigen *expression* (Block, Product, etc.), not
// a concrete Matrix. sol2 sees an unknown type, falls back to automagic +
// container probing, and hits the "cannot form a reference to void"
// cascade because Eigen iterators don't satisfy sol2's `decltype(*it)`.
//
// Rather than chase every such site with `-> Mat3` annotations, we tell
// sol2 how to push an Eigen expression by evaluating it first: the
// concrete type is `Eigen::internal::eval<Expr>::type`, which the
// registered usertype pusher then handles.

// Use the ADL-based customization hook (`sol_lua_push` in the same
// namespace as the type) rather than a partial specialization of
// `unqualified_pusher` — the latter is ambiguous with sol2's existing
// container-detected specialization. Putting the hook in namespace
// Eigen means ADL on any Eigen::Product / Block / etc. picks it up
// before sol2 falls back to its default machinery.
namespace Eigen {

namespace occ_lua_detail {
template <typename E>
inline int push_via_eval(lua_State *L, const E &expr) {
  using Eval = typename Eigen::internal::eval<E>::type;
  return sol::stack::push<Eval>(L, Eval(expr));
}
} // namespace occ_lua_detail

template <typename Lhs, typename Rhs, int Option>
inline int sol_lua_push(sol::types<Eigen::Product<Lhs, Rhs, Option>>,
                         lua_State *L,
                         const Eigen::Product<Lhs, Rhs, Option> &expr) {
  return occ_lua_detail::push_via_eval(L, expr);
}

template <typename XprType, int BR, int BC, bool IP>
inline int
sol_lua_push(sol::types<Eigen::Block<XprType, BR, BC, IP>>, lua_State *L,
              const Eigen::Block<XprType, BR, BC, IP> &expr) {
  return occ_lua_detail::push_via_eval(L, expr);
}

template <typename UnaryOp, typename XprType>
inline int sol_lua_push(sol::types<Eigen::CwiseUnaryOp<UnaryOp, XprType>>,
                         lua_State *L,
                         const Eigen::CwiseUnaryOp<UnaryOp, XprType> &expr) {
  return occ_lua_detail::push_via_eval(L, expr);
}

template <typename BinaryOp, typename Lhs, typename Rhs>
inline int sol_lua_push(
    sol::types<Eigen::CwiseBinaryOp<BinaryOp, Lhs, Rhs>>, lua_State *L,
    const Eigen::CwiseBinaryOp<BinaryOp, Lhs, Rhs> &expr) {
  return occ_lua_detail::push_via_eval(L, expr);
}

template <typename XprType, int Direction>
inline int sol_lua_push(sol::types<Eigen::Reverse<XprType, Direction>>,
                         lua_State *L,
                         const Eigen::Reverse<XprType, Direction> &expr) {
  return occ_lua_detail::push_via_eval(L, expr);
}

} // namespace Eigen

namespace occ::lua_bindings {

// Convert an Eigen vector to a 1-indexed Lua table. Takes `sol::this_state`
// so callers can use it directly inside binding lambdas without having to
// capture a sol::state_view (which would dangle once the registrar
// function returns).
template <typename Derived>
sol::table vec_to_table(sol::this_state s,
                        const Eigen::MatrixBase<Derived> &v) {
  sol::state_view lua(s);
  sol::table t = lua.create_table(static_cast<int>(v.size()), 0);
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    t[i + 1] = v(i);
  }
  return t;
}

// Convert an Eigen matrix to a row-major table-of-tables (1-indexed).
template <typename Derived>
sol::table mat_to_table(sol::this_state s,
                        const Eigen::MatrixBase<Derived> &m) {
  sol::state_view lua(s);
  sol::table t = lua.create_table(static_cast<int>(m.rows()), 0);
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    sol::table row = lua.create_table(static_cast<int>(m.cols()), 0);
    for (Eigen::Index j = 0; j < m.cols(); ++j) {
      row[j + 1] = m(i, j);
    }
    t[i + 1] = row;
  }
  return t;
}

// Pull a 1-indexed numeric table into a fixed-size Eigen vector.
template <int N> Eigen::Matrix<double, N, 1> table_to_vec(const sol::table &t) {
  Eigen::Matrix<double, N, 1> v;
  for (int i = 0; i < N; ++i) v(i) = t.get<double>(i + 1);
  return v;
}

// Pull a 1-indexed numeric table into a dynamic-size Eigen vector.
inline Vec table_to_vecx(const sol::table &t) {
  const int n = static_cast<int>(t.size());
  Vec v(n);
  for (int i = 0; i < n; ++i) v(i) = t.get<double>(i + 1);
  return v;
}

// Mat3N convention in Lua: 3×N column-major, matching the underlying
// Eigen storage and the Python (numpy) view. `t[1]` / `t[2]` / `t[3]`
// are the x / y / z rows; `t[1][j]` is x of atom j. The earlier N×3
// "atoms-as-rows" attempt hit a fatal ambiguity for 3-atom systems
// (gradients silently got transposed on round-trip), so we go with the
// Eigen-native shape and use the generic `mat_to_table` everywhere.

// Pull a 3×N nested table (3 outer rows of N inner) back into a Mat3N.
// Empty table → 3×0. Throws if any inner row's length disagrees.
inline Mat3N table_to_mat3n(const sol::table &t) {
  const int rows = static_cast<int>(t.size());
  if (rows == 0) return Mat3N(3, 0);
  if (rows != 3) {
    throw std::runtime_error(
        "expected a 3×N table (rows = x/y/z, columns = atoms); got " +
        std::to_string(rows) + " outer rows");
  }
  sol::table first = t.get<sol::table>(1);
  const int n = static_cast<int>(first.size());
  Mat3N m(3, n);
  for (int i = 0; i < 3; ++i) {
    sol::table row = t.get<sol::table>(i + 1);
    for (int j = 0; j < n; ++j) m(i, j) = row.get<double>(j + 1);
  }
  return m;
}

// Pull a 3×3 row-major table into a Mat3.
inline Mat3 table_to_mat3(const sol::table &t) {
  Mat3 m;
  for (int i = 0; i < 3; ++i) {
    sol::table row = t.get<sol::table>(i + 1);
    for (int j = 0; j < 3; ++j) m(i, j) = row.get<double>(j + 1);
  }
  return m;
}

// Pull a 4×4 row-major table into a Mat4.
inline Mat4 table_to_mat4(const sol::table &t) {
  Mat4 m;
  for (int i = 0; i < 4; ++i) {
    sol::table row = t.get<sol::table>(i + 1);
    for (int j = 0; j < 4; ++j) m(i, j) = row.get<double>(j + 1);
  }
  return m;
}

// Pull a 1-indexed 3-element table into a Vec3.
inline Vec3 table_to_vec3(const sol::table &t) {
  return Vec3(t.get<double>(1), t.get<double>(2), t.get<double>(3));
}

// Pull a 1-indexed 3-element integer table into an IVec3.
inline IVec3 table_to_ivec3(const sol::table &t) {
  return IVec3(t.get<int>(1), t.get<int>(2), t.get<int>(3));
}

} // namespace occ::lua_bindings
