#include "eigen_matrix.h"
#include <complex>

namespace occ::lua_bindings {

namespace lb = luabridge;
using Complex = std::complex<double>;

void register_complex_userdata(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginClass<Complex>("Complex")
      .addConstructor<void (*)(double, double)>()
      .addProperty(
          "re", +[](const Complex *z) { return z->real(); },
          +[](Complex *z, double x) { z->real(x); })
      .addProperty(
          "im", +[](const Complex *z) { return z->imag(); },
          +[](Complex *z, double x) { z->imag(x); })
      .addFunction(
          "abs", +[](const Complex *z) { return std::abs(*z); })
      // |z|^2, i.e. std::norm -- not a vector norm
      .addFunction(
          "squared_abs", +[](const Complex *z) { return std::norm(*z); })
      .addFunction(
          "arg", +[](const Complex *z) { return std::arg(*z); })
      .addFunction(
          "conjugate", +[](const Complex *z) { return std::conj(*z); })
      .addFunction(
          "__add", +[](const Complex *a, const Complex &b) { return *a + b; })
      .addFunction(
          "__sub", +[](const Complex *a, const Complex &b) { return *a - b; })
      .addFunction(
          "__mul", +[](const Complex *a, const Complex &b) { return *a * b; })
      .addFunction(
          "__div", +[](const Complex *a, const Complex &b) { return *a / b; })
      .addFunction(
          "__unm", +[](const Complex *a) { return -*a; })
      .addFunction(
          "__eq", +[](const Complex *a, const Complex &b) { return *a == b; })
      .addFunction(
          "__tostring",
          +[](const Complex *z) {
            return fmt::format("{:.6f}{:+.6f}i", z->real(), z->imag());
          })
      .endClass()
      .endNamespace();
}

void register_eigen_matrix_types(lua_State *L) {
  // must precede any complex-valued Eigen type
  register_complex_userdata(L);

  register_matrix_userdata<occ::Mat>(L, "Matrix");
  register_matrix_userdata<occ::Mat3N>(L, "Mat3N");
  register_matrix_userdata<occ::Mat3>(L, "Mat3");
  register_matrix_userdata<occ::Mat4>(L, "Mat4");
  register_matrix_userdata<occ::Mat6>(L, "Mat6");
  register_matrix_userdata<Eigen::MatrixXi>(L, "MatrixI");
  register_matrix_userdata<occ::CMat>(L, "CMatrix");

  register_vector_userdata<occ::Vec>(L, "Vector");
  register_vector_userdata<occ::Vec3>(L, "Vec3");
  register_vector_userdata<occ::IVec>(L, "IVector");
  register_vector_userdata<occ::IVec3>(L, "IVec3");
  register_vector_userdata<occ::CVec>(L, "CVector");
}

} // namespace occ::lua_bindings
