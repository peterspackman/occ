#include "eigen_matrix.h"

namespace occ::lua_bindings {

void register_eigen_matrix_types(sol::table &m) {
  // Matrices — registered once each. Dynamic-size first (most common),
  // then the fixed-size shapes we actually return from binding code.
  register_matrix_userdata<occ::Mat>(m, "Matrix");        // MatrixXd
  register_matrix_userdata<occ::Mat3N>(m, "Mat3N");       // 3 × dynamic double
  register_matrix_userdata<occ::Mat3>(m, "Mat3");         // 3×3
  register_matrix_userdata<occ::Mat4>(m, "Mat4");         // 4×4
  register_matrix_userdata<occ::Mat6>(m, "Mat6");         // 6×6
  register_matrix_userdata<Eigen::MatrixXi>(m, "MatrixI"); // int dynamic

  register_vector_userdata<occ::Vec>(m, "Vector");        // VectorXd
  register_vector_userdata<occ::Vec3>(m, "Vec3");         // 3-vector
  register_vector_userdata<occ::IVec>(m, "IVector");      // VectorXi
  register_vector_userdata<occ::IVec3>(m, "IVec3");       // 3-int-vector
}

} // namespace occ::lua_bindings
