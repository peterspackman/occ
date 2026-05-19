#include "eigen_matrix.h"

namespace occ::lua_bindings {

void register_eigen_matrix_types(lua_State *L) {
  register_matrix_userdata<occ::Mat>(L, "Matrix");
  register_matrix_userdata<occ::Mat3N>(L, "Mat3N");
  register_matrix_userdata<occ::Mat3>(L, "Mat3");
  register_matrix_userdata<occ::Mat4>(L, "Mat4");
  register_matrix_userdata<occ::Mat6>(L, "Mat6");
  register_matrix_userdata<Eigen::MatrixXi>(L, "MatrixI");

  register_vector_userdata<occ::Vec>(L, "Vector");
  register_vector_userdata<occ::Vec3>(L, "Vec3");
  register_vector_userdata<occ::IVec>(L, "IVector");
  register_vector_userdata<occ::IVec3>(L, "IVec3");
}

} // namespace occ::lua_bindings
