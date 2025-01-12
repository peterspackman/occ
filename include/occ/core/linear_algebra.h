#pragma once
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fmt/core.h>

namespace occ {
using DMatRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IMatRM =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FMatRM =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Array = Eigen::ArrayXd;
using DArray = Eigen::ArrayXd;
using CArray = Eigen::ArrayXcd;
using IArray = Eigen::ArrayXi;
using MaskArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
using MaskMat = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;

using IMat = Eigen::MatrixXi;
using IMat3N = Eigen::Matrix3Xi;
using Mat = Eigen::MatrixXd;
using MatRef = Eigen::Ref<Eigen::MatrixXd>;
using MatConstRef = Eigen::Ref<const Eigen::MatrixXd>;
using Mat3N = Eigen::Matrix3Xd;
using Mat3NConstRef = Eigen::Ref<const Mat3N>;
using MatN3 = Eigen::MatrixX3d;
using MatN3ConstRef = Eigen::Ref<const MatN3>;
using Mat3 = Eigen::Matrix3d;
using Mat2 = Eigen::Matrix2d;
using Mat3RM = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Mat3ConstRef = Eigen::Ref<const Mat3>;
using Mat4 = Eigen::Matrix4d;
using Mat4ConstRef = Eigen::Ref<const Mat4>;
using Mat4N = Eigen::Matrix4Xd;
using MatN4 = Eigen::MatrixX4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat6N = Eigen::Matrix<double, 6, Eigen::Dynamic>;

using CMat = Eigen::MatrixXcd;
using CMat3N = Eigen::MatrixX3cd;
using CMatN3 = Eigen::Matrix3Xcd;
using CMat3 = Eigen::Matrix3cd;
using CMat4 = Eigen::Matrix4cd;

using FMat = Eigen::MatrixXf;
using FMat3N = Eigen::Matrix3Xf;
using FMatN3 = Eigen::MatrixX3f;
using FMat3 = Eigen::Matrix3f;
using FMat4 = Eigen::Matrix3f;

using DMat = Eigen::MatrixXd;
using DMat3N = Eigen::Matrix3Xd;
using DMatN3 = Eigen::MatrixX3d;
using DMat3 = Eigen::Matrix3d;
using DMat4 = Eigen::Matrix4d;

using RowVec = Eigen::RowVectorXd;
using Vec = Eigen::VectorXd;
using RowVec3 = Eigen::RowVector3d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using RowVec4 = Eigen::RowVector4d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec10 = Eigen::Matrix<double, 10, 1>;

using CVec = Eigen::VectorXcd;
using DVec = Eigen::VectorXd;
using DVec3 = Eigen::Vector3d;
using FVec = Eigen::VectorXf;
using FVec3 = Eigen::Vector3f;
using FVec2 = Eigen::Vector2f;

using IVec = Eigen::VectorXi;
using IVec3 = Eigen::Vector3i;

struct MatTriple {

  Mat x, y, z;

  MatTriple operator+(const MatTriple &rhs) const;
  MatTriple operator-(const MatTriple &rhs) const;

  void scale_by(double fac);
  void symmetrize();
};

template <typename Derived>
std::string format_matrix(const Eigen::DenseBase<Derived> &matrix,
                          std::string_view fmt_str = "{:12.5f}") {
  const auto &derived = matrix.derived();
  fmt::memory_buffer out;

  // For vectors, always format as a row vector
  const Eigen::Index rows = derived.cols() == 1 ? 1 : derived.rows();
  const Eigen::Index cols =
      derived.cols() == 1 ? derived.rows() : derived.cols();

  // Pre-allocate with rough estimate of size needed
  out.reserve(rows * cols * (fmt_str.size() + 2));

  for (Eigen::Index i = 0; i < rows; ++i) {
    if (i != 0)
      fmt::format_to(std::back_inserter(out), "\n");
    for (Eigen::Index j = 0; j < cols; ++j) {
      if (j != 0)
        fmt::format_to(std::back_inserter(out), " ");
      // For vectors, transpose the access if it's a column vector
      const auto val = derived.cols() == 1 ? derived(j, 0) : derived(i, j);
      fmt::format_to(std::back_inserter(out), fmt::runtime(fmt_str), val);
    }
  }

  return fmt::to_string(out);
}

}; // namespace occ
