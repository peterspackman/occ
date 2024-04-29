#pragma once
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fmt/ostream.h>

template <typename T>
requires std::is_base_of_v<Eigen::DenseBase<T>, T> struct fmt::formatter<T>
    : ostream_formatter {};

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
using Mat3RM = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Mat3ConstRef = Eigen::Ref<const Mat3>;
using Mat4 = Eigen::Matrix4d;
using Mat4ConstRef = Eigen::Ref<const Mat4>;
using Mat4N = Eigen::Matrix4Xd;
using MatN4 = Eigen::MatrixX4d;

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

}; // namespace occ
