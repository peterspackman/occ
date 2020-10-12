#pragma once
#include <Eigen/Dense>

namespace tonto {
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

using Mat = Eigen::MatrixXd;
using Mat3N = Eigen::Matrix3Xd;
using MatN3 = Eigen::MatrixX3d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat4N = Eigen::Matrix4Xd;
using MatN4 = Eigen::MatrixX4d;

using CMat = Eigen::MatrixXcd;
using CMat3N = Eigen::MatrixX3cd;
using CMatN3 = Eigen::Matrix3Xcd;
using CMat3 = Eigen::Matrix3cd;
using CMat4 = Eigen::Matrix4cd;

using DMat = Eigen::MatrixXd;
using DMat3N = Eigen::Matrix3Xd;
using DMatN3 = Eigen::MatrixX3d;
using DMat3 = Eigen::Matrix3d;
using DMat4 = Eigen::Matrix4d;

using Vec = Eigen::VectorXd;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

using DVec = Eigen::VectorXd;
using DVec3 = Eigen::Vector3d;

using IVec = Eigen::VectorXi;
using IVec3 = Eigen::Vector3i;

}; // namespace tonto
