#pragma once
#define EIGEN_MATRIXBASE_PLUGIN "opmatrix.h"
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
using MaskArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

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

std::tuple<MatRM, MatRM, double> conditioning_orthogonalizer(const MatRM&, double);

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
std::tuple<MatRM, MatRM, size_t, double, double>
gensqrtinv(const MatRM&, bool symmetric = false, double max_condition_number = 1e8);

}; // namespace tonto
