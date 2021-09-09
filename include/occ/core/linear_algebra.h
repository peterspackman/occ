#pragma once
#define EIGEN_MATRIXBASE_PLUGIN "occ/core/opmatrix.h"
#include <Eigen/Dense>
#include <Eigen/StdVector>

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

using RowVec = Eigen::RowVectorXd;
using Vec = Eigen::VectorXd;
using RowVec3 = Eigen::RowVector3d;
using Vec3 = Eigen::Vector3d;
using RowVec4 = Eigen::RowVector4d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec10 = Eigen::Matrix<double, 10, 1>;

using DVec = Eigen::VectorXd;
using DVec3 = Eigen::Vector3d;

using IVec = Eigen::VectorXi;
using IVec3 = Eigen::Vector3i;

std::tuple<Mat, Mat, double> conditioning_orthogonalizer(const Mat &, double);

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
std::tuple<Mat, Mat, size_t, double, double>
gensqrtinv(const Mat &, bool symmetric = false,
           double max_condition_number = 1e8);

Mat3 inertia_tensor(Eigen::Ref<const Vec> masses,
                    Eigen::Ref<const Mat3N> positions);

}; // namespace occ
