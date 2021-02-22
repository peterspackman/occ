#include <tonto/core/kabsch.h>
#include <Eigen/SVD>

namespace tonto::linalg {

tonto::Mat3 kabsch_rotation_matrix(const tonto::Mat3N &a, const tonto::Mat3N &b)
{
    /*
    Calculate the optimal rotation matrix `R` to rotate
    `A` onto `B`, minimising root-mean-square deviation so that
    this may be then calculated.

    See: https://en.wikipedia.org/wiki/Kabsch_algorithm

    Reference:
    ```
    Kabsch, W. Acta Cryst. A, 32, 922-923, (1976)
    DOI: http://dx.doi.org/10.1107/S0567739476001873
    ```
    Args:
        A : (N,D) matrix where N is the number of vectors and D
            is the dimension of each vector
        B : (N,D) matrix where N is the number of
            vectors and D is the dimension of each vector
    Returns:
        (D,D) rotation matrix where D is the dimension of each vector
    */

    // Calculate the covariance matrix
    auto cov = a.transpose() * b;

    // Use singular value decomposition to calculate
    // the optimal rotation matrix
    Eigen::JacobiSVD<tonto::Mat> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // auto v, s, w = np.linalg.svd(cov)
    tonto::Mat u = svd.matrixU();
    tonto::Mat v = svd.matrixV();

    // check the determinant to ensure a right-handed
    // coordinate system
    if (u.determinant() * v.determinant() < 0.0)
    {
        u.rightCols(1).array() = - u.rightCols(1).array();
    }
    return u * v;
}

}
