#include <occ/crystal/muldin.h>

namespace occ::crystal {

Eigen::MatrixXd muldin(const Eigen::VectorXd &x) {

    /* Implementation of the MULDIN algorithm from Gorfman 2020 [1]
     *
     * References:
     *    [1] Gorfman, Acta Cryst. A. 2020, 76 (6) 713-718
     * https://doi.org/10.1107/S2053273320012668
     */
    int N = x.rows();

    // Initialize i1, which has 1 for non-negative x and -1 for negative x
    Eigen::VectorXd i1 =
        x.unaryExpr([](double val) { return val >= 0.0 ? 1.0 : -1.0; });

    // Create the matrix S0
    Eigen::MatrixXd S = i1.asDiagonal();

    // Ensure right-handed coordinate system
    if (S.determinant() < 0)
        S.col(0).swap(S.col(1));

    Eigen::VectorXd X = S.inverse() * x;

    std::vector<int> nonzeros;

    auto find_nonzeros = [&nonzeros](const Eigen::VectorXd &vec) {
        nonzeros.clear();
        for (int i = 0; i < vec.rows(); i++)
            if (vec(i) != 0)
                nonzeros.push_back(i);
    };

    find_nonzeros(X);
    if (!nonzeros.empty()) {
        int m1 = nonzeros.back();
        if (m1 < N - 1) {
            int a1 = m1 + 1;
            int a2 = m1 + N + 1;
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(N);
            for (int i = a1; i < a2; i++)
                perm.indices()[i - a1] = i % N;
            S = S * perm;
            X = S.inverse() * x;
        }
    }

    bool finished = nonzeros.size() <= 1;

    find_nonzeros(X);

    while (!finished) {
        double minimal_nonzero_value = std::numeric_limits<double>::max();
        int m1 = -1;

        for (const auto &index : nonzeros) {
            if (X(index) <= minimal_nonzero_value) {
                minimal_nonzero_value = X(index);
                m1 = index;
            }
        }
        // should be impossible
        if (m1 < 0)
            break;

        Eigen::MatrixXd T = Eigen::MatrixXd::Identity(N, N);
        T.col(m1).setZero();
        for (const auto &index : nonzeros)
            T(index, m1) = 1;

        S *= T;
        X = T.inverse() * X;

        find_nonzeros(X);
        finished = nonzeros.size() == 1;
    }

    return S;
}
} // namespace occ::crystal
