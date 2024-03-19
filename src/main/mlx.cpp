#include <chrono>
#include <cmath>
#include <iostream>

#include "mlx/mlx.h"
#include <Eigen/Dense>

namespace timer {

using namespace std::chrono;

template <typename R, typename P>
inline double seconds(duration<R, P> x) {
  return duration_cast<nanoseconds>(x).count() / 1e9;
}

inline auto time() {
  return high_resolution_clock::now();
}

} // namespace timer

using namespace mlx::core;

int main() {
  int n = 20000;
  int n2 = 400;
  {
      auto mat1 = mlx::core::random::normal({n, n}) + 1.0;
      auto mat2 = mlx::core::random::normal({n, n2});

      auto tic = timer::time();
      auto prod = mlx::core::matmul(mat1, mat2);

      auto result = mlx::core::sum(prod);
      auto toc = timer::time();
      std::cout << "Result (MLX)" << result << ", took: " 
		<< timer::seconds(toc - tic) << '\n';

  }
  {
      Eigen::MatrixXf mat1 = Eigen::MatrixXf::Random(n, n).array() + 1.0;
      auto mat2 = Eigen::MatrixXf::Random(n, n2);

      auto tic = timer::time();
      auto result = (mat1 * mat2).sum();
      auto toc = timer::time();
      std::cout << "Result (Eigen)" << result << ", took: " 
		<< timer::seconds(toc - tic) << '\n';
  }
}
