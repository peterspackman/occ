#include <Eigen/Eigenvalues>
#include <fmt/core.h>
#include <occ/core/eigh.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <stdexcept>
#include <vector>

#ifdef OCC_HAVE_LAPACK
extern "C" void dsyevd_(const char *jobz, const char *uplo, const int *n,
                        double *a, const int *lda, double *w, double *work,
                        const int *lwork, int *iwork, const int *liwork,
                        int *info);
#if defined(__linux__)
// Present only when the LAPACK brings its own threads (OpenBLAS, or an
// OpenMP build such as threaded LibSci); they would otherwise size themselves
// to the whole node rather than to --threads.
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
extern "C" void omp_set_num_threads(int) __attribute__((weak));
#endif
#endif

namespace occ::core {

namespace {

SymmetricEigenResult eigen_eigh(Eigen::Ref<const Mat> A) {
  Eigen::SelfAdjointEigenSolver<Mat> solver(A);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("Symmetric eigendecomposition failed to converge");
  return {solver.eigenvalues(), solver.eigenvectors()};
}

#ifdef OCC_HAVE_LAPACK
void match_lapack_threads() {
#if defined(__linux__)
  const int n = occ::parallel::get_num_threads();
  if (openblas_set_num_threads)
    openblas_set_num_threads(n);
  else if (omp_set_num_threads)
    omp_set_num_threads(n);
#endif
}

SymmetricEigenResult lapack_eigh(Eigen::Ref<const Mat> A) {
  const int n = static_cast<int>(A.rows());
  SymmetricEigenResult result{Vec(n), A};
  const char jobz = 'V', uplo = 'L';
  int info = 0, lwork = -1, liwork = -1, iwork_query = 0;
  double work_query = 0.0;
  dsyevd_(&jobz, &uplo, &n, result.eigenvectors.data(), &n,
          result.eigenvalues.data(), &work_query, &lwork, &iwork_query,
          &liwork, &info);
  if (info != 0)
    throw std::runtime_error(
        fmt::format("dsyevd workspace query failed (info = {})", info));
  lwork = static_cast<int>(work_query);
  liwork = iwork_query;
  std::vector<double> work(lwork);
  std::vector<int> iwork(liwork);
  match_lapack_threads();
  dsyevd_(&jobz, &uplo, &n, result.eigenvectors.data(), &n,
          result.eigenvalues.data(), work.data(), &lwork, iwork.data(),
          &liwork, &info);
  if (info < 0)
    throw std::runtime_error(
        fmt::format("dsyevd: argument {} had an illegal value", -info));
  if (info > 0) {
    occ::log::warn("dsyevd failed to converge (info = {}); falling back to "
                   "Eigen's eigensolver",
                   info);
    return eigen_eigh(A);
  }
  return result;
}
#endif

} // namespace

SymmetricEigenResult eigh(Eigen::Ref<const Mat> A) {
  if (A.rows() != A.cols())
    throw std::invalid_argument(fmt::format(
        "eigh: matrix is {}x{}, not square", A.rows(), A.cols()));
  if (A.rows() == 0)
    return {Vec(0), Mat(0, 0)};
#ifdef OCC_HAVE_LAPACK
  return lapack_eigh(A);
#else
  return eigen_eigh(A);
#endif
}

} // namespace occ::core
