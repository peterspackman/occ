#include <occ/qm/density_fitting.h>
#include <occ/core/parallel.h>

namespace occ::df {

DFFockEngine::DFFockEngine(const BasisSet& _obs, const BasisSet& _dfbs)
    : obs(_obs), dfbs(_dfbs), nbf(_obs.nbf()), ndf(_dfbs.nbf())
{
    Mat V = occ::ints::compute_2body_2index_ints(dfbs);
    Eigen::LLT<Mat> V_LLt(V);
    Mat I = Mat::Identity(ndf, ndf);
    auto L = V_LLt.matrixL();
    Linv_t = L.solve(I).transpose();
}

Mat DFFockEngine::compute_2body_fock_dfC(const Mat& Cocc)
{

  // using first time? compute 3-center ints and transform to inv sqrt
  // representation
  if (xyK.size() == 0) {

    xyK_dims = {nbf, nbf, ndf};
    xyK.resize(nbf * nbf * ndf);
    std::fill(xyK.begin(), xyK.end(), 0.0);

    auto lambda = [&](int thread_id, size_t bf1, size_t n1,
                      size_t bf2, size_t n2,
                      size_t bf3, size_t n3,
                      const double * buf)
    {
        size_t offset = 0;
        for(size_t i = bf1; i < bf1 + n1; i++)
        {
            for(size_t j = bf2; j < bf2+ n2; j++)
            {
                for(size_t k = bf3; k < bf3 + n3; k++)
                {
                    for(size_t l = 0; l < ndf; l++)
                    {
                        xyK[j * (nbf * ndf) + k * ndf + l] += buf[offset] * Linv_t(i, l);
                    }
                    offset++;
                }
            }
        }
    };

    three_center_integral_helper(lambda);

  }  // if (xyK.size() == 0)

  // compute exchange
  const size_t nocc = Cocc.cols();
  MatRM cr = Cocc;
  const double * Co = cr.data();
  std::array<size_t, 2> Co_dims{nbf, nocc};
  std::array<size_t, 3> xiK_dims{xyK_dims[0], Co_dims[1], xyK_dims[2]};
  std::vector<double> xiK(xiK_dims[0] * xiK_dims[1] * xiK_dims[2], 0.0);

  // xiK(i,l,k) = xyK(i,j,k) * Co(j,l)
  for(size_t i = 0; i < xyK_dims[0]; i++) {
      for(size_t j = 0; j < Co_dims[0]; j++) {
          size_t jxyK = i * xyK_dims[1] + j;
          for(size_t l = 0; l < Co_dims[1]; l++) {
              size_t lxiK = i * xiK_dims[1] + l;
              size_t lCo = j * Co_dims[1] + l;
              for(size_t k = 0; k < xyK_dims[2]; k++) {
                  size_t kxiK = lxiK * xiK_dims[2] + k;
                  size_t kxyK = jxyK * xyK_dims[2] + k;
                  xiK[kxiK] = xiK[kxiK] + xyK[kxyK] * Co[lCo];
              }
          }
      }
  }

  std::array<size_t, 2> G_dims{xiK_dims[0], xiK_dims[0]};
  std::vector<double> KK(G_dims[0] * G_dims[1]);
  //exchange
  // G(i,l) = xiK(i,j,k) * xiK(l,j,k)
  for(size_t i = 0; i < xiK_dims[0]; i++) {
      for(size_t l = 0; l < xiK_dims[0]; l++) {
          size_t lG = i * G_dims[1] + l;
          double tjG_val = 0.0;
          for(size_t j = 0; j < xiK_dims[1]; j++) {
              size_t jxiK = i * xiK_dims[1] + j;
              size_t jxiK0 = l * xiK_dims[1] + j;
              for(size_t k = 0; k < xiK_dims[2]; k++) {
                  size_t kxiK = jxiK * xiK_dims[2] + k;
                  size_t kxiK0 = jxiK0 * xiK_dims[2] + k;
                  tjG_val += xiK[kxiK] * xiK[kxiK0];
              }
          }
          KK[lG] = tjG_val;
      }
  }

  //coulomb
  //J(k) = xiK(i,j,k) * Co(i,j)
  size_t J_dim = xiK_dims[2];
  std::vector<double> J(J_dim, 0);
  for(size_t i = 0; i < Co_dims[0]; i++) {
      for(size_t j = 0; j < Co_dims[1]; j++) {
          size_t jxiK = i * xiK_dims[1] + j;
          size_t jCo = i * Co_dims[1] + j;
          for(size_t k = 0; k < xiK_dims[2]; k++) {
              size_t kxiK = jxiK * xiK_dims[2] + k;
              J[k] = J[k] + xiK[kxiK] * Co[jCo];
          }
      }
  }
  std::vector<double> JJ(KK.size());
  // G(i,j) = 2 * xyK(i,j,k) * J(k) - G(i,j)
  for(size_t i = 0; i < G_dims[0]; i++) {
      for(size_t j = 0; j < G_dims[1]; j++) {
          size_t jG = i * G_dims[1] + j;
          size_t jxyK = i * xyK_dims[1] + j;
          size_t jG0 = i * G_dims[1] + j;
          double tk_val = 0.0;
          for(size_t k = 0; k < J_dim; k++) {
              size_t kxyK = jxyK * xyK_dims[2] + k;
              tk_val += (2 * xyK[kxyK] * J[k]);
          }
          JJ[jG] = tk_val;
      }
  }
  // copy result to an Eigen::Matrix
  Mat Jm = Eigen::Map<const MatRM>(JJ.data(), nbf, nbf);
  Mat Km = Eigen::Map<const MatRM>(KK.data(), nbf, nbf);
  Mat result = (Jm - Km);
  return result;
}

inline int upper_triangle_index(const int N, const int i, const int j)
{
    return (2 * N * i - i * i - i + 2 * j) / 2;
}

inline int lower_triangle_index(const int N, const int i, const int j)
{
    return upper_triangle_index(N, j, i);
}

}
