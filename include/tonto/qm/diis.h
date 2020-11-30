#pragma once
#include <tonto/core/linear_algebra.h>
#include <deque>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace tonto::diis {

template <typename T>
struct diis_traits;

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct diis_traits<Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >> {
    typedef Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > T;
    typedef typename T::Scalar element_type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
typename diis_traits<Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >>::element_type
dot_product(const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& d1,
            const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& d2) {
  return d1.cwiseProduct(d2).sum();
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void zero(Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& d) {
  d.setZero(d.rows(), d.cols());
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename Scalar>
void axpy(Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& y,
     Scalar a,
     const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& x) {
  y += a*x;
}

template <typename T>
class DIIS {
public:
  typedef typename diis_traits<T>::element_type value_t;
  DIIS(size_t start=1,
       size_t ndiis=5,
       value_t damping_factor=0,
       size_t ngroup=1,
       size_t ngroup_diis=1,
       value_t mixing_fraction=0) :
         m_error{0}, m_error_is_set{false},
         m_start{start}, m_num_diis{ndiis},
         m_num_group{ngroup},
         m_num_group_diis{ngroup},
         m_damping_factor{damping_factor},
         m_mixing_fraction{mixing_fraction}
  {
    init();
  }
  ~DIIS() {
    m_x.clear();
    m_errors.clear();
    m_extrapolated.clear();
  }

  void extrapolate(T& x, T& error, bool extrapolate_error = false)
  {
    const value_t zero_determinant = std::numeric_limits<value_t>::epsilon();
    const value_t zero_norm = 1.0e-10;
    m_iter++;
    const bool do_mixing = (m_mixing_fraction != 0.0);
    const value_t scale = 1.0 + m_damping_factor;

    // if have ndiis vectors
    if (m_errors.size() == m_num_diis) {
      // holding max # of vectors already? drop the least recent {x, error} pair
      m_x.pop_front();
      m_errors.pop_front();
      if (!m_extrapolated.empty()) m_extrapolated.pop_front();
      MatX Bcrop = m_B.bottomRightCorner(m_num_diis-1,m_num_diis-1);
      Bcrop.conservativeResize(m_num_diis,m_num_diis);
      m_B = Bcrop;
    }

    // push {x, error} to the set
    m_x.push_back(x);
    m_errors.push_back(error);
    const auto nvec = m_errors.size();
    assert(m_x.size() == m_errors.size());

    // and compute the most recent elements of B, B(i,j) = <ei|ej>
    for (size_t i = 0; i < nvec - 1; i++) {
      m_B(i, nvec - 1) = m_B(nvec - 1, i) = dot_product(m_errors[i], m_errors[nvec - 1]);
    }
    m_B(nvec - 1, nvec - 1) = dot_product(m_errors[nvec - 1], m_errors[nvec - 1]);

    if (m_iter == 1) { // the first iteration
      if (!m_extrapolated.empty() && do_mixing) {
        zero(x);
        axpy(x, (1.0 - m_mixing_fraction), m_x[0]);
        axpy(x, m_mixing_fraction, m_extrapolated[0]);
      }
    }
    else if (m_iter > m_start && (((m_iter - m_start) % m_num_group) < m_num_group_diis)) { // not the first iteration and need to extrapolate?

      VecX c;

      value_t absdetA;
      size_t nskip = 0; // how many oldest vectors to skip for the sake of conditioning?
                        // try zero
      // skip oldest vectors until found a numerically stable system
      do {

        const size_t rank = nvec - nskip + 1; // size of matrix A

        // set up the DIIS linear system: A c = rhs
        MatX A(rank, rank);
        A.col(0).setConstant(-1.0);
        A.row(0).setConstant(-1.0);
        A(0,0) = 0.0;
        VecX rhs = VecX::Zero(rank);
        rhs(0) = -1.0;

        value_t norm = 1.0;
        if (m_B(nskip, nskip) > zero_norm)
          norm = 1.0 / m_B(nskip, nskip);

        A.block(1, 1, rank-1, rank-1) = m_B.block(nskip, nskip, rank-1, rank-1) * norm;
        A.diagonal() *= scale;

        // finally, solve the DIIS linear system
        Eigen::ColPivHouseholderQR<MatX> A_QR = A.colPivHouseholderQr();
        c = A_QR.solve(rhs);
        absdetA = A_QR.absDeterminant();

        ++nskip;
      } while (absdetA < zero_determinant && nskip < nvec); // while (system is poorly conditioned)

      // failed?
      if (absdetA < zero_determinant) {
        throw std::domain_error(fmt::format(
            "DIIS::extrapolate: poorly-conditioned system, |A| = {}", absdetA)
        );
      }
      --nskip; // undo the last ++ :-(
      {
        zero(x);
        for (size_t k = nskip, kk = 1; k < nvec; ++k, ++kk) {
          if (!do_mixing || m_extrapolated.empty()) {
            //std::cout << "contrib " << k << " c=" << c[kk] << ":" << std::endl << x_[k] << std::endl;
            axpy(x, c[kk], m_x[k]);
            if (extrapolate_error)
              axpy(error, c[kk], m_errors[k]);
          } else {
            axpy(x, c[kk] * (1.0 - m_mixing_fraction), m_x[k]);
            axpy(x, c[kk] * m_mixing_fraction, m_extrapolated[k]);
          }
        }
      }
    } // do DIIS

    // only need to keep extrapolated x if doing mixing
    if (do_mixing) m_extrapolated.push_back(x);
  }


private:
      value_t m_error;
      bool m_error_is_set{false};
      size_t m_start{1};
      size_t m_num_diis{5};
      size_t m_iter{0};
      size_t m_num_group{1};
      size_t m_num_group_diis{1};
      value_t m_damping_factor;
      value_t m_mixing_fraction;

      typedef Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatX;
      typedef Eigen::Matrix<value_t, Eigen::Dynamic, 1> VecX;

      MatX m_B; //!< B(i,j) = <ei|ej>

      std::deque<T> m_x; //!< set of most recent x given as input (i.e. not exrapolated)
      std::deque<T> m_errors; //!< set of most recent errors
      std::deque<T> m_extrapolated; //!< set of most recent extrapolated x

      void set_error(value_t e) { m_error = e; m_error_is_set = true; }
      value_t error() const { return m_error; }

      void init() {
        m_iter = 0;
        m_B = MatX::Zero(m_num_diis, m_num_diis);
        m_x.clear();
        m_errors.clear();
        m_extrapolated.clear();
      }



};
}
