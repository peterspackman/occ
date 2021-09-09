#pragma once
#include <deque>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>

namespace occ::diis {

template <typename T> struct diis_traits;

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
struct diis_traits<
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
    typedef Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
        T;
    typedef typename T::Scalar element_type;
};

template <typename T> class DIIS {
  public:
    typedef typename diis_traits<T>::element_type value_t;
    DIIS(size_t start = 1, size_t diis_subspace = 6, value_t damping_factor = 0,
         size_t ngroup = 1, size_t ngroup_diis = 1, value_t mixing_fraction = 0)
        : m_error{0}, m_error_is_set{false}, m_start{start},
          m_diis_subspace_size{diis_subspace}, m_num_group{ngroup},
          m_num_group_diis{ngroup}, m_damping_factor{damping_factor},
          m_mixing_fraction{mixing_fraction} {
        init();
    }
    ~DIIS() {
        m_x.clear();
        m_errors.clear();
        m_extrapolated.clear();
    }

    void extrapolate(T &x, T &error, bool extrapolate_error = false) {
        occ::timing::start(occ::timing::category::diis);
        const value_t zero_determinant =
            std::numeric_limits<value_t>::epsilon();
        const value_t zero_norm = 1.0e-10;
        m_iter++;
        const bool do_mixing = (m_mixing_fraction != 0.0);
        const value_t scale = 1.0 + m_damping_factor;

        // if have ndiis vectors
        if (m_errors.size() == m_diis_subspace_size) {
            // holding max # of vectors already? drop the least recent {x,
            // error} pair
            m_x.pop_front();
            m_errors.pop_front();
            if (!m_extrapolated.empty())
                m_extrapolated.pop_front();
            Mat Bcrop = m_B.bottomRightCorner(m_diis_subspace_size - 1,
                                              m_diis_subspace_size - 1);
            Bcrop.conservativeResize(m_diis_subspace_size,
                                     m_diis_subspace_size);
            m_B = Bcrop;
        }

        // push {x, error} to the set
        m_x.push_back(x);
        m_errors.push_back(error);
        const auto nvec = m_errors.size();
        assert(m_x.size() == m_errors.size());

        // and compute the most recent elements of B, B(i,j) = <ei|ej>
        for (size_t i = 0; i < nvec - 1; i++) {
            m_B(i, nvec - 1) = m_B(nvec - 1, i) =
                m_errors[i].cwiseProduct(m_errors[nvec - 1]).sum();
        }
        m_B(nvec - 1, nvec - 1) =
            m_errors[nvec - 1].cwiseProduct(m_errors[nvec - 1]).sum();

        if (m_iter == 1) { // the first iteration
            if (!m_extrapolated.empty() && do_mixing) {
                x.setZero();
                x.noalias() += (1.0 - m_mixing_fraction) * m_x[0];
                x.noalias() += m_mixing_fraction * m_extrapolated[0];
            }
        } else if (m_iter > m_start &&
                   (((m_iter - m_start) % m_num_group) <
                    m_num_group_diis)) { // not the first iteration and need to
                                         // extrapolate?

            Vec c;

            value_t absdetA;
            size_t nskip = 0; // how many oldest vectors to skip for the sake of
                              // conditioning? try zero
            // skip oldest vectors until found a numerically stable system
            do {

                const size_t rank = nvec - nskip + 1; // size of matrix A

                // set up the DIIS linear system: A c = rhs
                Mat A(rank, rank);
                A.col(0).setConstant(-1.0);
                A.row(0).setConstant(-1.0);
                A(0, 0) = 0.0;
                Vec rhs = Vec::Zero(rank);
                rhs(0) = -1.0;

                value_t norm = 1.0;
                if (m_B(nskip, nskip) > zero_norm)
                    norm = 1.0 / m_B(nskip, nskip);

                A.block(1, 1, rank - 1, rank - 1) =
                    m_B.block(nskip, nskip, rank - 1, rank - 1) * norm;
                A.diagonal() *= scale;

                // finally, solve the DIIS linear system
                Eigen::ColPivHouseholderQR<Mat> A_QR = A.colPivHouseholderQr();
                c = A_QR.solve(rhs);
                absdetA = A_QR.absDeterminant();

                ++nskip;
            } while (absdetA < zero_determinant &&
                     nskip < nvec); // while (system is poorly conditioned)

            // failed?
            if (absdetA < zero_determinant) {
                throw std::domain_error(fmt::format(
                    "DIIS::extrapolate: poorly-conditioned system, |A| = {}",
                    absdetA));
            }
            --nskip; // undo the last ++ :-(
            {
                x.setZero();
                for (size_t k = nskip, kk = 1; k < nvec; ++k, ++kk) {
                    if (!do_mixing || m_extrapolated.empty()) {
                        // std::cout << "contrib " << k << " c=" << c[kk] << ":"
                        // << std::endl << x_[k] << std::endl;
                        x.noalias() += c[kk] * m_x[k];
                        if (extrapolate_error)
                            error.noalias() += c[kk] * m_errors[k];
                    } else {
                        x.noalias() += (1.0 - m_mixing_fraction) * m_x[k];
                        x.noalias() +=
                            c[kk] * m_mixing_fraction * m_extrapolated[k];
                    }
                }
            }
        } // do DIIS

        // only need to keep extrapolated x if doing mixing
        if (do_mixing)
            m_extrapolated.push_back(x);
        occ::timing::stop(occ::timing::category::diis);
    }

  private:
    value_t m_error;
    bool m_error_is_set{false};
    size_t m_start{1};
    size_t m_diis_subspace_size{6};
    size_t m_iter{0};
    size_t m_num_group{1};
    size_t m_num_group_diis{1};
    value_t m_damping_factor;
    value_t m_mixing_fraction;

    Mat m_B; //!< B(i,j) = <ei|ej>

    std::deque<T>
        m_x; //!< set of most recent x given as input (i.e. not exrapolated)
    std::deque<T> m_errors;       //!< set of most recent errors
    std::deque<T> m_extrapolated; //!< set of most recent extrapolated x

    void set_error(value_t e) {
        m_error = e;
        m_error_is_set = true;
    }
    value_t error() const { return m_error; }

    void init() {
        m_iter = 0;
        m_B = Mat::Zero(m_diis_subspace_size, m_diis_subspace_size);
        m_x.clear();
        m_errors.clear();
        m_extrapolated.clear();
    }
};
} // namespace occ::diis
