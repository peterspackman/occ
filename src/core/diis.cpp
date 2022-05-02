#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/diis.h>
#include <occ/core/timings.h>

namespace occ::core::diis {

DIIS::DIIS(size_t start, size_t diis_subspace, double damping_factor,
           size_t ngroup, size_t ngroup_diis, double mixing_fraction)
    : m_error{0}, m_error_is_set{false}, m_start{start},
      m_diis_subspace_size{diis_subspace}, m_num_group{ngroup},
      m_num_group_diis{ngroup}, m_damping_factor{damping_factor},
      m_mixing_fraction{mixing_fraction} {
    init();
}

void DIIS::set_error(double e) {
    m_error = e;
    m_error_is_set = true;
}

double DIIS::error() const { return m_error; }

void DIIS::init() {
    m_iter = 0;
    m_B = Mat::Zero(m_diis_subspace_size, m_diis_subspace_size);
    m_x.clear();
    m_errors.clear();
    m_extrapolated.clear();
}

void DIIS::extrapolate(Mat &x, Mat &error, bool extrapolate_error) {
    occ::timing::start(occ::timing::category::diis);
    const double zero_determinant = std::numeric_limits<double>::epsilon();
    const double zero_norm = 1.0e-10;
    m_iter++;
    const bool do_mixing = (m_mixing_fraction != 0.0);
    const double scale = 1.0 + m_damping_factor;

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
        Bcrop.conservativeResize(m_diis_subspace_size, m_diis_subspace_size);
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

        double absdetA;
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

            double norm = 1.0;
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

} // namespace occ::core::diis
