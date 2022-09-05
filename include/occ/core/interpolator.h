#include <cmath>
#include <fmt/core.h>

namespace occ::core {

/**
 * An enum to specify the mapping of the domain of inputs for interpolation.
 *
 * Can help improve precision of linear interpolation while minimizing
 * number of points.
 */
enum DomainMapping {
    Linear,     /**< The typical f(x) mapping */
    SquareRoot, /**< Make f a function of x^2 -> f(x*x) mapping */
    Log,        /**< Make f a function of e^x -> f(e^x) mapping */
};

/**
 * Class for interpolating one-dimensional functions.
 *
 * Templated by the type of data to be interpolated, and the DomainMapping
 * (default = Linear)
 */
template <typename T, DomainMapping mapping = Linear> class Interpolator1D {
  public:
    /**
     * Default constructor
     */
    Interpolator1D() {}

    /**
     * Construct an Interpolator1D from a given function.
     *
     * \param f the function to interpolate
     * \param left the lowest (left most) value in the domain
     * \param right the highest (right most) value in the domain
     * \param N the number of points to sample in the domain [left, right]
     *
     * Will evaluate the function f N times, and map the domain values
     * internally based on the specified DomainMapping....
     *
     */
    template <typename F>
    Interpolator1D(const F &f, T left, T right, size_t N) {
        m_domain.reserve(N);
        m_range.reserve(N);

        T l_mapped, u_mapped;

        if constexpr (mapping == Log) {
            l_mapped = std::log(left);
            u_mapped = std::log(right);
        } else if constexpr (mapping == SquareRoot) {
            l_mapped = std::sqrt(left);
            u_mapped = std::sqrt(right);
        } else {
            l_mapped = left;
            u_mapped = right;
        }

        for (size_t i = 0; i < N; i++) {
            T x = l_mapped + i * (u_mapped - l_mapped) / N;

            if constexpr (mapping == Log) {
                x = std::exp(x);
            } else if constexpr (mapping == SquareRoot) {
                x = x * x;
            }
            T y = f(x);
            m_domain.push_back(x);
            m_range.push_back(y);
        }

        if constexpr (mapping == Log) {
            l_domain = std::log(m_domain[0]);
            u_domain = std::log(m_domain[m_domain.size() - 1]);
        } else if constexpr (mapping == SquareRoot) {
            l_domain = std::sqrt(m_domain[0]);
            u_domain = std::sqrt(m_domain[m_domain.size() - 1]);
        } else {
            l_domain = m_domain[0];
            u_domain = m_domain[m_domain.size() - 1];
        }

        l_fill = m_range[0];
        u_fill = m_range[m_range.size() - 1];
    }

    /**
     * Evaluate the interpolated function at the value provided.
     *
     * \param x the value where the interpolated function should be evaluated.
     *
     * The provided value will be mapped based on DomainMapping.
     * If the provided value falls outside the domain, it will be
     * yield the f(left) and f(right) values from the original
     * construction of the Interpolator1D.
     */
    T operator()(T x) const {
        size_t N = m_domain.size();
        T domain_distance = u_domain - l_domain;
        T dval;
        if constexpr (mapping == Log) {
            dval = std::log(x);
        } else if constexpr (mapping == SquareRoot) {
            dval = std::sqrt(x);
        } else {
            dval = x;
        }
        T guess = N * (dval - l_domain) / domain_distance;
        size_t j = static_cast<size_t>(std::floor(guess));

        if (j <= 0)
            return l_fill;
        if (j >= (N - 1))
            return u_fill;
        do
            j++;
        while (m_domain[j] < x);

        T slope =
            (m_range[j] - m_range[j - 1]) / (m_domain[j] - m_domain[j - 1]);
        return m_range[j - 1] + (x - m_domain[j - 1]) * slope;
    }

  private:
    T l_domain, u_domain, l_fill, u_fill;
    std::vector<T> m_domain, m_range;
};

} // namespace occ::core
