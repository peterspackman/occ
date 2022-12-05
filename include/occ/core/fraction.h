#pragma once
#include <numeric>
#include <string>

namespace occ::core {

/**
 * Class representing and holding data a rational number i.e. Fraction
 *
 */
class Fraction {

  public:
    /// Default constructor 0/1
    Fraction() = default;
    /// Constructor from two ints
    Fraction(int64_t, int64_t);
    /// Constructor from double, will find the nearest representation
    Fraction(double);
    /// Constructor from one int i.e. x/1
    Fraction(int64_t);
    /// Constructor from string, will try to parse expressions like x/y
    Fraction(const std::string &);
    /// represent this fraction as a string
    std::string to_string() const;
    template <typename T> T cast() const {
        return static_cast<T>(m_numerator) / static_cast<T>(m_denominator);
    }
    /// reduce this fraction by removing common factors in numerator/denominator
    const Fraction simplify() const;
    /// restrict this fraction to have a maximum denominator (i.e. round it)
    const Fraction limit_denominator(int64_t max_denominator = 1000000) const;
    /// Absolute value of this fraction
    const Fraction abs() const;
    /// Add this fraction to another
    const Fraction add(const Fraction &other) const;
    /// Subtract another fraction from this fraction
    const Fraction subtract(const Fraction &other) const;
    /// Multiply this fraction with another fraction
    const Fraction multiply(const Fraction &other) const;
    /// Divide this fraction by another fraction
    const Fraction divide(const Fraction &other) const;
    /// operator overload for Fraction::add
    const Fraction operator+(const Fraction &) const;
    /// operator overload for Fraction::subtract
    const Fraction operator-(const Fraction &) const;
    /// operator overload for Fraction::multiply
    const Fraction operator*(const Fraction &) const;
    /// operator overload for Fraction::divide
    const Fraction operator/(const Fraction &) const;
    auto operator==(const Fraction &) const -> bool;
    auto operator==(int64_t) const -> bool;
    auto operator<(const Fraction &) const -> bool;
    auto operator<=(const Fraction &) const -> bool;

  private:
    int64_t m_numerator{0};
    int64_t m_denominator{1};
};

} // namespace occ::core
