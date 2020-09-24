#include "fraction.h"
#include "util.h"
#include <fmt/core.h>
#include <math.h>
#include <vector>

namespace craso::numeric {

using craso::util::tokenize;

template <typename F, typename I>
std::tuple<I, I> as_integer_ratio(F f, I max_denominator) {
  I a, h[3] = {0, 1, 0}, k[3] = {1, 0, 0};
  I x, d, n = 1;
  I i, neg = 0;
  I num, denom;

  if (max_denominator <= 1) {
    return {0, 1};
  }

  if (f < 0) {
    neg = 1;
    f = -f;
  }

  while (f != floor(f)) {
    n <<= 1;
    f *= 2;
  }
  d = f;

  /* continued fraction and check denominator each step */
  for (i = 0; i < 64; i++) {
    a = n ? d / n : 0;
    if (i && !a)
      break;

    x = d;
    d = n;
    n = x % n;

    x = a;
    if (k[1] * a + k[0] >= max_denominator) {
      x = (max_denominator - k[0]) / k[1];
      if (x * 2 >= a || k[1] >= max_denominator)
        i = 65;
      else
        break;
    }

    h[2] = x * h[1] + h[0];
    h[0] = h[1];
    h[1] = h[2];
    k[2] = x * k[1] + k[0];
    k[0] = k[1];
    k[1] = k[2];
  }
  denom = k[1];
  num = neg ? -h[1] : h[1];
  return {num, denom};
}

Fraction::Fraction(int64_t numerator, int64_t denominator)
    : m_numerator(numerator), m_denominator(denominator) {}

Fraction::Fraction(int64_t numerator) : m_numerator(numerator) {}

Fraction::Fraction(const std::string &expr) {
  if (expr.find('/') != std::string::npos) {
    auto tokens = tokenize(expr, "/");
    double numerator = std::stod(tokens[0]);
    double denominator = std::stod(tokens[1]);
    Fraction f = (Fraction(numerator) / Fraction(denominator)).simplify();
    m_numerator = f.m_numerator;
    m_denominator = f.m_denominator;
  } else {
    Fraction f = Fraction(std::stod(expr)).simplify();
    m_numerator = f.m_numerator;
    m_denominator = f.m_denominator;
  }
}

Fraction::Fraction(double value) {
  std::tie(m_numerator, m_denominator) =
      as_integer_ratio<double, int64_t>(value, 1000000);
}

std::string Fraction::to_string() const {
  return fmt::format("{}/{}", m_numerator, m_denominator);
}

const Fraction Fraction::simplify() const {
  int64_t gcd = std::gcd(m_numerator, m_denominator);
  return Fraction(m_numerator / gcd, m_denominator / gcd);
}

const Fraction Fraction::limit_denominator(int64_t max_denominator) const {

  /*
   Algorithm notes: For any real number x, define a *best upper
   approximation* to x to be a rational number p/q such that:

     (1) p/q >= x, and
     (2) if p/q > r/s >= x then s > q, for any rational r/s.

   Define *best lower approximation* similarly.  Then it can be
   proved that a rational number is a best upper or lower
   approximation to x if, and only if, it is a convergent or
   semiconvergent of the (unique shortest) continued fraction
   associated to x.

   To find a best rational approximation with denominator <= M,
   we find the best upper and lower approximations with
   denominator <= M and take whichever of these is closer to x.
   In the event of a tie, the bound with smaller denominator is
   chosen.  If both denominators are equal (which can happen
   only when max_denominator == 1 and self is midway between
   two int64_tegers) the lower bound---i.e., the floor of self, is
   taken.
  */
  if (m_denominator <= max_denominator)
    return Fraction(m_numerator, m_denominator);

  int64_t p0 = 0, q0 = 1, p1 = 1, q1 = 0;
  int64_t n = m_numerator, d = m_denominator;
  while (true) {
    int64_t a = n / d;
    int64_t q2 = q0 + a * q1;
    if (q2 > max_denominator)
      break;
    int64_t p0tmp = p0, p1tmp = p1;
    p0 = p1tmp;
    q0 = q1;
    p1 = p0tmp + a * p1tmp;
    q1 = q2;
    int64_t ntmp = n;
    n = d;
    d = ntmp - a * d;
  }
  int64_t k = (max_denominator - q0) / q1;
  auto bound1 = Fraction(p0 + k * p1, q0 + k * q1);
  auto bound2 = Fraction(p1, q1);
  if ((subtract(bound2).abs()) <= (subtract(bound1).abs())) {
    return bound2;
  } else {
    return bound1;
  }
}

const Fraction Fraction::abs() const {
  return Fraction(std::abs(m_numerator), std::abs(m_denominator));
}

const Fraction Fraction::add(const Fraction &other) const {
  int64_t numeratorTemp(m_numerator);
  int64_t denominatorTemp(m_denominator);

  if (m_denominator == other.m_denominator) {
    numeratorTemp = m_numerator + other.m_numerator;
  } else {
    numeratorTemp =
        m_numerator * other.m_denominator + other.m_numerator * m_denominator;
    denominatorTemp = m_denominator * other.m_denominator;
  }
  return Fraction(numeratorTemp, denominatorTemp);
}

const Fraction Fraction::subtract(const Fraction &other) const {
  int64_t numeratorTemp(m_numerator);
  int64_t denominatorTemp(m_denominator);

  if (m_denominator == other.m_denominator) {
    numeratorTemp = m_numerator - other.m_numerator;
  } else {
    numeratorTemp =
        m_numerator * other.m_denominator - other.m_numerator * m_denominator;
    denominatorTemp = m_denominator * other.m_denominator;
  }
  return Fraction(numeratorTemp, denominatorTemp);
}

const Fraction Fraction::divide(const Fraction &other) const {
  int64_t numeratorTemp(m_numerator);
  int64_t denominatorTemp(m_denominator);

  numeratorTemp = m_numerator * other.m_denominator;
  denominatorTemp = m_denominator * other.m_numerator;
  return Fraction(numeratorTemp, denominatorTemp);
}

const Fraction Fraction::multiply(const Fraction &other) const {
  int64_t numeratorTemp(m_numerator);
  int64_t denominatorTemp(m_denominator);

  numeratorTemp = m_numerator * other.m_numerator;
  denominatorTemp = m_denominator * other.m_denominator;

  return Fraction(numeratorTemp, denominatorTemp);
}

const Fraction Fraction::operator+(const Fraction &other) const {
  return add(other);
}

const Fraction Fraction::operator-(const Fraction &other) const {
  return subtract(other);
}

const Fraction Fraction::operator*(const Fraction &other) const {
  return multiply(other);
}

const Fraction Fraction::operator/(const Fraction &other) const {
  return divide(other);
}

bool Fraction::operator==(const Fraction &other) const {
  return (m_numerator == other.m_numerator) &&
         (m_denominator == other.m_denominator);
}

bool Fraction::operator==(int64_t n) const {
  return (m_denominator == 1) && (m_numerator == n);
}

bool Fraction::operator<(const Fraction &other) const {
  return cast<double>() < other.cast<double>();
}

bool Fraction::operator<=(const Fraction &other) const {
  return cast<double>() <= other.cast<double>();
}
} // namespace craso::numeric
