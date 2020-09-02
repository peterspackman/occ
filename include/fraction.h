#pragma once
#include <numeric>
#include <QString>

class Fraction {

public:
     Fraction() = default;
     Fraction(int64_t, int64_t);
     Fraction(double);
     Fraction(int64_t);
     Fraction(const QString&);
     QString toString() const;
     double toDouble() const;
     const Fraction simplify() const;
     const Fraction limitDenominator(int64_t max_denominator=1000000) const;
     const Fraction abs() const;
     const Fraction add(const Fraction& other) const;
     const Fraction subtract(const Fraction& other) const;
     const Fraction multiply(const Fraction& other) const;
     const Fraction divide(const Fraction& other) const;
     const Fraction operator+(const Fraction&) const;
     const Fraction operator-(const Fraction&) const;
     const Fraction operator*(const Fraction&) const;
     const Fraction operator/(const Fraction&) const;
     bool operator==(const Fraction&) const;
     bool operator==(int64_t) const;
     bool operator<(const Fraction&) const;
     bool operator<=(const Fraction&) const;

private:
     int64_t m_numerator{0};
     int64_t m_denominator{1};
};
