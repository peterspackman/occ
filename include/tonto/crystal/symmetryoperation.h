#pragma once
#include <tonto/core/linear_algebra.h>
#include <string>

namespace tonto::crystal {

using tonto::Mat3;
using tonto::Mat3N;
using tonto::Mat4;
using tonto::Vec3;

class SymmetryOperation {
public:
    SymmetryOperation(const tonto::Mat4 &);
    SymmetryOperation(const std::string &);
    SymmetryOperation(int);

    int to_int() const;
    std::string to_string() const;

    SymmetryOperation inverted() const;
    SymmetryOperation translated(const Vec3 &) const;

    bool is_identity() const { return to_int() == 16484; }

    Mat3N apply(const Mat3N &frac) const;
    const auto &seitz() const { return m_seitz; }
    Mat3 rotation() const { return m_seitz.block<3, 3>(0, 0); }
    Vec3 translation() const { return m_seitz.block<3, 1>(0, 3); }

    // Operators
    auto operator()(const Mat3N &frac) const { return apply(frac); }
    bool operator==(const SymmetryOperation &other) const {
        return to_int() == other.to_int();
    }
    bool operator<(const SymmetryOperation &other) const {
        return to_int() < other.to_int();
    }
    bool operator>(const SymmetryOperation &other) const {
        return to_int() > other.to_int();
    }
    bool operator<=(const SymmetryOperation &other) const {
        return to_int() <= other.to_int();
    }
    bool operator>=(const SymmetryOperation &other) const {
        return to_int() >= other.to_int();
    }

    const SymmetryOperation operator*(const SymmetryOperation &other) const
    {
        return SymmetryOperation(seitz() * other.seitz());
    }

private:
    Mat4 m_seitz;
};

} // namespace tonto::crystal
