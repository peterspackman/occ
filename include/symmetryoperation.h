#pragma once
#include <string>
#include <Eigen/Dense>

namespace craso::crystal {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Matrix3Xd;


class SymmetryOperation
{
public:
    SymmetryOperation(const std::string&);
    SymmetryOperation(int);

    int to_int() const { return m_int; }
    void set_from_int(int);

    const std::string& to_string() const { return m_str; }
    void set_from_string(const std::string&);
    SymmetryOperation inverted() const;
    SymmetryOperation translated(const Vector3d&) const;
    bool is_identity() const { return m_int == 16484; }

    auto apply(const Matrix3Xd& frac) const {
        Eigen::Matrix3Xd tmp = m_rotation * frac;
        tmp.colwise() += m_translation;
        return tmp;
    }

    const auto& seitz() const { return m_seitz; }
    const auto& rotation() const { return m_rotation; }
    const auto& translation() const { return m_translation; }

    // Operators
    auto operator()(const Matrix3Xd& frac) const { return apply(frac); }
    bool operator==(const SymmetryOperation& other) const { return m_int == other.m_int; }
    bool operator<(const SymmetryOperation& other) const { return m_int < other.m_int; }
    bool operator>(const SymmetryOperation& other) const { return m_int > other.m_int; }
    bool operator<=(const SymmetryOperation& other) const { return m_int <= other.m_int; }
    bool operator>=(const SymmetryOperation& other) const { return m_int >= other.m_int; }
private:
    void update_from_seitz();
    int m_int;
    std::string m_str;
    Matrix4d m_seitz;
    // above is the core data, this is just convenience
    Matrix3d m_rotation;
    Vector3d m_translation;
};

}
