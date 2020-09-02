#pragma once
#include <QString>
#include <Eigen/Dense>
#include <QMatrix4x4>
#include <QMatrix3x3>
#include <QVector3D>

class SymmetryOperation
{

public:
    SymmetryOperation(const QString&);
    SymmetryOperation(int);

    int integerCode() const;
    int toInt() const { return integerCode(); }
    void setIntegerCode(int);

    const QString& stringCode() const;
    void setStringCode(const QString&);
    SymmetryOperation inverted() const;
    SymmetryOperation translated(const QVector3D&) const;
    bool isIdentity() const { return m_intCode == 16484; }

    Eigen::Matrix3Xd apply(const Eigen::Matrix3Xd&) const;
    const QMatrix4x4 seitz() const { return QMatrix4x4(m_seitzf.data()); }
    const QMatrix3x3 rotation() const { return QMatrix3x3(m_rotationf.data()); }
    const QVector3D translation() const { return QVector3D(m_seitzf(0, 3), m_seitzf(1, 3), m_seitzf(2, 3)); }
    // Operators
    Eigen::Matrix3Xd operator()(const Eigen::Matrix3Xd& frac) const { return apply(frac); }
    bool operator==(const SymmetryOperation& other) const { return m_intCode == other.m_intCode; }
    bool operator<(const SymmetryOperation& other) const { return m_intCode < other.m_intCode; }
    bool operator>(const SymmetryOperation& other) const { return m_intCode > other.m_intCode; }
    bool operator<=(const SymmetryOperation& other) const { return m_intCode <= other.m_intCode; }
    bool operator>=(const SymmetryOperation& other) const { return m_intCode >= other.m_intCode; }
private:
    void updateFromSeitz();
    int m_intCode;
    QString m_stringCode;
    Eigen::Matrix4d m_seitz;
    // above is the core data, this is just convenience
    Eigen::Matrix4f m_seitzf;
    Eigen::Matrix3d m_rotation;
    Eigen::Matrix3f m_rotationf;
    Eigen::Vector3d m_translation;
};
