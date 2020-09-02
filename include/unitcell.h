#pragma once

#include <QObject>
#include <QString>
#include <Eigen/Dense>
#include <QVector3D>
#include <QMatrix3x3>
#include "util.h"

using cx::isclose;


class UnitCell
{
public:
    UnitCell(double, double, double, double, double, double);
    UnitCell(const Eigen::Vector3d&, const Eigen::Vector3d&);

    double a() const { return m_lengths[0]; }
    double b() const { return m_lengths[1]; }
    double c() const { return m_lengths[2]; }
    double alpha() const { return m_angles[0]; }
    double beta() const { return m_angles[1]; }
    double gamma() const { return m_angles[2]; }
    void setA(double);
    void setB(double);
    void setC(double);
    void setAlpha(double);
    void setBeta(double);
    void setGamma(double);

    bool isCubic() const;
    bool isTriclinic() const;
    bool isMonoclinic() const;
    bool isOrthorhombic() const;
    bool isTetragonal() const;
    bool isRhombohedral() const;
    bool isHexagonal() const;
    bool isOrthogonal() const { return _a_90() && _b_90() && _c_90(); }
    QString cellTypeString() const;

    Eigen::Matrix3Xd toCartesian(const Eigen::Matrix3Xd& coords) const { return m_direct * coords; }
    Eigen::Matrix3Xd toFractional(const Eigen::Matrix3Xd& coords) const { return m_inverse * coords; }
    const QMatrix3x3 direct() const { return QMatrix3x3(m_directf.data()); }
    const QMatrix3x3 reciprocal() const { return QMatrix3x3(m_reciprocalf.data()); }
    const QMatrix3x3 inverse() const { return QMatrix3x3(m_inversef.data()); }
    const QVector3D aVector() const { return QVector3D(m_directf(0, 0), m_directf(1, 0), m_directf(2, 0)); }
    const QVector3D bVector() const { return QVector3D(m_directf(0, 1), m_directf(1, 1), m_directf(2, 1)); }
    const QVector3D cVector() const { return QVector3D(m_directf(0, 2), m_directf(1, 2), m_directf(2, 2)); }
    const QVector3D aStarVector() const { return QVector3D(m_reciprocalf(0, 0), m_reciprocalf(1, 0), m_reciprocalf(2, 0)); }
    const QVector3D bStarVector() const { return QVector3D(m_reciprocalf(0, 1), m_reciprocalf(1, 1), m_reciprocalf(2, 1)); }
    const QVector3D cStarVector() const { return QVector3D(m_reciprocalf(0, 2), m_reciprocalf(1, 2), m_reciprocalf(2, 2)); }

private:
    void updateCellMatrices();
    inline bool _ab_close() const { return isclose(m_lengths[0], m_lengths[1]); }
    inline bool _ac_close() const { return isclose(m_lengths[0], m_lengths[2]); }
    inline bool _bc_close() const { return isclose(m_lengths[1], m_lengths[2]); }
    inline bool _abc_close() const { return _ab_close() && _ac_close() && _bc_close(); }
    inline bool _abc_different() const { return !_ab_close() && !_ac_close() && !_bc_close(); }
    inline bool _a_ab_close() const { return isclose(m_angles[0], m_angles[1]); }
    inline bool _a_ac_close() const { return isclose(m_angles[0], m_angles[2]); }
    inline bool _a_bc_close() const { return isclose(m_angles[1], m_angles[2]); }
    inline bool _a_abc_close() const { return _a_ab_close() && _a_ac_close() && _a_bc_close(); }
    inline bool _a_abc_different() const { return !_a_ab_close() && !_a_ac_close() && !_a_bc_close(); }
    inline bool _a_90() const { return isclose(m_angles[0], M_PI/2); }
    inline bool _b_90() const { return isclose(m_angles[1], M_PI/2); }
    inline bool _c_90() const { return isclose(m_angles[2], M_PI/2); }

    Eigen::Vector3d m_lengths;
    Eigen::Vector3d m_angles;
    Eigen::Vector3d m_sin;
    Eigen::Vector3d m_cos;
    double m_volume = 0.0;

    Eigen::Matrix3d m_direct;
    Eigen::Matrix3d m_inverse;
    Eigen::Matrix3d m_reciprocal;
    Eigen::Matrix3f m_directf;
    Eigen::Matrix3f m_inversef;
    Eigen::Matrix3f m_reciprocalf;
};

