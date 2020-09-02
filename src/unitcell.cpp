#include "unitcell.h"
#include <cmath>


UnitCell::UnitCell(const Eigen::Vector3d& lengths, const Eigen::Vector3d& angles) : m_lengths{lengths}, m_angles{angles}
{
    updateCellMatrices();
}

UnitCell::UnitCell(double a, double b, double c, double alpha, double beta, double gamma) :
    m_lengths{a, b, c}, m_angles{alpha, beta, gamma}
{
    updateCellMatrices();
}

void UnitCell::setA(double a)
{
    if(a >= 0.0) {
        m_lengths(0) = a;
        updateCellMatrices();
    }
}

void UnitCell::setB(double b)
{
    if(b >= 0.0) {
        m_lengths[1] = b;
        updateCellMatrices();
    }
}

void UnitCell::setC(double c)
{
    if(c >= 0.0) {
        m_lengths[2] = c;
        updateCellMatrices();
    }
}

void UnitCell::setAlpha(double a)
{
    if(a >= 0.0) {
        m_angles[0] = a;
        updateCellMatrices();
    }
}

void UnitCell::setBeta(double b)
{
    if(b >= 0.0) {
        m_angles[1] = b;
        updateCellMatrices();
    }
}

void UnitCell::setGamma(double c)
{
    if(c >= 0.0) {
        m_angles[2] = c;
        updateCellMatrices();
    }
}

void UnitCell::updateCellMatrices()
{
    for(int i = 0; i < 3; i++) {
        m_sin[i] = sin(m_angles[i]);
        m_cos[i] = cos(m_angles[i]);
    }

    const double a = m_lengths[0], ca = m_cos[0];
    const double b = m_lengths[1], cb = m_cos[1];
    const double c = m_lengths[2], sg = m_sin[2], cg = m_cos[2];
    m_volume = a * b * c * sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);

    m_direct <<
          a, b * cg, c * cb,
        0.0, b * sg, c * (ca - cb * cg) / sg,
        0.0,    0.0, m_volume / (a * b * sg);

    m_reciprocal <<
                  1 / a, 0.0, 0.0,
         -cg / (a * sg),  1 / (b * sg), 0.0,
         b * c * (ca * cg - cb) / (m_volume * sg),
         a * c * (cb * cg - ca) / (m_volume * sg),
         a * b * sg / m_volume;

    m_inverse = m_reciprocal.transpose();
    m_directf = m_direct.cast<float>();
    m_inversef = m_inversef.cast<float>();
    m_reciprocalf = m_reciprocalf.cast<float>();
}

bool UnitCell::isCubic() const
{
    return _abc_close() && isOrthogonal();
}

bool UnitCell::isTriclinic() const
{
    return _abc_different() && _a_abc_different();
}

bool UnitCell::isMonoclinic() const
{
    return _a_ac_close() && _abc_different();
}

bool UnitCell::isOrthorhombic() const
{
    return isOrthogonal() && _abc_different();
}

bool UnitCell::isTetragonal() const
{
    return _ab_close() && !_ac_close() && isOrthogonal();
}

bool UnitCell::isRhombohedral() const
{
    return _abc_close() && _a_abc_close() && ! _a_90();
}

bool UnitCell::isHexagonal() const
{
    return _ab_close() && !_ac_close() && _a_90() && isclose(m_angles[2], 2 * M_PI / 3);
}

QString UnitCell::cellTypeString() const {
    if(isCubic()) return "cubic";
    if(isRhombohedral()) return "rhombohedral";
    if(isHexagonal()) return "hexagonal";
    if(isTetragonal()) return "tetragonal";
    if(isOrthorhombic()) return "orthorhombic";
    if(isMonoclinic()) return "monoclinic";
    return "triclinic";
}
