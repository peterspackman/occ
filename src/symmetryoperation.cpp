#include "symmetryoperation.h"
#include <QRegularExpression>
#include "fraction.h"

namespace craso::crystal {

void decodeInteger(int code, Eigen::Matrix4d& seitz)
{
    /*Decode an integer encoded symmetry operation.

    A space group operation is compressed using ternary numerical system for
    rotation and duodecimal system for translation. This is achieved because
    each element of rotation matrix can have only one of {-1,0,1}, and the
    translation can have one of {0,2,3,4,6,8,9,10} divided by 12.  Therefore
    3^9 * 12^3 = 34012224 different values can map space group operations. In
    principle, octal numerical system can be used for translation, but
    duodecimal system is more convenient.
    */
    seitz.block(3, 0, 1, 3).setZero();
    seitz(3, 3) = 1.0;
    int r = code % 19683; // 19683 = 3**9
    int shift = 6561; // 6561 = 3**8
    int t = code / 19683;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // we need integer division here
            seitz(i, j) = (r % (shift * 3)) / shift - 1;
            shift /= 3;
        }
    }

    shift = 144;
    for (int i = 0; i < 3; i++) {
        // we need integer division here by shift
        seitz(i, 3) = ((t % (shift * 12)) / shift) / 12.0;
        shift /= 12;
    }
}

QStringList getSymbols(QString s) {
    QStringList symbols;
    QRegularExpression re(".*?([+-]*[xyzXYZ0-9/\\.]+)");
    auto i = re.globalMatch(s);
    while(i.hasNext()) {
        auto match = i.next();
        symbols << QString(match.captured(1));
    }
    return symbols;
}

void decodeString(QString code, Eigen::Matrix4d& seitz)
{
    seitz.fill(0.0);
    QStringList tokens = code.toLower().replace(" ", "").split(",");
    int i = 0;
    int idx = 0;
    for (const auto& x: tokens)
    {
        int fac = 1;
        auto symbols = getSymbols(x);
        for(const auto& symbol: symbols)
        {
            if(symbol.contains("x")) {
                idx = 0;
                fac = symbol.contains("-x") ? -1 : 1;
                seitz(i, idx) = fac;
            }
            else if(symbol.contains("y")) {
                idx = 1;
                fac = symbol.contains("-y") ? -1 : 1;
                seitz(i, idx) = fac;
            }
            else if(symbol.contains("z")) {
                idx = 2;
                fac = symbol.contains("-z") ? -1 : 1;
                seitz(i, idx) = fac;
            }
            else {
                seitz(i, 3) = fmod(Fraction(symbol).toDouble(), 1.0);
            }
        }
        i++;
    }
    seitz.block(3, 0, 1, 3).setZero();
    seitz(3, 3) = 1.0;
}

QString encodeString(const Eigen::Matrix4d& seitz) {
    /* Encode a rotation matrix (of -1, 0, 1s) and (rational) translation vector
    into string form e.g. 1/2-x,z-1/3,-y-1/6
    */

    QString symbols = "xyz";
    QStringList res;
    for (int i = 0; i < 3; i++) {
        auto t = Fraction(seitz(i, 3)).limitDenominator(12);
        QString v = "";
        if (!(t == 0)) {
            v += t.toString();
        }
        for (int j = 0; j < 3; j++) {
            auto c = seitz(i, j);
            if (c < 0) {
                v += "-" + symbols[j];
            }
            else if (c > 0) {
                v += "+" + symbols[j];
            }
        }
        res.append(v);
    }
    return res.join(",");

}

int encodeInteger(const Eigen::Matrix4d& seitz) {
    /* Encode an integer encoded symmetry from a rotation matrix and translation
    vector.

    A space group operation is compressed using ternary numerical system for
    rotation and duodecimal system for translation. This is achieved because
    each element of rotation matrix can have only one of {-1,0,1}, and the
    translation can have one of {0,2,3,4,6,8,9,10} divided by 12.  Therefore
    3^9 * 12^3 = 34012224 different values can map space group operations. In
    principle, octal numerical system can be used for translation, but
    duodecimal system is more convenient.
    */

    int r = 0, t = 0, shift = 1;
    // encode rotation component
    for (int i = 2; i >= 0; i--) {
        for (int j = 2; j >= 0; j--) {
            r += (static_cast<int>(seitz.coeff(i, j)) + 1) * shift;
            shift *= 3;
        }
    }
    shift = 1;
    for (int i = 2; i >= 0; i--) {
        t += static_cast<int>(seitz.coeff(i, 3) * 12) * shift;
        shift *= 12;
    }
    return r + t * 19683;
}

SymmetryOperation::SymmetryOperation(int code)
{
    set_from_int(code);
}

SymmetryOperation::SymmetryOperation(const QString& str)
{
    set_from_string(str);
}

int SymmetryOperation::integerCode() const
{
    return m_int;
}

SymmetryOperation SymmetryOperation::translated(const QVector3D& t) const
{
    SymmetryOperation result = *this;
    result.m_seitz(0, 3) += t[0];
    result.m_seitz(1, 3) += t[1];
    result.m_seitz(2, 3) += t[2];
    result.update_from_seitz();
    result.m_str = encodeString(result.m_seitz);
    result.m_int = encodeInteger(result.m_seitz);
    return result;
}

SymmetryOperation SymmetryOperation::inverted() const
{
    SymmetryOperation result = *this;
    result.m_seitz.block(0, 0, 3, 4) *= -1;
    result.update_from_seitz();
    result.m_str = encodeString(result.m_seitz);
    result.m_int = encodeInteger(result.m_seitz);
    return result;
}

void SymmetryOperation::set_from_int(int code)
{
    if(code == m_int) return;
    m_int = code;
    decodeInteger(code, m_seitz);
    update_from_seitz();
    m_str = encodeString(m_seitz);
}

void SymmetryOperation::set_from_string(const QString &code) {
    if(code == m_str) return;
    decodeString(code, m_seitz);
    m_str = code;
    update_from_seitz();
    m_int = encodeInteger(m_seitz);
}

void SymmetryOperation::update_from_seitz()
{
    m_seitzf = m_seitz.cast<float>();
    m_rotationf = m_seitzf.block(0, 0, m_rotationf.rows(), m_rotationf.cols());
    m_rotation = m_seitz.block(0, 0, m_rotation.rows(), m_rotation.cols());
    m_translation = m_seitz.block(0, 3, m_translation.rows(), m_translation.cols());
}

Eigen::Matrix3Xd SymmetryOperation::apply(const Eigen::Matrix3Xd &frac) const
{
    Eigen::Matrix3Xd tmp = m_rotation * frac;
    tmp.colwise() += m_translation;
    return tmp;
}

}
