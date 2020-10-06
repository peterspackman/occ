#pragma once
#include "linear_algebra.h"
#include "numgrid.h"
#include <vector>
#include <xc.h>
#include <string>

namespace libint2 {
class BasisSet;
class Atom;
}

namespace tonto::dft {

using tonto::Mat3N;
using tonto::MatRM;
using tonto::MatN4;
using tonto::Vec;
using tonto::IVec;

class DFTGrid {
public:
    DFTGrid(const libint2::BasisSet&, const std::vector<libint2::Atom>&);
    const auto& atomic_numbers() const { return m_atomic_numbers; }
    const auto n_atoms() const { return m_atomic_numbers.size(); }
    MatN4 grid_points(size_t idx) const;
    void set_radial_precision(double prec) { m_radial_precision = prec; }
    void set_min_angular_points(size_t n) { m_min_angular = n; }
    void set_max_angular_points(size_t n) { m_max_angular = n; }

private:
    double m_radial_precision{1e-12};
    size_t m_min_angular{86};
    size_t m_max_angular{302};
    IVec m_l_max;
    Vec m_x;
    Vec m_y;
    Vec m_z;
    IVec m_atomic_numbers;
    Vec m_alpha_max;
    MatRM m_alpha_min;
};


class DensityFunctional {
public:
    enum Family {
        LDA = XC_FAMILY_LDA,
        GGA = XC_FAMILY_GGA,
        HGGA = XC_FAMILY_HYB_GGA,
        MGGA = XC_FAMILY_MGGA,
        HMGGA = XC_FAMILY_HYB_MGGA
    };
    enum Kind {
        Exchange = XC_EXCHANGE,
        Correlation = XC_CORRELATION,
        ExchangeCorrelation = XC_EXCHANGE_CORRELATION,
        Kinetic = XC_KINETIC
    };

    DensityFunctional(const std::string&);
    ~DensityFunctional();

    Family family() const { return static_cast<Family>(m_func.info->family); }
    Kind kind() const { return static_cast<Kind>(m_func.info->kind); }

    std::string kind_string() const {
        switch(kind()) {
        case Exchange: return "exchange";
        case Correlation: return "correlation";
        case ExchangeCorrelation: return "exchange-correlation";
        case Kinetic: return "kinetic";
        default: return "unknown kind";
        }
    }

    tonto::Vec energy(const tonto::Vec& rho) const;
    tonto::Vec potential(const tonto::Vec& rho) const;

    std::string family_string() const {
        switch(family()) {
        case LDA: return "LDA";
        case GGA: return "GGA";
        case HGGA: return "hybrid GGA";
        case MGGA: return "meta-GGA";
        case HMGGA: return "hybrid meta-GGA";
        default: return "unknown family";
        }
    }
private:
    xc_func_type m_func;
};
}
