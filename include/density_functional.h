#pragma once
#include "linear_algebra.h"
#include <string>
#include <xc.h>
#include <memory>

namespace tonto::dft {


using tonto::Vec;
using tonto::IVec;
using tonto::MatRM;


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

    Family family() const { return static_cast<Family>(m_func.get()->info->family); }
    Kind kind() const { return static_cast<Kind>(m_func.get()->info->kind); }

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
    std::string m_func_name;
    std::unique_ptr<xc_func_type> m_func;
};

}
