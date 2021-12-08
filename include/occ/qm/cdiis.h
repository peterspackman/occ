#pragma once
#include <occ/core/diis.h>

namespace occ::qm {

class CDIIS : public occ::core::diis::DIIS {
public:
    Mat update(const Mat &overlap, const Mat &D, const Mat &F);
    double error() const { return m_error; }
private:
    double m_error{1.0};
};


}
