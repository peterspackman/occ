#pragma once

namespace craso::hf {

class FockMatrixBuilder {

public:
    FockMatrixBuilder() : m_nthreads{1} {}
private:
    int m_nthreads{1};
};

}