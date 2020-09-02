#include "molecule.h"
#include <Eigen/Core>

Molecule::Molecule(const Eigen::VectorXi& nums, const Eigen::Matrix3Xd& pos) :
    m_atomicNumbers(nums), m_positions(pos)
{
    for(size_t i = 0; i < size(); i++) {
        m_elements.push_back(Elements::Element(m_atomicNumbers(i)));
    }
    m_name = Elements::chemicalFormula(m_elements);
}


