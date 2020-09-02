#pragma once
#include <QStandardItem>
#include <Eigen/Dense>
#include "element.h"
#include <QDebug>

class Molecule
{
public:
    Molecule(const Eigen::VectorXi&, const Eigen::Matrix3Xd&);
    size_t size() const { return m_atomicNumbers.size(); }
    void setName(const QString&);
    const QString& name() const { return m_name; }
    const Eigen::Matrix3Xd& positions() const { return m_positions; }
    const Eigen::VectorXi& atomicNumbers() const {return m_atomicNumbers;}
    void addBond(size_t l, size_t r) { m_bonds.push_back({l, r}); }
    void setBonds(const QVector<QPair<size_t, size_t>>& bonds) { m_bonds = bonds; }
    const QVector<QPair<size_t, size_t>>& bonds() const { return m_bonds; }
    void setUnitCellIndex(const Eigen::VectorXi& idx) { m_ucIdx = idx; }
    void setAsymmetricUnitIndex(const Eigen::VectorXi& idx) { m_asymIdx = idx; }
    void dump() const {
        for(size_t i = 0; i < size(); i++) {
            qDebug() << m_atomicNumbers(i) << m_positions(0, i) << m_positions(1, i) << m_positions(2, i);
        }
    }
private:
    QString m_name{""};
    Eigen::VectorXi m_atomicNumbers;
    Eigen::Matrix3Xd m_positions;
    Eigen::VectorXi m_ucIdx;
    Eigen::VectorXi m_asymIdx;
    QVector<QPair<size_t, size_t>> m_bonds;
    QVector<Elements::Element> m_elements;
};
