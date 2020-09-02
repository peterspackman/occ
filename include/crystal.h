#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <QVector>
#include <QString>
#include "spacegroup.h"
#include "unitcell.h"
#include "periodicbondgraph.h"
#include "molecule.h"

struct HKL {
    int h{0}, k{0}, l{0};
};

struct AtomSlab {
    Eigen::Matrix3Xd fracPos;
    Eigen::Matrix3Xd cartPos;
    Eigen::VectorXi asymIndex;
    Eigen::VectorXi atomicNumbers;
    Eigen::VectorXi symop;
    void resize(size_t n) {
        fracPos.resize(3, n);
        cartPos.resize(3, n);
        asymIndex.resize(n);
        atomicNumbers.resize(n);
        symop.resize(n);
    }
    size_t size() const { return fracPos.cols(); }
};

struct AsymmetricUnit
{
    Eigen::Matrix3Xd positions;
    Eigen::VectorXi atomicNumbers;
    Eigen::VectorXd occupations;
    QVector<QString> labels;
    QString chemicalFormula() const;
    Eigen::VectorXd covalentRadii() const;
};

class Crystal
{
public:
    Crystal(const AsymmetricUnit&, const SpaceGroup&, const UnitCell&);
    const QVector<QString>& labels() const { return m_asymmetricUnit.labels; }
    const Eigen::Matrix3Xd& frac() const { return m_asymmetricUnit.positions; }
    inline auto toFractional(const Eigen::Matrix3Xd& p) const { return m_unitCell.toFractional(p); }
    inline auto toCartesian(const Eigen::Matrix3Xd& p) const { return m_unitCell.toCartesian(p); }
    inline int numSites() const { return m_asymmetricUnit.atomicNumbers.size(); }
    inline const QVector<SymmetryOperation>& symmetryOperations() const { return m_spaceGroup.symmetryOperations(); }
    const SpaceGroup& spaceGroup() const { return m_spaceGroup; }
    const AsymmetricUnit& asymmetricUnit() const { return m_asymmetricUnit; }
    const UnitCell& unitCell() const { return m_unitCell; }
    AtomSlab slab(const HKL&, const HKL&) const;
    const AtomSlab& unitCellAtoms() const;
    const PeriodicBondGraph& unitCellConnectivity() const;
    const QVector<Molecule>& unitCellMolecules() const;
private:
    AsymmetricUnit m_asymmetricUnit;
    SpaceGroup m_spaceGroup;
    UnitCell m_unitCell;
    mutable PeriodicBondGraph m_bondGraph;
    mutable AtomSlab m_unitCellAtoms;
    mutable bool m_unitCellAtomsNeedUpdate{true};
    mutable bool m_unitCellConnectivityNeedUpdate{true};
    mutable QVector<Molecule> m_unitCellMolecules{};
};

