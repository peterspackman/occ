#pragma once
#include "symmetryoperation.h"
#include <QString>
#include <QVector>
#include <QMap>
#include <QPair>
#include <tuple>

class SpaceGroup
{
    friend class SpaceGroupTable;
public:
    SpaceGroup() = default;
    SpaceGroup(const QVector<SymmetryOperation>& symops);
    int internationalTablesNumber() const { return m_number;}
    const QString& symbol() const { return m_symbol; }
    void setSymbol(const QString &sym) { m_symbol = sym; }
    const QString& choice() const { return m_choice; }
    const QVector<SymmetryOperation>& symmetryOperations() const { return m_symops; }
    std::pair<Eigen::VectorXi, Eigen::Matrix3Xd> applyAllSymmetryOperations(const Eigen::Matrix3Xd&) const;
    const QString crystalSystem() const;
    int latticeNumber() const;
    bool hasHexagonalRhombohedralChoice() const;
private:
    SpaceGroup(int, QString, QString, QString, QString, bool, QVector<SymmetryOperation>);
    int m_number = -1;
    QString m_symbol;
    QString m_fullName;
    QString m_choice;
    QString m_centering;
    bool m_centrosymmetric;
    QVector<SymmetryOperation> m_symops;
};

class SpaceGroupTable {
private:
    static SpaceGroupTable *instance;
    SpaceGroupTable();
    QVector<SpaceGroup> m_spaceGroups;
    QMap<int, int> m_intToSpaceGroup;
    QMap<QString, int> m_symbolToSpaceGroup;
    QMap<QPair<int, QString>, int> m_intChoiceToSpaceGroup;
public:
    static SpaceGroupTable *table();
    const SpaceGroup& get(int) const;
    const SpaceGroup& get(int, const QString&) const;
    SpaceGroup get(QVector<SymmetryOperation>) const;
};



