#pragma once
#include <QMap>
#include <QString>
#include <QRegularExpression>
#include <QTextStream>
#include "crystal.h"
#include <gemmi/cif.hpp>

struct AtomData {
    QString element;
    QString site_label;
    QString residue_name;
    QString chain_id;
    int residue_number;
    double position[3];
};

struct CellData {
    double a{-1}, b{-1}, c{-1};
    double alpha{-1}, beta{-1}, gamma{-1};
    bool valid() const {
        if(a <= 0) return false;
        if(b <= 0) return false;
        if(c <= 0) return false;
        if(alpha <= 0) return false;
        if(beta <= 0) return false;
        if(gamma <= 0) return false;
        return true;
    }
};

struct SymmetryData {
    int number{-1};
    QString nameHM{"Not set"};
    QString nameHall{"Not set"};
    QVector<QString> symops;
    size_t symopCount() const {
        return symops.length();
    }
    bool valid() const {
        if(number > 0) return true;
        if(nameHM != "Not set") return true;
        if(nameHall != "Not set") return true;
        if(symopCount() > 0) return true;
        return false;
    }

};

class CifParser
{
public:
    CifParser();
    void dump();
    const QVector<AtomData>& atomData() const { return m_atoms; }
    const SymmetryData& symmetryData() const { return m_sym; }
    const CellData& cellData() const { return m_cell; }
    bool symmetryValid() const { return m_sym.valid(); }
    bool cellValid() const { return m_cell.valid(); }
    bool crystalValid() const { return symmetryValid() && cellValid(); }
    const QString& failureDescription() const { return m_failureDescription; }
    size_t atomCount() const { return m_atoms.length(); }
    Crystal* parseCrystal(const QString&);
private:
    void extractAtomSites(const gemmi::cif::Loop&);
    void extractCellParameter(const gemmi::cif::Pair&);
    void extractSymmetryOperations(const gemmi::cif::Loop&);
    QString m_failureDescription;
    QVector<AtomData> m_atoms;
    SymmetryData m_sym;
    CellData m_cell;
};
