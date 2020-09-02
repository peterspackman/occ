#include "cifparser.h"
#include <QDebug>
#include "util.h"
#include "element.h"
#include <gemmi/numb.hpp>
#include <iostream>

CifParser::CifParser()
{
}

void CifParser::extractAtomSites(const gemmi::cif::Loop &loop)
{
    int label_idx = loop.find_tag("_atom_site_label");
    int symbol_idx = loop.find_tag("_atom_site_type_symbol");
    int x_idx = loop.find_tag("_atom_site_fract_x");
    int y_idx = loop.find_tag("_atom_site_fract_y");
    int z_idx = loop.find_tag("_atom_site_fract_z");
    qDebug() << label_idx << symbol_idx << x_idx << y_idx << z_idx;
    for(size_t i = 0; i < loop.length(); i ++)
    {
        AtomData atom;
        if(label_idx >= 0) atom.site_label += QString(loop.val(i, label_idx).c_str());
        if(symbol_idx >= 0) atom.element += QString(loop.val(i, symbol_idx).c_str());
        if(x_idx >= 0) atom.position[0] = gemmi::cif::as_number(loop.val(i, x_idx));
        if(y_idx >= 0) atom.position[1] = gemmi::cif::as_number(loop.val(i, y_idx));
        if(z_idx >= 0) atom.position[2] = gemmi::cif::as_number(loop.val(i, z_idx));
        m_atoms.push_back(atom);
    }

}

void CifParser::extractCellParameter(const gemmi::cif::Pair &pair)
{
    const auto& tag = pair.front();
    if(tag == "_cell_length_a") m_cell.a = gemmi::cif::as_number(pair.back());
    else if(tag == "_cell_length_b") m_cell.b = gemmi::cif::as_number(pair.back());
    else if(tag == "_cell_length_c") m_cell.c = gemmi::cif::as_number(pair.back());
    else if(tag == "_cell_angle_alpha") m_cell.alpha = cx::deg2rad(gemmi::cif::as_number(pair.back()));
    else if(tag == "_cell_angle_beta") m_cell.beta = cx::deg2rad(gemmi::cif::as_number(pair.back()));
    else if(tag == "_cell_angle_gamma") m_cell.gamma = cx::deg2rad(gemmi::cif::as_number(pair.back()));
}

void CifParser::extractSymmetryOperations(const gemmi::cif::Loop &loop)
{
    int idx = loop.find_tag("_symmetry_equiv_pos_as_xyz");
    if(idx < 0) return;

    for(size_t i = 0; i < loop.length(); i ++)
    {
        QString symop;
        symop = QString::fromStdString(loop.val(i, idx));
        m_sym.symops.push_back(symop);
    }
}

void CifParser::dump()
{
    for(const auto& atom: m_atoms)
    {
        qDebug() << "Atom" << atom.element << atom.site_label << atom.position[0] << atom.position[1] << atom.position[2];
    }
    qDebug() << "Unit cell" << m_cell.a << m_cell.b << m_cell.c << m_cell.alpha << m_cell.beta << m_cell.gamma;
    qDebug() << "Symmetry" << m_sym.nameHM << m_sym.nameHall << m_sym.number;
    for(const auto& symop: m_sym.symops)
    {
        qDebug() << symop;
    }
}

Crystal* CifParser::parseCrystal(const QString& filename)
{
    std::string s = filename.toStdString();
    try {
        auto doc = gemmi::cif::read_file(s);
        auto block = doc.blocks.front();
        for(const auto& item: block.items)
        {
            if(item.type == gemmi::cif::ItemType::Pair) {
                if(item.has_prefix("_cell")) extractCellParameter(item.pair);
            }
            if(item.type == gemmi::cif::ItemType::Loop) {
                if(item.has_prefix("_atom_site")) extractAtomSites(item.loop);
                else if(item.has_prefix("_symmetry_equiv_pos")) extractSymmetryOperations(item.loop);
                else {
                    qDebug() << "Skipping loop";
                }
            }
        }
    }
    catch (const std::exception& e) {
        m_failureDescription = QString(e.what());
        return nullptr;
    }
    if(!cellValid()) {
        m_failureDescription = "Missing unit cell data";
        return nullptr;
    }
    if(!symmetryValid()) {
        m_failureDescription = "Missing symmetry data";
        return nullptr;
    }
    AsymmetricUnit asym;
    if(atomCount() > 0) {
        asym.atomicNumbers.conservativeResize(atomCount());
        asym.positions.conservativeResize(3, atomCount());
        int i = 0;

        for(const auto& atom: m_atoms) {
            asym.positions(0, i) = atom.position[0];
            asym.positions(1, i) = atom.position[1];
            asym.positions(2, i) = atom.position[2];
            asym.atomicNumbers(i) = Elements::Element(atom.element).n();
            asym.labels.push_back(atom.site_label);
            i++;
        }
    }
    UnitCell uc(m_cell.a, m_cell.b, m_cell.c, m_cell.alpha, m_cell.beta, m_cell.gamma);
    QVector<SymmetryOperation> symops;
    if(m_sym.symopCount() > 0) {
        for(const auto &s: m_sym.symops) {
            symops.push_back(SymmetryOperation(s));
        }
    }
    if(symops.length() > 0) {
        auto sg = SpaceGroupTable::table()->get(symops);
        if(sg.internationalTablesNumber() < 1) {
            sg.setSymbol(m_sym.nameHM);
            qDebug() << QString("Could not determine space group %1 from table, some data may be missing").arg(sg.symbol());
        }
        qDebug() << "Space group: " << sg.symbol();
        return new Crystal(asym, sg, uc);
    }
    SpaceGroup sg;
    return new Crystal(asym, sg, uc);
}
