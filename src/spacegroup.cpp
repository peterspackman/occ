#include "spacegroup.h"
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <iostream>

static QMap<QString, QVector<QVariant>> LATTICE_VARIABLES({
    {"triclinic", QVector<QVariant>{"a", "b", "c", "alpha", "beta", "gamma"}},
    {"monoclinic", QVector<QVariant>{"a", "b", "c", 90, "beta", 90}},
    {"orthorhombic", QVector<QVariant>{"a", "b", "c", 90, 90, 90}},
    {"tetragonal", QVector<QVariant>{"a", "a", "c", 90, 90, 90}},
    {"rhombohedral", QVector<QVariant>{"a", "a", "a", "alpha", "alpha", "alpha"}},
    {"hexagonal", QVector<QVariant>{"a", "a", "c", 90, 120, 90}},
    {"cubic", QVector<QVariant>{"a", "a", "a", 90, 90, 90}}
});

static QMap<QString, int> CENTERING_TO_LATTICE({
    {"primitive", 1},
    {"body", 2},
    {"rcenter", 3},
    {"face", 4},
    {"aface", 5},
    {"bface", 6},
    {"cface", 7},
});


SpaceGroupTable *SpaceGroupTable::instance = nullptr;

SpaceGroupTable* SpaceGroupTable::table()
{
    if (instance == nullptr)
    {
        qDebug() << "Loading space group data";
        instance = new SpaceGroupTable();
        qDebug() << "Done loading space group data";
    }
    return instance;
}


SpaceGroupTable::SpaceGroupTable()
{
    QFile jsonFile(QStringLiteral(":/resources/sgdata.json"));
    if (!jsonFile.open(QIODevice::ReadOnly)) {
        qWarning("Couldn't open space group data file.");
    }
    QByteArray jsonData = jsonFile.readAll();
    QJsonObject spaceGroupData = QJsonDocument::fromJson(jsonData).object();
    int idx = 0;
    for(const auto& sg : spaceGroupData.keys()) {
        QJsonValue value = spaceGroupData.value(sg);
        QJsonArray choices = value.toArray();
        for(const auto& x : choices) {
            QJsonArray obj = x.toArray();
            int number = obj[0].toInt();
            QString shortName = obj[1].toString();
            //QString schoenflies = obj[2].toString();
            QString full = obj[3].toString();
            //QString international = obj[4].toString();
            //int pointGroup = obj[5].toInt();
            QString choice = obj[6].toString();
            QString centering = obj[7].toString();
            bool centrosymmetric = obj[9].toBool();
            QVector<SymmetryOperation> symops;
            for(const auto& symopCode : obj[8].toArray()) {
                symops.push_back(SymmetryOperation(symopCode.toInt()));
            }
            m_spaceGroups.push_back(
                SpaceGroup(
                    number,
                    shortName,
                    full,
                    choice,
                    centering,
                    centrosymmetric,
                    symops
                )
            );
            if(!m_intToSpaceGroup.contains(number)) m_intToSpaceGroup.insert(number, idx);
            if(!m_symbolToSpaceGroup.contains(shortName)) m_symbolToSpaceGroup.insert(shortName, idx);
            m_intChoiceToSpaceGroup.insert({number, choice}, idx);
            idx++;
        }
    }
}

const SpaceGroup& SpaceGroupTable::get(int number) const
{
    return m_spaceGroups[m_intToSpaceGroup[number]];
}

const SpaceGroup& SpaceGroupTable::get(int number, const QString& choice) const
{
    return m_spaceGroups[m_intChoiceToSpaceGroup[{number, choice}]];
}

SpaceGroup SpaceGroupTable::get(QVector<SymmetryOperation> symops) const
{
    std::sort(symops.begin(), symops.end());
    for(const auto& sg: m_spaceGroups) {
        auto sgsymops = sg.symmetryOperations();
        if(sgsymops.length() != symops.length()) continue;
        std::sort(sgsymops.begin(), sgsymops.end());
        if(sgsymops == symops) return sg;
    }
    return SpaceGroup(-1, "Unknown", "Unknown", "Unknown", "Unknown", false, symops);
}

SpaceGroup::SpaceGroup(int n, QString shortName, QString full, QString choice, QString centering, bool centrosymmetric, QVector<SymmetryOperation> symops) :
    m_number(n), m_symbol(shortName), m_fullName(full), m_choice(choice), m_centering(centering), m_centrosymmetric(centrosymmetric), m_symops(symops)
{

}

const QString SpaceGroup::crystalSystem() const
{
    int sg = m_number;
    if(sg <= 2) return "triclinic";
    if(sg <= 16) return "monoclinic";
    if(sg <= 74) return "orthorhombic";
    if(sg <= 142) return "tetragonal";
    if(sg <= 167) return "trigonal";
    if(sg <= 194) return "hexagonal";
    return "cubic";
}

int SpaceGroup::latticeNumber() const
{
    int latt = CENTERING_TO_LATTICE[m_centering];
    if(m_centrosymmetric) return latt;
    else return - latt;
}

bool SpaceGroup::hasHexagonalRhombohedralChoice() const
{
    switch(m_number) {
    case 146:
    case 148:
    case 155:
    case 160:
    case 161:
    case 166:
    case 167:
        return true;
    default:
        return false;
    }
}

std::pair<Eigen::VectorXi, Eigen::Matrix3Xd> SpaceGroup::applyAllSymmetryOperations(const Eigen::Matrix3Xd& frac) const
{
    int nSites = frac.cols();
    int nSymops = m_symops.length();
    Eigen::Matrix3Xd transformed(3, nSites * nSymops);
    Eigen::VectorXi generators(nSites * nSymops);
    transformed.block(0, 0, 3, nSites) = frac;
    for(int i = 0; i < nSites; i++) {
        generators(i) = 16484;
    }
    int offset = nSites;
    for(const auto& symop : m_symops) {
        if(symop.isIdentity()) continue;
        int code = symop.toInt();
        generators.block(offset, 0, nSites, 1).setConstant(code);
        transformed.block(0, offset, frac.rows(), frac.cols()) = symop(frac);
        offset += nSites;
    }
    return {generators, transformed};
}
