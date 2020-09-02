#include "element.h"
#include <QDebug>
using namespace Elements;

Element::Element(int atomicNumber) : m_data(&ELEMENTDATA_TABLE[atomicNumber])
{}

Element::Element(const QString& s) : m_data(&ELEMENTDATA_TABLE[0])
{
    //capitalize the symbol first
    auto symbol = s.trimmed();
    symbol = symbol.toLower();
    symbol[0] = symbol[0].toUpper();
    auto ptr = ELEMENT_MAP.value(symbol, nullptr);
    if (ptr != nullptr) {
        m_data = ptr;
        return;
    }
    int idx = 0;
    for(const auto& el : ELEMENTDATA_TABLE) {
        if (symbol == el.symbol) {
            auto ptr = &ELEMENTDATA_TABLE[idx];
            ELEMENT_MAP[symbol] = ptr;
            m_data = ptr;
            break;
        }
        idx++;
    }
}

QString Elements::chemicalFormula(const QVector<Element>& els)
{
    QMap<QString, int> counts;
    for(const auto& el : els) {
        counts[el.symbol()]++;
    }
    QStringList components;
    for(const auto& key : counts.keys()) {
        QString comp = key;
        auto count = counts[key];
        if (count > 1) comp += QString::number(count);
        components.push_back(comp);
    }
    components.sort();
    return components.join("");
}
