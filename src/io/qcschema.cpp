#include <occ/io/qcschema.h>
#include <occ/core/timings.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <occ/io/occ_input.h>

namespace occ::io {


void from_json(const nlohmann::json &J, QCSchemaModel &model) {
    J.at("method").get_to(model.method);
    J.at("basis").get_to(model.basis);
}

void from_json(const nlohmann::json &J, QCSchemaMolecule &mol) {
    std::vector<double> positions;
    J.at("geometry").get_to(positions);
    for(size_t i = 0; i < positions.size(); i += 3) {
        mol.positions.emplace_back(std::array<double, 3>{
                positions[i],
                positions[i + 1],
                positions[i + 2]});
    }
    std::vector<std::string> symbols;
    J.at("symbols").get_to(symbols);
    for(const auto& sym: symbols) {
        mol.elements.emplace_back(occ::chem::Element(sym));
    }
}

void from_json(const nlohmann::json &J, QCSchemaInput &qc) {
    J.at("molecule").get_to(qc.molecule);
    J.at("model").get_to(qc.model);
    J.at("driver").get_to(qc.driver);
}

QCSchemaReader::QCSchemaReader(const std::string &filename) : m_filename(filename) {
    occ::timing::start(occ::timing::category::io);
    std::ifstream file(filename);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

QCSchemaReader::QCSchemaReader(std::istream &file) : m_filename("_istream_") {
    occ::timing::start(occ::timing::category::io);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

void QCSchemaReader::parse(std::istream &is) {
    nlohmann::json j;
    is >> j;
    input = j.get<QCSchemaInput>();
}

void QCSchemaReader::update_occ_input(OccInput &result) const {
    result.geometry.elements = input.molecule.elements;
    result.geometry.positions = input.molecule.positions;
    result.method.name = input.model.method;
    result.basis.name = input.model.basis;
}

OccInput QCSchemaReader::as_occ_input() const {
    OccInput result;
    update_occ_input(result);
    return result;
}


} // namespace occ::io
