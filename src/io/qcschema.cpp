#include <occ/io/qcschema.h>
#include <occ/core/timings.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <occ/io/occ_input.h>
#include <fmt/core.h>

namespace occ::io {


void from_json(const nlohmann::json &J, QCSchemaModel &model) {
    J.at("method").get_to(model.method);
    J.at("basis").get_to(model.basis);
}

void from_json(const nlohmann::json &J, QCSchemaTopology &mol) {
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
        mol.elements.emplace_back(occ::core::Element(sym));
    }

    if(J.contains("fragments")) {
        J.at("fragments").get_to(mol.fragments);
    }

    if(J.contains("fragment_multiplicities")) {
        J.at("fragment_multiplicities").get_to(mol.fragment_multiplicities);
    }

    if(J.contains("molecular_charge")) {
        J.at("molecular_charge").get_to(mol.charge);
    }

    if(J.contains("molecular_multiplicity")) {
        J.at("molecular_multiplicity").get_to(mol.multiplicity);
    }
}

void from_json(const nlohmann::json &J, QCSchemaInput &qc) {
    J.at("molecule").get_to(qc.topology);
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
    result.geometry.elements = input.topology.elements;
    result.geometry.positions = input.topology.positions;
    result.electronic.multiplicity = input.topology.multiplicity;
    result.electronic.charge = input.topology.charge;
    result.method.name = input.model.method;
    result.basis.name = input.model.basis;
}

OccInput QCSchemaReader::as_occ_input() const {
    OccInput result;
    update_occ_input(result);
    return result;
}


} // namespace occ::io
