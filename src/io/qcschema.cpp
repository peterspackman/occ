#include <occ/io/qcschema.h>
#include <occ/core/timings.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <fmt/core.h>

namespace occ::io {

struct QCSchemaBond {
    int idx_a{-1};
    int idx_b{-1};
    double bond_length{0.0};
};

struct QCSchemaMolecule {
    std::vector<double> positions;
    std::vector<std::string> symbols;
};

struct QCSchemaModel {
    std::string method;
    std::string basis;
};

struct QCSchemaInput {
    QCSchemaMolecule molecule;
    QCSchemaModel model;
    std::string driver;
};

void from_json(const nlohmann::json &J, QCSchemaModel &model) {
    J.at("method").get_to(model.method);
    J.at("basis").get_to(model.basis);
}

void from_json(const nlohmann::json &J, QCSchemaMolecule &mol) {
    J.at("geometry").get_to(mol.positions);
    J.at("symbols").get_to(mol.symbols);
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
    auto inp = j.get<QCSchemaInput>();
    fmt::print("Molecule size: {}\n",  inp.molecule.positions.size());
}


} // namespace occ::io
