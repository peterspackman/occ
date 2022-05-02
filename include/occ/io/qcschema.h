#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/io/occ_input.h>

namespace occ::io {

using occ::core::Element;

struct OccInput;

struct QCSchemaBond {
    int idx_a{-1};
    int idx_b{-1};
    double bond_length{0.0};
};

struct QCSchemaTopology {
    std::vector<std::array<double, 3>> positions;
    std::vector<Element> elements;
    std::vector<std::vector<int>> fragments;
    std::vector<int> fragment_multiplicities;
    int charge{0};
    int multiplicity{1};
};

struct QCSchemaModel {
    std::string method;
    std::string basis;
};

struct QCSchemaInput {
    std::string name{"input"};
    QCSchemaTopology topology;
    QCSchemaModel model;
    PairInput pair_input;
    std::string driver;
    int threads{1};
};

class QCSchemaReader {
  public:
    QCSchemaReader(const std::string &);
    QCSchemaReader(std::istream &);
    OccInput as_occ_input() const;
    void update_occ_input(OccInput &) const;
    QCSchemaInput input;

  private:
    void parse(std::istream &);
    std::string m_filename;
};

} // namespace occ::io
