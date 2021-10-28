#pragma once
#include <vector>
#include <occ/qm/spinorbital.h>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <array>
#include <istream>

namespace occ::io {
using occ::qm::SpinorbitalKind;
using occ::chem::Element;

using Position = std::array<double, 3>;

struct ElectronInput {
    int multiplicity{1};
    double charge{0.0};
    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
};

struct GeometryInput {
    std::vector<Position> positions;
    std::vector<Element> elements;
    occ::chem::Molecule molecule() const;
};

struct MethodInput {
    std::string name{"hf"};
};

struct BasisSetInput {
    std::string name{"3-21G"};
    std::string df_name{""};
    bool spherical{false};
};

struct SolventInput {
    std::string solvent_name{""};
    std::string output_surface_filename{""};
};

struct RuntimeInput {
    int nthreads{1};
    std::string output_filename{""};
};

struct OccInput {
    RuntimeInput runtime;
    ElectronInput electronic;
    GeometryInput geometry;
    MethodInput method;
    BasisSetInput basis;
    SolventInput solvent;
    std::string filename{""};
};

template<typename T>
OccInput build(const std::string &filename) {
    return T(filename).as_occ_input();
}

template<typename T>
OccInput build(std::istream &file) {
    return T(file).as_occ_input();
}

} // namespace occ::io
