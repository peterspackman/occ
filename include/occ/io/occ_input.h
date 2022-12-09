#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/qm/spinorbital.h>
#include <vector>

namespace occ::io {
using occ::core::Element;
using occ::qm::SpinorbitalKind;

using Position = std::array<double, 3>;

struct ElectronInput {
    int multiplicity{1};
    double charge{0.0};
    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
};

struct GeometryInput {
    std::vector<Position> positions;
    std::vector<Element> elements;
    occ::core::Molecule molecule() const;
    void set_molecule(const occ::core::Molecule &);
};

struct DriverInput {
    std::string driver{"energy"};
    int threads{1};
};

struct MethodInput {
    std::string name{"rhf"};
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

struct PairInput {
    std::string source_a{"none"};
    Mat3 rotation_a{Mat3::Identity()};
    Vec3 translation_a{Vec3::Zero()};

    std::string source_b{"none"};
    Mat3 rotation_b{Mat3::Identity()};
    Vec3 translation_b{Vec3::Zero()};
    std::string model_name{"ce-b3lyp"};
};

struct OccInput {
    DriverInput driver;
    RuntimeInput runtime;
    ElectronInput electronic;
    GeometryInput geometry;
    PairInput pair;
    MethodInput method;
    BasisSetInput basis;
    SolventInput solvent;
    std::string name{""};
    std::string filename{""};
};

template <typename T> OccInput build(const std::string &filename) {
    return T(filename).as_occ_input();
}

template <typename T> OccInput build(std::istream &file) {
    return T(file).as_occ_input();
}

} // namespace occ::io
