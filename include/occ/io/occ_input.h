#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/point_charge.h>
#include <occ/crystal/crystal.h>
#include <occ/io/grid_settings.h>
#include <occ/qm/spinorbital.h>
#include <vector>

namespace occ::io {
using occ::core::Element;
using occ::qm::SpinorbitalKind;

using Position = std::array<double, 3>;
using PointChargeList = std::vector<occ::core::PointCharge>;

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
    PointChargeList point_charges;
    std::string point_charge_filename{""};
};

struct OutputInput {
    std::vector<std::string> formats{"json"};
};

struct DriverInput {
    std::string driver{"energy"};
};

struct MethodInput {
    std::string name{"rhf"};
    BeckeGridSettings dft_grid;
    double integral_precision{1e-12};
    double orbital_smearing_sigma{0.0};
};

struct BasisSetInput {
    std::string name{"3-21G"};
    std::string df_name{""};
    std::string basis_set_directory{""};
    bool spherical{false};
};

struct SolventInput {
    std::string solvent_name{""};
    std::string output_surface_filename{""};
    bool radii_scaling{false};
};

struct RuntimeInput {
    int threads{1};
    std::string output_filename{""};
};

struct DispersionCorrectionInput {
    bool evaluate_correction{false};
    double xdm_a1{1.0};
    double xdm_a2{1.0};
};

struct CrystalInput {
    occ::crystal::AsymmetricUnit asymmetric_unit;
    occ::crystal::SpaceGroup space_group;
    occ::crystal::UnitCell unit_cell;
};

struct PairInput {
    std::string source_a{"none"};
    Mat3 rotation_a{Mat3::Identity()};
    Vec3 translation_a{Vec3::Zero()};
    int ecp_electrons_a{0};
    std::string source_b{"none"};
    Mat3 rotation_b{Mat3::Identity()};
    Vec3 translation_b{Vec3::Zero()};
    int ecp_electrons_b{0};
    std::string model_name{"ce-b3lyp"};
};

struct IsosurfaceInput {
    GeometryInput interior_geometry;
    GeometryInput exterior_geometry;
};

struct OccInput {
    std::string verbosity{"normal"};
    DriverInput driver;
    RuntimeInput runtime;
    ElectronInput electronic;
    GeometryInput geometry;
    PairInput pair;
    MethodInput method;
    BasisSetInput basis;
    SolventInput solvent;
    DispersionCorrectionInput dispersion;
    CrystalInput crystal;
    IsosurfaceInput isosurface;
    OutputInput output;
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
