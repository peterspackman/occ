#include "core_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/atom.h>
#include <occ/core/dimer.h>
#include <occ/core/eem.h>
#include <occ/core/eeq.h>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/point_charge.h>
#include <occ/core/point_group.h>
#include <occ/core/data_directory.h>
#include <occ/io/xyz.h>

using namespace emscripten;
using namespace occ::core;
using namespace occ;

void set_data_directory_wrapper(std::string path) {
    occ::set_data_directory(path);
}

std::string get_data_directory_wrapper() {
    return std::string(occ::get_data_directory() ?: "");
}

void register_core_bindings() {
    // Eigen matrix bindings using factory functions
    class_<Vec3>("Vec3")
        .function("x", optional_override([](const Vec3& v) { return v.x(); }))
        .function("y", optional_override([](const Vec3& v) { return v.y(); }))
        .function("z", optional_override([](const Vec3& v) { return v.z(); }))
        .function("setX", optional_override([](Vec3& v, double val) { v.x() = val; }))
        .function("setY", optional_override([](Vec3& v, double val) { v.y() = val; }))
        .function("setZ", optional_override([](Vec3& v, double val) { v.z() = val; }))
        .class_function("Zero", optional_override([]() { 
            Vec3 result = Vec3::Zero(); 
            return result; 
        }))
        .class_function("create", optional_override([](double x, double y, double z) { 
            Vec3 result(x, y, z); 
            return result; 
        }));
    
    class_<Mat3N>("Mat3N")
        .function("set", optional_override([](Mat3N& m, int row, int col, double val) {
            m(row, col) = val;
        }))
        .function("get", optional_override([](const Mat3N& m, int row, int col) {
            return m(row, col);
        }))
        .function("rows", optional_override([](const Mat3N& m) { return m.rows(); }))
        .function("cols", optional_override([](const Mat3N& m) { return m.cols(); }))
        .class_function("create", optional_override([](int cols) { 
            Mat3N result = Mat3N::Zero(3, cols); 
            return result; 
        }));
        
    class_<IVec>("IVec")
        .function("size", optional_override([](const IVec& v) { return v.size(); }))
        .function("get", optional_override([](const IVec& v, int i) {
            return v(i);
        }))
        .function("set", optional_override([](IVec& v, int i, int val) {
            v(i) = val;
        }))
        .class_function("fromArray", optional_override([](const emscripten::val& jsArray) {
            const int length = jsArray["length"].as<int>();
            IVec result(length);
            for (int i = 0; i < length; ++i) {
                result(i) = jsArray[i].as<int>();
            }
            return result;
        }));
        
    class_<Vec>("Vec")
        .function("size", optional_override([](const Vec& v) { return v.size(); }))
        .function("get", optional_override([](const Vec& v, int i) {
            return v(i);
        }))
        .function("set", optional_override([](Vec& v, int i, double val) {
            v(i) = val;
        }))
        .class_function("create", optional_override([](int size) { 
            Vec result = Vec::Zero(size); 
            return result; 
        }));

    class_<Mat>("Mat")
        .function("rows", optional_override([](const Mat& m) { return m.rows(); }))
        .function("cols", optional_override([](const Mat& m) { return m.cols(); }))
        .function("get", optional_override([](const Mat& m, int row, int col) {
            return m(row, col);
        }))
        .function("set", optional_override([](Mat& m, int row, int col, double val) {
            m(row, col) = val;
        }))
        .class_function("create", optional_override([](int rows, int cols) { 
            Mat result = Mat::Zero(rows, cols); 
            return result; 
        }));

    // Vector binding for std::vector<Atom>
    register_vector<Atom>("VectorAtom");

    // Element class binding
    class_<Element>("Element")
        .constructor<const std::string&>()
        .property("symbol", &Element::symbol)
        .property("mass", &Element::mass)
        .property("name", &Element::name)
        .property("vanDerWaalsRadius", &Element::van_der_waals_radius)
        .property("covalentRadius", &Element::covalent_radius)
        .property("atomicNumber", &Element::atomic_number)
        .function("toString", optional_override([](const Element& el) {
            return std::string("<Element '") + el.symbol() + "'>";
        }))
        .class_function("fromAtomicNumber", optional_override([](int atomic_number) {
            return Element(atomic_number);
        }));

    // Atom class binding
    class_<Atom>("Atom")
        .constructor<int, double, double, double>()
        .property("atomicNumber", &Atom::atomic_number)
        .property("x", &Atom::x)
        .property("y", &Atom::y)
        .property("z", &Atom::z)
        .function("getPosition", &Atom::position)
        .function("setPosition", &Atom::set_position)
        .function("toString", optional_override([](const Atom& a) {
            return std::string("<Atom ") + std::to_string(a.atomic_number) + 
                   " [" + std::to_string(a.x) + ", " + std::to_string(a.y) + 
                   ", " + std::to_string(a.z) + "]>";
        }));

    // PointCharge class binding
    class_<PointCharge>("PointCharge")
        .constructor<double, double, double, double>()
        .constructor<double, const Vec3&>()
        .property("charge", &PointCharge::charge)
        .function("getPosition", &PointCharge::position)
        .function("setCharge", &PointCharge::set_charge)
        .function("setPosition", &PointCharge::set_position)
        .function("toString", optional_override([](const PointCharge& pc) {
            const auto& pos = pc.position();
            return std::string("<PointCharge q=") + std::to_string(pc.charge()) + 
                   " [" + std::to_string(pos.x()) + ", " + std::to_string(pos.y()) + 
                   ", " + std::to_string(pos.z()) + "]>";
        }));

    // Molecule Origin enum
    enum_<Molecule::Origin>("Origin")
        .value("CARTESIAN", Molecule::Origin::Cartesian)
        .value("CENTROID", Molecule::Origin::Centroid)
        .value("CENTEROFMASS", Molecule::Origin::CenterOfMass);

    // Molecule class binding
    class_<Molecule>("Molecule")
        .constructor<const IVec&, const Mat3N&>()
        .function("size", &Molecule::size)
        .function("elements", &Molecule::elements)
        .function("positions", &Molecule::positions)
        .property("name", &Molecule::name)
        .function("setName", &Molecule::set_name)
        .function("partialCharges", &Molecule::partial_charges)
        .function("setPartialCharges", &Molecule::set_partial_charges)
        .function("espPartialCharges", &Molecule::esp_partial_charges)
        .function("atomicMasses", &Molecule::atomic_masses)
        .function("atomicNumbers", &Molecule::atomic_numbers)
        .function("vdwRadii", &Molecule::vdw_radii)
        .function("molarMass", &Molecule::molar_mass)
        .function("atoms", &Molecule::atoms)
        .function("centerOfMass", &Molecule::center_of_mass)
        .function("centroid", &Molecule::centroid)
        .function("rotate", select_overload<void(const Mat3&, Molecule::Origin)>(&Molecule::rotate))
        .function("translate", &Molecule::translate)
        .function("rotated", select_overload<Molecule(const Mat3&, Molecule::Origin) const>(&Molecule::rotated))
        .function("translated", &Molecule::translated)
        .function("centered", optional_override([](const Molecule& mol, Molecule::Origin origin) {
            Vec3 center;
            switch (origin) {
                case Molecule::Origin::Centroid:
                    center = mol.centroid();
                    break;
                case Molecule::Origin::CenterOfMass:
                    center = mol.center_of_mass();
                    break;
                default:
                    center = Vec3::Zero();
            }
            return mol.translated(-center);
        }))
        .class_function("fromXyzFile", optional_override([](const std::string& filename) {
            return occ::io::molecule_from_xyz_file(filename);
        }))
        .class_function("fromXyzString", optional_override([](const std::string& contents) {
            return occ::io::molecule_from_xyz_string(contents);
        }))
        .function("translationalFreeEnergy", &Molecule::translational_free_energy)
        .function("rotationalFreeEnergy", &Molecule::rotational_free_energy)
        .function("toString", optional_override([](const Molecule& mol) {
            auto com = mol.center_of_mass();
            return std::string("<Molecule ") + mol.name() + " @[" + 
                   std::to_string(com.x()) + ", " + std::to_string(com.y()) + 
                   ", " + std::to_string(com.z()) + "]>";
        }));

    // Dimer class binding
    class_<Dimer>("Dimer")
        .constructor<const Molecule&, const Molecule&>()
        .property("a", &Dimer::a)
        .property("b", &Dimer::b)
        .property("nearestDistance", &Dimer::nearest_distance)
        .property("centerOfMassDistance", &Dimer::center_of_mass_distance)
        .property("centroidDistance", &Dimer::centroid_distance)
        .function("symmetryRelation", &Dimer::symmetry_relation)
        .property("name", &Dimer::name)
        .function("setName", &Dimer::set_name);

    // Point group enums and classes
    enum_<PointGroup>("PointGroup")
        .value("C1", PointGroup::C1)
        .value("Ci", PointGroup::Ci)
        .value("Cs", PointGroup::Cs)
        .value("C2", PointGroup::C2)
        .value("C3", PointGroup::C3)
        .value("C4", PointGroup::C4)
        .value("C5", PointGroup::C5)
        .value("C6", PointGroup::C6)
        .value("C2v", PointGroup::C2v)
        .value("C3v", PointGroup::C3v)
        .value("C4v", PointGroup::C4v)
        .value("C5v", PointGroup::C5v)
        .value("C6v", PointGroup::C6v)
        .value("D2", PointGroup::D2)
        .value("D3", PointGroup::D3)
        .value("D4", PointGroup::D4)
        .value("D5", PointGroup::D5)
        .value("D6", PointGroup::D6)
        .value("D2h", PointGroup::D2h)
        .value("D3h", PointGroup::D3h)
        .value("D4h", PointGroup::D4h)
        .value("D5h", PointGroup::D5h)
        .value("D6h", PointGroup::D6h)
        .value("Td", PointGroup::Td)
        .value("Oh", PointGroup::Oh);

    class_<MolecularPointGroup>("MolecularPointGroup")
        .constructor<const Molecule&>()
        .function("getDescription", optional_override([](const MolecularPointGroup& pg) {
            return std::string(pg.description());
        }))
        .function("getPointGroupString", optional_override([](const MolecularPointGroup& pg) {
            return std::string(pg.point_group_string());
        }))
        .property("pointGroup", &MolecularPointGroup::point_group)
        .property("symmetryNumber", &MolecularPointGroup::symmetry_number)
        .function("toString", optional_override([](const MolecularPointGroup& pg) {
            return std::string("<MolecularPointGroup '") + pg.point_group_string() + "'>";
        }));

    // Utility functions
    function("eemPartialCharges", &occ::core::charges::eem_partial_charges);
    function("eeqPartialCharges", &occ::core::charges::eeq_partial_charges);
    function("eeqCoordinationNumbers", &occ::core::charges::eeq_coordination_numbers);
    
    // Data directory functions
    function("setDataDirectory", &set_data_directory_wrapper);
    function("getDataDirectory", &get_data_directory_wrapper);
}
