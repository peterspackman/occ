#include "core_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/atom.h>
#include <occ/core/data_directory.h>
#include <occ/core/dimer.h>
#include <occ/core/eem.h>
#include <occ/core/eeq.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/element.h>
#include <occ/crystal/crystal.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/core/point_charge.h>
#include <occ/core/point_group.h>
#include <occ/io/xyz.h>

using namespace emscripten;
using namespace occ::core;
using namespace occ::crystal;
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
      .function("x", optional_override([](const Vec3 &v) { return v.x(); }))
      .function("y", optional_override([](const Vec3 &v) { return v.y(); }))
      .function("z", optional_override([](const Vec3 &v) { return v.z(); }))
      .function("setX",
                optional_override([](Vec3 &v, double val) { v.x() = val; }))
      .function("setY",
                optional_override([](Vec3 &v, double val) { v.y() = val; }))
      .function("setZ",
                optional_override([](Vec3 &v, double val) { v.z() = val; }))
      .function("toString", optional_override([](const Vec3 &v) {
                  return occ::format_matrix(v);
                }))
      .function("toStringFormatted", optional_override([](const Vec3 &v, const std::string &fmt) {
                  return occ::format_matrix(v, fmt);
                }))
      .class_function("Zero", optional_override([]() {
                        Vec3 result = Vec3::Zero();
                        return result;
                      }))
      .class_function("create",
                      optional_override([](double x, double y, double z) {
                        Vec3 result(x, y, z);
                        return result;
                      }));

  class_<Mat3N>("Mat3N")
      .function("set", optional_override([](Mat3N &m, int row, int col,
                                            double val) { m(row, col) = val; }))
      .function("get", optional_override([](const Mat3N &m, int row, int col) {
                  return m(row, col);
                }))
      .function("rows",
                optional_override([](const Mat3N &m) { return m.rows(); }))
      .function("cols",
                optional_override([](const Mat3N &m) { return m.cols(); }))
      .function("toString", optional_override([](const Mat3N &m) {
                  return occ::format_matrix(m);
                }))
      .function("toStringFormatted", optional_override([](const Mat3N &m, const std::string &fmt) {
                  return occ::format_matrix(m, fmt);
                }))
      .class_function("create", optional_override([](int cols) {
                        Mat3N result = Mat3N::Zero(3, cols);
                        return result;
                      }));

  class_<IVec>("IVec")
      .function("size",
                optional_override([](const IVec &v) { return v.size(); }))
      .function("get",
                optional_override([](const IVec &v, int i) { return v(i); }))
      .function("set",
                optional_override([](IVec &v, int i, int val) { v(i) = val; }))
      .function("toString", optional_override([](const IVec &v) {
                  return occ::format_matrix(v);
                }))
      .function("toStringFormatted", optional_override([](const IVec &v, const std::string &fmt) {
                  return occ::format_matrix(v, fmt);
                }))
      .class_function("fromArray",
                      optional_override([](const emscripten::val &jsArray) {
                        const int length = jsArray["length"].as<int>();
                        IVec result(length);
                        for (int i = 0; i < length; ++i) {
                          result(i) = jsArray[i].as<int>();
                        }
                        return result;
                      }));

  class_<Vec>("Vec")
      .function("size",
                optional_override([](const Vec &v) { return v.size(); }))
      .function("get",
                optional_override([](const Vec &v, int i) { return v(i); }))
      .function("set", optional_override(
                           [](Vec &v, int i, double val) { v(i) = val; }))
      .function("toString", optional_override([](const Vec &v) {
                  return occ::format_matrix(v);
                }))
      .function("toStringFormatted", optional_override([](const Vec &v, const std::string &fmt) {
                  return occ::format_matrix(v, fmt);
                }))
      .class_function("create", optional_override([](int size) {
                        Vec result = Vec::Zero(size);
                        return result;
                      }));

  class_<Vec6>("Vec6")
      .function("size",
                optional_override([](const Vec6 &v) { return v.size(); }))
      .function("get",
                optional_override([](const Vec6 &v, int i) { return v(i); }))
      .function("set", optional_override(
                           [](Vec6 &v, int i, double val) { v(i) = val; }))
      .function("toString", optional_override([](const Vec6 &v) {
                  return occ::format_matrix(v);
                }))
      .function("toStringFormatted", optional_override([](const Vec6 &v, const std::string &fmt) {
                  return occ::format_matrix(v, fmt);
                }))
      .class_function("create", optional_override([]() {
                        Vec6 result = Vec6::Zero();
                        return result;
                      }));

  class_<Mat>("Mat")
      .function("rows",
                optional_override([](const Mat &m) { return m.rows(); }))
      .function("cols",
                optional_override([](const Mat &m) { return m.cols(); }))
      .function("get", optional_override([](const Mat &m, int row, int col) {
                  return m(row, col);
                }))
      .function("set", optional_override([](Mat &m, int row, int col,
                                            double val) { m(row, col) = val; }))
      .function("toString", optional_override([](const Mat &m) {
                  return occ::format_matrix(m);
                }))
      .function("toStringFormatted", optional_override([](const Mat &m, const std::string &fmt) {
                  return occ::format_matrix(m, fmt);
                }))
      .class_function("create", optional_override([](int rows, int cols) {
                        Mat result = Mat::Zero(rows, cols);
                        return result;
                      }));

  // Mat3 typedef for 3x3 matrices
  class_<Mat3>("Mat3")
      .function("rows",
                optional_override([](const Mat3 &m) { return m.rows(); }))
      .function("cols",
                optional_override([](const Mat3 &m) { return m.cols(); }))
      .function("get", optional_override([](const Mat3 &m, int row, int col) {
                  return m(row, col);
                }))
      .function("set", optional_override([](Mat3 &m, int row, int col,
                                            double val) { m(row, col) = val; }))
      .function("toString", optional_override([](const Mat3 &m) {
                  return occ::format_matrix(m);
                }))
      .function("toStringFormatted", optional_override([](const Mat3 &m, const std::string &fmt) {
                  return occ::format_matrix(m, fmt);
                }))
      .class_function("create", optional_override([]() {
                        Mat3 result = Mat3::Zero();
                        return result;
                      }));

  // Mat6 typedef for 6x6 matrices
  class_<Mat6>("Mat6")
      .function("rows",
                optional_override([](const Mat6 &m) { return m.rows(); }))
      .function("cols",
                optional_override([](const Mat6 &m) { return m.cols(); }))
      .function("get", optional_override([](const Mat6 &m, int row, int col) {
                  return m(row, col);
                }))
      .function("set", optional_override([](Mat6 &m, int row, int col,
                                            double val) { m(row, col) = val; }))
      .function("toString", optional_override([](const Mat6 &m) {
                  return occ::format_matrix(m);
                }))
      .function("toStringFormatted", optional_override([](const Mat6 &m, const std::string &fmt) {
                  return occ::format_matrix(m, fmt);
                }))
      .class_function("create", optional_override([](int rows, int cols) {
                        Mat6 result = Mat6::Zero(rows, cols);
                        return result;
                      }));

  // Vector bindings
  register_vector<Atom>("VectorAtom");
  register_vector<std::string>("VectorString");

  // Element class binding
  class_<Element>("Element")
      .constructor<const std::string &>()
      .property("symbol", &Element::symbol)
      .property("mass", &Element::mass)
      .property("name", &Element::name)
      .property("vanDerWaalsRadius", &Element::van_der_waals_radius)
      .property("covalentRadius", &Element::covalent_radius)
      .property("atomicNumber", &Element::atomic_number)
      .function("toString", optional_override([](const Element &el) {
                  return std::string("<Element '") + el.symbol() + "'>";
                }))
      .class_function("fromAtomicNumber",
                      optional_override([](int atomic_number) {
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
      .function("toString", optional_override([](const Atom &a) {
                  return std::string("<Atom ") +
                         std::to_string(a.atomic_number) + " [" +
                         std::to_string(a.x) + ", " + std::to_string(a.y) +
                         ", " + std::to_string(a.z) + "]>";
                }));

  // PointCharge class binding
  class_<PointCharge>("PointCharge")
      .constructor<double, double, double, double>()
      .constructor<double, const Vec3 &>()
      .property("charge", &PointCharge::charge)
      .function("getPosition", &PointCharge::position)
      .function("setCharge", &PointCharge::set_charge)
      .function("setPosition", &PointCharge::set_position)
      .function("toString", optional_override([](const PointCharge &pc) {
                  const auto &pos = pc.position();
                  return std::string("<PointCharge q=") +
                         std::to_string(pc.charge()) + " [" +
                         std::to_string(pos.x()) + ", " +
                         std::to_string(pos.y()) + ", " +
                         std::to_string(pos.z()) + "]>";
                }));

  // Molecule Origin enum
  enum_<Molecule::Origin>("Origin")
      .value("CARTESIAN", Molecule::Origin::Cartesian)
      .value("CENTROID", Molecule::Origin::Centroid)
      .value("CENTEROFMASS", Molecule::Origin::CenterOfMass);

  // Molecule class binding
  class_<Molecule>("Molecule")
      .constructor<const IVec &, const Mat3N &>()
      .class_function("fromAtoms",
                      optional_override([](const std::vector<Atom> &atoms) {
                        return Molecule(atoms);
                      }))
      .function("size", &Molecule::size)
      .function("elements", &Molecule::elements)
      .function("positions", &Molecule::positions)
      .function("name", &Molecule::name)
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
      .function("charge", &Molecule::charge)
      .function("setCharge", &Molecule::set_charge)
      .function("multiplicity", &Molecule::multiplicity)
      .function("setMultiplicity", &Molecule::set_multiplicity)
      .function("numElectrons", &Molecule::num_electrons)
      .function("centroid", &Molecule::centroid)
      .function("rotate", select_overload<void(const Mat3 &, Molecule::Origin)>(
                              &Molecule::rotate))
      .function("translate", &Molecule::translate)
      .function("rotated",
                select_overload<Molecule(const Mat3 &, Molecule::Origin) const>(
                    &Molecule::rotated))
      .function("translated", &Molecule::translated)
      .function("centered", optional_override([](const Molecule &mol,
                                                 Molecule::Origin origin) {
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
      .class_function("fromXyzFile",
                      optional_override([](const std::string &filename) {
                        return occ::io::molecule_from_xyz_file(filename);
                      }))
      .class_function(
          "fromXyzString",
          optional_override([](const emscripten::val &contents_val) {
            std::string contents = contents_val.as<std::string>();
            return occ::io::molecule_from_xyz_string(contents);
          }))
      .function("translationalFreeEnergy", &Molecule::translational_free_energy)
      .function("rotationalFreeEnergy", &Molecule::rotational_free_energy)
      .function("toString", optional_override([](const Molecule &mol) {
                  auto com = mol.center_of_mass();
                  return std::string("<Molecule ") + mol.name() + " @[" +
                         std::to_string(com.x()) + ", " +
                         std::to_string(com.y()) + ", " +
                         std::to_string(com.z()) + "]>";
                }));

  // Dimer class binding
  class_<Dimer>("Dimer")
      .constructor<const Molecule &, const Molecule &>()
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
      .constructor<const Molecule &>()
      .function("getDescription",
                optional_override([](const MolecularPointGroup &pg) {
                  return std::string(pg.description());
                }))
      .function("getPointGroupString",
                optional_override([](const MolecularPointGroup &pg) {
                  return std::string(pg.point_group_string());
                }))
      .property("pointGroup", &MolecularPointGroup::point_group)
      .property("symmetryNumber", &MolecularPointGroup::symmetry_number)
      .function("toString",
                optional_override([](const MolecularPointGroup &pg) {
                  return std::string("<MolecularPointGroup '") +
                         pg.point_group_string() + "'>";
                }));

  // Utility functions
  function("eemPartialCharges", &occ::core::charges::eem_partial_charges);
  function("eeqPartialCharges", &occ::core::charges::eeq_partial_charges);
  function("eeqCoordinationNumbers",
           &occ::core::charges::eeq_coordination_numbers);

  // Data directory functions
  function("setDataDirectory", &set_data_directory_wrapper);
  function("getDataDirectory", &get_data_directory_wrapper);

  // Parallel threading functions
  function("setNumThreads", &occ::parallel::set_num_threads);
  function("getNumThreads", &occ::parallel::get_num_threads);

  // Logging level enum
  enum_<spdlog::level::level_enum>("LogLevel")
      .value("TRACE", spdlog::level::trace)
      .value("DEBUG", spdlog::level::debug)
      .value("INFO", spdlog::level::info)
      .value("WARN", spdlog::level::warn)
      .value("ERROR", spdlog::level::err)
      .value("CRITICAL", spdlog::level::critical)
      .value("OFF", spdlog::level::off);

  // Logging functions
  function("setLogLevel", optional_override([](int level) {
             occ::log::set_log_level(
                 static_cast<spdlog::level::level_enum>(level));
           }));

  function("setLogLevelString", optional_override([](const std::string &level) {
             occ::log::set_log_level(level);
           }));

  function("registerLogCallback",
           optional_override([](emscripten::val callback) {
             occ::log::register_log_callback(
                 [callback](spdlog::level::level_enum level,
                            const std::string &message) {
                   callback(static_cast<int>(level), message);
                 });
           }));

  function("clearLogCallbacks", &occ::log::clear_log_callbacks);

  function("getBufferedLogs", optional_override([]() {
             auto logs = occ::log::get_buffered_logs();
             emscripten::val result = emscripten::val::array();
             for (size_t i = 0; i < logs.size(); ++i) {
               emscripten::val log_entry = emscripten::val::object();
               log_entry.set("level", static_cast<int>(logs[i].first));
               log_entry.set("message", logs[i].second);
               result.set(i, log_entry);
             }
             return result;
           }));

  function("clearLogBuffer", &occ::log::clear_log_buffer);
  function("setLogBuffering", &occ::log::set_log_buffering);

  // Direct logging functions
  function("logTrace", optional_override([](const std::string &msg) {
             occ::log::trace(msg);
           }));
  function("logDebug", optional_override([](const std::string &msg) {
             occ::log::debug(msg);
           }));
  function("logInfo", optional_override(
                          [](const std::string &msg) { occ::log::info(msg); }));
  function("logWarn", optional_override(
                          [](const std::string &msg) { occ::log::warn(msg); }));
  function("logError", optional_override([](const std::string &msg) {
             occ::log::error(msg);
           }));
  function("logCritical", optional_override([](const std::string &msg) {
             occ::log::critical(msg);
           }));

  // ElasticTensor averaging schemes
  enum_<ElasticTensor::AveragingScheme>("AveragingScheme")
      .value("VOIGT", ElasticTensor::AveragingScheme::Voigt)
      .value("REUSS", ElasticTensor::AveragingScheme::Reuss)
      .value("HILL", ElasticTensor::AveragingScheme::Hill)
      .value("NUMERICAL", ElasticTensor::AveragingScheme::Numerical);

  // ElasticTensor class
  class_<ElasticTensor>("ElasticTensor")
      .constructor<const Mat6 &>()

      // Young's modulus methods
      .function("youngsModulus",
                optional_override([](const ElasticTensor &et, const Vec3 &dir) {
                  return et.youngs_modulus(dir);
                }))

      // Linear compressibility methods
      .function("linearCompressibility",
                optional_override([](const ElasticTensor &et, const Vec3 &dir) {
                  return et.linear_compressibility(dir);
                }))

      // Shear modulus methods
      .function("shearModulus",
                optional_override([](const ElasticTensor &et, const Vec3 &dir1,
                                     const Vec3 &dir2) {
                  return et.shear_modulus(dir1, dir2);
                }))
      .function("shearModulusMinMax",
                optional_override([](const ElasticTensor &et, const Vec3 &dir) {
                  auto [min_val, max_val] = et.shear_modulus_minmax(dir);
                  emscripten::val result = emscripten::val::object();
                  result.set("min", min_val);
                  result.set("max", max_val);
                  return result;
                }))

      // Poisson's ratio methods
      .function("poissonRatio",
                optional_override([](const ElasticTensor &et, const Vec3 &dir1,
                                     const Vec3 &dir2) {
                  return et.poisson_ratio(dir1, dir2);
                }))
      .function("poissonRatioMinMax",
                optional_override([](const ElasticTensor &et, const Vec3 &dir) {
                  auto [min_val, max_val] = et.poisson_ratio_minmax(dir);
                  emscripten::val result = emscripten::val::object();
                  result.set("min", min_val);
                  result.set("max", max_val);
                  return result;
                }))

      // Average properties
      .function("averageBulkModulus",
                optional_override([](const ElasticTensor &et,
                                     ElasticTensor::AveragingScheme avg) {
                  return et.average_bulk_modulus(avg);
                }))
      .function("averageShearModulus",
                optional_override([](const ElasticTensor &et,
                                     ElasticTensor::AveragingScheme avg) {
                  return et.average_shear_modulus(avg);
                }))
      .function("averageYoungsModulus",
                optional_override([](const ElasticTensor &et,
                                     ElasticTensor::AveragingScheme avg) {
                  return et.average_youngs_modulus(avg);
                }))
      .function("averagePoissonRatio",
                optional_override([](const ElasticTensor &et,
                                     ElasticTensor::AveragingScheme avg) {
                  return et.average_poisson_ratio(avg);
                }))

      // Directional averages and reduced modulus
      .function("averagePoissonRatioDirection",
                optional_override([](const ElasticTensor &et, const Vec3 &dir,
                                     int num_samples) {
                  return et.average_poisson_ratio_direction(dir, num_samples);
                }))
      .function("reducedYoungsModulus",
                optional_override([](const ElasticTensor &et, const Vec3 &dir,
                                     int num_samples) {
                  return et.reduced_youngs_modulus(dir, num_samples);
                }))

      // Matrix access
      .property("voigtC", &ElasticTensor::voigt_c)
      .property("voigtS", &ElasticTensor::voigt_s)
      
      // Eigenvalues
      .function("eigenvalues", &ElasticTensor::eigenvalues)

      // Rotation methods
      .function("voigtRotationMatrix",
                optional_override([](const ElasticTensor &et, const Mat3 &rotation) {
                  return et.voigt_rotation_matrix(rotation);
                }))
      .function("rotateVoigtStiffness",
                optional_override([](const ElasticTensor &et, const Mat3 &rotation) {
                  return et.rotate_voigt_stiffness(rotation);
                }))
      .function("rotateVoigtCompliance",
                optional_override([](const ElasticTensor &et, const Mat3 &rotation) {
                  return et.rotate_voigt_compliance(rotation);
                }))

      // Acoustic velocities
      .function("transverseAcousticVelocity",
                optional_override([](const ElasticTensor &et, double bulk_modulus_gpa,
                                     double shear_modulus_gpa, double density_g_cm3) {
                  return et.transverse_acoustic_velocity(bulk_modulus_gpa, shear_modulus_gpa, density_g_cm3);
                }))
      .function("longitudinalAcousticVelocity",
                optional_override([](const ElasticTensor &et, double bulk_modulus_gpa,
                                     double shear_modulus_gpa, double density_g_cm3) {
                  return et.longitudinal_acoustic_velocity(bulk_modulus_gpa, shear_modulus_gpa, density_g_cm3);
                }))
      .function("acousticVelocitiesWithCrystal",
                optional_override([](const ElasticTensor &et, const Crystal &crystal, ElasticTensor::AveragingScheme scheme) {
                  double density = crystal.density();
                  double K = et.average_bulk_modulus(scheme);
                  double G = et.average_shear_modulus(scheme);
                  double v_s = et.transverse_acoustic_velocity(K, G, density);
                  double v_p = et.longitudinal_acoustic_velocity(K, G, density);
                  emscripten::val result = emscripten::val::object();
                  result.set("vs", v_s);
                  result.set("vp", v_p);
                  result.set("density", density);
                  return result;
                }));

  // Helper function to generate directional data for visualization
  function("generateDirectionalData",
           optional_override([](const ElasticTensor &et,
                                const std::string &property, int num_points) {
             emscripten::val result = emscripten::val::array();

             // Generate points on a sphere
             for (int i = 0; i < num_points; ++i) {
               double theta =
                   (static_cast<double>(i) / num_points) * 2.0 * M_PI;
               Vec3 direction(std::cos(theta), std::sin(theta),
                              0.0); // For 2D visualization in xy plane

               double value = 0.0;
               if (property == "youngs") {
                 value = et.youngs_modulus(direction);
               } else if (property == "linear_compressibility") {
                 value = et.linear_compressibility(direction);
               } else if (property == "shear") {
                 // Use perpendicular direction for shear
                 Vec3 shear_dir(-std::sin(theta), std::cos(theta), 0.0);
                 value = et.shear_modulus(direction, shear_dir);
               } else if (property == "poisson") {
                 Vec3 shear_dir(-std::sin(theta), std::cos(theta), 0.0);
                 value = et.poisson_ratio(direction, shear_dir);
               }

               emscripten::val point = emscripten::val::object();
               point.set("x", direction.x() * value);
               point.set("y", direction.y() * value);
               point.set("value", value);
               point.set("angle", theta);
               result.set(i, point);
             }

             return result;
           }));

  // Float matrix types for isosurface calculations
  class_<FMat3N>("FMat3N")
      .function("set", optional_override([](FMat3N &m, int row, int col,
                                            float val) { m(row, col) = val; }))
      .function("get", optional_override([](const FMat3N &m, int row, int col) {
                  return m(row, col);
                }))
      .function("rows",
                optional_override([](const FMat3N &m) { return m.rows(); }))
      .function("cols",
                optional_override([](const FMat3N &m) { return m.cols(); }))
      .class_function("create", optional_override([](int cols) {
                        FMat3N result = FMat3N::Zero(3, cols);
                        return result;
                      }));

  class_<FVec>("FVec")
      .function("size",
                optional_override([](const FVec &v) { return v.size(); }))
      .function("get",
                optional_override([](const FVec &v, int i) { return v(i); }))
      .function("set", optional_override(
                           [](FVec &v, int i, float val) { v(i) = val; }))
      .class_function("create", optional_override([](int size) {
                        FVec result = FVec::Zero(size);
                        return result;
                      }));
}
