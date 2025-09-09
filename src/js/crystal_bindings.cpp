#include "crystal_bindings.h"
#include <ankerl/unordered_dense.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <memory>
#include <occ/crystal/asymmetric_unit.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/crystal/unitcell.h>
#include <occ/io/cifparser.h>
#include <sstream>

using namespace emscripten;
using namespace occ;
using namespace occ::crystal;

void register_crystal_bindings() {
  // HKL (Miller Indices) class
  class_<HKL>("HKL")
      .constructor<int, int, int>()
      .constructor<>()
      .property("h", &HKL::h)
      .property("k", &HKL::k)
      .property("l", &HKL::l)
      .function("d",
                optional_override([](const HKL &hkl, const UnitCell &cell) {
                  return hkl.d(cell.direct());
                }))
      .function("vector", optional_override([](const HKL &hkl) {
                  Vec3 v = hkl.vector();
                  val result = val::array();
                  result.set(0, v(0));
                  result.set(1, v(1));
                  result.set(2, v(2));
                  return result;
                }))
      .function("toString", optional_override([](const HKL &hkl) {
                  std::ostringstream ss;
                  ss << "(" << hkl.h << " " << hkl.k << " " << hkl.l << ")";
                  return ss.str();
                }))
      .class_function("floor", optional_override([](const val &vecArray) {
                        Vec3 v(vecArray[0].as<double>(),
                               vecArray[1].as<double>(),
                               vecArray[2].as<double>());
                        return HKL::floor(v);
                      }))
      .class_function("ceil", optional_override([](const val &vecArray) {
                        Vec3 v(vecArray[0].as<double>(),
                               vecArray[1].as<double>(),
                               vecArray[2].as<double>());
                        return HKL::ceil(v);
                      }));

  // UnitCell class
  class_<UnitCell>("UnitCell")
      .constructor<double, double, double, double, double, double>()
      .constructor<>()
      .function("a", &UnitCell::a)
      .function("b", &UnitCell::b)
      .function("c", &UnitCell::c)
      .function("alpha", &UnitCell::alpha)
      .function("beta", &UnitCell::beta)
      .function("gamma", &UnitCell::gamma)
      .function("volume", &UnitCell::volume)
      .function("setA", &UnitCell::set_a)
      .function("setB", &UnitCell::set_b)
      .function("setC", &UnitCell::set_c)
      .function("setAlpha", &UnitCell::set_alpha)
      .function("setBeta", &UnitCell::set_beta)
      .function("setGamma", &UnitCell::set_gamma)
      .function("isCubic", &UnitCell::is_cubic)
      .function("isTriclinic", &UnitCell::is_triclinic)
      .function("isMonoclinic", &UnitCell::is_monoclinic)
      .function("isOrthorhombic", &UnitCell::is_orthorhombic)
      .function("isTetragonal", &UnitCell::is_tetragonal)
      .function("isRhombohedral", &UnitCell::is_rhombohedral)
      .function("isHexagonal", &UnitCell::is_hexagonal)
      .function(
          "toCartesian",
          optional_override([](const UnitCell &cell, const val &fracArray) {
            // Convert fractional coordinates to cartesian
            Mat3N frac_coords(3, fracArray["length"].as<int>() / 3);
            for (int i = 0; i < fracArray["length"].as<int>() / 3; ++i) {
              frac_coords(0, i) = fracArray[i * 3 + 0].as<double>();
              frac_coords(1, i) = fracArray[i * 3 + 1].as<double>();
              frac_coords(2, i) = fracArray[i * 3 + 2].as<double>();
            }
            Mat3N cart_coords = cell.to_cartesian(frac_coords);

            val result = val::global("Float64Array").new_(cart_coords.size());
            for (int i = 0; i < cart_coords.cols(); ++i) {
              result.set(i * 3 + 0, cart_coords(0, i));
              result.set(i * 3 + 1, cart_coords(1, i));
              result.set(i * 3 + 2, cart_coords(2, i));
            }
            return result;
          }))
      .function(
          "toFractional",
          optional_override([](const UnitCell &cell, const val &cartArray) {
            // Convert cartesian coordinates to fractional
            Mat3N cart_coords(3, cartArray["length"].as<int>() / 3);
            for (int i = 0; i < cartArray["length"].as<int>() / 3; ++i) {
              cart_coords(0, i) = cartArray[i * 3 + 0].as<double>();
              cart_coords(1, i) = cartArray[i * 3 + 1].as<double>();
              cart_coords(2, i) = cartArray[i * 3 + 2].as<double>();
            }
            Mat3N frac_coords = cell.to_fractional(cart_coords);

            val result = val::global("Float64Array").new_(frac_coords.size());
            for (int i = 0; i < frac_coords.cols(); ++i) {
              result.set(i * 3 + 0, frac_coords(0, i));
              result.set(i * 3 + 1, frac_coords(1, i));
              result.set(i * 3 + 2, frac_coords(2, i));
            }
            return result;
          }))
      .function("getDirect", optional_override([](const UnitCell &cell) {
                  const Mat3 &direct = cell.direct();
                  val result = val::global("Float64Array").new_(9);
                  for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                      result.set(i * 3 + j, direct(i, j));
                    }
                  }
                  return result;
                }))
      .function("getReciprocal", optional_override([](const UnitCell &cell) {
                  const Mat3 &reciprocal = cell.reciprocal();
                  val result = val::global("Float64Array").new_(9);
                  for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                      result.set(i * 3 + j, reciprocal(i, j));
                    }
                  }
                  return result;
                }))
      .function("getAVector", optional_override([](const UnitCell &cell) {
                  Vec3 a_vec = cell.a_vector();
                  val result = val::array();
                  result.set(0, a_vec(0));
                  result.set(1, a_vec(1));
                  result.set(2, a_vec(2));
                  return result;
                }))
      .function("getBVector", optional_override([](const UnitCell &cell) {
                  Vec3 b_vec = cell.b_vector();
                  val result = val::array();
                  result.set(0, b_vec(0));
                  result.set(1, b_vec(1));
                  result.set(2, b_vec(2));
                  return result;
                }))
      .function("getCVector", optional_override([](const UnitCell &cell) {
                  Vec3 c_vec = cell.c_vector();
                  val result = val::array();
                  result.set(0, c_vec(0));
                  result.set(1, c_vec(1));
                  result.set(2, c_vec(2));
                  return result;
                }));

  // SymmetryOperation class
  class_<SymmetryOperation>("SymmetryOperation")
      .constructor<std::string>()
      .class_function("fromInt", optional_override([](int symop_int) {
                        return SymmetryOperation(symop_int);
                      }))
      .function("toInt", &SymmetryOperation::to_int)
      .function("toString", &SymmetryOperation::to_string)
      .function("inverted", &SymmetryOperation::inverted)
      .function("isIdentity", &SymmetryOperation::is_identity)
      .function("getRotation",
                optional_override([](const SymmetryOperation &symop) {
                  const Mat3 &rot = symop.rotation();
                  val result = val::global("Float64Array").new_(9);
                  for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                      result.set(i * 3 + j, rot(i, j));
                    }
                  }
                  return result;
                }))
      .function("getTranslation",
                optional_override([](const SymmetryOperation &symop) {
                  const Vec3 &trans = symop.translation();
                  val result = val::array();
                  result.set(0, trans(0));
                  result.set(1, trans(1));
                  result.set(2, trans(2));
                  return result;
                }))
      .function(
          "apply", optional_override([](const SymmetryOperation &symop,
                                        const val &coordsArray) {
            // Apply symmetry operation to fractional coordinates
            Mat3N coords(3, coordsArray["length"].as<int>() / 3);
            for (int i = 0; i < coordsArray["length"].as<int>() / 3; ++i) {
              coords(0, i) = coordsArray[i * 3 + 0].as<double>();
              coords(1, i) = coordsArray[i * 3 + 1].as<double>();
              coords(2, i) = coordsArray[i * 3 + 2].as<double>();
            }
            Mat3N result_coords = symop.apply(coords);

            val result = val::global("Float64Array").new_(result_coords.size());
            for (int i = 0; i < result_coords.cols(); ++i) {
              result.set(i * 3 + 0, result_coords(0, i));
              result.set(i * 3 + 1, result_coords(1, i));
              result.set(i * 3 + 2, result_coords(2, i));
            }
            return result;
          }));

  // SpaceGroup class
  class_<SpaceGroup>("SpaceGroup")
      .constructor<std::string>()
      .class_function("fromNumber", optional_override([](int number) {
                        return SpaceGroup(number);
                      }))
      .function("number", &SpaceGroup::number)
      .function("symbol", &SpaceGroup::symbol)
      .function("shortName", &SpaceGroup::short_name)
      .function("getSymmetryOperations",
                optional_override([](const SpaceGroup &sg) {
                  const auto &symops = sg.symmetry_operations();
                  val result = val::array();
                  for (size_t i = 0; i < symops.size(); ++i) {
                    result.set(i, symops[i]);
                  }
                  return result;
                }))
      .function("numSymmetryOperations",
                optional_override([](const SpaceGroup &sg) {
                  return sg.symmetry_operations().size();
                }))
      .function("hasHRChoice", &SpaceGroup::has_H_R_choice);

  // AsymmetricUnit class
  class_<AsymmetricUnit>("AsymmetricUnit")
      .constructor<>()
      .function("size", &AsymmetricUnit::size)
      .function("chemicalFormula", &AsymmetricUnit::chemical_formula)
      .function("generateDefaultLabels",
                &AsymmetricUnit::generate_default_labels)
      .function("getPositions",
                optional_override([](const AsymmetricUnit &asym) {
                  const Mat3N &pos = asym.positions;
                  val result = val::global("Float64Array").new_(pos.size());
                  for (int i = 0; i < pos.cols(); ++i) {
                    result.set(i * 3 + 0, pos(0, i));
                    result.set(i * 3 + 1, pos(1, i));
                    result.set(i * 3 + 2, pos(2, i));
                  }
                  return result;
                }))
      .function("getAtomicNumbers",
                optional_override([](const AsymmetricUnit &asym) {
                  const IVec &nums = asym.atomic_numbers;
                  val result = val::global("Int32Array").new_(nums.size());
                  for (int i = 0; i < nums.size(); ++i) {
                    result.set(i, nums(i));
                  }
                  return result;
                }))
      .function("getLabels", optional_override([](const AsymmetricUnit &asym) {
                  const auto &labels = asym.labels;
                  val result = val::array();
                  for (size_t i = 0; i < labels.size(); ++i) {
                    result.set(i, labels[i]);
                  }
                  return result;
                }))
      .function("setPositions", optional_override([](AsymmetricUnit &asym,
                                                     const val &posArray) {
                  int numAtoms = posArray["length"].as<int>() / 3;
                  asym.positions = Mat3N(3, numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    asym.positions(0, i) = posArray[i * 3 + 0].as<double>();
                    asym.positions(1, i) = posArray[i * 3 + 1].as<double>();
                    asym.positions(2, i) = posArray[i * 3 + 2].as<double>();
                  }
                }))
      .function("setAtomicNumbers", optional_override([](AsymmetricUnit &asym,
                                                         const val &numsArray) {
                  int numAtoms = numsArray["length"].as<int>();
                  asym.atomic_numbers = IVec(numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    asym.atomic_numbers(i) = numsArray[i].as<int>();
                  }
                }))
      .function("setLabels", optional_override([](AsymmetricUnit &asym,
                                                  const val &labelsArray) {
                  int numLabels = labelsArray["length"].as<int>();
                  asym.labels.clear();
                  asym.labels.reserve(numLabels);
                  for (int i = 0; i < numLabels; ++i) {
                    asym.labels.push_back(labelsArray[i].as<std::string>());
                  }
                }));

  // Main Crystal class
  class_<Crystal>("Crystal")
      .constructor<AsymmetricUnit, SpaceGroup, UnitCell>()
      .function("asymmetricUnit",
                optional_override(
                    [](const Crystal &crystal) -> const AsymmetricUnit & {
                      return crystal.asymmetric_unit();
                    }),
                allow_raw_pointers())
      .function(
          "spaceGroup",
          optional_override([](const Crystal &crystal) -> const SpaceGroup & {
            return crystal.space_group();
          }),
          allow_raw_pointers())
      .function(
          "unitCell",
          optional_override([](const Crystal &crystal) -> const UnitCell & {
            return crystal.unit_cell();
          }),
          allow_raw_pointers())
      .function("numSites", &Crystal::num_sites)
      .function("volume", &Crystal::volume)
      .function("chemicalFormula",
                optional_override([](const Crystal &crystal) {
                  return crystal.asymmetric_unit().chemical_formula();
                }))
      .function(
          "toFractional",
          optional_override([](const Crystal &crystal, const val &cartArray) {
            Mat3N cart_coords(3, cartArray["length"].as<int>() / 3);
            for (int i = 0; i < cartArray["length"].as<int>() / 3; ++i) {
              cart_coords(0, i) = cartArray[i * 3 + 0].as<double>();
              cart_coords(1, i) = cartArray[i * 3 + 1].as<double>();
              cart_coords(2, i) = cartArray[i * 3 + 2].as<double>();
            }
            Mat3N frac_coords = crystal.to_fractional(cart_coords);

            val result = val::global("Float64Array").new_(frac_coords.size());
            for (int i = 0; i < frac_coords.cols(); ++i) {
              result.set(i * 3 + 0, frac_coords(0, i));
              result.set(i * 3 + 1, frac_coords(1, i));
              result.set(i * 3 + 2, frac_coords(2, i));
            }
            return result;
          }))
      .function(
          "toCartesian",
          optional_override([](const Crystal &crystal, const val &fracArray) {
            Mat3N frac_coords(3, fracArray["length"].as<int>() / 3);
            for (int i = 0; i < fracArray["length"].as<int>() / 3; ++i) {
              frac_coords(0, i) = fracArray[i * 3 + 0].as<double>();
              frac_coords(1, i) = fracArray[i * 3 + 1].as<double>();
              frac_coords(2, i) = fracArray[i * 3 + 2].as<double>();
            }
            Mat3N cart_coords = crystal.to_cartesian(frac_coords);

            val result = val::global("Float64Array").new_(cart_coords.size());
            for (int i = 0; i < cart_coords.cols(); ++i) {
              result.set(i * 3 + 0, cart_coords(0, i));
              result.set(i * 3 + 1, cart_coords(1, i));
              result.set(i * 3 + 2, cart_coords(2, i));
            }
            return result;
          }))
      .function("normalizeHydrogenBondlengths",
                optional_override([](Crystal &crystal) {
                  return crystal.normalize_hydrogen_bondlengths();
                }))
      .function("normalizeHydrogenBondlengthsCustom",
                optional_override([](Crystal &crystal, const val &customLengths) {
                  ankerl::unordered_dense::map<int, double> lengths_map;
                  val keys = val::global("Object")["keys"](customLengths);
                  int length = keys["length"].as<int>();
                  for (int i = 0; i < length; ++i) {
                    int atomic_number = keys[i].as<int>();
                    double bond_length = customLengths[keys[i]].as<double>();
                    lengths_map[atomic_number] = bond_length;
                  }
                  return crystal.normalize_hydrogen_bondlengths(lengths_map);
                }))
      .class_function("fromCifFile",
                      optional_override([](const std::string &filename) {
                        occ::io::CifParser parser;
                        return parser.parse_crystal_from_file(filename).value();
                      }))
      .class_function(
          "fromCifString", optional_override([](const std::string &cifContent) {
            occ::io::CifParser parser;
            return parser.parse_crystal_from_string(cifContent).value();
          }));

  // Utility functions
  function("parseSpaceGroupNumber",
           optional_override([](const std::string &symbol) {
             SpaceGroup sg(symbol);
             return sg.number();
           }));

  function("parseSpaceGroupSymbol", optional_override([](int number) {
             SpaceGroup sg(number);
             return sg.symbol();
           }));
}