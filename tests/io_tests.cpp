#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/crystal/crystal.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/io/crystalgrower.h>
#include <occ/io/cube.h>
#include <occ/io/dftb_gen.h>
#include <occ/io/eigen_json.h>
#include <occ/io/gmf.h>
#include <occ/io/isosurface_json.h>
#include <occ/io/orca_json.h>
#include <occ/io/ply.h>
#include <occ/io/qcschema.h>
#include <occ/io/shelxfile.h>
#include <occ/io/cifparser.h>
#include <occ/io/cifwriter.h>
#include <gemmi/cif.hpp>

using occ::format_matrix;
using occ::util::all_close;

auto acetic_crystal() {
  const std::vector<std::string> labels = {"C1", "C2", "H1", "H2",
                                           "H3", "H4", "O1", "O2"};
  occ::IVec nums(labels.size());
  occ::Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++) {
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  }
  positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200,
      0.05100, -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900,
      0.05300, 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030,
      0.17690;

  occ::crystal::AsymmetricUnit asym(positions.transpose(), nums, labels);
  occ::crystal::SpaceGroup sg(33);
  occ::crystal::UnitCell cell =
      occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);

  return occ::crystal::Crystal(asym, sg, cell);
}

TEST_CASE("Write acetic CrystalGrower structure file", "[write]") {
  auto acetic = acetic_crystal();
  auto dimers = acetic.unit_cell_dimers(3.8);
  occ::io::crystalgrower::StructureWriter writer(std::cout);
  occ::io::crystalgrower::NetWriter net_writer(std::cout);
  writer.write(acetic, dimers);
  net_writer.write(acetic, dimers);
  REQUIRE(true);
}

// Eigen JSON
using nlohmann::json;

namespace test {
struct test_struct {
  occ::Vec vector;
  occ::Mat3 matrix3d;
  occ::RowVec3 rvec3;

  bool operator==(const test_struct &other) const {
    using occ::util::all_close;
    return all_close(vector, other.vector) &&
           all_close(matrix3d, other.matrix3d) && all_close(rvec3, other.rvec3);
  }
};

void to_json(json &js, const test_struct &t) {
  js = {{"vector", t.vector}, {"matrix3d", t.matrix3d}, {"rvec3", t.rvec3}};
}

void from_json(const json &j, test_struct &t) {
  j.at("vector").get_to(t.vector);
  j.at("matrix3d").get_to(t.matrix3d);
  j.at("rvec3").get_to(t.rvec3);
}

} // namespace test

TEST_CASE("eigen serialize/deserialize as part of struct",
          "[serialize,deserialize]") {
  auto t = test::test_struct{occ::Vec::Zero(10), occ::Mat3::Identity(),
                             occ::RowVec3::Zero()};
  nlohmann::json j = t;
  auto t2 = j.get<test::test_struct>();
  REQUIRE(t == t2);
}

// QCSchema

const char *json_input_contents = R""""({
    "schema_name": "qcschema_input",
    "schema_version": 1,
    "return_output": true,
    "molecule": {
        "geometry": [
            0.0,
            0.0,
            -0.1294769411935893,
            0.0,
            -1.494187339479985,
            1.0274465079245698,
            0.0,
            1.494187339479985,
            1.0274465079245698
        ],
        "symbols": [
            "O",
            "H",
            "H"
        ]
    },
    "driver": "energy",
    "model": {
        "method": "b3lyp",
        "basis": "6-31g"
    },
    "keywords": {
        "scf_type": "df",
        "mp2_type": "df",
        "cc_type": "df",
        "scf_properties": ["mayer_indices"]
    }
}
)"""";

TEST_CASE("Read QCSchema formatted input water b3lyp/6-31G", "[read]") {
  std::istringstream json(json_input_contents);
  occ::io::QCSchemaReader reader(json);
}

TEST_CASE("Serial Molecule & Dimer to JSON", "[json,write]") {
  fmt::print("Molecule to JSON\n");
  auto acetic = acetic_crystal();
  auto dimers = acetic.unit_cell_dimers(3.8);
  const auto &mols = acetic.symmetry_unique_molecules();
  nlohmann::json j;
  j["symmetry_unique_molecules"] = mols;
  fmt::print("{}\n", j.dump());
  fmt::print("Dimer to JSON\n");
  nlohmann::json jd;
  jd["symmetry_unique_dimers"] = dimers.unique_dimers;
  fmt::print("{}\n", jd.dump());
}

const char *gmf_file_contents = R""""(
 title: Created by occ
  name: acetic
 space: P n a 21
  cell: 13.310000 4.100000 5.750000  90.000000 90.000000 90.000000
 morph: unrelaxed equilibrium

miller:    1   1   0
 0.000000   1  1      0.0000     0.0000     0.0000     0.0000  -1
)"""";

TEST_CASE("GMFWriter Tests", "[GMFWriter]") {
  using occ::io::GMFWriter;

  GMFWriter::Facet facet = {{1, 1, 0}, 0.0, 1, 1, 0.0, 0.0, 0.0, 0.0, -1};
  auto crystal = acetic_crystal();

  SECTION("Setting properties") {
    GMFWriter gmf(crystal);

    gmf.set_title("Created by occ");
    REQUIRE(gmf.title() == "Created by occ");
    gmf.set_name("acetic");
    REQUIRE(gmf.name() == "acetic");

    gmf.set_morphology_kind("unrelaxed equilibrium");
    REQUIRE(gmf.morphology_kind() == "unrelaxed equilibrium");

    gmf.add_facet(facet);
    REQUIRE(gmf.number_of_facets() == 1);
  }

  SECTION("Checking file content") {
    GMFWriter gmf(crystal);
    gmf.set_title("Created by occ");
    gmf.set_name("acetic");
    gmf.set_morphology_kind("unrelaxed equilibrium");
    gmf.add_facet(facet);

    std::ostringstream gmf_os;
    gmf.write(gmf_os);
    std::string contents = gmf_os.str();
    REQUIRE(gmf_os.str() == gmf_file_contents);
  }
}

TEST_CASE("read/write dftb gen format", "[write,read]") {
  auto acetic = acetic_crystal();
  SECTION("Crystal") {
    std::string gen_contents;

    {
      occ::io::DftbGenFormat gen;
      gen.set_crystal(acetic);
      std::ostringstream gen_contents_stream;
      gen.write(gen_contents_stream);
      gen_contents = gen_contents_stream.str();
    }
    fmt::print("dftb gen contents (crystal):\n{}\n", gen_contents);

    {
      std::istringstream gen_contents_stream(gen_contents);
      occ::io::DftbGenFormat gen;
      gen.parse(gen_contents_stream);
      auto maybe_crystal = gen.crystal();
      REQUIRE(maybe_crystal);
      auto acetic_read = maybe_crystal.value();
      REQUIRE(all_close(acetic_read.unit_cell().direct(),
                        acetic.unit_cell().direct()));
      REQUIRE(acetic_read.asymmetric_unit().size() ==
              4 * acetic.asymmetric_unit().size());
    }
  }
  SECTION("Molecule") {
    std::string gen_contents;
    auto mol = acetic.symmetry_unique_molecules()[0];
    {
      occ::io::DftbGenFormat gen;
      gen.set_molecule(mol);
      std::ostringstream gen_contents_stream;
      gen.write(gen_contents_stream);
      gen_contents = gen_contents_stream.str();
    }
    fmt::print("dftb gen contents (molecule):\n{}\n", gen_contents);

    {
      std::istringstream gen_contents_stream(gen_contents);
      occ::io::DftbGenFormat gen;
      gen.parse(gen_contents_stream);
      auto maybe_molecule = gen.molecule();
      REQUIRE(maybe_molecule);
      auto acetic_read = maybe_molecule.value();
      REQUIRE(all_close(acetic_read.positions(), mol.positions()));
      REQUIRE(acetic_read.atomic_numbers() == mol.atomic_numbers());
    }
  }
}

TEST_CASE("crystal_json", "[write,read]") {
  SECTION("SymmetryOperation") {
    using occ::crystal::SymmetryOperation;
    SymmetryOperation s("x,y,z");
    nlohmann::json j;
    j["symop"] = s;
    SymmetryOperation d = j["symop"].get<SymmetryOperation>();
    REQUIRE(s == d);
  }

  SECTION("SpaceGroup") {
    using occ::crystal::SpaceGroup;
    SpaceGroup s(61);
    nlohmann::json j;
    j["sg"] = s;
    auto d = j["sg"].get<SpaceGroup>();
    REQUIRE(s.symmetry_operations() == d.symmetry_operations());
    REQUIRE(s.symbol() == d.symbol());
  }

  SECTION("UnitCell") {
    using occ::crystal::UnitCell;
    UnitCell s(3.0, 2.0, 1.0, M_PI / 2, M_PI / 2, M_PI / 2);
    nlohmann::json j;
    j["uc"] = s;
    auto d = j["uc"].get<UnitCell>();
    REQUIRE(all_close(s.direct(), d.direct()));
  }

  SECTION("AsymmetricUnit") {
    using occ::crystal::AsymmetricUnit;
    auto acetic = acetic_crystal();
    auto s = acetic.asymmetric_unit();
    nlohmann::json j;
    j["asym"] = s;
    auto d = j["asym"].get<AsymmetricUnit>();
    REQUIRE(all_close(s.positions, d.positions));
    REQUIRE(s.labels == d.labels);
  }

  SECTION("CrystalAtomRegion") {
    auto acetic = acetic_crystal();
    using occ::crystal::CrystalAtomRegion;
    auto s = acetic.unit_cell_atoms();
    nlohmann::json j;
    j["uc"] = s;
    CrystalAtomRegion d = j["uc"].get<CrystalAtomRegion>();
    REQUIRE(d.size() == s.size());
    REQUIRE(d.frac_pos.isApprox(s.frac_pos));
    REQUIRE(d.cart_pos.isApprox(s.cart_pos));
    REQUIRE(d.asym_idx == s.asym_idx);
    REQUIRE(d.atomic_numbers == s.atomic_numbers);
    REQUIRE(d.symop == s.symop);
  }

  SECTION("Crystal") {
    using occ::crystal::Crystal;
    auto s = acetic_crystal();
    nlohmann::json j;
    j["crystal"] = s;
    Crystal d = j["crystal"].get<Crystal>();
    REQUIRE(d.labels() == s.labels());
    REQUIRE(d.unit_cell().direct().isApprox(s.unit_cell().direct()));
  }
}

TEST_CASE("Write PLY mesh validation", "[io][ply]") {
  using occ::io::write_ply_mesh;
  using occ::isosurface::Isosurface;
  SECTION("Valid small mesh") {
    Isosurface mesh;
    mesh.vertices.resize(3, 3);
    mesh.vertices << 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
    mesh.faces.resize(3, 1);
    mesh.faces << 0, 1, 2;
    mesh.normals.resize(3, 3);
    mesh.normals << 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f;

    occ::FVec fprop(3);
    fprop << 1.0f, 2.0f, 3.0f;
    occ::IVec iprop(3);
    iprop << 1, 2, 3;
    mesh.properties.add("test_float", fprop);
    mesh.properties.add("test_int", iprop);

    // Test both binary and ASCII formats
    SECTION("ASCII format") {
      std::stringstream ss;
      REQUIRE_NOTHROW(write_ply_mesh("test.ply", mesh, false));
    }

    SECTION("Binary format") {
      std::stringstream ss;
      REQUIRE_NOTHROW(write_ply_mesh("test.ply", mesh, true));
    }
  }

  SECTION("Error handling") {
    Isosurface empty_mesh;

    SECTION("Empty mesh") {
      REQUIRE_THROWS(write_ply_mesh("test.ply", empty_mesh, false));
    }

    SECTION("Mismatched property sizes") {
      Isosurface mesh;
      mesh.vertices.resize(3, 2);
      mesh.vertices << 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f;
      occ::FVec bad_prop(1);
      bad_prop << 1.0f;
      mesh.properties.add("test", bad_prop); // Wrong size
      REQUIRE_THROWS(write_ply_mesh("test.ply", mesh, false));
    }
  }

  SECTION("Large mesh handling") {
    Isosurface large_mesh;
    // Create a larger mesh with many vertices/faces
    const int num_vertices = 1000000;
    large_mesh.vertices = occ::FMat3N::Random(3, num_vertices);
    large_mesh.faces = occ::IMat3N::Random(3, num_vertices);
    large_mesh.normals = occ::FMat3N::Random(3, num_vertices);
    occ::FVec prop = occ::FVec::Random(num_vertices);
    large_mesh.properties.add("test", prop);

    REQUIRE_NOTHROW(write_ply_mesh("large_test.ply", large_mesh, true));
  }
};

class CubeApproxMatcher : public Catch::Matchers::MatcherBase<occ::io::Cube> {
  const occ::io::Cube &m_cube;
  double m_tol;
  mutable std::string m_error_message;

public:
  CubeApproxMatcher(const occ::io::Cube &cube, double tol = 1e-5)
      : m_cube(cube), m_tol(tol) {}

  bool match(const occ::io::Cube &other) const override {
    if (m_cube.steps != other.steps) {
      m_error_message =
          fmt::format("\nGrid dimensions don't match"
                      "\nExpected: ({}, {}, {})"
                      "\nFound: ({}, {}, {})",
                      m_cube.steps.x(), m_cube.steps.y(), m_cube.steps.z(),
                      other.steps.x(), other.steps.y(), other.steps.z());
      return false;
    }
    if (!m_cube.origin.isApprox(other.origin, m_tol)) {
      m_error_message =
          fmt::format("\nOrigins don't match"
                      "\nExpected: ({}, {}, {})"
                      "\nFound: ({}, {}, {})",
                      m_cube.origin.x(), m_cube.origin.y(), m_cube.origin.z(),
                      other.origin.x(), other.origin.y(), other.origin.z());
      return false;
    }
    if (!m_cube.basis.isApprox(other.basis, m_tol)) {
      m_error_message =
          fmt::format("\nBasis vectors don't match"
                      "\nExpected:\n{}\nFound:\n{}",
                      format_matrix(m_cube.basis), format_matrix(other.basis));
      return false;
    }
    if (m_cube.atoms.size() != other.atoms.size()) {
      m_error_message = fmt::format("\nNumber of atoms don't match"
                                    "\nExpected: {}"
                                    "\nFound: {}",
                                    m_cube.atoms.size(), other.atoms.size());
      return false;
    }

    // Check grid data
    for (int x = 0; x < m_cube.steps(0); x++) {
      for (int y = 0; y < m_cube.steps(1); y++) {
        for (int z = 0; z < m_cube.steps(2); z++) {
          float diff = std::abs(m_cube.grid()(x, y, z) - other.grid()(x, y, z));
          if (diff > m_tol) {
            m_error_message = fmt::format(
                "\nGrid values don't match at ({}, {}, {})"
                "\nExpected: {}"
                "\nFound: {}"
                "\nDiff: {}",
                x, y, z, m_cube.grid()(x, y, z), other.grid()(x, y, z), diff);
            return false;
          }
        }
      }
    }
    return true;
  }

  std::string describe() const override {
    return "Approximately equals cube within tolerance " +
           std::to_string(m_tol) + m_error_message;
  }
};

TEST_CASE("Cube", "[cube]") {
  SECTION("Parse example cube file") {
    std::stringstream ss;
    ss << R"(Generated by OCC from file: water.owf.json
Scalar values for property 'electron_density'
    3    -1.123993     0.224611    -0.397687
    2     0.200000     0.000000     0.000000
    3     0.000000     0.200000     0.000000
    4     0.000000     0.000000     0.200000
    8     0.000000    -1.326958    -0.105939     0.018788
    1     0.000000    -1.931665     1.600174    -0.021710
    1     0.000000     0.486644     0.079598     0.009862
    0.929560     0.846625     0.763947     0.713051 
    1.598553     0.518882     0.931769     0.437888 
    0.706532     0.627551     0.595849     0.527861 
    1.218836     0.821015     0.877120     0.692500 
    1.299062     0.597163     0.894824     0.502266 
    0.806580     0.605963     0.681185     0.509903)";

    auto cube = occ::io::Cube::load(ss);

    // Check header info
    REQUIRE(cube.atoms.size() == 3);
    REQUIRE_THAT(cube.origin.x(), Catch::Matchers::WithinRel(-1.123993));
    REQUIRE_THAT(cube.origin.y(), Catch::Matchers::WithinRel(0.224611));
    REQUIRE_THAT(cube.origin.z(), Catch::Matchers::WithinRel(-0.397687));

    // Check grid dimensions
    REQUIRE(cube.steps.x() == 2);
    REQUIRE(cube.steps.y() == 3);
    REQUIRE(cube.steps.z() == 4);

    // Check atoms
    REQUIRE(cube.atoms[0].atomic_number == 8); // Oxygen
    REQUIRE(cube.atoms[1].atomic_number == 1); // Hydrogen
    REQUIRE(cube.atoms[2].atomic_number == 1); // Hydrogen

    // Check some data points
    REQUIRE_THAT(cube.grid()(0, 0, 0), Catch::Matchers::WithinRel(0.929560f));
    REQUIRE_THAT(cube.grid()(1, 2, 3), Catch::Matchers::WithinRel(0.509903f));

    // Test round-trip
    std::stringstream out;
    cube.save(out);

    // Parse the output and compare
    auto cube2 = occ::io::Cube::load(out);
    REQUIRE_THAT(cube2, CubeApproxMatcher(cube));
  }
}

TEST_CASE("Isosurface JSON", "[isosurface_json]") {
  SECTION("Convert simple isosurface to JSON") {
    // Create a simple test isosurface
    occ::isosurface::Isosurface surf;
    surf.kind = "test_surface";
    surf.description = "Test isosurface for JSON export";
    surf.isovalue = 0.002f;
    surf.separation = 0.2f;

    // Create simple triangle mesh
    surf.vertices.resize(3, 3); // 3 vertices
    surf.vertices << 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f;

    surf.faces.resize(3, 1); // 1 triangle
    surf.faces << 0, 1, 2;

    surf.normals.resize(3, 3);
    surf.normals << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f;

    // Convert to JSON
    auto json_str = occ::io::isosurface_to_json_string(surf);
    auto j = nlohmann::json::parse(json_str);

    // Check metadata
    REQUIRE(j["kind"] == "test_surface");
    REQUIRE(j["description"] == "Test isosurface for JSON export");
    REQUIRE_THAT(j["isovalue"].get<float>(),
                 Catch::Matchers::WithinRel(0.002f));
    REQUIRE_THAT(j["separation"].get<float>(),
                 Catch::Matchers::WithinRel(0.2f));

    // Check dimensions
    REQUIRE(j["numVertices"] == 3);
    REQUIRE(j["numFaces"] == 1);

    // Check vertices (flattened array)
    auto vertices = j["vertices"];
    REQUIRE(vertices.size() == 9); // 3 vertices * 3 components
    REQUIRE_THAT(vertices[0].get<float>(), Catch::Matchers::WithinRel(0.0f));
    REQUIRE_THAT(vertices[1].get<float>(), Catch::Matchers::WithinRel(0.0f));
    REQUIRE_THAT(vertices[2].get<float>(), Catch::Matchers::WithinRel(0.0f));

    // Check faces
    auto faces = j["faces"];
    REQUIRE(faces.size() == 3); // 1 triangle * 3 indices
    REQUIRE(faces[0] == 0);
    REQUIRE(faces[1] == 1);
    REQUIRE(faces[2] == 2);

    // Check normals
    auto normals = j["normals"];
    REQUIRE(normals.size() == 9); // 3 vertices * 3 components
    REQUIRE_THAT(normals[2].get<float>(), Catch::Matchers::WithinRel(1.0f));
    REQUIRE_THAT(normals[5].get<float>(), Catch::Matchers::WithinRel(1.0f));
    REQUIRE_THAT(normals[8].get<float>(), Catch::Matchers::WithinRel(1.0f));
  }
}

TEST_CASE("ShelxFile unified read/write", "[shelx][file]") {

  SECTION("Round-trip test with acetic acid") {
    // Create a simple crystal structure
    auto original_crystal = acetic_crystal();

    // Write to SHELX format
    occ::io::ShelxFile shelx_file;
    shelx_file.set_title("Acetic acid test structure");
    std::string shelx_content =
        shelx_file.write_crystal_to_string(original_crystal);

    INFO("Generated SHELX content:\n" << shelx_content);

    // Basic format checks
    REQUIRE(shelx_content.find("TITL") != std::string::npos);
    REQUIRE(shelx_content.find("CELL") != std::string::npos);
    REQUIRE(shelx_content.find("LATT") != std::string::npos);
    REQUIRE(shelx_content.find("SFAC") != std::string::npos);
    REQUIRE(shelx_content.find("END") != std::string::npos);

    // Read back the structure
    auto read_crystal = shelx_file.read_crystal_from_string(shelx_content);

    REQUIRE(read_crystal.has_value());

    const auto &rc = read_crystal.value();

    // Check unit cell parameters are preserved
    const auto &orig_uc = original_crystal.unit_cell();
    const auto &read_uc = rc.unit_cell();

    REQUIRE_THAT(orig_uc.a(), Catch::Matchers::WithinRel(read_uc.a(), 1e-3));
    REQUIRE_THAT(orig_uc.b(), Catch::Matchers::WithinRel(read_uc.b(), 1e-3));
    REQUIRE_THAT(orig_uc.c(), Catch::Matchers::WithinRel(read_uc.c(), 1e-3));
    REQUIRE_THAT(orig_uc.alpha(),
                 Catch::Matchers::WithinRel(read_uc.alpha(), 1e-3));
    REQUIRE_THAT(orig_uc.beta(),
                 Catch::Matchers::WithinRel(read_uc.beta(), 1e-3));
    REQUIRE_THAT(orig_uc.gamma(),
                 Catch::Matchers::WithinRel(read_uc.gamma(), 1e-3));

    // Check space group is preserved
    REQUIRE(original_crystal.space_group().symbol() ==
            rc.space_group().symbol());

    // Check number of atoms is preserved
    REQUIRE(original_crystal.asymmetric_unit().size() ==
            rc.asymmetric_unit().size());
  }

  SECTION("File write/read test") {
    auto crystal = acetic_crystal();

    std::string test_filename = "test_shelx_write.res";

    // Write to file
    occ::io::ShelxFile shelx_file;
    shelx_file.set_title("Test file write");
    shelx_file.set_wavelength(1.5406); // Different wavelength

    bool write_success =
        shelx_file.write_crystal_to_file(crystal, test_filename);
    REQUIRE(write_success);

    // Read back from file
    auto read_crystal = shelx_file.read_crystal_from_file(test_filename);
    REQUIRE(read_crystal.has_value());

    // Check that wavelength was written/read correctly
    // (This would need to be verified by looking at the actual file content
    // since wavelength isn't stored in the Crystal object)

    // Clean up
    std::remove(test_filename.c_str());
  }

  SECTION("Test filename detection") {
    REQUIRE(occ::io::ShelxFile::is_likely_shelx_filename("test.res"));
    REQUIRE(occ::io::ShelxFile::is_likely_shelx_filename("test.ins"));
    REQUIRE_FALSE(occ::io::ShelxFile::is_likely_shelx_filename("test.cif"));
    REQUIRE_FALSE(occ::io::ShelxFile::is_likely_shelx_filename("test.xyz"));
  }

  SECTION("SHELX format demonstration") {
    auto crystal = acetic_crystal();

    occ::io::ShelxFile shelx;
    shelx.set_title("Acetic acid demonstration");
    shelx.set_wavelength(1.54178); // Cu Kα

    std::string output = shelx.write_crystal_to_string(crystal);

    // Print the generated SHELX format for inspection
    INFO("Generated SHELX format:\n" << output);

    // Basic validation that all required sections are present
    REQUIRE(output.find("TITL Acetic acid demonstration") != std::string::npos);
    REQUIRE(output.find("CELL 1.541780") != std::string::npos);
    REQUIRE(output.find("LATT -1") !=
            std::string::npos); // Should be negative for non-centrosymmetric
    REQUIRE(output.find("SYMM") != std::string::npos);
    REQUIRE(output.find("SFAC C H O") != std::string::npos);
    REQUIRE(output.find("C1 1") != std::string::npos); // Carbon atoms
    REQUIRE(output.find("H1 2") != std::string::npos); // Hydrogen atoms
    REQUIRE(output.find("O1 3") != std::string::npos); // Oxygen atoms
    REQUIRE(output.find("END") != std::string::npos);

    // Test that round-trip works
    auto rebuilt = shelx.read_crystal_from_string(output);
    REQUIRE(rebuilt.has_value());
    REQUIRE(rebuilt->space_group().symbol() == crystal.space_group().symbol());
  }
  
  SECTION("Parse SHELX string with detailed checks") {
    const std::string acetic_acid_res = R"(TITL acetic_acid
CELL 0.71073 13.31 4.09 5.769 90 90 90
ZERR 4 0.001 0.001 0.001 0 0 0

LATT -1
SYMM 1/2-x,1/2+y,1/2+z
SYMM 1/2+x,1/2-y,z
SYMM -x,-y,1/2+z
SFAC C H O
UNIT 8 16 8
FVAR 1.00
C1      1  0.165100  0.285800  0.170900 1.0
C2      1  0.089400  0.376200  0.348100 1.0
H1      2  0.182000  0.051000 -0.116000 1.0
H2      2  0.128000  0.510000  0.491000 1.0
H3      2  0.033000  0.540000  0.279000 1.0
H4      2  0.053000  0.168000  0.421000 1.0
O1      3  0.128700  0.107500  0.000000 1.0
O2      3  0.252900  0.370300  0.176900 1.0
END
)";

    occ::io::ShelxFile shelx_file;
    auto crystal = shelx_file.read_crystal_from_string(acetic_acid_res);
    REQUIRE(crystal.has_value());

    const auto& c = crystal.value();
    const auto& uc = c.unit_cell();
    const auto& asym = c.asymmetric_unit();
    const auto& sg = c.space_group();

    // Check unit cell parameters
    REQUIRE_THAT(uc.a(), Catch::Matchers::WithinRel(13.31, 1e-3));
    REQUIRE_THAT(uc.b(), Catch::Matchers::WithinRel(4.09, 1e-3));
    REQUIRE_THAT(uc.c(), Catch::Matchers::WithinRel(5.769, 1e-3));
    
    // Check number and types of atoms
    REQUIRE(asym.size() == 8);
    REQUIRE(asym.atomic_numbers(0) == 6); // C1
    REQUIRE(asym.atomic_numbers(1) == 6); // C2
    REQUIRE(asym.atomic_numbers(2) == 1); // H1
    REQUIRE(asym.atomic_numbers(6) == 8); // O1
    REQUIRE(asym.atomic_numbers(7) == 8); // O2
    
    // Check space group detection
    INFO("Detected space group: " << sg.symbol());
    REQUIRE(sg.number() >= 1);
    REQUIRE(sg.number() <= 230);
  }
  
  SECTION("Parse R3c rhombohedral structure") {
    const std::string r3c_res = R"(TITL r3c
CELL 0.71073 34.4501 34.4501 11.2367 90 90 120
ZERR 18 0.0004 0.0004 0.0003 0.00 0.00 0.00
LATT -3
SYMM -y,x-y,z
SYMM -x+y,-x,z
SYMM -y,-x,1/2+z
SYMM -x+y,y,1/2+z
SYMM x,x-y,1/2+z
SFAC C H O
UNIT 594 432 72
FVAR 1.00
O1      3  0.048360  0.884950  0.153850  1.000000 0.050600
C1      1  0.103390  0.778460 -0.089900  1.000000 0.080900
H1A     2  0.124600  0.803400 -0.131700  1.000000 0.097000
END
)";

    occ::io::ShelxFile shelx_file;
    auto crystal = shelx_file.read_crystal_from_string(r3c_res);
    REQUIRE(crystal.has_value());

    const auto& c = crystal.value();
    const auto& uc = c.unit_cell();
    const auto& sg = c.space_group();

    // Check hexagonal unit cell
    REQUIRE_THAT(uc.a(), Catch::Matchers::WithinRel(34.4501, 1e-3));
    REQUIRE_THAT(uc.b(), Catch::Matchers::WithinRel(34.4501, 1e-3));
    REQUIRE_THAT(uc.c(), Catch::Matchers::WithinRel(11.2367, 1e-3));
    REQUIRE_THAT(uc.gamma(), Catch::Matchers::WithinRel(120.0 * M_PI / 180.0, 1e-3));

    // Check space group detection
    INFO("R3c detected space group: " << sg.symbol());
    INFO("R3c space group number: " << sg.number());
    REQUIRE(sg.number() > 1); // Should not be P1
  }
}


TEST_CASE("CIF Parser HM space group symbols", "[cif][spacegroup]") {
  SECTION("Parse HM symbol 'P C M 21' with correct setting") {
    const std::string test_cif = R"(data_test
_symmetry_space_group_name_H-M     'P C M 21        '
_symmetry_Int_Tables_number        26
_cell_length_a                     8.290000
_cell_length_b                     14.000000
_cell_length_c                     7.225000
_cell_angle_alpha                  90.000000
_cell_angle_beta                   90.000000
_cell_angle_gamma                  90.000000
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C 0.46040 0.41640 0.00000 1.0000
)";

    occ::io::CifParser parser;
    auto crystal = parser.parse_crystal_from_string(test_cif);
    
    REQUIRE(crystal.has_value());
    const auto& c = crystal.value();
    const auto& sg = c.space_group();
    
    // Check that the space group is correctly recognized with exact setting
    INFO("Actual space group number: " << sg.number());
    INFO("Actual space group symbol: " << sg.symbol());
    REQUIRE(sg.number() == 26);
    // Input 'P C M 21        ' should be parsed as 'P c m 21' (ba-c setting)
    REQUIRE(sg.symbol() == "P c m 21");
    
    // Verify that it's the correct setting by checking symmetry operations count
    // P c m 21 should have 4 symmetry operations
    REQUIRE(sg.symmetry_operations().size() == 4);
    
    // Verify unit cell parameters
    const auto& uc = c.unit_cell();
    REQUIRE_THAT(uc.a(), Catch::Matchers::WithinRel(8.29, 1e-3));
    REQUIRE_THAT(uc.b(), Catch::Matchers::WithinRel(14.0, 1e-3));
    REQUIRE_THAT(uc.c(), Catch::Matchers::WithinRel(7.225, 1e-3));
  }
  
  
  SECTION("Parse space group 26 different settings") {
    // Test that we get the correct setting for different orientations of space group 26
    struct TestCase {
      std::string hm_input;
      std::string expected_hm;
      std::string expected_hall;
    };
    
    std::vector<TestCase> test_cases = {
      {"'P C M 21'", "P c m 21", "P 2c -2c"},      // ba-c setting  
      {"'P M C 21'", "P m c 21", "P 2c -2"},       // standard setting
      {"'P 21 M A'", "P 21 m a", "P -2a 2a"},      // cab setting
      {"'P 21 A M'", "P 21 a m", "P -2 2a"},       // -cba setting
      {"'P B 21 M'", "P b 21 m", "P -2 -2b"},      // bca setting  
      {"'P M 21 B'", "P m 21 b", "P -2b -2"}       // a-cb setting
    };
    
    for (const auto& test_case : test_cases) {
      std::string test_cif = fmt::format(R"(data_test
_symmetry_space_group_name_H-M     {}
_cell_length_a                     8.290000
_cell_length_b                     14.000000
_cell_length_c                     7.225000
_cell_angle_alpha                  90.000000
_cell_angle_beta                   90.000000
_cell_angle_gamma                  90.000000
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C 0.46040 0.41640 0.00000 1.0000
)", test_case.hm_input);

      occ::io::CifParser parser;
      auto crystal = parser.parse_crystal_from_string(test_cif);
      
      INFO("Testing HM symbol: " << test_case.hm_input);
      REQUIRE(crystal.has_value());
      
      const auto& sg = crystal->space_group();
      REQUIRE(sg.number() == 26);
      REQUIRE(sg.symbol() == test_case.expected_hm);
      
      // All should have 4 symmetry operations
      REQUIRE(sg.symmetry_operations().size() == 4);
    }
  }
  
}

TEST_CASE("CIF writer", "[io][cif][write]") {
  SECTION("Write simple crystal to CIF") {
    // Create simple test crystal
    occ::crystal::AsymmetricUnit asym;
    asym.atomic_numbers.resize(4);
    asym.atomic_numbers << 6, 1, 1, 1; // CH3
    
    occ::crystal::UnitCell uc(10.0, 10.0, 10.0, 90.0, 90.0, 90.0);
    
    // Create positions in Cartesian then convert
    occ::Mat3N cart_coords(3, 4);
    cart_coords.col(0) = occ::Vec3(5.0, 5.0, 5.0); // C at center
    cart_coords.col(1) = occ::Vec3(6.0, 5.0, 5.0); // H1
    cart_coords.col(2) = occ::Vec3(4.5, 5.8, 5.0); // H2
    cart_coords.col(3) = occ::Vec3(4.5, 4.2, 5.0); // H3
    
    asym.positions = uc.to_fractional(cart_coords);
    asym.labels = {"C1", "H1", "H2", "H3"};
    
    occ::crystal::SpaceGroup sg(1);
    occ::crystal::Crystal crystal(asym, sg, uc);
    
    // Write to CIF
    occ::io::CifWriter writer;
    std::string cif_content = writer.to_string(crystal, "Test CH3");
    
    // Basic checks that CIF contains expected content
    REQUIRE(cif_content.find("_atom_site_label") != std::string::npos);
    REQUIRE(cif_content.find("_atom_site_fract_x") != std::string::npos);
    REQUIRE(cif_content.find("_atom_site_fract_y") != std::string::npos);
    REQUIRE(cif_content.find("_atom_site_fract_z") != std::string::npos);
    REQUIRE(cif_content.find("C1") != std::string::npos);
    REQUIRE(cif_content.find("H1") != std::string::npos);
    REQUIRE(cif_content.find("H2") != std::string::npos);
    REQUIRE(cif_content.find("H3") != std::string::npos);
  }
  
  SECTION("Write crystal with normalized hydrogens") {
    // Create test crystal with wrong H bond lengths
    occ::crystal::AsymmetricUnit asym;
    asym.atomic_numbers.resize(2);
    asym.atomic_numbers << 8, 1; // OH
    asym.positions.resize(3, 2);
    asym.positions.col(0) = occ::Vec3(0.5, 0.5, 0.5);
    asym.positions.col(1) = occ::Vec3(0.5, 0.55, 0.5); // Wrong O-H distance
    asym.labels = {"O1", "H1"};
    
    occ::crystal::UnitCell uc(8.0, 8.0, 12.0, occ::units::radians(90.0), 
                              occ::units::radians(90.0), occ::units::radians(120.0)); // Hexagonal
    occ::crystal::SpaceGroup sg(194); // P6_3/mmc
    occ::crystal::Crystal crystal(asym, sg, uc);
    
    // Normalize hydrogens first
    int normalized = crystal.normalize_hydrogen_bondlengths();
    REQUIRE(normalized == 1);
    
    // Write to CIF
    occ::io::CifWriter writer;
    std::string cif_content = writer.to_string(crystal, "Normalized_OH");
    
    // Check space group is preserved
    REQUIRE(cif_content.find("P 63/m m c") != std::string::npos);
    
    // Parse the CIF content to check unit cell parameters properly
    gemmi::cif::Document doc = gemmi::cif::read_string(cif_content);
    REQUIRE(doc.blocks.size() > 0);
    
    const auto& block = doc.blocks[0];
    
    // Check unit cell parameters with appropriate tolerances
    auto a_pair = block.find_pair("_cell_length_a");
    auto b_pair = block.find_pair("_cell_length_b");
    auto c_pair = block.find_pair("_cell_length_c");
    auto gamma_pair = block.find_pair("_cell_angle_gamma");
    
    REQUIRE(a_pair != nullptr);
    REQUIRE(b_pair != nullptr);
    REQUIRE(c_pair != nullptr);
    REQUIRE(gamma_pair != nullptr);
    
    CHECK_THAT(std::stod((*a_pair)[1]), Catch::Matchers::WithinAbs(8.0, 0.001));
    CHECK_THAT(std::stod((*b_pair)[1]), Catch::Matchers::WithinAbs(8.0, 0.001));
    CHECK_THAT(std::stod((*c_pair)[1]), Catch::Matchers::WithinAbs(12.0, 0.001));
    CHECK_THAT(std::stod((*gamma_pair)[1]), Catch::Matchers::WithinAbs(120.0, 0.001));
    
    // Check O-H bond length is normalized
    occ::Mat3N cart_pos = crystal.to_cartesian(crystal.asymmetric_unit().positions);
    occ::Vec3 oh_bond = cart_pos.col(1) - cart_pos.col(0);
    CHECK_THAT(oh_bond.norm(), Catch::Matchers::WithinAbs(0.983, 0.001));
  }
  
  SECTION("CIF writer file output") {
    // Test writing to actual file
    auto acetic = acetic_crystal();
    
    occ::io::CifWriter writer;
    std::string temp_file = "/tmp/test_acetic.cif";
    
    // This should not throw
    REQUIRE_NOTHROW(writer.write(temp_file, acetic, "Acetic acid test"));
    
    // File should exist and have content
    std::ifstream file(temp_file);
    REQUIRE(file.good());
    
    std::string line;
    bool found_atom_site = false;
    while (std::getline(file, line)) {
      if (line.find("_atom_site_label") != std::string::npos) {
        found_atom_site = true;
        break;
      }
    }
    REQUIRE(found_atom_site);
  }
}
