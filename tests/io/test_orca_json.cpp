#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/io/orca_json.h>
#include <occ/qm/hf.h>
#include <sstream>

using occ::Mat;

const char *json_contents = R"(
{
    "Molecule": {
        "Atoms": [
            {
                "BasisFunctions": [
                    {
                        "Coefficients": [
                            0.1543289707029839,
                            0.5353281424384732,
                            0.44463454202535485
                        ],
                        "Exponents": [
                            3.42525091,
                            0.62391373,
                            0.1688554
                        ],
                        "Shell": "s"
                    }
                ],
                "Coords": [
                    0.0,
                    0.0,
                    0.0
                ],
                "ElementLabel": "H",
                "ElementNumber": 1,
                "Idx": 0,
                "NuclearCharge": 1.0
            },
            {
                "BasisFunctions": [
                    {
                        "Coefficients": [
                            0.1543289707029839,
                            0.5353281424384732,
                            0.44463454202535485
                        ],
                        "Exponents": [
                            3.42525091,
                            0.62391373,
                            0.1688554
                        ],
                        "Shell": "s"
                    }
                ],
                "Coords": [
                    0.0,
                    0.0,
                    2.6456165874897524
                ],
                "ElementLabel": "H",
                "ElementNumber": 1,
                "Idx": 1,
                "NuclearCharge": 1.0
            }
        ],
        "BaseName": "h2",
        "Charge": 0,
        "CoordinateUnits": "Bohrs",
        "H-Matrix": [
            [
                -0.8422289108173067,
                -0.3785030189433606
            ],
            [
                -0.3785030189433606,
                -0.8422289108173067
            ]
        ],
        "HFTyp": "RHF",
        "MolecularOrbitals": {
            "EnergyUnit": "Eh",
            "MOs": [
                {
                    "MOCoefficients": [
                        0.621202118970972,
                        0.621202118970972
                    ],
                    "Occupancy": 2.0,
                    "OrbitalEnergy": -0.3773228237944341
                },
                {
                    "MOCoefficients": [
                        -0.8425697665943784,
                        0.8425697665943784
                    ],
                    "Occupancy": 0.0,
                    "OrbitalEnergy": 0.25890197203084375
                }
            ],
            "OrbitalLabels": [
                "0H   1s",
                "1H   1s"
            ]
        },
        "Multiplicity": 1,
        "S-Matrix": [
            [
                1.0,
                0.29569907102006404
            ],
            [
                0.29569907102006404,
                1.0
            ]
        ],
        "T-Matrix": [
            [
                0.7600318835666087,
                0.019745102188373418
            ],
            [
                0.019745102188373418,
                0.7600318835666087
            ]
        ]
    },
    "ORCA Header": {
        "Version": "5.0 - current"
    }
}
)";

TEST_CASE("H2 orca json", "[read]") {

    std::istringstream json_istream(json_contents);
    occ::io::OrcaJSONReader reader(json_istream);
    fmt::print("Atomic numbers:\n{}\n", reader.atomic_numbers());
    fmt::print("Atomic positions:\n{}\n", reader.atom_positions());
    std::vector<occ::core::Atom> atoms = reader.atoms();
    Mat S1 = reader.overlap_matrix();
    fmt::print("ORCA Overlap matrix:\n{}\n", S1);

    occ::hf::HartreeFock hf(reader.basis_set());
    Mat S2 = hf.compute_overlap_matrix();
    fmt::print("OUR Overlap matrix:\n{}\n", S2);
    fmt::print("Difference\n{}\n", S2 - S1);
    REQUIRE(occ::util::all_close(S1, S2));
}
