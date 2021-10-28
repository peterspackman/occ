#include "catch.hpp"
#include <occ/io/qcschema.h>
#include <sstream>
#include <fmt/ostream.h>

const char * json_input_contents = R""""({
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


TEST_CASE("water qcschema", "[read]")
{
    std::istringstream json(json_input_contents);
    occ::io::QCSchemaReader reader(json);
}
