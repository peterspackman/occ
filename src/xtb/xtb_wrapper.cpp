#include <cstdint>
#include <cstdlib>
#include <fmt/core.h>
#include <occ/core/molecule.h>
#include <occ/xtb/xtb_wrapper.h>

extern "C" {
#include "tblite/calculator.h"
#include "tblite/error.h"
#include "tblite/structure.h"
#include "tblite/version.h"
}

namespace occ::xtb {

void print_tblite_version() {
    auto v = tblite_get_version();
    fmt::print("TBLITE VERSION: {}\n", v);
}

void single_point_h2o() {
    int natoms = 3;
    Mat3N positions(3, natoms);
    IVec nums(natoms);
    nums << 8, 1, 1;
    positions << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
        0.07959806, 0.01878821, -0.02171049, 0.00986248;

    tblite_error error = nullptr;
    tblite_context ctx = nullptr;
    tblite_calculator calc = nullptr;
    tblite_result result = nullptr;

    error = tblite_new_error();
    ctx = tblite_new_context();
    result = tblite_new_result();

    tblite_structure structure =
        tblite_new_structure(error, natoms, nums.data(), positions.data(),
                             nullptr, nullptr, nullptr, nullptr);
    calc = tblite_new_gfn2_calculator(ctx, structure);
    tblite_set_calculator_accuracy(ctx, calc, 2.0);
    tblite_set_calculator_max_iter(ctx, calc, 50);
    tblite_set_calculator_mixer_damping(ctx, calc, 0.2);
    tblite_set_calculator_temperature(ctx, calc, 0.0);

    double energy;
    tblite_get_singlepoint(ctx, structure, calc, result);
    tblite_get_result_energy(error, result, &energy);

    fmt::print("Singlepoint energy: {}\n", energy);

    tblite_delete_error(&error);
    tblite_delete_structure(&structure);
}

} // namespace occ::xtb
