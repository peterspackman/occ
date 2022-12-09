#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/xtb/xtb_wrapper.h>

int main(int argc, char *argv[]) {

    occ::log::set_level(occ::log::level::debug);
    occ::xtb::print_tblite_version();
    occ::xtb::single_point_h2o();
    return 0;
}
