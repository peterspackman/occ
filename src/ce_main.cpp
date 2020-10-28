#include "logger.h"
#include "argparse.hpp"
#include "fchkreader.h"
#include <fmt/ostream.h>

using tonto::io::FchkReader;

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("ce");
    parser.add_argument("fchk_a").help("Input file fchk A");
    parser.add_argument("fchk_b").help("Input file fchk B");

    parser.add_argument("--model")
            .default_value(std::string("ce-b3lyp"));

    tonto::log::set_level(tonto::log::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }
    const std::string fchk_filename_a = parser.get<std::string>("fchk_a");
    const std::string fchk_filename_b = parser.get<std::string>("fchk_b");
    FchkReader fchk_a(fchk_filename_a);
    tonto::log::info("Opened fchk file: {}", fchk_filename_a);
    FchkReader fchk_b(fchk_filename_b);
    tonto::log::info("Opened fchk file: {}", fchk_filename_b);
}
