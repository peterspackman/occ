#include <tonto/crystal/crystal.h>
#include <tonto/io/cifparser.h>
#include <tonto/3rdparty/argparse.hpp>
#include <tonto/core/logger.h>

using tonto::crystal::Crystal;
using tonto::crystal::SymmetryOperation;

SymmetryOperation dimer_symop(const tonto::chem::Dimer &dimer, const Crystal &crystal)
{
    const auto& a = dimer.a();
    const auto& b = dimer.b();

    int sa_int = a.asymmetric_unit_symop()(0);
    int sb_int = b.asymmetric_unit_symop()(0);

    SymmetryOperation symop_a(sa_int);
    SymmetryOperation symop_b(sb_int);

    auto symop_ab = symop_b * symop_a.inverted();
    tonto::Vec3 c_a = symop_ab(crystal.to_fractional(a.positions())).rowwise().mean();
    tonto::Vec3 v_ab = crystal.to_fractional(b.centroid()) - c_a;

    symop_ab = symop_ab.translated(v_ab);
    return symop_ab;
}

Crystal read_crystal(const std::string &filename)
{
    tonto::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("interactions");
    parser.add_argument("input").help("Input CIF");
    parser.add_argument("-j", "--threads")
            .help("Number of threads")
            .default_value(2)
            .action([](const std::string& value) { return std::stoi(value); });
    tonto::log::set_level(tonto::log::level::debug);
    spdlog::set_level(spdlog::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }


    const std::string error_format = "Exception:\n    {}\nTerminating program.\n";
    try {
        std::string filename = parser.get<std::string>("input");
        Crystal c = read_crystal(filename);
        fmt::print("Loaded crystal from {}\n", filename);
        auto crystal_dimers = c.symmetry_unique_dimers(3.8);
        const auto &dimers = crystal_dimers.unique_dimers;
        fmt::print("Dimers\n");
        for(const auto& dimer: dimers)
        {
            auto s_ab = dimer_symop(dimer, c);
            fmt::print("R = {:.3f}, symop = {}\n", dimer.nearest_distance(), s_ab.to_string());
        }

        const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
        for(size_t i = 0; i < mol_neighbors.size(); i++)
        {
            const auto& n = mol_neighbors[i];
            fmt::print("Neighbors for molecule {}\n", i);
            size_t j = 0;
            for(const auto& dimer: n)
            {
                auto s_ab = dimer_symop(dimer, c);
                fmt::print("R = {:.3f}, symop = {}, unique_idx = {}\n",
                           dimer.nearest_distance(), s_ab.to_string(),
                           crystal_dimers.unique_dimer_idx[i][j]);
                j++;
            }
        }

     } catch (const char *ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::string &ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::exception &ex) {
        fmt::print(error_format, ex.what());
        return 1;
    } catch (...) {
        fmt::print("Exception:\n- Unknown...\n");
        return 1;
    }
   
    return 0;
}
