#include "logger.h"
#include "argparse.hpp"
#include "fchkreader.h"
#include <fmt/ostream.h>
#include "hf.h"
#include "wavefunction.h"
#include "element.h"
#include "util.h"
#include "gto.h"
#include "pairinteraction.h"
#include "disp.h"
#include "polarization.h"
#include <toml.hpp>

using tonto::qm::Wavefunction;
using tonto::io::FchkReader;
using tonto::interaction::CEModelInteraction;
using tonto::chem::Element;

constexpr double kjmol_per_hartree{2625.46};

namespace impl
{
struct rotation
{
    Eigen::Matrix3d mat{Eigen::Matrix3d::Identity(3,3)};
};

struct translation
{
    Eigen::Vector3d vec{Eigen::Vector3d::Zero(3)};
};

}

namespace toml {
template<>
struct from<impl::rotation>
{
    static impl::rotation from_toml(const value& v)
    {
        impl::rotation rot;
        auto arr = toml::get<toml::array>(v);
        if(arr.size() == 9) {
            for(size_t i = 0; i < 9; i++) {
                double v = arr[i].is_floating() ? arr[i].as_floating(std::nothrow) :
                                              static_cast<double>(arr[i].as_integer());
                rot.mat(i / 3, i % 3) = v;
            }
        }
        else if(arr.size() == 3)
        {
            for(size_t i = 0; i < 3; i++){
                auto row = toml::get<toml::array>(arr[i]);
                for(size_t j = 0; j < 3; j++)
                {
                    double v = row[j].is_floating() ? row[j].as_floating(std::nothrow) :
                                                  static_cast<double>(row[j].as_integer());
                    rot.mat(i, j) = v;
                }
            }
        }
        else {
            std::cerr << toml::format_error(
            "[error] 3D rotation matrix has invalid length",
            v, "Expecting a [3, 3] or [9] array of int or double")
            << std::endl;
        }
        return rot;
    }
};

template<>
struct from<impl::translation>
{
    static impl::translation from_toml(const value& v)
    {
        impl::translation trans;
        auto arr = toml::get<toml::array>(v);
        if(arr.size() == 3)
        {
            for(size_t i = 0; i < 3; i++) {
                double v = arr[i].is_floating() ? arr[i].as_floating(std::nothrow) :
                                              static_cast<double>(arr[i].as_integer());
                trans.vec(i) = v;
            }
        }
        else
        {
            std::cerr << toml::format_error(
            "[error] 3D translation vector has invalid length",
            v, "Expecting a [3] array of int or double")
            << std::endl;
        }
        return trans;
    }
};
}

int main(int argc, const char **argv) {
    const auto input = toml::parse((argc > 1) ? argv[1] : "ce.toml");
    const auto pair_interaction_table = toml::find(input, "interaction");
    const auto global_settings_table = toml::find(input, "global");

    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();

    using tonto::parallel::nthreads;
    nthreads = toml::find_or<int>(global_settings_table, "threads", 1);
    omp_set_num_threads(nthreads);

    const std::string model_name = toml::find_or<std::string>(pair_interaction_table, "model", "ce-b3lyp");
    const std::string fchk_filename_a = toml::find_or<std::string>(pair_interaction_table, "monomer_a", "a.fchk");
    const std::string fchk_filename_b = toml::find_or<std::string>(pair_interaction_table, "monomer_b", "b.fchk");

    tonto::Mat3 rotation_a = toml::find_or<impl::rotation>(pair_interaction_table, "rotation_a", impl::rotation{}).mat;
    tonto::Mat3 rotation_b = toml::find_or<impl::rotation>(pair_interaction_table, "rotation_b", impl::rotation{}).mat;
    fmt::print("Rotation of monomer A:\n{}\n", rotation_a);
    fmt::print("Rotation of monomer B:\n{}\n", rotation_b);

    tonto::Vec3 translation_a = toml::find_or<impl::translation>(pair_interaction_table, "translation_a", impl::translation{}).vec;
    tonto::Vec3 translation_b = toml::find_or<impl::translation>(pair_interaction_table, "translation_b", impl::translation{}).vec;
    fmt::print("Translation of monomer A:\n{}\n", translation_a);
    fmt::print("Translation of monomer B:\n{}\n", translation_b);

    FchkReader fchk_a(fchk_filename_a);
    tonto::log::info("Parsed fchk file: {}", fchk_filename_a);
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_a, "sym", "x", "y", "z");
    for (const auto &atom : fchk_a.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    FchkReader fchk_b(fchk_filename_b);
    tonto::log::info("Parsed fchk file: {}", fchk_filename_b);
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_b, "sym", "x", "y", "z");
    for (const auto &atom : fchk_b.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    Wavefunction A(fchk_a);
    tonto::log::info("Finished reading {}", fchk_filename_a);
    A.apply_transformation(rotation_a, translation_a);

    fmt::print("Geometry after transformation ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_a, "sym", "x", "y", "z");
    for (const auto &atom : A.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    Wavefunction B(fchk_b);
    B.apply_transformation(rotation_b, translation_b);
    tonto::log::info("Finished reading {}", fchk_filename_b);
    fmt::print("Geometry after transformation ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_b, "sym", "x", "y", "z");
    for (const auto &atom : B.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    CEModelInteraction interaction(tonto::interaction::CE_B3LYP_631Gdp);
    auto interaction_energy = interaction(A, B);
    fmt::print("Total\n");
    fmt::print("Coulomb             {: 12.6f}\n", interaction_energy.coulomb * kjmol_per_hartree);
    fmt::print("Exchange-repulsion  {: 12.6f}\n", interaction_energy.exchange_repulsion * kjmol_per_hartree);
    fmt::print("Polarization        {: 12.6f}\n", interaction_energy.polarization * kjmol_per_hartree);
    fmt::print("Dispersion          {: 12.6f}\n", interaction_energy.dispersion * kjmol_per_hartree);
    fmt::print("Scaled total        {: 12.6f}\n", interaction_energy.total * kjmol_per_hartree);
}
