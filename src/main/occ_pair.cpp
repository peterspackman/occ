#include <occ/core/element.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/3rdparty/argparse.hpp>
#include <occ/io/fchkreader.h>
#include <occ/io/moldenreader.h>
#include <fmt/ostream.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/polarization.h>
#include <toml.hpp>
#include <filesystem>

using occ::qm::Wavefunction;
using occ::interaction::CEModelInteraction;

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

Wavefunction load_wavefunction(const std::string& filename)
{
    namespace fs = std::filesystem;
    using occ::util::to_lower;
    std::string ext = fs::path(filename).extension();
    to_lower(ext);
    if(ext == ".fchk")
    {
        using occ::io::FchkReader;
        FchkReader fchk(filename);
        return Wavefunction(fchk);
    }
    if(ext == ".molden")
    {
        using occ::io::MoldenReader;
        MoldenReader molden(filename);
        return Wavefunction(molden);
    }
    throw std::runtime_error("Unknown file extension when reading wavefunction: " + ext);
}

int main(int argc, const char **argv) {
    const auto input = toml::parse((argc > 1) ? argv[1] : "ce.toml");
    const auto pair_interaction_table = toml::find(input, "interaction");
    const auto global_settings_table = toml::find(input, "global");
    occ::timing::start(occ::timing::category::global);
    libint2::Shell::do_enforce_unit_normalization(true);
    libint2::initialize();

    using occ::parallel::nthreads;
    nthreads = toml::find_or<int>(global_settings_table, "threads", 1);

    const std::string model_name = toml::find_or<std::string>(pair_interaction_table, "model", "ce-b3lyp");
    const std::string filename_a = toml::find_or<std::string>(pair_interaction_table, "monomer_a", "a.fchk");
    const std::string filename_b = toml::find_or<std::string>(pair_interaction_table, "monomer_b", "b.fchk");
    occ::Mat3 rotation_a = toml::find_or<impl::rotation>(pair_interaction_table, "rotation_a", impl::rotation{}).mat;
    occ::Mat3 rotation_b = toml::find_or<impl::rotation>(pair_interaction_table, "rotation_b", impl::rotation{}).mat;
    occ::Vec3 translation_a = toml::find_or<impl::translation>(pair_interaction_table, "translation_a", impl::translation{}).vec;
    occ::Vec3 translation_b = toml::find_or<impl::translation>(pair_interaction_table, "translation_b", impl::translation{}).vec;


    Wavefunction A = load_wavefunction(filename_a);
    A.apply_transformation(rotation_a, translation_a);
    Wavefunction B = load_wavefunction(filename_b);
    B.apply_transformation(rotation_b, translation_b);
    auto model = occ::interaction::ce_model_from_string(model_name);
    CEModelInteraction interaction(model);
    auto interaction_energy = interaction(A, B);
    occ::timing::stop(occ::timing::category::global);

    fmt::print("Component              Energy (kJ/mol)\n\n");
    fmt::print("Coulomb               {: 12.6f}\n", interaction_energy.coulomb * kjmol_per_hartree);
    fmt::print("Exchange-repulsion    {: 12.6f}\n", interaction_energy.exchange_repulsion * kjmol_per_hartree);
    fmt::print("Polarization          {: 12.6f}\n", interaction_energy.polarization * kjmol_per_hartree);
    fmt::print("Dispersion            {: 12.6f}\n", interaction_energy.dispersion * kjmol_per_hartree);
    fmt::print("__________________________________\n");
    fmt::print("Total {:^8s}        {: 12.6f}\n", model_name, interaction_energy.total * kjmol_per_hartree);

    fmt::print("\n");
    occ::timing::print_timings();
}