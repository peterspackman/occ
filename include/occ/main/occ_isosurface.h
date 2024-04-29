#pragma once
#include <CLI/App.hpp>
#include <vector>

namespace occ::main {

struct IsosurfaceConfig {
    enum class Surface {
	PromoleculeDensity,
	Hirshfeld,
	EEQ_ESP,
	ElectronDensity,
	ESP,
	SpinDensity,
	DeformationDensity,
	Orbital,
	CrystalVoid
    };

    enum class Property {
	Dnorm,
	Dint_norm,
	Dext_norm,
	Dint,
	Dext,
	FragmentPatch,
	ShapeIndex,
	Curvedness,
	EEQ_ESP,
	PromoleculeDensity,
	ESP,
	ElectronDensity,
	SpinDensity,
	DeformationDensity,
	Orbital
    };

    std::string geometry_filename{""};
    std::string environment_filename{""};
    size_t max_depth{4};
    double separation{0.2};
    double isovalue{0.02};
    double background_density{0.0};
    bool use_hashed_mc{false};
    std::string wavefunction_filename{""};
    std::vector<double> wfn_rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> wfn_translation{0.0, 0.0, 0.0};
    int orbital_index{0};
    bool binary_output{true};
    std::string kind{"promolecule_density"};
    std::string output_filename{"surface.ply"};
    std::vector<std::string> additional_properties{};

    std::vector<Property> surface_properties() const;
    Surface surface_type() const;

    bool requires_crystal() const;
    bool requires_environment() const;
    bool requires_wavefunction() const;
    bool have_environment_file() const;
};

CLI::App *add_isosurface_subcommand(CLI::App &app);
void run_isosurface_subcommand(IsosurfaceConfig const &);
} // namespace occ::main
