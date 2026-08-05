#pragma once
#include <map>
#include <occ/dma/dma.h>
#include <occ/qm/wavefunction.h>
#include <string>

namespace occ::driver {

struct DMAConfig {
    // Input/output files
    std::string wavefunction_filename;
    std::string punch_filename{"dma.punch"};
    std::string json_filename; // structured results; empty = don't write

    // Basic DMA settings
    occ::dma::DMASettings settings;

    // Per-element overrides. Keys are element symbols in canonical case ("Cl").
    // Radii are in Angstrom; limits are ranks, clamped to settings.max_rank.
    std::map<std::string, double> atom_radii;
    std::map<std::string, int> atom_limits;

    // Rigid-body transform applied to the wavefunction before any analysis,
    // for reusing a wavefunction computed on a symmetry-equivalent copy.
    // x -> rotation * x + translation. Same convention as occ isosurface.
    occ::Mat3 wfn_rotation{occ::Mat3::Identity()};
    occ::Vec3 wfn_translation{occ::Vec3::Zero()}; // Angstrom

    // Molecular orientation options
    std::string axis_method{"none"};  // "none", "nc", "pca", "moi"
    std::vector<int> axis_atoms;      // atom indices for nc method (0-based)
    std::string oriented_xyz_filename; // output filename for oriented molecule
    std::string axis_filename;        // output filename for neighcrys axis file

    // Output options
    bool write_punch{true};
    bool write_oriented_xyz{false};
    bool write_axis_file{false};
    std::string csp_input_filename;  // write force-field JSON for CSP programs
    // Short-range parameter set written into that JSON. Resolved by
    // occ::mults::short_range_model_from_string(), see
    // occ/mults/dimer_interaction.h.
    std::string csp_force_field{"w99"};
};

class DMADriver {
public:
    DMADriver() = default;
    explicit DMADriver(const DMAConfig& config) : m_config(config) {}

    void set_config(const DMAConfig& config) { m_config = config; }
    const DMAConfig& config() const { return m_config; }

    struct DMAOutput {
        occ::dma::DMAResult result;
        occ::dma::DMASites sites;

        /// Total multipoles about the origin of the analysis frame. Computed
        /// with the same sites as `result`, so it is consistent with it -- do
        /// not recompute this from a separately loaded wavefunction.
        occ::dma::Mult total;

        /// Composite transform taking the input wavefunction frame to the
        /// frame everything above is expressed in: x_analysis = rotation *
        /// x_input + translation. Translation is in Bohr.
        occ::Mat3 rotation{occ::Mat3::Identity()};
        occ::Vec3 translation{occ::Vec3::Zero()};

        /// Axis method that actually ran, which is not always the one asked
        /// for (--write-csp-input promotes "none" to "moi").
        std::string axis_method{"none"};
        std::vector<int> axis_atoms;
    };

    // Main driver function that loads wavefunction and performs DMA
    DMAOutput run();

    // Alternative: run with already loaded wavefunction
    DMAOutput run(const occ::qm::Wavefunction& wfn);

    // Generate punch file content as string
    static std::string generate_punch_file(const occ::dma::DMAResult& result,
                                          const occ::dma::DMASites& sites);

    // Write punch file output (convenience wrapper)
    static void write_punch_file(const std::string& filename,
                                const occ::dma::DMAResult& result,
                                const occ::dma::DMASites& sites);

    /// Structured results: settings actually used, the frame they are in, and
    /// per-site effective radii/limits. Serialized as JSON text.
    static std::string generate_json(const DMAConfig &config,
                                     const DMAOutput &output);

    static void write_json(const std::string &filename,
                           const DMAConfig &config, const DMAOutput &output);

private:
    DMAConfig m_config;
};

} // namespace occ::driver
