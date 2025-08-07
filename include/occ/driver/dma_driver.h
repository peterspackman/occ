#pragma once
#include <occ/core/log.h>
#include <occ/dma/dma.h>
#include <occ/qm/wavefunction.h>
#include <ankerl/unordered_dense.h>
#include <string>
#include <optional>

namespace occ::driver {

struct DMAConfig {
    // Input/output files
    std::string wavefunction_filename;
    std::string punch_filename{"dma.punch"};
    
    // Basic DMA settings
    occ::dma::DMASettings settings;
    
    // Atom-specific settings
    ankerl::unordered_dense::map<std::string, double> atom_radii;  // Element symbol -> radius in Angstrom
    ankerl::unordered_dense::map<std::string, int> atom_limits;    // Element symbol -> max rank
    
    // Molecular orientation options
    std::string axis_method{"none"};  // "none", "nc", "pca", "moi"
    std::vector<int> axis_atoms;      // atom indices for nc method (0-based)
    std::string oriented_xyz_filename; // output filename for oriented molecule
    std::string axis_filename;        // output filename for neighcrys axis file
    
    // Output options
    bool write_punch{true};
    bool write_oriented_xyz{false};
    bool write_axis_file{false};
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
    
private:
    DMAConfig m_config;
};

} // namespace occ::driver