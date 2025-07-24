#include <occ/driver/dma_driver.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <fmt/core.h>
#include <sstream>
#include <fstream>

namespace occ::driver {

DMADriver::DMAOutput DMADriver::run() {
    if (m_config.wavefunction_filename.empty()) {
        throw std::runtime_error("Wavefunction filename not set");
    }
    
    occ::log::info("Loading wavefunction from: {}", m_config.wavefunction_filename);
    auto wfn = occ::qm::Wavefunction::load(m_config.wavefunction_filename);
    return run(wfn);
}

DMADriver::DMAOutput DMADriver::run(const occ::qm::Wavefunction& wfn) {
    occ::dma::DMACalculator calc(wfn);
    calc.update_settings(m_config.settings);
    
    // Apply atom-specific settings
    for (const auto& [element, radius] : m_config.atom_radii) {
        int atomic_number = occ::core::Element(element).atomic_number();
        calc.set_radius_for_element(atomic_number, radius);
        occ::log::debug("Setting radius for {} to {:.3f} Angstrom", element, radius);
    }
    
    for (const auto& [element, limit] : m_config.atom_limits) {
        int atomic_number = occ::core::Element(element).atomic_number();
        calc.set_limit_for_element(atomic_number, limit);
        occ::log::debug("Setting max rank for {} to {}", element, limit);
    }
    
    // Set default H settings if not specified
    if (m_config.atom_radii.find("H") == m_config.atom_radii.end()) {
        calc.set_radius_for_element(1, 0.35);
    }
    if (m_config.atom_limits.find("H") == m_config.atom_limits.end()) {
        calc.set_limit_for_element(1, 1);
    }
    
    occ::log::debug("Running DMA calculation with max_rank={}, big_exponent={}", 
                    m_config.settings.max_rank, m_config.settings.big_exponent);
    
    auto result = calc.compute_multipoles();
    DMAOutput output{result, calc.sites()};
    
    if (m_config.write_punch && !m_config.punch_filename.empty()) {
        write_punch_file(m_config.punch_filename, output.result, output.sites);
        occ::log::info("Wrote punch file to: {}", m_config.punch_filename);
    }
    
    return output;
}

std::string DMADriver::generate_punch_file(const occ::dma::DMAResult& result,
                                          const occ::dma::DMASites& sites) {
    std::stringstream ss;
    
    ss << fmt::format("! Distributed multipoles from occ dma\n");
    ss << fmt::format("! Max rank: {}\n", result.max_rank);
    ss << fmt::format("\n");
    ss << fmt::format("Units angstrom\n\n");
    
    // Write individual site multipoles
    for (int i = 0; i < result.multipoles.size(); i++) {
        const auto& m = result.multipoles[i];
        const auto pos = sites.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;
        
        ss << fmt::format("{:<8s} {:12.8f} {:12.8f} {:12.8f}\n",
                         sites.name[i], pos.x(), pos.y(), pos.z());
        ss << fmt::format("Rank {}\n", m.max_rank);
        
        // Write multipoles in order: Q00, Q10, Q11c, Q11s, Q20, Q21c, Q21s, Q22c, Q22s, etc.
        int idx = 0;
        for (int rank = 0; rank <= m.max_rank; rank++) {
            int num_components = 2 * rank + 1;
            for (int comp = 0; comp < num_components; comp++) {
                ss << fmt::format(" {:16.10f}", m.q(idx++));
                if ((comp + 1) % 3 == 0 || comp == num_components - 1) {
                    ss << "\n";
                }
            }
        }
        ss << "\n";
    }
    
    return ss.str();
}

void DMADriver::write_punch_file(const std::string& filename,
                                const occ::dma::DMAResult& result,
                                const occ::dma::DMASites& sites) {
    std::ofstream punch(filename);
    if (!punch.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open punch file: {}", filename));
    }
    
    punch << generate_punch_file(result, sites);
    punch.close();
}

} // namespace occ::driver