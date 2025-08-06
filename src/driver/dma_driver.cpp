#include <occ/driver/dma_driver.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecular_axis.h>
#include <fmt/core.h>
#include <sstream>
#include <fstream>

namespace occ::driver {

// Convert string axis method to enum for the new API
occ::core::AxisMethod string_to_axis_method(const std::string& method_str) {
    return occ::core::MolecularAxisCalculator::string_to_axis_method(method_str);
}

DMADriver::DMAOutput DMADriver::run() {
    if (m_config.wavefunction_filename.empty()) {
        throw std::runtime_error("Wavefunction filename not set");
    }
    
    occ::log::info("Loading wavefunction from: {}", m_config.wavefunction_filename);
    auto wfn = occ::qm::Wavefunction::load(m_config.wavefunction_filename);
    return run(wfn);
}

DMADriver::DMAOutput DMADriver::run(const occ::qm::Wavefunction& wfn) {
    // Create a copy of the wavefunction that we can modify
    occ::qm::Wavefunction oriented_wfn = wfn;
    
    // Apply molecular orientation if requested
    if (m_config.axis_method != "none") {
        occ::log::info("Applying molecular orientation using {} method", m_config.axis_method);
        
        // Create molecular axis calculator
        occ::core::MolecularAxisCalculator axis_calc(oriented_wfn);
        occ::core::AxisMethod method = string_to_axis_method(m_config.axis_method);
        
        // Calculate molecular axes
        occ::core::MolecularAxisResult axis_result;
        if (method == occ::core::AxisMethod::Neighcrys) {
            std::vector<int> atoms_to_use = m_config.axis_atoms;
            if (atoms_to_use.empty()) {
                // Default to first 3 atoms if no axis atoms specified
                if (oriented_wfn.positions().cols() < 3) {
                    throw std::runtime_error("Neighcrys axis method requires at least 3 atoms");
                }
                atoms_to_use = {0, 1, 2};
                occ::log::info("Using default axis atoms: 0, 1, 2");
            } else {
                occ::log::info("Using specified axis atoms: {}, {}, {}", 
                              atoms_to_use[0], atoms_to_use[1], atoms_to_use[2]);
            }
            axis_result = axis_calc.calculate_axes(method, atoms_to_use);
        } else {
            axis_result = axis_calc.calculate_axes(method);
            if (method == occ::core::AxisMethod::PCA) {
                occ::log::info("Using PCA-based molecular axes");
            } else if (method == occ::core::AxisMethod::MOI) {
                occ::log::info("Using moment of inertia-based molecular axes");
            }
        }
        
        // Apply transformation
        occ::core::MolecularAxisCalculator::apply_molecular_transformation(oriented_wfn, axis_result);
        
        occ::log::debug("Applied rotation matrix (det = {}):\n{}", 
                        axis_result.determinant, format_matrix(axis_result.axes));
        occ::log::debug("Applied translation: [{:.6f}, {:.6f}, {:.6f}]", 
                        -axis_result.center_of_mass.x(), -axis_result.center_of_mass.y(), -axis_result.center_of_mass.z());
                        
        // Write oriented XYZ file if requested
        if (m_config.write_oriented_xyz || !m_config.oriented_xyz_filename.empty()) {
            std::string xyz_filename = m_config.oriented_xyz_filename.empty() ? 
                                     "oriented.xyz" : m_config.oriented_xyz_filename;
            occ::core::MolecularAxisCalculator::write_oriented_xyz(xyz_filename, oriented_wfn);
            occ::log::info("Wrote oriented molecule to: {}", xyz_filename);
        }
        
        // Write neighcrys axis file if requested
        if (m_config.write_axis_file || !m_config.axis_filename.empty()) {
            std::string axis_filename = m_config.axis_filename.empty() ? 
                                      "molecule.mols" : m_config.axis_filename;
            
            // Use the axis atoms that were used for orientation, or default to first 3
            std::vector<int> axis_atoms_for_file = axis_result.axis_atoms;
            if (axis_atoms_for_file.empty() && oriented_wfn.positions().cols() >= 3) {
                axis_atoms_for_file = {0, 1, 2};
            }
            
            auto axis_info = axis_calc.generate_neighcrys_info(axis_atoms_for_file);
            occ::core::MolecularAxisCalculator::write_neighcrys_axis_file(axis_filename, axis_info);
            occ::log::info("Wrote neighcrys axis file to: {}", axis_filename);
        }
    }
    
    occ::dma::DMACalculator calc(oriented_wfn);
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
    
    // Write neighcrys axis file if requested but no orientation was applied
    if ((m_config.write_axis_file || !m_config.axis_filename.empty()) && m_config.axis_method == "none") {
        std::string axis_filename = m_config.axis_filename.empty() ? 
                                  "molecule.mols" : m_config.axis_filename;
        
        std::vector<int> axis_atoms_for_file = m_config.axis_atoms;
        if (axis_atoms_for_file.empty() && oriented_wfn.positions().cols() >= 3) {
            axis_atoms_for_file = {0, 1, 2};
        }
        
        occ::core::MolecularAxisCalculator axis_calc(oriented_wfn);
        auto axis_info = axis_calc.generate_neighcrys_info(axis_atoms_for_file);
        occ::core::MolecularAxisCalculator::write_neighcrys_axis_file(axis_filename, axis_info);
        occ::log::info("Wrote neighcrys axis file to: {}", axis_filename);
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