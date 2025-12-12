#include <occ/io/cifwriter.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <gemmi/to_cif.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace occ::io {

void CifWriter::write(const std::string& filename, 
                      const occ::crystal::Crystal& crystal,
                      const std::string& title) {
    std::string cif_content = to_string(crystal, title);
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    file << cif_content;
}

std::string CifWriter::to_string(const occ::crystal::Crystal& crystal,
                                 const std::string& title) {
    gemmi::cif::Document doc = crystal_to_cif_document(crystal, title);
    
    std::ostringstream oss;
    gemmi::cif::write_cif_to_stream(oss, doc);
    return oss.str();
}

gemmi::cif::Document CifWriter::crystal_to_cif_document(const occ::crystal::Crystal& crystal,
                                                        const std::string& title) const {
    gemmi::cif::Document doc;
    
    // Create block name
    std::string block_name = title;
    if (block_name.empty()) {
        block_name = crystal.asymmetric_unit().chemical_formula();
    }
    
    auto& block = doc.add_new_block(block_name);
    
    // Add unit cell parameters
    const auto& uc = crystal.unit_cell();
    block.set_pair("_cell_length_a", std::to_string(uc.a()));
    block.set_pair("_cell_length_b", std::to_string(uc.b()));
    block.set_pair("_cell_length_c", std::to_string(uc.c()));
    block.set_pair("_cell_angle_alpha", std::to_string(occ::units::degrees(uc.alpha())));
    block.set_pair("_cell_angle_beta", std::to_string(occ::units::degrees(uc.beta())));
    block.set_pair("_cell_angle_gamma", std::to_string(occ::units::degrees(uc.gamma())));
    block.set_pair("_cell_volume", std::to_string(crystal.volume()));
    
    // Add space group
    const auto& sg = crystal.space_group();
    block.set_pair("_space_group_name_H-M_alt", "'" + sg.symbol() + "'");
    block.set_pair("_space_group_IT_number", std::to_string(sg.number()));
    
    // Add atom site loop
    const auto& asym = crystal.asymmetric_unit();
    const auto& positions = asym.positions;
    const auto& atomic_numbers = asym.atomic_numbers;
    const auto& labels = asym.labels;
    
    // Create loop item
    block.items.emplace_back(gemmi::cif::LoopArg{});
    auto& loop_item = block.items.back();
    auto& loop = loop_item.loop;
    
    // Set up loop tags
    loop.tags = {
        "_atom_site_label",
        "_atom_site_type_symbol", 
        "_atom_site_fract_x",
        "_atom_site_fract_y", 
        "_atom_site_fract_z",
        "_atom_site_occupancy"
    };
    
    for (int i = 0; i < asym.size(); ++i) {
        // Generate atom label
        std::string atom_label;
        if (i < labels.size() && !labels[i].empty()) {
            atom_label = labels[i];
        } else {
            occ::core::Element element(atomic_numbers(i));
            atom_label = element.symbol() + std::to_string(i + 1);
        }
        
        // Get element symbol
        occ::core::Element element(atomic_numbers(i));
        std::string element_symbol = element.symbol();
        
        // Format coordinates with precision
        std::ostringstream x_ss, y_ss, z_ss;
        x_ss << std::fixed << std::setprecision(m_precision) << positions(0, i);
        y_ss << std::fixed << std::setprecision(m_precision) << positions(1, i);
        z_ss << std::fixed << std::setprecision(m_precision) << positions(2, i);
        
        // Get occupancy
        double occupancy = (i < asym.occupations.size()) ? asym.occupations(i) : 1.0;
        std::ostringstream occ_ss;
        occ_ss << std::fixed << std::setprecision(4) << occupancy;
        
        // Add row to loop
        loop.values.insert(loop.values.end(), {
            atom_label,
            element_symbol,
            x_ss.str(),
            y_ss.str(),
            z_ss.str(),
            occ_ss.str()
        });
    }
    
    return doc;
}

} // namespace occ::io