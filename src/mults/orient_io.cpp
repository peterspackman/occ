#include <occ/mults/orient_io.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <fmt/core.h>

namespace occ::mults {

dma::Mult parse_orient_multipoles(int max_rank, const std::vector<std::string>& lines) {
    dma::Mult m(max_rank);
    std::vector<double> values;

    // Parse all floating point values from the lines
    for (const auto& line : lines) {
        std::istringstream iss(line);
        double val;
        while (iss >> val) {
            values.push_back(val);
        }
    }

    // Copy values in order (Orient format matches our internal order)
    int expected = m.num_components();
    if (values.size() != expected) {
        throw std::runtime_error(
            fmt::format("Expected {} multipole components for rank {}, got {}",
                       expected, max_rank, values.size()));
    }

    for (int i = 0; i < expected; i++) {
        m.q(i) = values[i];
    }

    return m;
}

std::vector<OrientSite> parse_orient_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open Orient file: {}", filename));
    }

    std::vector<OrientSite> sites;
    std::string line;
    Vec3 molecule_offset(0.0, 0.0, 0.0);  // Current molecule position offset

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '!') continue;

        // Check for "Molecule ... at X Y Z" directive
        if (line.find("Molecule") != std::string::npos && line.find(" at ") != std::string::npos) {
            std::istringstream mol_iss(line);
            std::string word;
            while (mol_iss >> word) {
                if (word == "at") {
                    double mx, my, mz;
                    mol_iss >> mx >> my >> mz;
                    molecule_offset = Vec3(mx, my, mz);
                    break;
                }
            }
            continue;
        }

        // Look for lines with "Rank" keyword (site definition)
        if (line.find("Rank") != std::string::npos) {
            OrientSite site;

            // Parse: "C  -0.10287   0.00000   0.00001      Rank 4  Type C"
            std::istringstream site_iss(line);
            double x, y, z;
            int rank;
            site_iss >> site.name >> x >> y >> z;

            // Apply molecule offset to site position
            site.position = Vec3(x, y, z) + molecule_offset;

            // Find "Rank" and read the number
            std::string word;
            while (site_iss >> word) {
                if (word == "Rank") {
                    site_iss >> rank;
                    break;
                }
            }

            // Read multipole lines
            std::vector<std::string> multipole_lines;
            int num_components = (rank + 1) * (rank + 1);
            int components_read = 0;

            while (components_read < num_components && std::getline(file, line)) {
                if (line.empty() || line[0] == '!') continue;

                // Count how many numbers are on this line
                std::istringstream count_iss(line);
                double val;
                int line_count = 0;
                while (count_iss >> val) {
                    line_count++;
                }

                if (line_count > 0) {
                    multipole_lines.push_back(line);
                    components_read += line_count;
                }
            }

            // Parse the multipoles
            site.multipole = parse_orient_multipoles(rank, multipole_lines);
            sites.push_back(site);
        }
    }

    return sites;
}

} // namespace occ::mults
