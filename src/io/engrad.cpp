#include <fstream>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/io/engrad.h>
#include <scn/scn.h>

namespace occ::io {

EngradReader::EngradReader(const std::string &filename) {
    occ::timing::start(occ::timing::category::io);
    std::ifstream file(filename);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

EngradReader::EngradReader(std::istream &file) {
    occ::timing::start(occ::timing::category::io);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

void EngradReader::parse(std::istream &stream) {
    constexpr int expected_max_line_length{1024};
    std::string current_line;
    current_line.reserve(expected_max_line_length);

    auto skip_comment_lines = [&]() {
        int num_skipped = 0;
        do {
            std::getline(stream, current_line);
            num_skipped++;
        } while (current_line[0] == '#');
        occ::log::trace("Skipped {} comment lines in engrad file",
                        num_skipped - 1);
    };

    skip_comment_lines();

    auto scan_result = scn::scan_default(current_line, m_num_atoms);

    m_gradient = Mat3N(3, m_num_atoms);
    m_positions = Mat3N(3, m_num_atoms);
    m_atomic_numbers = IVec(m_num_atoms);

    skip_comment_lines();
    scan_result = scn::scan_default(current_line, m_energy);

    skip_comment_lines();
    for (int i = 0; i < m_num_atoms * 3; i++) {
        int component = i % 3;
        int atom = i / 3;
        scan_result =
            scn::scan_default(current_line, m_gradient(component, atom));
        std::getline(stream, current_line);
    }

    skip_comment_lines();
    for (int i = 0; i < m_num_atoms; i++) {
        scan_result = scn::scan_default(current_line, m_atomic_numbers(i),
                                        m_positions(0, i), m_positions(1, i),
                                        m_positions(2, i));
        std::getline(stream, current_line);
    }
}

} // namespace occ::io
