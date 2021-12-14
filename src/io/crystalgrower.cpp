#include <fmt/ostream.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/io/crystalgrower.h>

namespace occ::io::crystalgrower {

StructureWriter::StructureWriter(const std::string &filename)
    : m_owned_destination(filename), m_dest(m_owned_destination) {}

StructureWriter::StructureWriter(std::ostream &stream) : m_dest(stream) {}

void StructureWriter::write(const occ::crystal::Crystal &crystal,
                            const occ::crystal::CrystalDimers &uc_dimers) {
    using occ::units::degrees;
    fmt::print(m_dest, "{}\n\n", "title");
    const auto &uc_molecules = crystal.unit_cell_molecules();
    fmt::print(m_dest, "{}\n\n", uc_molecules.size());
    const auto &neighbors = uc_dimers.molecule_neighbors;
    size_t uc_idx = 0;
    for (const auto &mol : uc_molecules) {
        size_t num_neighbors = neighbors[uc_idx].size();
        fmt::print(m_dest, "{} {} (1,0) {}\n\n", mol.name(),
                   mol.unit_cell_molecule_idx() + 1, num_neighbors);
        for (const auto &n : neighbors[uc_idx]) {
            const auto uc_shift = n.b().cell_shift();
            const auto uc_idx = n.b().unit_cell_molecule_idx() + 1;
            fmt::print(m_dest, "{}({},{},{}) ", uc_idx, uc_shift[0], uc_shift[1],
                       uc_shift[2]);
        }
        fmt::print(m_dest, "\n\nX 0[0]\n\n");

        uc_idx++;
    }

    const auto &uc = crystal.unit_cell();
    double a = uc.a(), b = uc.b(), c = uc.c();
    double alpha = degrees(uc.alpha()), beta = degrees(uc.beta()),
           gamma = degrees(uc.gamma());
    fmt::print(m_dest, "{:8.4f} {:8.4f} {:8.4f} P\n", a, b, c);
    fmt::print(m_dest, "{:7.3f} {:7.3f} {:7.3f}\n", alpha, beta, gamma);

    fmt::print(m_dest, "\nNon primitive data\n");
    fmt::print(m_dest, "{:8.4f} {:8.4f} {:8.4f}\n", a, b, c);
    fmt::print(m_dest, "{:7.3f} {:7.3f} {:7.3f}\n", alpha, beta, gamma);
    fmt::print(m_dest, "\n");

    for (const auto &mol : uc_molecules) {
        const auto &bonds = mol.bonds();
        const auto &elements = mol.elements();
        const auto &pos = crystal.to_fractional(mol.positions());
        fmt::print(m_dest, "{} {} {}/{}\n", mol.name(),
                   mol.unit_cell_molecule_idx() + 1, mol.size(),
                   bonds.size() / 2);
        for (size_t i = 0; i < mol.size(); i++) {
            fmt::print(m_dest, "{} {:s} {: 8.5f} {: 8.5f} {: 8.5f}\n", i + 1,
                       elements[i].symbol(), pos(0, i), pos(1, i), pos(2, i));
        }
        fmt::print(m_dest, "\n");
        for (const auto &b : bonds) {
            if (b.first < b.second)
                fmt::print(m_dest, " {} {}\n", b.first + 1, b.second + 1);
        }
        fmt::print(m_dest, "\n");
    }
}

NetWriter::NetWriter(const std::string &filename)
    : m_owned_destination(filename), m_dest(m_owned_destination) {}

NetWriter::NetWriter(std::ostream &stream) : m_dest(stream) {}

void NetWriter::write(const occ::crystal::Crystal &crystal,
                      const occ::crystal::CrystalDimers &uc_dimers) {
    const auto &uc_molecules = crystal.unit_cell_molecules();
    const auto &neighbors = uc_dimers.molecule_neighbors;
    size_t uc_idx = 0;
    constexpr double max_de = 1e-4;

    for (const auto &mol : uc_molecules) {
        std::vector<double> unique_interaction_energies;

        for (const auto &n : neighbors[uc_idx]) {
            const auto uc_shift = n.b().cell_shift();
            const auto uc_idx = n.b().unit_cell_molecule_idx() + 1;
            const double e_int = n.interaction_energy();

            auto match = std::find_if(
                unique_interaction_energies.begin(),
                unique_interaction_energies.end(),
                [&e_int, &max_de](double x){ return std::abs(x - e_int) < max_de; }
            );
            size_t interaction_idx = 1;

            if(match == std::end(unique_interaction_energies)) {
                unique_interaction_energies.push_back(e_int);
                interaction_idx = unique_interaction_energies.size();
            }
            else {
                interaction_idx = 1 + std::distance(unique_interaction_energies.begin(), match);
            }

            fmt::print(m_dest, "{}:[1A][{}-{}]({},{},{}) R={:.3f}\n",
                       interaction_idx,
                       n.a().name(), n.b().name(),
                       uc_shift[0], uc_shift[1], uc_shift[2],
                       n.centroid_distance()
            );
            interaction_idx++;
        }
        for(double e: unique_interaction_energies) {
            fmt::print(m_dest, "{:.4f}\n", e);
        }
        uc_idx++;
    }
}

} // namespace occ::io::crystalgrower
