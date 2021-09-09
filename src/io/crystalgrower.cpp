#include <occ/io/crystalgrower.h>
#include <occ/crystal/crystal.h>
#include <fmt/ostream.h>
#include <occ/core/units.h>

namespace occ::io::crystalgrower {

StructureWriter::StructureWriter(const std::string &filename) : m_owned_destination(filename), m_dest(m_owned_destination)
{
}

StructureWriter::StructureWriter(std::ostream &stream) : m_dest(stream)
{
}


void StructureWriter::write(const occ::crystal::Crystal &crystal)
{
    using occ::units::degrees;
    fmt::print(m_dest, "{}\n\n", "title");
    const auto& uc_molecules = crystal.unit_cell_molecules();
    fmt::print(m_dest, "{}\n\n", uc_molecules.size());
    for(const auto& mol: uc_molecules)
    {
        size_t num_neighbors = 0;
        fmt::print(m_dest, "{} {} {}\n\n", mol.unit_cell_molecule_idx() + 1, mol.name(), num_neighbors);
    }

    const auto& uc = crystal.unit_cell();
    double a = uc.a(), b = uc.b(), c = uc.c();
    double alpha = degrees(uc.alpha()), beta = degrees(uc.beta()), gamma = degrees(uc.gamma());
    fmt::print(m_dest, "{:8.4f} {:8.4f} {:8.4f} P\n", a, b, c);
    fmt::print(m_dest, "{:7.3f} {:7.3f} {:7.3f}\n", alpha, beta, gamma);

    fmt::print(m_dest, "\nNon primitive data\n");
    fmt::print(m_dest, "{:8.4f} {:8.4f} {:8.4f}\n", a, b, c);
    fmt::print(m_dest, "{:7.3f} {:7.3f} {:7.3f}\n", alpha, beta, gamma);
    fmt::print(m_dest, "\n");

    for(const auto& mol: uc_molecules)
    {
        const auto& bonds = mol.bonds();
        const auto& elements = mol.elements();
        const auto& pos = crystal.to_fractional(mol.positions());
        fmt::print(m_dest, "{} {} {}/{}\n", mol.name(), mol.unit_cell_molecule_idx() + 1, mol.size(), bonds.size() / 2);
        for(size_t i = 0; i < mol.size(); i++)
        {
            fmt::print(m_dest, "{} {:<2s} {: 8.5f} {: 8.5f} {: 8.5f}\n", i + 1, elements[i].symbol(),
                    pos(0, i), pos(1, i), pos(2, i));
        }
        fmt::print(m_dest, "\n");
        for(const auto& b: bonds)
        {
            if(b.first < b.second)
                fmt::print(m_dest, " {} {}\n", b.first + 1, b.second + 1);
        }
        fmt::print(m_dest, "\n");
    }
}

}
