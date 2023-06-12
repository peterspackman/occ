#include <fmt/core.h>
#include <occ/core/units.h>
#include <occ/io/gmf.h>

namespace occ::io {
GMFWriter::GMFWriter(const occ::crystal::Crystal &crystal)
    : m_crystal(crystal){};

bool GMFWriter::write(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open())
        return false;
    write(file);
    file.close();
    return true;
}

bool GMFWriter::write(std::ostream &output) const {

    const auto &sg = m_crystal.space_group();
    const auto &uc = m_crystal.unit_cell();

    output << fmt::format("\n title: {}\n", m_title);
    output << fmt::format("  name: {}\n", m_name);
    output << fmt::format(" space: {}\n", sg.symbol());
    output << fmt::format("  cell: {:f} {:f} {:f}  {:f} {:f} {:f}\n", uc.a(),
                          uc.b(), uc.c(), occ::units::degrees(uc.alpha()),
                          occ::units::degrees(uc.beta()),
                          occ::units::degrees(uc.gamma()));

    output << fmt::format(" morph: {}\n\n", m_morphology_kind);

    for (const auto &facet : m_facets) {
        output << fmt::format("miller:  {:3d} {:3d} {:3d}\n", facet.hkl.h,
                              facet.hkl.k, facet.hkl.l);
        output << fmt::format(" {:8.6f}  {:2d} {:2d}  {:10.4f} {:10.4f} "
                              "{:10.4f} {:10.4f}  {:g}\n",
                              facet.shift, facet.region0, facet.region1,
                              facet.surface, facet.attachment,
                              facet.surface_relaxed, facet.attachment_relaxed,
                              facet.gnorm);
    }
    return true;
}

} // namespace occ::io
