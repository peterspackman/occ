#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/io/fchkwriter.h>

namespace occ::io {

namespace impl {

static const std::array<const char *, 15> fchk_type_strings{
    "SP",
    "FOPT",
    "POPT",
    "FTS",
    "FSADDLE",
    "PSADDLE",
    "FORCE",
    "FREQ",
    "SCAN",
    "GUESS=ONLY",
    "LST",
    "STABILITY",
    "REARCHIVE/MS-RESTART",
    "MIXED"};

static const std::vector<std::string> fchk_key_order{
    "Number of atoms",
    "Info1-9",
    "Full Title",
    "Route",
    "Charge",
    "Multiplicity",
    "Number of electrons",
    "Number of alpha electrons",
    "Number of beta electrons",
    "Number of basis functions",
    "Number of independent functions",
    "Number of point charges in /Mol/",
    "Number of translation vectors",
    "Atomic numbers",
    "Nuclear charges",
    "Current cartesian coordinates",
    "Number of symbols in /Mol/",
    "Force Field",
    "Atom Types",
    "Int Atom Types",
    "MM Charges",
    "Integer atomic weights",
    "Real atomic weights",
    "Atom fragment info",
    "Atom residue num",
    "Nuclear spins",
    "Nuclear ZEff",
    "Nuclear ZNuc",
    "Nuclear QMom",
    "Nuclear GFac",
    "MicOpt",
    "Number of residues",
    "Number of secondary structures",
    "Number of contracted shells",
    "Number of primitive shells",
    "Pure/Cartesian d shells",
    "Pure/Cartesian f shells",
    "Highest angular momentum",
    "Largest degree of contraction",
    "Shell types",
    "Number of primitives per shell",
    "Shell to atom map",
    "Primitive exponents",
    "Contraction coefficients",
    "P(S=P) Contraction coefficients",
    "Coordinates of each shell",
    "Constraint Structure",
    "Num ILSW",
    "ILSW",
    "Num RLSW",
    "RLSW",
    "ECP-MxAtEC",
    "ECP-MaxLECP",
    "ECP-MaxAtL",
    "ECP-MxTECP",
    "ECP-LenNCZ",
    "ECP-KFirst",
    "ECP-KLast",
    "ECP-Lmax",
    "ECP-LPSkip",
    "ECP-RNFroz",
    "ECP-NLP",
    "ECP-CLP1",
    "ECP-CLP2",
    "ECP-ZLP",
    "MxBond",
    "NBond",
    "IBond",
    "RBond",
    "Virial ratio",
    "SCF Energy",
    "Total Energy",
    "RMS Density",
    "External E-field",
    "IOpCl",
    "IROHF",
    "Alpha Orbital Energies",
    "Beta Orbital Energies",
    "Alpha MO coefficients",
    "Beta MO coefficients",
    "Total SCF Density",
    "Total MP2 Density",
    "Spin SCF Density",
    "Spin MP2 Density",
    "Mulliken Charges",
    "ONIOM Charges",
    "ONIOM Multiplicities",
    "Atom Layers",
    "Atom Modifiers",
    "Atom Modified Types",
    "Int Atom Modified Types",
    "Link Atoms",
    "Atom Modified MM Charges",
    "Link Distances",
    "Cartesian Gradient",
    "Dipole Moment",
    "Quadrupole Moment",
    "QEq coupling tensors"};

void FchkScalarWriter::operator()(int value) {
    fmt::print(destination, "{:40s}   I     {:12d}\n", key, value);
}

void FchkScalarWriter::operator()(double value) {
    fmt::print(destination, "{:40s}   R     {:22.15E}\n", key, value);
}

void FchkScalarWriter::operator()(const std::string &value) {
    fmt::print(destination, "{:40s}   C     {:12s}\n", key, value);
}

void FchkScalarWriter::operator()(bool value) {
    fmt::print(destination, "{:40s}   L     {:1d}\n", key, value);
}

void FchkVectorWriter::operator()(const std::vector<int> &values) {
    constexpr int num_per_line{6};
    const std::string value_format{"{:12d}"};
    fmt::print(destination, "{:40s}   I   N={:12d}\n", key, values.size());
    int count = 0;
    for (const auto &value : values) {
        fmt::print(destination, fmt::runtime(value_format), value);
        count++;
        if (count % num_per_line == 0)
            fmt::print(destination, "\n");
    }
    if (count % num_per_line != 0)
        fmt::print(destination, "\n");
}

void FchkVectorWriter::operator()(const std::vector<double> &values) {
    constexpr int num_per_line{5};
    const std::string value_format{"{:16.8e}"};
    fmt::print(destination, "{:40s}   R   N={:12d}\n", key, values.size());
    int count = 0;
    for (const auto &value : values) {
        fmt::print(destination, fmt::runtime(value_format), value);
        count++;
        if (count % num_per_line == 0)
            fmt::print(destination, "\n");
    }
    if (count % num_per_line != 0)
        fmt::print(destination, "\n");
}

void FchkVectorWriter::operator()(const std::vector<std::string> &values) {
    constexpr int num_per_line{5};
    const std::string value_format{"{:12s}"};
    fmt::print(destination, "{:40s}   C   N={:12d}\n", key, values.size());
    int count = 0;
    for (const auto &value : values) {
        fmt::print(destination, fmt::runtime(value_format), value);
        count++;
        if (count % num_per_line == 0)
            fmt::print(destination, "\n");
    }
    if (count % num_per_line != 0)
        fmt::print(destination, "\n");
}

void FchkVectorWriter::operator()(const std::vector<bool> &values) {
    constexpr int num_per_line{72};
    const std::string value_format{"{:1d}"};
    fmt::print(destination, "{:40s}   L   N={:12d}\n", key, values.size());
    int count = 0;
    for (bool value : values) {
        fmt::print(destination, fmt::runtime(value_format), value);
        count++;
        if (count % num_per_line == 0)
            fmt::print(destination, "\n");
    }
    if (count % num_per_line != 0)
        fmt::print(destination, "\n");
}
} // namespace impl

void FchkWriter::set_basis(const occ::qm::AOBasis &basis) {
    int largest_contraction{0};
    int l_max = 0;
    std::vector<int> shell_types, nprim_per_shell;
    shell_types.reserve(basis.size());
    std::vector<double> shell_coords;
    shell_coords.reserve(3 * basis.size());
    std::vector<double> primitive_exponents, contraction_coefficients;
    int number_primitive_shells{0};
    for (size_t i = 0; i < basis.size(); i++) {
        const auto &sh = basis.shells()[i];
        int l = sh.l;
        if (l > 1 && sh.is_pure())
            l = -l;
        l_max = std::max(l, l_max);
        int nprim = sh.num_primitives();
        number_primitive_shells += nprim;
        largest_contraction = std::max(nprim, largest_contraction);
        shell_types.push_back(l);
        nprim_per_shell.push_back(nprim);
        shell_coords.push_back(sh.origin(0));
        shell_coords.push_back(sh.origin(1));
        shell_coords.push_back(sh.origin(2));
        for (size_t p = 0; p < nprim; p++) {
            primitive_exponents.push_back(sh.exponents(p));
            contraction_coefficients.push_back(sh.coeff_normalized(0, p));
        }
    }
    set_scalar("Number of contracted shells", basis.size());
    bool spherical = basis.is_pure();
    set_scalar("Pure/Cartesian d shells", spherical ? 0 : 1);
    set_scalar("Pure/Cartesian f shells", spherical ? 0 : 1);
    set_scalar("Highest angular momentum", l_max);
    set_scalar("Largest degree of contraction", largest_contraction);
    set_vector("Shell types", shell_types);
    set_vector("Number of primitives per shell", nprim_per_shell);
    set_scalar("Number of primitive shells", number_primitive_shells);
    set_vector("Primitive exponents", primitive_exponents);
    set_vector("Contraction coefficients", contraction_coefficients);
    set_vector("P(S=P) Contraction coefficients",
               occ::Vec::Zero(contraction_coefficients.size()));
    set_vector("Coordinates of each shell", shell_coords);
}

FchkWriter::FchkWriter(const std::string &filename)
    : m_owned_destination(filename), m_dest(m_owned_destination) {}

FchkWriter::FchkWriter(std::ostream &stream) : m_dest(stream) {}

void FchkWriter::write() {
    fmt::print(m_dest, "{:<72s}\n", m_title);
    fmt::print(m_dest, "{:10s} {:<30s} {:>30s}\n",
               impl::fchk_type_strings[m_type], m_method, m_basis_name);
    impl::FchkScalarWriter scalar_writer{m_dest, ""};
    impl::FchkVectorWriter vector_writer{m_dest, ""};

    for (const auto &key : impl::fchk_key_order) {
        if (m_scalars.contains(key)) {
            scalar_writer.key = key;
            std::visit(scalar_writer, m_scalars[key]);
        }
        if (m_vectors.contains(key)) {
            vector_writer.key = key;
            std::visit(vector_writer, m_vectors[key]);
        }
    }
}

} // namespace occ::io
