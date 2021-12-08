#include <occ/io/orca_json.h>
#include <occ/core/util.h>
#include <occ/core/timings.h>
#include <occ/core/logger.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>

namespace occ::io {


/*
  Scaling factors for Orca

         class                           factor
    1    s                             * 1.0
         p
         d
         f(0,+1,-1,+2,-2)
         g(0,+1,-1,+2,-2)
         h(0,+1,-1,+2,-2,+5,-5)
    2    f(+3,-3)                      *-1.0
         g(+3,-3,+4,-4)
*/


double normalization_factor(double alpha, int l, int m, int n) {
    using occ::util::double_factorial;
    return std::sqrt(std::pow(4 * alpha, l + m + n) * std::pow(2 * alpha / M_PI, 1.5) /
                     (double_factorial(2 * l - 1) * 
                      double_factorial(2 * m - 1) * 
                      double_factorial(2 * n - 1)));
}

OrcaJSONReader::OrcaJSONReader(const std::string &filename) {
    occ::timing::start(occ::timing::category::io);
    open(filename);
    parse(m_json_file);
    occ::timing::stop(occ::timing::category::io);
}

OrcaJSONReader::OrcaJSONReader(std::istream &filehandle) {
    occ::timing::start(occ::timing::category::io);
    parse(filehandle);
    occ::timing::stop(occ::timing::category::io);
}

void OrcaJSONReader::open(const std::string &filename) {
    m_json_file.open(filename);
    if (m_json_file.fail() | m_json_file.bad()) {
        throw std::runtime_error("Unable to open fchk file: " + filename);
    }
}

void OrcaJSONReader::close() { m_json_file.close(); }


int string_to_l(const std::string &shell_label) {
    char label = shell_label[0];
    switch(label) {
        case 's': return 0;
        case 'S': return 0;
        case 'p': return 1;
        case 'P': return 1;
        case 'd': return 2;
        case 'D': return 2;
        case 'f': return 3;
        case 'F': return 3;
        case 'g': return 4;
        case 'G': return 4;
        case 'h': return 5;
        case 'H': return 5;
        case 'i': return 6;
        case 'I': return 6;
        case 'k': return 7;
        case 'K': return 7;
        default: return 0;
    }
}

void OrcaJSONReader::parse(std::istream &stream) {
    using occ::qm::Shell;
    using libint2::svector;

    auto j = nlohmann::json::parse(stream);
    const auto& mol = j["Molecule"];
    const auto& atoms = mol["Atoms"];

    size_t num_atoms = atoms.size();
    m_atomic_numbers = IVec(num_atoms);
    m_atom_labels.resize(num_atoms);
    m_atom_positions = Mat3N(3, num_atoms);


    for(const auto &atom: atoms) {
        size_t idx = atom["Idx"];
        m_atomic_numbers(idx) = atom["ElementNumber"];
        m_atom_labels[idx] = atom["ElementLabel"];
        const auto& pos = atom["Coords"];
        m_atom_positions(0, idx) = pos[0];
        m_atom_positions(1, idx) = pos[1];
        m_atom_positions(2, idx) = pos[2];

        std::array<double, 3> pos_array = {pos[0], pos[1], pos[2]};

        for(const auto& bf : atom["BasisFunctions"]) {
            svector<double> alpha;
            svector<double> coeffs;
            const auto& coeff = bf["Coefficients"];
            const auto& exp = bf["Exponents"];
            const auto& shell_kind = bf["Shell"];
            for(size_t i = 0; i < coeff.size(); i++) {
                coeffs.push_back(coeff[i]);
            }
            for(size_t i = 0; i < exp.size(); i++) {
                alpha.push_back(exp[i]);
            }

            int l = string_to_l(shell_kind);
            m_basis.emplace_back(Shell{std::move(alpha),
                                 {{l, true, std::move(coeffs)}},
                                 {std::move(pos_array)}});
        }
    }
    m_basis.update();
    m_basis.set_pure(true);
    size_t nbf = m_basis.nbf();
    occ::log::debug("num atoms {}", num_atoms);

    bool unrestricted = mol["HFTyp"] == "UHF";

    occ::log::debug("unrestricted: {}", unrestricted);
    occ::log::debug("nbf: {}", nbf);

    const auto& mos = mol["MolecularOrbitals"]["MOs"];
    const auto& mo_labels = mol["MolecularOrbitals"]["OrbitalLabels"];
    size_t num_mos = mos.size();
    m_alpha_energies = Vec(nbf);
    m_alpha_coeffs = Mat(nbf, nbf);
    m_alpha_labels.reserve(nbf);
    if(unrestricted) {
        m_beta_energies = Vec(nbf);
        m_beta_coeffs = Mat(nbf, nbf);
        m_beta_labels.reserve(nbf);
    }

    size_t mo_idx = 0;
    for(const auto& mo: mos) {
        bool alpha_block = mo_idx < nbf;
        auto& e = alpha_block ? m_alpha_energies : m_beta_energies;
        auto& c = alpha_block ? m_alpha_coeffs : m_beta_coeffs;
        auto& n = alpha_block ? m_num_alpha : m_num_beta;
        auto& l = alpha_block ? m_alpha_labels : m_beta_labels;

        const auto& coeffs = mo["MOCoefficients"];
        Eigen::Index j = mo_idx % nbf;
        for(size_t i = 0; i < coeffs.size(); i++) {
            c(i, j) = coeffs[i];
        }
        e(j) = mo["OrbitalEnergy"];
        size_t occn = static_cast<size_t>(mo["Occupancy"]);
        l.push_back(mo_labels[mo_idx]);
        m_num_electrons += occn;
        n += unrestricted ? occn : occn / 2;
        mo_idx++;
    }

    if(!unrestricted) m_num_beta = m_num_alpha;

    occ::log::debug("Num electrons: {}", m_num_electrons);
    occ::log::debug("Num alpha electrons {}", m_num_alpha);
    occ::log::debug("Num beta electrons {}", m_num_beta);

    const auto &S = mol["S-Matrix"];
    size_t bf1 = 0;
    m_overlap = Mat(nbf, nbf);
    for(const auto &row: S) {
        size_t bf2 = 0;
        for(const auto &x: row) {
            m_overlap(bf1, bf2) = x;
            bf2++;
        }
        bf1++;
    }
}

std::vector<occ::core::Atom> OrcaJSONReader::atoms() const {
    std::vector<occ::core::Atom> atoms;
    atoms.reserve(m_atomic_numbers.size());
    for (size_t i = 0; i < m_atomic_numbers.size(); i++) {
        atoms.emplace_back(occ::core::Atom{
            m_atomic_numbers(i), m_atom_positions(0, i),
            m_atom_positions(1, i), m_atom_positions(2, i)});
    }
    return atoms;
}



}
