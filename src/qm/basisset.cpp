#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/qm/basisset.h>
#include <stdexcept>
#include <vector>

#include <occ/io/basis_g94.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace occ::qm {

namespace fs = std::filesystem;

struct canonicalizer {
    char operator()(char c) {
        char cc = ::tolower(c);
        switch (cc) {
        case '/':
            cc = 'I';
            break;
        }
        return cc;
    }
};

BasisSet::BasisSet(std::string name, const std::vector<Atom> &atoms)
    : m_name(std::move(name)) {

    std::string basis_lib_path = data_path();
    auto canonical_name = canonicalize_name(m_name);
    auto force_cartesian_d = gaussian_cartesian_d_convention(canonical_name);

    std::vector<std::string> basis_component_names =
        decompose_name_into_components(canonical_name);

    std::vector<std::vector<std::vector<libint2::Shell>>> component_basis_sets;
    component_basis_sets.reserve(basis_component_names.size());

    for (const auto &basis_component_name : basis_component_names) {
        std::string g94_filepath = basis_component_name + ".g94";
        if (!fs::exists(basis_component_name + ".g94")) {
            g94_filepath = basis_lib_path + "/" + g94_filepath;
        }
        component_basis_sets.emplace_back(
            occ::io::basis::g94::read(g94_filepath, force_cartesian_d));
    }

    for (auto a = 0ul; a < atoms.size(); ++a) {
        const std::size_t Z = atoms[a].atomic_number;

        for (auto comp_idx = 0ul; comp_idx != component_basis_sets.size();
             ++comp_idx) {
            const auto &component_basis_set = component_basis_sets[comp_idx];
            if (!component_basis_set.at(Z).empty()) {
                for (auto s : component_basis_set.at(Z)) {
                    this->push_back(std::move(s));
                    this->back().move({{atoms[a].x, atoms[a].y, atoms[a].z}});
                }
            } else {
                std::string basis_filename = basis_lib_path + "/" +
                                             basis_component_names[comp_idx] +
                                             ".g94";
                std::string errmsg =
                    fmt::format("No matching basis for element (z={}) in {}", Z,
                                basis_filename);
                throw std::logic_error(errmsg);
            }
        }
    }

    update();
}

BasisSet::BasisSet(const std::vector<Atom> &atoms,
                   const std::vector<std::vector<Shell>> &element_bases,
                   std::string name)
    : m_name(std::move(name)) {
    for (auto a = 0ul; a < atoms.size(); ++a) {
        auto Z = atoms[a].atomic_number;
        if (decltype(Z)(element_bases.size()) > Z &&
            !element_bases.at(Z).empty()) {
            for (auto s : element_bases.at(Z)) {
                this->push_back(std::move(s));
                this->back().move({{atoms[a].x, atoms[a].y, atoms[a].z}});
            }
        } else {
            std::string errmsg = fmt::format("No matching basis for element "
                                             "(z={}) in provided element basis",
                                             Z);

            throw std::logic_error(errmsg);
        }
    }
}

std::vector<long> BasisSet::shell2atom(const std::vector<Shell> &shells,
                                       const std::vector<Atom> &atoms) {
    std::vector<long> result;
    result.reserve(shells.size());
    size_t shell_idx = 0;
    for (const auto &s : shells) {
        auto a = std::find_if(atoms.begin(), atoms.end(), [&s](const Atom &a) {
            return s.O[0] == a.x && s.O[1] == a.y && s.O[2] == a.z;
        });
        const auto found_match = (a != atoms.end());
        if (!found_match)
            throw std::logic_error(
                fmt::format("no matching atom found for shell {}", shell_idx));
        result.push_back(a - atoms.begin());
        shell_idx++;
    }
    return result;
}

std::vector<long> BasisSet::bf2atom(const std::vector<Atom> &atoms) const {
    auto shell_map = atom2shell(atoms);
    auto bf_map = shell2bf();
    std::vector<long> result(nbf());
    for (size_t a = 0; a < atoms.size(); a++) {
        for (const auto &sh : shell_map[a]) {
            for (size_t bf = bf_map[sh]; bf < bf_map[sh] + (at(sh)).size();
                 bf++)
                result[bf] = a;
        }
    }
    return result;
}

std::vector<std::vector<long>>
BasisSet::atom2shell(const std::vector<Atom> &atoms,
                     const std::vector<Shell> &shells) {
    std::vector<std::vector<long>> result;
    result.resize(atoms.size());
    size_t iatom = 0;
    for (const auto &a : atoms) {
        auto s = shells.begin();
        while (s != shells.end()) {
            s = std::find_if(s, shells.end(), [&a](const Shell &s) {
                return s.O[0] == a.x && s.O[1] == a.y && s.O[2] == a.z;
            });
            if (s != shells.end()) {
                result[iatom].push_back(s - shells.begin());
                ++s;
            }
        }
        ++iatom;
    }
    return result;
}

std::string BasisSet::canonicalize_name(const std::string &name) {
    auto result = name;
    std::transform(name.begin(), name.end(), result.begin(), canonicalizer());
    return result;
}

bool BasisSet::gaussian_cartesian_d_convention(
    const std::string &canonical_name) {
    // 3-21??g??, 4-31g??
    if (canonical_name.find("3-21") == 0 || canonical_name.find("4-31g") == 0)
        return true;
    // 6-31??g?? but not 6-311 OR 6-31g()
    if (canonical_name.find("6-31") == 0 && canonical_name[4] != '1') {
        // to exclude 6-31??g() find the g, then check the next character
        auto g_pos = canonical_name.find('g');
        if (g_pos == std::string::npos) // wtf, I don't even know what this is,
                                        // assume spherical d is OK
            return false;
        if (g_pos + 1 == canonical_name.size()) // 6-31??g uses cartesian d
            return true;
        if (canonical_name[g_pos + 1] == '*') // 6-31??g*? uses cartesian d
            return true;
    }
    return false;
}

std::vector<std::string>
BasisSet::decompose_name_into_components(std::string name) {
    std::vector<std::string> component_names;
    // aug-cc-pvxz* = cc-pvxz* + augmentation-... , except aug-cc-pvxz-cabs
    if ((name.find("aug-cc-pv") == 0) &&
        (name.find("cabs") == std::string::npos)) {
        std::string base_name = name.substr(4);
        component_names.push_back(base_name);
        component_names.push_back(std::string("augmentation-") + base_name);
    } else
        component_names.push_back(name);

    return component_names;
}

std::string BasisSet::data_path() {
    std::string path;
    const char *data_path_env = getenv("OCC_BASIS_PATH");
    if (data_path_env) {
        path = data_path_env;
    } else {
#if defined(DATADIR)
        path = std::string{DATADIR};
#elif defined(SRCDATADIR)
        path = std::string{SRCDATADIR};
#else
        path = std::string("/usr/local/share/libint/2.7.0");
#endif
    }
    // validate basis_path = path + "/basis"
    std::string basis_path = path + std::string("/basis");
    bool error = true;
    std::error_code ec;
    auto validate_basis_path = [&basis_path, &error, &ec]() -> void {
        if (not basis_path.empty()) {
            struct stat sb;
            error = (::stat(basis_path.c_str(), &sb) == -1);
            error = error || not S_ISDIR(sb.st_mode);
            if (error)
                ec = std::error_code(errno, std::generic_category());
        }
    };
    validate_basis_path();
    if (error) { // try without "/basis"
        basis_path = path;
        validate_basis_path();
    }
    if (error) {
        occ::log::warn("There is a problem with BasisSet::data_path(), the "
                       "path '{}' is not valid ({})",
                       basis_path, ec.message());
        basis_path = fs::current_path().string();
    }
    return basis_path;
}

std::vector<size_t>
compute_shell2bf(const std::vector<libint2::Shell> &shells) {
    std::vector<size_t> result;
    result.reserve(shells.size());

    size_t n = 0;
    for (auto shell : shells) {
        result.push_back(n);
        n += shell.size();
    }

    return result;
}

void BasisSet::update() {
    m_nbf = occ::qm::nbf(*this);
    m_max_nprim = occ::qm::max_nprim(*this);
    m_max_l = occ::qm::max_l(*this);
    m_shell2bf = compute_shell2bf(*this);
}

Mat rotate_molecular_orbitals_cart(const BasisSet &basis,
                                   const occ::Mat3 &rotation, const Mat &C) {
    const auto shell2bf = basis.shell2bf();
    Mat result(C.rows(), C.cols());
    for (size_t s = 0; s < basis.size(); s++) {
        const auto &shell = basis[s];
        size_t bf_first = shell2bf[s];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        Mat rot;
        switch (l) {
        case 0:
            result.block(bf_first, 0, shell_size, C.cols()).noalias() =
                C.block(bf_first, 0, shell_size, C.cols());
            continue;
        case 1:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<1>(rotation);
            break;
        case 2:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<2>(rotation);
            break;
        case 3:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<3>(rotation);
            break;
        case 4:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<4>(rotation);
            break;
        case 5:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<5>(rotation);
            break;
        case 6:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<6>(rotation);
            break;
            // trivial to implement for higher angular momenta, but the template
            // is not instantiated
        default:
            throw std::runtime_error(
                "MO rotation not implemented for angular momentum > 6");
        }
        result.block(bf_first, 0, shell_size, C.cols()).noalias() =
            rot * C.block(bf_first, 0, shell_size, C.cols());
    }
    return result;
}

void BasisSet::rotate(const occ::Mat3 &rotation) {
    for (auto &shell : *this) {
        occ::Vec3 pos{shell.O[0], shell.O[1], shell.O[2]};
        auto rot_pos = rotation * pos;
        shell.O[0] = rot_pos(0);
        shell.O[1] = rot_pos(1);
        shell.O[2] = rot_pos(2);
    }
    update();
}

void BasisSet::translate(const occ::Vec3 &translation) {
    for (auto &shell : *this) {
        shell.O[0] += translation(0);
        shell.O[1] += translation(1);
        shell.O[2] += translation(2);
    }
    update();
}

std::vector<size_t> pople_sp_shells(const BasisSet &basis) {
    std::vector<bool> visited(basis.size(), false);
    std::vector<size_t> shells(basis.size());
    size_t current_sp_shell{0};
    auto same_site = [](const libint2::Shell &sh1, const libint2::Shell &sh2) {
        return (sh1.O[0] == sh2.O[0]) && (sh1.O[1] == sh2.O[1]) &&
               (sh1.O[2] == sh2.O[2]);
    };

    auto same_primitives = [](const libint2::Shell &sh1,
                              const libint2::Shell &sh2) {
        if (sh1.alpha.size() != sh2.alpha.size())
            return false;
        for (size_t i = 0; i < sh1.alpha.size(); i++) {
            if (sh1.alpha[i] != sh2.alpha[i])
                return false;
        }
        return true;
    };

    for (size_t i = 0; i < basis.size(); i++) {
        if (visited[i])
            continue;
        const auto &sh1 = basis[i];
        shells[i] = current_sp_shell;
        visited[i] = true;
        if (sh1.contr[0].l != 0)
            continue;
        // if we have an S shell, look for matching P shells
        for (size_t j = i + 1; j < basis.size(); j++) {
            if (visited[j])
                continue;
            const auto &sh2 = basis[j];
            if (sh2.contr[0].l != 1)
                continue;
            if (!same_site(sh1, sh2))
                continue;
            if (same_primitives(sh1, sh2)) {
                shells[j] = current_sp_shell;
                visited[j] = true;
            }
        }
        current_sp_shell++;
    }
    return shells;
}

} // namespace occ::qm
