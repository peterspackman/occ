#include "basisset.h"
#include "util.h"
#include "gto.h"
#include <cerrno>
#include <iostream>
#include <fstream>
#include <locale>
#include <vector>
#include <stdexcept>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tonto::qm {

struct canonicalizer {
    char operator()(char c) {
      char cc = ::tolower(c);
      switch (cc) {
        case '/': cc = 'I'; break;
      }
      return cc;
    }
};

BasisSet::BasisSet(std::string name,
                   const std::vector<Atom>& atoms,
                   const bool throw_if_no_match) : m_name(std::move(name))
{

    // read in the library file contents
    std::string basis_lib_path = data_path();
    auto canonical_name = canonicalize_name(m_name);
    // some basis sets use cartesian d shells by convention, the convention is taken from Gaussian09
    auto force_cartesian_d = gaussian_cartesian_d_convention(canonical_name);

    // parse the name into components
    std::vector<std::string> basis_component_names = decompose_name_into_components(canonical_name);

    // ref_shells[component_idx][Z] => vector of Shells
    std::vector<std::vector<std::vector<libint2::Shell>>> component_basis_sets;
    component_basis_sets.reserve(basis_component_names.size());

    // read in ALL basis set components
    for(const auto& basis_component_name: basis_component_names) {
        auto file_dot_g94 = basis_lib_path + "/" + basis_component_name + ".g94";

        // use same cartesian_d convention for all components!
        component_basis_sets.emplace_back(read_g94_basis_library(file_dot_g94, force_cartesian_d, throw_if_no_match));
    }

    // for each atom find the corresponding basis components
    for(auto a=0ul; a<atoms.size(); ++a) {

        const std::size_t Z = atoms[a].atomic_number;

        // add each component in order
        for(auto comp_idx=0ul; comp_idx!=component_basis_sets.size(); ++comp_idx) {
            const auto& component_basis_set = component_basis_sets[comp_idx];
            if (!component_basis_set.at(Z).empty()) {  // found? add shells in order
                for(auto s: component_basis_set.at(Z)) {
                    this->push_back(std::move(s));
                    this->back().move({{atoms[a].x, atoms[a].y, atoms[a].z}});
                } // shell loop
            }
            else if (throw_if_no_match) {  // not found? throw, if needed
                std::string errmsg(std::string("did not find the basis for this Z in ") +
                                   basis_lib_path + "/" + basis_component_names[comp_idx] + ".g94");
                throw std::logic_error(errmsg);
            }
        } // basis component loop
    } // atom loop

    update();
}



BasisSet::BasisSet(const std::vector<Atom>& atoms,
                   const std::vector<std::vector<Shell>>& element_bases,
                   std::string name,
                   const bool throw_if_no_match) : m_name(std::move(name))
{
    // for each atom find the corresponding basis components
    for(auto a=0ul; a<atoms.size(); ++a) {

        auto Z = atoms[a].atomic_number;

        if (decltype(Z)(element_bases.size()) > Z && !element_bases.at(Z).empty()) {  // found? add shells in order
            for(auto s: element_bases.at(Z)) {
                this->push_back(std::move(s));
                this->back().move({{atoms[a].x, atoms[a].y, atoms[a].z}});
            } // shell loop
        }
        else if (throw_if_no_match) {  // not found? throw, if needed
            throw std::logic_error(std::string("did not find the basis for Z=") + std::to_string(Z) + " in the element_bases");
        }
    } // atom loop
}


std::vector<long> BasisSet::shell2atom(const std::vector<Shell>& shells, const std::vector<Atom>& atoms, bool throw_if_no_match)
{
  std::vector<long> result;
  result.reserve(shells.size());
  for(const auto& s: shells) {
    auto a = std::find_if(atoms.begin(), atoms.end(), [&s](const Atom& a){ return s.O[0] == a.x && s.O[1] == a.y && s.O[2] == a.z; } );
    const auto found_match = (a != atoms.end());
    if (throw_if_no_match && !found_match)
      throw std::logic_error("shell2atom: no matching atom found");
    result.push_back( found_match ? a - atoms.begin() : -1);
  }
  return result;
}

std::vector<std::vector<long>> BasisSet::atom2shell(const std::vector<Atom>& atoms, const std::vector<Shell>& shells)
{
  std::vector<std::vector<long>> result;
  result.resize(atoms.size());
  size_t iatom = 0;
  for(const auto& a: atoms) {
    auto s = shells.begin();
    while (s != shells.end()) {
      s = std::find_if(s, shells.end(), [&a](const Shell& s){ return s.O[0] == a.x && s.O[1] == a.y && s.O[2] == a.z; } );
      if (s != shells.end()) {
        result[iatom].push_back( s - shells.begin());
        ++s;
      }
    }
    ++iatom;
  }
  return result;
}


std::string BasisSet::canonicalize_name(const std::string& name)
{
  auto result = name;
  std::transform(name.begin(), name.end(),
                 result.begin(), canonicalizer());
  return result;
}


bool BasisSet::gaussian_cartesian_d_convention(const std::string& canonical_name)
{
  // 3-21??g??, 4-31g??
  if (canonical_name.find("3-21")    == 0 ||
      canonical_name.find("4-31g")   == 0)
    return true;
  // 6-31??g?? but not 6-311 OR 6-31g()
  if (canonical_name.find("6-31") == 0 && canonical_name[4] != '1') {
    // to exclude 6-31??g() find the g, then check the next character
    auto g_pos = canonical_name.find('g');
    if (g_pos == std::string::npos) // wtf, I don't even know what this is, assume spherical d is OK
      return false;
    if (g_pos+1 == canonical_name.size()) // 6-31??g uses cartesian d
      return true;
    if (canonical_name[g_pos+1] == '*') // 6-31??g*? uses cartesian d
      return true;
  }
  return false;
}

std::vector<std::string> BasisSet::decompose_name_into_components(std::string name)
{
  std::vector<std::string> component_names;
  // aug-cc-pvxz* = cc-pvxz* + augmentation-... , except aug-cc-pvxz-cabs
  if ( (name.find("aug-cc-pv") == 0) && (name.find("cabs")==std::string::npos)  ) {
    std::string base_name = name.substr(4);
    component_names.push_back(base_name);
    component_names.push_back(std::string("augmentation-") + base_name);
  }
  else
    component_names.push_back(name);

  return component_names;
}


std::string BasisSet::data_path()
{
  std::string path;
  const char* data_path_env = getenv("LIBINT_DATA_PATH");
  if (data_path_env) {
    path = data_path_env;
  }
  else {
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
    std::ostringstream oss; oss << "BasisSet::data_path(): path \"" << path << "{/basis}\" is not valid";
    throw std::system_error(ec, oss.str());
  }
  return basis_path;
}

void BasisSet::fortran_dfloats_to_efloats(std::string& str)
{
  for(auto& ch: str) {
    if (ch == 'd') ch = 'e';
    if (ch == 'D') ch = 'E';
  }
}


std::vector<std::vector<libint2::Shell>> BasisSet::read_g94_basis_library(
        std::string file_dot_g94,
        bool force_cartesian_d,
        bool throw_if_missing,
        std::string locale_name)
{
  std::locale locale(locale_name.c_str());  // TODO omit c_str() with up-to-date stdlib
  std::vector<std::vector<libint2::Shell>> ref_shells(118); // 118 = number of chemical elements
  std::ifstream is(file_dot_g94);
  is.imbue(locale);

  if (is.good()) {
    if (libint2::verbose())
      libint2::verbose_stream() << "Will read basis set from " << file_dot_g94 << std::endl;

    std::string line, rest;

    // skip till first basis
    while (std::getline(is, line) && line != "****") {
    }

#define LIBINT2_LINE_TO_STRINGSTREAM(line) \
  std::istringstream iss(line); \
  iss.imbue(locale);

    size_t Z;
    auto nextbasis = true, nextshell = false;
    // read lines till end
    while (std::getline(is, line)) {
      // skipping empties and starting with '!' (the comment delimiter)
      if (line.empty() || line[0] == '!') continue;
      if (line == "****") {
        nextbasis = true;
        nextshell = false;
        continue;
      }
      if (nextbasis) {
        nextbasis = false;
        LIBINT2_LINE_TO_STRINGSTREAM(line);
        std::string elemsymbol;
        iss >> elemsymbol >> rest;

        bool found = false;
        for (const auto &e: libint2::chemistry::get_element_info()) {
          if (strcaseequal(e.symbol, elemsymbol)) {
            Z = e.Z;
            found = true;
            break;
          }
        }
        if (not found) {
          std::ostringstream oss;
          oss << "in file " << file_dot_g94
              << " found G94 basis set for element symbol \""
              << elemsymbol << "\", not found in Periodic Table.";
          throw std::logic_error(oss.str());
        }

        nextshell = true;
        continue;
      }
      if (nextshell) {
        LIBINT2_LINE_TO_STRINGSTREAM(line);
        std::string amlabel;
        std::size_t nprim;
        iss >> amlabel >> nprim >> rest;
        if (amlabel != "SP" && amlabel != "sp") {
          assert(amlabel.size() == 1);
          auto l = Shell::am_symbol_to_l(amlabel[0]);
          svector<double> exps;
          svector<double> coeffs;
          for (decltype(nprim) p = 0; p != nprim; ++p) {
            while (std::getline(is, line) && (line.empty() || line[0] == '!')) {}
            fortran_dfloats_to_efloats(line);
            LIBINT2_LINE_TO_STRINGSTREAM(line);
            double e, c;
            iss >> e >> c;
            exps.emplace_back(e);
            coeffs.emplace_back(c);
          }
          auto pure = force_cartesian_d ? (l > 2) : (l > 1);
          ref_shells.at(Z).push_back(
              libint2::Shell{
                  std::move(exps),
                  {
                      {l, pure, std::move(coeffs)}
                  },
                  {{0, 0, 0}}
              }
          );
        } else { // split the SP shells
          svector<double> exps;
          svector<double> coeffs_s, coeffs_p;
          for (decltype(nprim) p = 0; p != nprim; ++p) {
            while (std::getline(is, line) && (line.empty() || line[0] == '!')) {}
            fortran_dfloats_to_efloats(line);
            LIBINT2_LINE_TO_STRINGSTREAM(line);
            double e, c1, c2;
            iss >> e >> c1 >> c2;
            exps.emplace_back(e);
            coeffs_s.emplace_back(c1);
            coeffs_p.emplace_back(c2);
          }
          ref_shells.at(Z).push_back(
              libint2::Shell{exps,
                             {
                                 {0, false, coeffs_s}
                             },
                             {{0, 0, 0}}
              }
          );
          ref_shells.at(Z).push_back(
              libint2::Shell{std::move(exps),
                             {
                                 {1, false, std::move(coeffs_p)}
                             },
                             {{0, 0, 0}}
              }
          );
        }
      }
    }

#undef LIBINT2_LINE_TO_STRINGSTREAM
  }
  else {  // !is.good()
    if (throw_if_missing) {
      std::ostringstream oss;
      oss << "BasisSet::read_g94_basis_library(): could not open \"" << file_dot_g94 << "\"" << std::endl;
      throw std::ios_base::failure(oss.str());
    }
  }

  return ref_shells;
}

std::vector<size_t> compute_shell2bf(const std::vector<libint2::Shell>& shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

void BasisSet::update()
{
    m_nbf = tonto::qm::nbf(*this);
    m_max_nprim = tonto::qm::max_nprim(*this);
    m_max_l = tonto::qm::max_l(*this);
    m_shell2bf = compute_shell2bf(*this);
}


tonto::MatRM rotate_molecular_orbitals(const BasisSet& basis, const tonto::Mat3& rotation, const tonto::MatRM& C)
{
    assert(!basis.is_pure());
    const auto shell2bf = basis.shell2bf();
    tonto::MatRM result(C.rows(), C.cols());
    for(size_t s = 0; s < basis.size(); s++) {
        const auto& shell = basis[s];
        size_t bf_first = shell2bf[s];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        tonto::MatRM rot;
        switch(l) {
        case 0:
            result.block(bf_first, 0, shell_size, C.cols()).noalias() = C.block(bf_first, 0, shell_size, C.cols());
            continue;
        case 1:
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<1>(rotation);
            break;
        case 2:
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<2>(rotation);
            break;
        case 3:
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<3>(rotation);
            break;
        case 4:
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<4>(rotation);
            break;
        case 5:
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<5>(rotation);
            break;
        case 6:
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<6>(rotation);
            break;
            // trivial to implement for higher angular momenta, but the template is not instantiated
        default:
            throw std::runtime_error("MO rotation not implemented for angular momentum > 6");
        }
        result.block(bf_first, 0, shell_size, C.cols()).noalias() = rot * C.block(bf_first, 0, shell_size, C.cols());
    }
    return result;
}


void rotate_atoms(std::vector<libint2::Atom>& atoms, const tonto::Mat3& rotation)
{
    for(auto& atom: atoms) {
        tonto::Vec3 pos{atom.x, atom.y, atom.z};
        auto pos_rot = rotation * pos;
        atom.x = pos_rot(0);
        atom.y = pos_rot(1);
        atom.z = pos_rot(2);
    }
}

}
