#include "molecule.h"
#include <Eigen/Core>
#include <fstream>

namespace tonto::chem {

Molecule::Molecule(const IVec &nums, const Mat3N &pos)
    : m_atomicNumbers(nums), m_positions(pos) {
  for (size_t i = 0; i < size(); i++) {
    m_elements.push_back(Element(m_atomicNumbers(i)));
    m_atoms.push_back(libint2::Atom{m_atomicNumbers(i), m_positions(0, i),
                                    m_positions(1, i), m_positions(2, i)});
  }
  m_name = chemical_formula(m_elements);
}

Molecule::Molecule(const std::vector<libint2::Atom> &atoms)
    : m_atoms(atoms), m_positions(3, atoms.size()),
      m_atomicNumbers(atoms.size()) {
  m_elements.reserve(m_atoms.size());
  for (size_t i = 0; i < m_atoms.size(); i++) {
    const auto &atom = m_atoms[i];
    m_elements.push_back(Element(atom.atomic_number));
    m_atomicNumbers(i) = atom.atomic_number;
    m_positions(0, i) = atom.x;
    m_positions(1, i) = atom.y;
    m_positions(2, i) = atom.z;
  }
  m_name = chemical_formula(m_elements);
}

Molecule read_xyz_file(const std::string &filename) {
  std::ifstream is(filename);
  if (not is.good()) {
    char errmsg[256] = "Could not open file: ";
    strncpy(errmsg + 20, filename.c_str(), 235);
    errmsg[255] = '\0';
    throw std::runtime_error(errmsg);
  }

  // to prepare for MPI parallelization, we will read the entire file into a
  // string that can be
  // broadcast to everyone, then converted to an std::istringstream object that
  // can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise
  // throw an exception
  if (filename.rfind(".xyz") != std::string::npos)
    return Molecule(libint2::read_dotxyz(iss));
  else
    throw "only .xyz files are accepted";
}

} // namespace tonto::chem
