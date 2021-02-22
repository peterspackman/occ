#include <tonto/core/molecule.h>
#include <Eigen/Core>
#include <fstream>

namespace tonto::chem {

Molecule::Molecule(const IVec &nums, const Mat3N &pos)
    : m_atomicNumbers(nums), m_positions(pos) {
  for (size_t i = 0; i < size(); i++) {
    m_elements.push_back(Element(m_atomicNumbers(i)));
  }
  m_name = chemical_formula(m_elements);
}

Molecule::Molecule(const std::vector<libint2::Atom> &atoms)
    : m_positions(3, atoms.size()),
      m_atomicNumbers(atoms.size()) {
  m_elements.reserve(atoms.size());
  for (size_t i = 0; i < atoms.size(); i++) {
    const auto &atom = atoms[i];
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

std::vector<libint2::Atom> Molecule::atoms() const
{
    std::vector<libint2::Atom> result(size());
    for (size_t i = 0; i < size(); i++) {
        result.push_back(libint2::Atom{m_atomicNumbers(i), m_positions(0, i),
                                       m_positions(1, i), m_positions(2, i)});
    }
    return result;
}

const Vec Molecule::vdw_radii() const
{
    Vec radii(size());
    for(size_t i = 0; i < radii.size(); i++)
    {
        radii(i) = static_cast<double>(m_elements[i].vdw());
    }
    return radii;
}


const Vec Molecule::atomic_masses() const
{
    Vec masses(size());
    for(size_t i = 0; i < masses.size(); i++)
    {
        masses(i) = static_cast<double>(m_elements[i].mass());
    }
    return masses;
}

const tonto::Vec3 Molecule::centroid() const
{
    return m_positions.rowwise().mean();
}

const tonto::Vec3 Molecule::center_of_mass() const
{
    tonto::RowVec masses = atomic_masses();
    masses.array() /= masses.sum();
    return (m_positions.array().rowwise() * masses.array()).rowwise().sum();
}


bool Molecule::comparable_to(const Molecule &other) const
{
    if(size() != other.size()) return false;
    for(size_t i = 0; i < size(); i++)
    {
        if(m_atomicNumbers(i) != other.m_atomicNumbers(i)) return false;
    }
    return true;
}

void Molecule::rotate(const Eigen::Affine3d &rotation, Origin origin)
{
    rotate(rotation.linear(), origin);
}

void Molecule::rotate(const tonto::Mat3 &rotation, Origin origin)
{
    Vec3 O = {0, 0, 0};
    switch(origin)
    {
        case Centroid:
        {
            O = centroid();
            break;
        }
        case CenterOfMass:
        {
            O = center_of_mass();
            break;
        }
        default: break;
    }
    translate(-O);
    m_positions = rotation * m_positions;
    translate(O);
}

void Molecule::translate(const tonto::Vec3 &translation)
{
    m_positions.colwise() += translation;
}

void Molecule::transform(const Mat4 &transform, Origin origin)
{
    rotate(transform.block<3, 3>(0, 0), origin);
    translate(transform.block<3, 1>(0, 3));
}

} // namespace tonto::chem
