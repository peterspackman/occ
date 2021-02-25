#include <tonto/core/molecule.h>
#include <Eigen/Core>
#include <fstream>
#include <fmt/core.h>
#include <tonto/core/units.h>

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
    // Internally store in angstroms
    m_positions(0, i) = atom.x * tonto::units::BOHR_TO_ANGSTROM;
    m_positions(1, i) = atom.y * tonto::units::BOHR_TO_ANGSTROM;
    m_positions(2, i) = atom.z * tonto::units::BOHR_TO_ANGSTROM;
  }
  m_name = chemical_formula(m_elements);
}

Molecule read_xyz_file(const std::string &filename) {
  std::ifstream is(filename);
  if (not is.good()) {
    throw std::runtime_error(fmt::format("Could not open file: '{}'", filename));
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
    using tonto::units::ANGSTROM_TO_BOHR;
    for (size_t i = 0; i < size(); i++) {
        result[i] = {m_atomicNumbers(i), m_positions(0, i) * ANGSTROM_TO_BOHR,
                     m_positions(1, i) * ANGSTROM_TO_BOHR, m_positions(2, i) * ANGSTROM_TO_BOHR};
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

Molecule Molecule::rotated(const Eigen::Affine3d &rotation, Origin origin) const
{
    return rotated(rotation.linear(), origin);
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

Molecule Molecule::rotated(const tonto::Mat3 &rotation, Origin origin) const
{
    Molecule result = *this;
    result.rotate(rotation, origin);
    return result;
}

void Molecule::translate(const tonto::Vec3 &translation)
{
    m_positions.colwise() += translation;
}

Molecule Molecule::translated(const tonto::Vec3 &translation) const
{
    Molecule result = *this;
    result.translate(translation);
    return result;
}

void Molecule::transform(const Mat4 &transform, Origin origin)
{
    rotate(transform.block<3, 3>(0, 0), origin);
    translate(transform.block<3, 1>(0, 3));
}

Molecule Molecule::transformed(const Mat4 &transform, Origin origin) const
{
    Molecule result = *this;
    result.transform(transform, origin);
    return result;
}

std::tuple<size_t, size_t, double> Molecule::nearest_atom(const Molecule &other) const
{
    std::tuple<size_t, size_t, double> result{0, 0, std::numeric_limits<double>::max()};
    for(size_t i = 0; i < size(); i++)
    {
        const tonto::Vec3& p1 = m_positions.col(i);
        for(size_t j = 0; j < other.size(); j++)
        {
            const tonto::Vec3& p2 = other.m_positions.col(j);
            double d = (p2 - p1).norm();
            if(d < std::get<2>(result)) result = {i, j, d};
        }
    }
    return result;
}

} // namespace tonto::chem
