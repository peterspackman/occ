#pragma once
#include <vector>

// modified routines from libint2
// include/libint2/chemistry/sto3g_atomic_density.h

namespace occ::qm::guess {

namespace impl {

/* compute orbital occupation numbers for a subshell created
 * by smearing at most num_electrons_remaining
 * (corresponds to spherical averaging)
 */
template <typename output_iterator>
void update_occupation_subshell(output_iterator &destination, size_t size, size_t& num_electrons_remaining) {
  const size_t electrons_allocated = (num_electrons_remaining > 2 * size)
      ? 2 * size
      : num_electrons_remaining;
  num_electrons_remaining -= electrons_allocated;
  const double electrons_per_orbital = (electrons_allocated % size == 0)
                                ? static_cast<double>(electrons_allocated / size)
                                : (static_cast<double>(electrons_allocated)) / size;
  for (size_t f = 0; f != size; ++f, destination++) *destination = electrons_per_orbital;
}

}  // namespace impl

inline size_t minimal_basis_nao(size_t Z) {
  size_t nao;
  if (Z == 1 || Z == 2)  // H, He
    nao = 1;
  else if (Z <= 10)  // Li - Ne
    nao = 5;         // 2p is included even for Li and Be
  else if (Z <= 18)  // Na - Ar
    nao = 9;         // 3p is included even for Na and Mg
  else if (Z <= 20)  // K, Ca
    nao = 13;        // 4p is included
  else if (Z <= 36)  // Sc - Kr
    nao = 18;
  else if (Z <= 38)  // Rb, Sr
    nao = 22;        // 5p is included
  else if (Z <= 54)  // Y - I
    nao = 27;
  else if (Z <= 86)
    nao = 49;
  else
    throw std::runtime_error("minimal basis not defined for elements Z > 86");
  return nao;
}

std::vector<double> minimal_basis_occupation_vector(size_t Z) {

  using impl::update_occupation_subshell;
  std::vector<double> occvec(minimal_basis_nao(Z));
  auto iter = occvec.begin();

  size_t num_of_electrons = Z;

  // Electronic configurations from NIST:
  // http://www.nist.gov/pml/data/images/illo_for_2014_PT_1.PNG
  update_occupation_subshell(iter, 1, num_of_electrons);            // 1s
  if (Z > 2)   {  // Li+
    update_occupation_subshell(iter, 1, num_of_electrons);          // 2s
    update_occupation_subshell(iter, 3, num_of_electrons);          // 2p
  }
  if (Z > 10)  {  // Na+
    update_occupation_subshell(iter, 1, num_of_electrons);          // 3s
    update_occupation_subshell(iter, 3, num_of_electrons);          // 3p
  }
  if (18 < Z && Z <= 36)  {  // K .. Kr
    // 4s is singly occupied in K, Cr, and Cu
    size_t num_of_4s_electrons = (Z == 19 || Z == 24 || Z == 29) ? 1 : 2;
    num_of_electrons -= num_of_4s_electrons;
    update_occupation_subshell(iter, 1, num_of_4s_electrons);       // 4s

    size_t num_of_4p_electrons =
        std::min(static_cast<decltype(Z)>(6), (Z > 30) ? Z - 30 : 0);
    num_of_electrons -= num_of_4p_electrons;
    update_occupation_subshell(iter, 3, num_of_4p_electrons);       // 4p

    update_occupation_subshell(iter, 5, num_of_electrons);          // 3d
  }
  if (36 < Z && Z <= 53)  {  // Rb .. I
    // 3d4s4p are fully occupied ...
    update_occupation_subshell(iter, 1, num_of_electrons);          // 4s
    update_occupation_subshell(iter, 3, num_of_electrons);          // 4p

    // 5s is singly occupied in Rb, Nb, Mo, Ru, Rh, and Ag
    size_t num_of_5s_electrons =
        (Z == 37 || Z == 41 || Z == 42 || Z == 44 || Z == 45 || Z == 47) ? 1
                                                                         : 2;
    num_of_electrons -= num_of_5s_electrons;
    update_occupation_subshell(iter, 1, num_of_5s_electrons);       // 5s

    size_t num_of_5p_electrons =
        std::min(static_cast<decltype(Z)>(6), (Z > 48) ? Z - 48 : 0);
    num_of_electrons -= num_of_5p_electrons;
    update_occupation_subshell(iter, 3, num_of_5p_electrons);       // 5p

    update_occupation_subshell(iter, 5, num_of_electrons);          // 3d
    update_occupation_subshell(iter, 5, num_of_electrons);          // 4d
  }
  if(53 < Z <= 86)
  {
    update_occupation_subshell(iter, 1, num_of_electrons);          // 4s
    update_occupation_subshell(iter, 3, num_of_electrons);          // 4p
    update_occupation_subshell(iter, 1, num_of_electrons);          // 5s
    update_occupation_subshell(iter, 3, num_of_electrons);          // 5p
    update_occupation_subshell(iter, 3, num_of_electrons);          // 6p
    update_occupation_subshell(iter, 3, num_of_electrons);          // 6p
    update_occupation_subshell(iter, 5, num_of_electrons);          // 3d
    update_occupation_subshell(iter, 5, num_of_electrons);          // 4d
    update_occupation_subshell(iter, 5, num_of_electrons);          // 5d
    update_occupation_subshell(iter, 9, num_of_electrons);          // 4f
  }

  return occvec;
}

}
