#include <occ/crystal/standard_bonds.h>

namespace occ::crystal {

// Static member initialization
ankerl::unordered_dense::map<int, double> StandardBondLengths::custom_lengths;

// Default bond lengths indexed by atomic number
// Values from Allen, F. H. Acta Cryst. (2010). B66, 380–386
const ankerl::unordered_dense::map<int, double> StandardBondLengths::default_lengths = {
    {5,  B_H},   // B
    {6,  C_H},   // C
    {7,  N_H},   // N
    {8,  O_H},   // O
};

double StandardBondLengths::get_hydrogen_bond_length(int atomic_number) {
    // Check custom lengths first
    auto custom_it = custom_lengths.find(atomic_number);
    if (custom_it != custom_lengths.end()) {
        return custom_it->second;
    }
    
    // Check default lengths
    auto default_it = default_lengths.find(atomic_number);
    if (default_it != default_lengths.end()) {
        return default_it->second;
    }
    
    // No standard length available
    return -1.0;
}

bool StandardBondLengths::has_standard_length(int atomic_number) {
    return custom_lengths.count(atomic_number) > 0 || 
           default_lengths.count(atomic_number) > 0;
}

void StandardBondLengths::set_custom_bond_length(int atomic_number, double length) {
    custom_lengths[atomic_number] = length;
}

void StandardBondLengths::clear_custom_lengths() {
    custom_lengths.clear();
}

} // namespace occ::crystal