/**
 * @mainpage Open Computational Chemistry
 *
 * \section welcome Welcome
 *
 * Welcome to the official API documentation for the Open Computational
 * Chemistry project.
 *
 * Note that this is the documentation for the code - not how to use the
 * programs. If you are looking for tutorials, please visit the website:
 *
 * \section example A simple Hartree-Fock program
 *
 * Here's a very brief example, indicative of the overall design of the program,
 * implementing a very simple spin-restricted Hartree-Fock program.
 *
 * \code
 *
 * #include <occ/qm/hf.h>
 * #include <occ/qm/scf.h>
 * #include <occ/io/xyz.h>
 *
 * int main(int argc, char argv**) {
 *    using occ::qm::HartreeFock;
 *    using occ::qm::SpinorbitalKind;
 *    using occ::scf::SCF;
 *
 *    // read in a molecule from a file
 *    auto mol = occ::io::molecule_from_xyz_file("water.xyz");
 *    // load the Gaussian-type basis set for the molecule
 *    auto bs = occ::qm::AOBasis::load(mol.atoms(), "6-31G");
 *
 *    // Initialize a Hartree-Fock object for restricted spinorbitals
 *    // with this basis set
 *    HartreeFock<SpinorbitalKind::Restricted> hf(bs);
 *
 *    // Initialize an SCF object to evaluate the ground state.
 *    SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
 *
 *    // Perform the SCF iterations and get the resulting ground state energy.
 *    double e = scf.compute_scf_energy();
 *
 *    return 0;
 * }
 *
 * \endcode
 *
 */

/**
 * @namespace occ::core
 * @brief fundamental functionality for linear algebra, utilities,
 * molecules and more
 * @details No dependencies on other modules in occ
 */

/**
 * @namespace occ::constants
 * @brief definitions of scientific and math constants
 * @details part of occ::core module
 */

/**
 * @namespace occ::crystal
 * @brief functionality related to periodic crystal structures, space groups,
 * symmetry operations
 * @details depends on occ::core and the gemmi library
 */

/**
 * @namespace occ::density
 * @brief functionality related to evaluation of electron density
 * @details part of the occ::gto module
 */

/**
 * @namespace occ::dft
 * @brief functionality related to Kohn-Sham density functional theory
 * @details part of the occ::qm, occ:gto, occ:io module
 */

/**
 * @namespace occ::disp
 * @brief dispersion corrections
 */

/**
 * @namespace occ::geometry
 * @brief computational geometry functionality - marching cubes, Morton codes
 * etc.
 */

/**
 * @namespace occ::gto
 * @brief evaluation Gaussian-type orbitals, their derivatives etc.
 */

/**
 * @namespace occ::interaction
 * @brief interactions energies including CrystalExplorer model energies
 */

/**
 * @namespace occ::io
 * @brief file input and output module including reading wavefunction files
 */

/**
 * @namespace occ::log
 * @brief logging for debug output, warnings, errors etc.
 */

/**
 * @namespace occ::main
 * @brief main module for functionaliity related to single point energies etc.
 */

/**
 * @namespace occ::parallel
 * @brief main module for functionaliity parallelism - threads etc.
 */

/**
 * @namespace occ::qm
 * @brief quantum mechanics/quantum chemistry functionality including
 * Hartree-Fock and more
 */

/**
 * @namespace occ::scf
 * @brief self-consistent field implementation
 */

/**
 * @namespace occ::sht
 * @brief spherical harmonic transforms
 */

/**
 * @namespace occ::slater
 * @brief evaluation of Slater-type orbitals, their derivatives etc.
 */

/**
 * @namespace occ::solvent
 * @brief solvation models for correction of QM methods (implicit only for now)
 */

/**
 * @namespace occ::timing
 * @brief routines for global timers, timers of sections and more
 */
