#include "catch.hpp"
#include "fchkreader.h"
#include <sstream>
#include <fmt/ostream.h>

const char* fchk_contents = R"(h2
SP        RB3LYP                                                      STO-3G
Number of atoms                            I                2
Info1-9                                    I   N=           9
           9           9           0           0           0         110
           1          18        -502
Charge                                     I                0
Multiplicity                               I                1
Number of electrons                        I                2
Number of alpha electrons                  I                1
Number of beta electrons                   I                1
Number of basis functions                  I                2
Number of independent functions            I                2
Number of point charges in /Mol/           I                0
Number of translation vectors              I                0
Atomic numbers                             I   N=           2
           1           1
Nuclear charges                            R   N=           2
  1.00000000E+00  1.00000000E+00
Current cartesian coordinates              R   N=           6
  0.00000000E+00  0.00000000E+00  6.99198669E-01  8.56271412E-17  0.00000000E+00
 -6.99198669E-01
Force Field                                I                0
Int Atom Types                             I   N=           2
           0           0
MM charges                                 R   N=           2
  0.00000000E+00  0.00000000E+00
Integer atomic weights                     I   N=           2
           1           1
Real atomic weights                        R   N=           2
  1.00782504E+00  1.00782504E+00
Atom fragment info                         I   N=           2
           0           0
Atom residue num                           I   N=           2
           0           0
Nuclear spins                              I   N=           2
           1           1
Nuclear ZEff                               R   N=           2
 -1.00000000E+00 -1.00000000E+00
Nuclear ZNuc                               R   N=           2
  1.00000000E+00  1.00000000E+00
Nuclear QMom                               R   N=           2
  0.00000000E+00  0.00000000E+00
Nuclear GFac                               R   N=           2
  2.79284600E+00  2.79284600E+00
MicOpt                                     I   N=           2
          -1          -1
Number of contracted shells                I                2
Number of primitive shells                 I                6
Pure/Cartesian d shells                    I                0
Pure/Cartesian f shells                    I                0
Highest angular momentum                   I                0
Largest degree of contraction              I                3
Shell types                                I   N=           2
           0           0
Number of primitives per shell             I   N=           2
           3           3
Shell to atom map                          I   N=           2
           1           2
Primitive exponents                        R   N=           6
  3.42525091E+00  6.23913730E-01  1.68855404E-01  3.42525091E+00  6.23913730E-01
  1.68855404E-01
Contraction coefficients                   R   N=           6
  1.54328967E-01  5.35328142E-01  4.44634542E-01  1.54328967E-01  5.35328142E-01
  4.44634542E-01
Coordinates of each shell                  R   N=           6
  0.00000000E+00  0.00000000E+00  6.99198669E-01  8.56271412E-17  0.00000000E+00
 -6.99198669E-01
Constraint Structure                       R   N=           6
  0.00000000E+00  0.00000000E+00  6.99198669E-01  8.56271412E-17  0.00000000E+00
 -6.99198669E-01
Num ILSW                                   I              100
ILSW                                       I   N=         100
           0           0           0           0           2           0
           0           0           0           0         402          -1
           0           0           0           0           0           0
           0           0           0           0           0           0
           1           0           0           0           0           0
           0           0      100000           0          -1           0
           0           0           0           0           0           0
           0           0           0           1           0           0
           0           0           1           0           0           0
           0           0           4          41           0           0
           0           0           5           0           0           0
           0           0           0           2           0           0
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0
Num RLSW                                   I               41
RLSW                                       R   N=          41
  8.00000000E-01  7.20000000E-01  1.00000000E+00  8.10000000E-01  2.00000000E-01
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  1.00000000E+00  1.00000000E+00
  0.00000000E+00
MxBond                                     I                1
NBond                                      I   N=           2
           1           1
IBond                                      I   N=           2
           2           1
RBond                                      R   N=           2
  1.00000000E+00  1.00000000E+00
Virial Ratio                               R      1.970141361625062E+00
SCF Energy                                 R     -1.165418375762579E+00
Total Energy                               R     -1.165418375762579E+00
External E-field                           R   N=          35
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
IOpCl                                      I                0
IROHF                                      I                0
Alpha Orbital Energies                     R   N=           2
 -4.14539570E-01  4.27590260E-01
Alpha MO coefficients                      R   N=           4
  5.48842275E-01  5.48842275E-01  1.21245192E+00 -1.21245192E+00
Total SCF Density                          R   N=           3
  6.02455687E-01  6.02455687E-01  6.02455687E-01
Mulliken Charges                           R   N=           2
 -2.77555756E-16 -3.33066907E-16
ONIOM Charges                              I   N=          16
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0
ONIOM Multiplicities                       I   N=          16
           1           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0
Atom Layers                                I   N=           2
           1           1
Atom Modifiers                             I   N=           2
           0           0
Force Field                                I                0
Int Atom Modified Types                    I   N=           2
           0           0
Link Atoms                                 I   N=           2
           0           0
Atom Modified MM Charges                   R   N=           2
  0.00000000E+00  0.00000000E+00
Link Distances                             R   N=           8
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
Cartesian Gradient                         R   N=           6
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00
Dipole Moment                              R   N=           3
 -1.23259516E-32  0.00000000E+00  5.55111512E-17
Quadrupole Moment                          R   N=           6
 -1.04128604E-01 -1.04128604E-01  2.08257207E-01  0.00000000E+00 -1.91281142E-17
  0.00000000E+00
QEq coupling tensors                       R   N=          12
  1.83153654E-01  0.00000000E+00  1.83153654E-01  4.89879653E-17  0.00000000E+00
 -3.66307308E-01  1.83153654E-01  0.00000000E+00  1.83153654E-01  1.34443277E-17
  0.00000000E+00 -3.66307308E-01
)";


TEST_CASE("H2 fchk", "[read]")
{
    std::istringstream fchk(fchk_contents);
    tonto::io::FchkReader reader(fchk);
    REQUIRE(reader.num_alpha() == 1);
    REQUIRE(reader.num_basis_functions() == 2);
    fmt::print("Alpha MOs:\n{}\n", reader.alpha_mo_coefficients());
    fmt::print("Alpha MO energies:\n{}\n", reader.alpha_mo_energies());
    fmt::print("Positions:\n{}\n", reader.atomic_positions());
    REQUIRE(reader.basis().primitive_exponents[0] == Approx(3.42525091));

    tonto::io::FchkReader::FchkBasis basis = reader.basis();
    basis.print();
    fmt::print("Libint2 basis:\n");
    for(const auto& shell: reader.libint_basis())
    {
        fmt::print("\n{}\n", shell);
    }
}
