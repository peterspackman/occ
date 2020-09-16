#include "molecule.h"
#include "hf.h"
#include "scf.h"
#include <fmt/core.h>

int main(int argc, char* argv[]) {
    using std::cout;
    using std::cerr;
    using std::endl;
    using craso::chem::Molecule;
    using craso::hf::HartreeFock;
    using craso::scf::SCF;


    try {
        libint2::Shell::do_enforce_unit_normalization(false);
        if (!libint2::initialized()) libint2::initialize();
        const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
        const auto basisname = (argc > 2) ? argv[2] : "6-31g*";
        Molecule m = craso::chem::read_xyz_file(filename);

        {
            using craso::parallel::nthreads;
            auto nthreads_cstr = getenv("OMP_NUM_THREADS");
            nthreads = 1;
            if (nthreads_cstr && strcmp(nthreads_cstr, "")) {
                std::istringstream iss(nthreads_cstr);
                iss >> nthreads;
                if (nthreads > 1 << 16 || nthreads <= 0) nthreads = 1;
            }
            omp_set_num_threads(nthreads);
            fmt::print("Using {} threads\n", nthreads);
        }
        fmt::print("Geometry loaded from {}\n", filename);
        fmt::print("Using {} basis on all atoms\n", basisname);

        libint2::BasisSet obs(basisname, m.atoms());

        fmt::print("Orbital basis set rank = {}\n", obs.nbf());

        HartreeFock hf(m.atoms(), obs);
        craso::scf::SCF<HartreeFock> scf(hf);
        double e = scf.compute_scf_energy();
        fmt::print("Total Energy (SCF): {:20.12f} hartree\n", e);

    }
    catch (const char* ex) {
        fmt::print("Caught exception when performing HF calculation:  {}\n",  ex);
        return 1;
    } catch (std::string& ex) {
        fmt::print("Caught exception when performing HF calculation:  {}\n",  ex);
        return 1;
    } catch (std::exception& ex) {
        fmt::print("Caught exception when performing HF calculation:  {}\n",  ex.what());
        return 1;
    } catch (...) {
        fmt::print("Unknown exception occurred...\n");
        return 1;
    }
    return 0;
}
