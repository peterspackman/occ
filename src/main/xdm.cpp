#include <cmath>
#include <filesystem>
#include <fmt/core.h>
#include <iostream>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/io/fchkreader.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/wavefunction.h>
#include <occ/xdm/xdm.h>
#include <vector>

using occ::qm::Wavefunction;

Wavefunction load_wavefunction(const std::string &filename) {
    namespace fs = std::filesystem;
    using occ::util::to_lower;
    std::string ext = fs::path(filename).extension().string();
    to_lower(ext);
    if (ext == ".fchk") {
        using occ::io::FchkReader;
        FchkReader fchk(filename);
        return Wavefunction(fchk);
    }
    if (ext == ".molden" || ext == ".input") {
        using occ::io::MoldenReader;
        MoldenReader molden(filename);
        return Wavefunction(molden);
    }
    throw std::runtime_error(
        "Unknown file extension when reading wavefunction: " + ext);
}

int main(int argc, char *argv[]) {

    occ::log::set_level(occ::log::level::debug);
    auto wfn = load_wavefunction(argv[1]);
    fmt::print("wfn loaded: {}\n", argv[1]);

    auto xdm = occ::xdm::XDM(wfn.basis);
    double e = xdm.energy(wfn.mo);
    fmt::print("XDM energy = {:20.16f} Hartree\n", e);
    return 0;
}
