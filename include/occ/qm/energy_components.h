#pragma once
#include <fmt/core.h>

namespace occ::qm
{
struct EnergyComponents
{
    double coulomb{0.0};
    double exchange{0.0};
    double correlation{0.0};
    double exchange_repulsion{0.0};
    double nuclear_repulsion{0.0};
    double nuclear_attraction{0.0};
    double kinetic{0.0};
    double one_electron{0.0};
    double two_electron{0.0};
    double electronic{0.0};
    double solvation{0.0};
    double exchange_correlation{0.0};
    double total{0.0};

    std::string energy_table_string() const {

        std::string result{""};
        std::string fmt_string{"{:32s} {:20.12f}\n"};

        result += fmt::format("{:32s} {:>20s}\n\n", "Component", "Energy (Hartree)");
        result += fmt::format(fmt_string, "Nuclear repulsion (NN)", nuclear_repulsion);
        if(nuclear_attraction != 0.0) result += fmt::format(fmt_string, "Nuclear attraction (V)", nuclear_attraction);
        if(kinetic != 0.0) result += fmt::format(fmt_string, "Kinetic (T)", kinetic);
        if(one_electron != 0.0) result += fmt::format(fmt_string, "One-electron (T + V)", one_electron);
        if(exchange_correlation != 0.0) result += fmt::format(fmt_string, "Exchange-correlation", solvation);
        if(two_electron != 0.0) result += fmt::format(fmt_string, "Two-electron", two_electron);
        if(solvation != 0.0) result += fmt::format(fmt_string, "Solvation", solvation);
        if(total != 0.0) result += fmt::format(fmt_string, "Total", total);
        return result;
    }
};

}
