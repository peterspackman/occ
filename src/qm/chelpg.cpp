#include <LBFGS.h>
#include <occ/core/logger.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

Vec chelpg_charges(const Wavefunction &wfn, Eigen::Ref<Mat3N> grid_points) {

    hf::HartreeFock hf(wfn.basis);
    Vec esp = hf.electronic_electric_potential_contribution(wfn.mo.kind, wfn.mo,
                                                            grid_points) +
              hf.nuclear_electric_potential_contribution(grid_points);

    const int num_atoms = wfn.atoms.size();
    double net_charge = hf.system_charge();
    Mat inverse_distances(grid_points.cols(), num_atoms);
    for (int i = 0; i < num_atoms; i++) {
        const Vec3 pi{wfn.atoms[i].x, wfn.atoms[i].y, wfn.atoms[i].z};
        inverse_distances.col(i) =
            1.0 / (grid_points.colwise() - pi).colwise().norm().array();
    }
    LBFGSpp::LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;

    // Create solver and function object
    LBFGSpp::LBFGSSolver<double> solver(param);

    Vec charges = wfn.mulliken_charges();
    Mat pred = inverse_distances * charges;

    auto func = [&esp, &pred, &inverse_distances](const Vec &x, Vec &grad) {
        pred = inverse_distances * x;
        double rmsd = (pred - esp).norm() / pred.size();
        Mat diff = pred - esp;
        grad.array() =
            (diff.transpose() * inverse_distances).colwise().sum().transpose();
        return rmsd;
    };
    double rmsd;
    int niter = solver.minimize(func, charges, rmsd);
    charges.array() -= (charges.sum() - net_charge) / charges.rows();
    pred = inverse_distances * charges;
    rmsd = (pred - esp).norm() / pred.size();

    occ::log::debug("CHELPG charges in {} iterations, RMSD = {:12.5g}, net "
                    "charge = {:12.5g}",
                    niter, rmsd, charges.sum());

    return charges;
}

} // namespace occ::qm
