#pragma once

namespace occ::qm {

struct SCFConvergenceSettings {
    double energy_threshold{1e-6};
    double commutator_threshold{1e-5};
    double incremental_fock_threshold{1e-4};

    inline bool energy_converged(double energy_difference) const {
        return energy_difference < energy_threshold;
    }

    inline bool commutator_converged(double commutator_difference) const {
        return commutator_difference < commutator_threshold;
    }

    inline bool energy_and_commutator_converged(double ediff,
                                                double cdiff) const {
        return energy_converged(ediff) && commutator_converged(cdiff);
    }

    inline bool start_incremental_fock(double diis_error) const {
        return diis_error < incremental_fock_threshold;
    }
};

} // namespace occ::qm
