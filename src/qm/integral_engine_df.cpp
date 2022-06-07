#include <occ/core/timings.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm {

IntegralEngineDF::IntegralEngineDF(const AtomList &atoms, const ShellList &ao,
                                   const ShellList &df)
    : m_ao_env(atoms, ao), m_aux_env(atoms, df) {
    m_ao_env.set_auxiliary_basis(df, false);
    occ::timing::start(occ::timing::category::df);
    Mat V =
        m_aux_env.one_electron_operator(Op::coulomb); // V = (P|Q) in df basis
    occ::timing::stop(occ::timing::category::df);

    occ::timing::start(occ::timing::category::la);
    V_LLt = Eigen::LLT<Mat>(V);
    Mat Vsqrt = Eigen::SelfAdjointEigenSolver<Mat>(V).operatorSqrt();
    Vsqrt_LLt = Eigen::LLT<Mat>(Vsqrt);
    occ::timing::stop(occ::timing::category::la);
}

} // namespace occ::qm
