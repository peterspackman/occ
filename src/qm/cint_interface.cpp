#include <occ/qm/cint_interface.h>

namespace occ::qm::cint {
Optimizer::Optimizer(IntegralEnvironment &env, Operator op, int num_center,
                     int grad)
    : m_op(op), m_num_center(num_center), m_grad(grad) {
  switch (m_num_center) {
  case 1:
  case 2:
    if (grad) {
      create1or2c_grad(env);
    } else
      create1or2c(env);
    break;
  case 3:
    if (grad) {
      create3c_grad(env);
    } else
      create3c(env);
    break;
  case 4:
    if (grad) {
      create4c_grad(env);
    } else
      create4c(env);
    break;
  default:
    throw std::runtime_error("Invalid num centers for cint::Optimizer");
  }
}

Optimizer::~Optimizer() { libcint::CINTdel_optimizer(&m_optimizer); }

void Optimizer::create1or2c(IntegralEnvironment &env) {
  switch (m_op) {
  case Operator::coulomb:
    libcint::int2c2e_optimizer(&m_optimizer, env.atom_data_ptr(),
                               env.num_atoms(), env.basis_data_ptr(),
                               env.num_basis(), env.env_data_ptr());
    break;
  case Operator::nuclear:
    libcint::int1e_nuc_optimizer(&m_optimizer, env.atom_data_ptr(),
                                 env.num_atoms(), env.basis_data_ptr(),
                                 env.num_basis(), env.env_data_ptr());
    break;
  case Operator::kinetic:
    libcint::int1e_kin_optimizer(&m_optimizer, env.atom_data_ptr(),
                                 env.num_atoms(), env.basis_data_ptr(),
                                 env.num_basis(), env.env_data_ptr());
    break;
  case Operator::overlap:
    libcint::int1e_ovlp_optimizer(&m_optimizer, env.atom_data_ptr(),
                                  env.num_atoms(), env.basis_data_ptr(),
                                  env.num_basis(), env.env_data_ptr());
    break;
  case Operator::dipole:
    libcint::int1e_r_optimizer(&m_optimizer, env.atom_data_ptr(),
                               env.num_atoms(), env.basis_data_ptr(),
                               env.num_basis(), env.env_data_ptr());
    break;
  case Operator::quadrupole:
    libcint::int1e_rr_optimizer(&m_optimizer, env.atom_data_ptr(),
                                env.num_atoms(), env.basis_data_ptr(),
                                env.num_basis(), env.env_data_ptr());
    break;
  case Operator::octapole:
    libcint::int1e_rrr_optimizer(&m_optimizer, env.atom_data_ptr(),
                                 env.num_atoms(), env.basis_data_ptr(),
                                 env.num_basis(), env.env_data_ptr());
    break;
  case Operator::hexadecapole:
    libcint::int1e_rrrr_optimizer(&m_optimizer, env.atom_data_ptr(),
                                  env.num_atoms(), env.basis_data_ptr(),
                                  env.num_basis(), env.env_data_ptr());
    break;
  case Operator::rinv:
    libcint::int1e_rinv_optimizer(&m_optimizer, env.atom_data_ptr(),
                                  env.num_atoms(), env.basis_data_ptr(),
                                  env.num_basis(), env.env_data_ptr());
    break;
  }
}

void Optimizer::create1or2c_grad(IntegralEnvironment &env) {
  switch (m_op) {
  case Operator::coulomb:
    libcint::int2c2e_ip1_optimizer(&m_optimizer, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
    break;
  case Operator::nuclear:
    libcint::int1e_ipnuc_optimizer(&m_optimizer, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
    break;
  case Operator::kinetic:
    libcint::int1e_ipkin_optimizer(&m_optimizer, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
    break;
  case Operator::overlap:
    libcint::int1e_ipovlp_optimizer(&m_optimizer, env.atom_data_ptr(),
                                    env.num_atoms(), env.basis_data_ptr(),
                                    env.num_basis(), env.env_data_ptr());
    break;
  case Operator::rinv:
    libcint::int1e_iprinv_optimizer(&m_optimizer, env.atom_data_ptr(),
                                    env.num_atoms(), env.basis_data_ptr(),
                                    env.num_basis(), env.env_data_ptr());
    break;
  default:
    throw std::runtime_error(
        "Invalid operator for gradient in cint::Optimizer");
  }
}

void Optimizer::create3c(IntegralEnvironment &env) {
  switch (m_op) {
  case Operator::coulomb:
    libcint::int3c2e_optimizer(&m_optimizer, env.atom_data_ptr(),
                               env.num_atoms(), env.basis_data_ptr(),
                               env.num_basis(), env.env_data_ptr());
    break;
  default:
    throw std::runtime_error(
        "Invalid operator for 3-center integral optimizer");
  }
}

void Optimizer::create3c_grad(IntegralEnvironment &env) {
  switch (m_op) {
  case Operator::coulomb:
    libcint::int3c2e_ip1_optimizer(&m_optimizer, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
    break;
  default:
    throw std::runtime_error(
        "Invalid operator for gradient in 3-center integral cint::Optimizer");
  }
}

void Optimizer::create4c(IntegralEnvironment &env) {
  switch (m_op) {
  case Operator::coulomb:
    libcint::int2e_optimizer(&m_optimizer, env.atom_data_ptr(), env.num_atoms(),
                             env.basis_data_ptr(), env.num_basis(),
                             env.env_data_ptr());
    break;
  default:
    throw std::runtime_error(
        "Invalid operator for 4-center integral optimizer");
  }
}

void Optimizer::create4c_grad(IntegralEnvironment &env) {
  switch (m_op) {
  case Operator::coulomb:
    libcint::int2e_ip1_optimizer(&m_optimizer, env.atom_data_ptr(),
                                 env.num_atoms(), env.basis_data_ptr(),
                                 env.num_basis(), env.env_data_ptr());
    break;
  default:
    throw std::runtime_error(
        "Invalid operator for 4-center integral optimizer");
  }
}
} // namespace occ::qm::cint
