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

void IntegralEnvironment::set_common_origin(const std::array<double, 3> &origin) {
  m_env_data[libcint::common_origin_offset] = origin[0];
  m_env_data[libcint::common_origin_offset + 1] = origin[1];
  m_env_data[libcint::common_origin_offset + 2] = origin[2];
}

void IntegralEnvironment::set_rinv_origin(const std::array<double, 3> &origin) {
  m_env_data[libcint::rinv_origin_offset] = origin[0];
  m_env_data[libcint::rinv_origin_offset + 1] = origin[1];
  m_env_data[libcint::rinv_origin_offset + 2] = origin[2];
}

void IntegralEnvironment::print() const {
  fmt::print("Atom Info {}\n", m_atom_info.size());
  for (const auto &atom : m_atom_info) {
    fmt::print("{} {} {} {} {} {}\n", atom.data[0], atom.data[1],
                atom.data[2], atom.data[3], atom.data[4], atom.data[5]);
  }
  fmt::print("Basis Info {}\n", m_basis_info.size());
  for (const auto &sh : m_basis_info) {
    fmt::print("{} {} {} {} {} {}\n", sh.data[0], sh.data[1], sh.data[2],
                sh.data[3], sh.data[4], sh.data[5], sh.data[6], sh.data[7]);
  }
  fmt::print("Env Data {}\n", m_env_data.size());
  for (size_t i = 0; i < m_env_data.size(); i++) {
    fmt::print("{:12.6f} ", m_env_data[i]);
    if (i > 0 && (i % 6 == 0))
      fmt::print("\n");
  }
  fmt::print("\n");
}

} // namespace occ::qm::cint
