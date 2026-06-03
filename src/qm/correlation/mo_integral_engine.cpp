#include <memory>
#include <occ/core/log.h>
#include <occ/qm/correlation/mo_integral_engine.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

MOIntegralEngine::MOIntegralEngine(const IntegralEngine &ao_engine,
                                   const MolecularOrbitals &mo)
    : m_ao_engine(ao_engine), m_mo(mo) {
  setup_mo_coefficients();
}

void MOIntegralEngine::setup_mo_coefficients() {
  if (m_mo.n_alpha > m_mo.n_ao) {
    occ::log::error("Invalid orbital counts: n_alpha ({}) > n_ao ({})",
                    m_mo.n_alpha, m_mo.n_ao);
    m_n_occ = 0;
    m_n_virt = 0;
    return;
  }
  m_n_occ = m_mo.n_alpha;
  m_n_virt = m_mo.n_ao - m_mo.n_alpha;
  occ::log::debug("MOIntegralEngine: {} occupied, {} virtual orbitals", m_n_occ,
                  m_n_virt);
}

double MOIntegralEngine::compute_mo_eri(size_t i, size_t j, size_t k,
                                        size_t l) const {
  const auto &basis = m_ao_engine.aobasis();
  const auto &first_bf = basis.first_bf();
  const size_t nsh = basis.size();
  constexpr auto op = cint::Operator::coulomb;

  // Hoist the optimizer + buffer out of the shell loops (single allocation).
  auto env = m_ao_engine.env();
  occ::qm::cint::Optimizer opt(env, op, 4);
  auto buffer = std::make_unique<double[]>(env.buffer_size_2e());
  const bool spherical = basis.is_pure();

  double result = 0.0;
  for (size_t p_sh = 0; p_sh < nsh; ++p_sh) {
    const size_t p0 = first_bf[p_sh];
    for (size_t q_sh = 0; q_sh < nsh; ++q_sh) {
      const size_t q0 = first_bf[q_sh];
      for (size_t r_sh = 0; r_sh < nsh; ++r_sh) {
        const size_t r0 = first_bf[r_sh];
        for (size_t s_sh = 0; s_sh < nsh; ++s_sh) {
          const size_t s0 = first_bf[s_sh];

          std::array<int, 4> shell_idxs{
              static_cast<int>(p_sh), static_cast<int>(q_sh),
              static_cast<int>(r_sh), static_cast<int>(s_sh)};

          std::array<int, 4> dims;
          if (spherical) {
            dims = env.four_center_helper<op, Shell::Kind::Spherical>(
                shell_idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
          } else {
            dims = env.four_center_helper<op, Shell::Kind::Cartesian>(
                shell_idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
          }
          if (dims[0] < 0)
            continue;

          // libcint buffer layout: first index varies fastest.
          for (size_t s_idx = 0; s_idx < static_cast<size_t>(dims[3]); ++s_idx) {
            for (size_t r_idx = 0; r_idx < static_cast<size_t>(dims[2]);
                 ++r_idx) {
              for (size_t q_idx = 0; q_idx < static_cast<size_t>(dims[1]);
                   ++q_idx) {
                for (size_t p_idx = 0; p_idx < static_cast<size_t>(dims[0]);
                     ++p_idx) {
                  const size_t buf_idx =
                      p_idx +
                      dims[0] * (q_idx + dims[1] * (r_idx + dims[2] * s_idx));
                  const double ao = buffer[buf_idx];
                  result += m_mo.C(p0 + p_idx, i) * m_mo.C(q0 + q_idx, j) *
                            m_mo.C(r0 + r_idx, k) * m_mo.C(s0 + s_idx, l) * ao;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

} // namespace occ::qm
