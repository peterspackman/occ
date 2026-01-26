#include <occ/mults/multipole_interactions.h>
#include <cmath>
#include <stdexcept>

namespace occ::mults {

MultipoleInteractions::MultipoleInteractions(const Config& config)
    : m_config(config),
      m_builder(config.max_rank, config.zero_tolerance),
      m_evaluator(config.max_rank)
{
}

double MultipoleInteractions::compute_esp(
    const occ::dma::Mult& multipole,
    const Vec3& site_position,
    const Vec3& eval_point) const
{
    // ESP is interaction energy with unit charge at eval_point
    // Create unit charge multipole (rank 0, Q00 = 1.0)
    occ::dma::Mult unit_charge(0);
    unit_charge.q(0) = 1.0;

    // Use interaction energy calculation
    return compute_interaction_energy(multipole, site_position,
                                     unit_charge, eval_point);
}

std::vector<double> MultipoleInteractions::compute_esp_grid(
    const occ::dma::Mult& multipole,
    const Vec3& site_position,
    Mat3NConstRef eval_points) const
{
    const int num_points = eval_points.cols();
    std::vector<double> results;
    results.reserve(num_points);

    // Build term list once for the multipole vs unit charge interaction
    occ::dma::Mult unit_charge(0);
    unit_charge.q(0) = 1.0;

    // Use max_interaction_rank if specified, otherwise use max_rank (Orient convention)
    int effective_max_interaction_rank = m_config.max_interaction_rank >= 0 ?
        m_config.max_interaction_rank : m_config.max_rank;
    auto term_list = m_builder.build_electrostatic_terms(
        multipole, unit_charge, effective_max_interaction_rank);

    // Evaluate at each point (each column)
    for (int i = 0; i < num_points; ++i) {
        Vec3 point = eval_points.col(i);
        Vec3 rab = point - site_position;
        double r = rab.norm();

        if (r < 1e-15) {
            throw std::runtime_error("Evaluation point too close to multipole site");
        }

        // Set coordinates and evaluate batch
        m_evaluator.set_coordinates(site_position, point);
        auto sfunction_results = m_evaluator.compute_batch(term_list, 0);

        // Accumulate energy
        double esp = accumulate_energy(term_list, sfunction_results, r);
        results.push_back(esp);
    }

    return results;
}

double MultipoleInteractions::compute_interaction_energy(
    const occ::dma::Mult& mult1, const Vec3& pos1,
    const occ::dma::Mult& mult2, const Vec3& pos2) const
{
    // Compute inter-site distance
    Vec3 rab = pos2 - pos1;
    double r = rab.norm();

    if (r < 1e-15) {
        throw std::runtime_error("Sites too close for interaction energy calculation");
    }

    // Build filtered term list (skips zero Q1*Q2 terms)
    // Use max_interaction_rank if specified, otherwise use max_rank (Orient convention)
    int effective_max_interaction_rank = m_config.max_interaction_rank >= 0 ?
        m_config.max_interaction_rank : m_config.max_rank;
    auto term_list = m_builder.build_electrostatic_terms(
        mult1, mult2, effective_max_interaction_rank);

    // Set up coordinates for S-function evaluation
    m_evaluator.set_coordinates(pos1, pos2);

    // Batch evaluate all non-zero S-functions
    auto results = m_evaluator.compute_batch(term_list, 0);

    // Accumulate energy contributions
    return accumulate_energy(term_list, results, r);
}

double MultipoleInteractions::accumulate_energy(
    const SFunctionTermList& terms,
    const std::vector<SFunctionResult>& results,
    double r)
{
    if (terms.size() != results.size()) {
        throw std::runtime_error("Term list and result size mismatch");
    }

    double energy = 0.0;

    for (size_t i = 0; i < terms.size(); ++i) {
        const auto& term = terms.terms[i];
        const auto& result = results[i];

        // Energy contribution: coeff * S(t1,t2,j) / r^power
        // coeff already contains q1 * q2 * binomial(j, l1)
        double r_power = std::pow(r, term.power);
        energy += term.coeff * result.s0 / r_power;
    }

    return energy;
}

} // namespace occ::mults
