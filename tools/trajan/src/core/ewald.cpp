#include <trajan/core/ewald.h>

#include <trajan/3rdparty/pocketfft.h>

namespace trajan::core {

Ewald::Ewald(double alpha)
    : m_alpha(alpha) {}

void Ewald::update(const Mat3N& positions, const std::vector<double>& charges, const UnitCell& unit_cell) {
    m_positions = &positions;
    m_charges = &charges;
    m_unit_cell = &unit_cell;
}

double Ewald::reciprocal_sum(bool use_fft) const {
    if (use_fft) {
        const auto structure_factor = calculate_structure_factor_fft();
        const double volume = m_unit_cell->volume();
        const int grid_size = 64; // Should be configurable

        double sum = 0.0;
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                for (int k = 0; k < grid_size; ++k) {
                    const Eigen::Vector3d k_vector_frac(i, j, k);
                    const auto k_vector = m_unit_cell->to_reciprocal(k_vector_frac);
                    const double k_squared = k_vector.squaredNorm();

                    if (k_squared > 1e-6) { // Skip k=0
                        const int grid_index = k + grid_size * (j + grid_size * i);
                        const double term = std::exp(-k_squared / (4.0 * m_alpha * m_alpha)) / k_squared;
                        sum += std::norm(structure_factor[grid_index]) * term;
                    }
                }
            }
        }

        return (1.0 / (2.0 * volume)) * 4.0 * M_PI * sum;
    } else {
        const auto structure_factor = calculate_structure_factor_direct();
        const auto& reciprocal_vectors = m_unit_cell->reciprocal();
        const double volume = m_unit_cell->volume();

        double sum = 0.0;
        for (int i = 0; i < reciprocal_vectors.cols(); ++i) {
            const auto k_vector = reciprocal_vectors.col(i);
            const double k_squared = k_vector.squaredNorm();

            if (k_squared > 1e-6) { // Skip k=0
                const double term = std::exp(-k_squared / (4.0 * m_alpha * m_alpha)) / k_squared;
                sum += std::norm(structure_factor[i]) * term;
            }
        }

        return (1.0 / (2.0 * volume)) * 4.0 * M_PI * sum;
    }
}

std::vector<std::complex<double>> Ewald::calculate_structure_factor_fft() const {
    const int grid_size = 64; // Should be configurable
    std::vector<std::complex<double>> charge_grid(grid_size * grid_size * grid_size, {0.0, 0.0});

    const auto& positions = *m_positions;
    const auto& charges = *m_charges;
    const auto& unit_cell = *m_unit_cell;

    for (int i = 0; i < positions.cols(); ++i) {
        const auto fractional_pos = unit_cell.to_fractional(positions.col(i));
        const int ix = static_cast<int>(fractional_pos.x() * grid_size) % grid_size;
        const int iy = static_cast<int>(fractional_pos.y() * grid_size) % grid_size;
        const int iz = static_cast<int>(fractional_pos.z() * grid_size) % grid_size;
        const int grid_index = iz + grid_size * (iy + grid_size * ix);
        charge_grid[grid_index] += charges[i];
    }

    pocketfft::shape_t shape = {grid_size, grid_size, grid_size};
    pocketfft::stride_t stride = {sizeof(std::complex<double>) * grid_size * grid_size, sizeof(std::complex<double>) * grid_size, sizeof(std::complex<double>)};
    pocketfft::c2c(shape, stride, stride, {0, 1, 2}, true, charge_grid.data(), charge_grid.data(), 1.0);

    return charge_grid;
}

std::vector<std::complex<double>> Ewald::calculate_structure_factor_direct() const {
    const auto& reciprocal_vectors = m_unit_cell->reciprocal();
    const auto& positions = *m_positions;
    const auto& charges = *m_charges;

    const int num_k_vectors = reciprocal_vectors.cols();
    std::vector<std::complex<double>> structure_factor(num_k_vectors);

    for (int i = 0; i < num_k_vectors; ++i) {
        const auto k_vector = reciprocal_vectors.col(i);
        std::complex<double> s_k(0.0, 0.0);
        for (int j = 0; j < positions.cols(); ++j) {
            const auto r_j = positions.col(j);
            const double k_dot_r = k_vector.dot(r_j);
            s_k += charges[j] * std::exp(std::complex<double>(0.0, k_dot_r));
        }
        structure_factor[i] = s_k;
    }

    return structure_factor;
}

} // namespace trajan::core