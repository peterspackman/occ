#include <fmt/core.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/element.h>
#include <occ/core/format_matrix.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/molecular_axis.h>
#include <occ/core/units.h>
#include <occ/driver/dma_driver.h>
#include <sstream>

namespace occ::driver {

// Convert string axis method to enum for the new API
occ::core::AxisMethod string_to_axis_method(const std::string& method_str) {
    return occ::core::MolecularAxisCalculator::string_to_axis_method(method_str);
}

DMADriver::DMAOutput DMADriver::run() {
    if (m_config.wavefunction_filename.empty()) {
        throw std::runtime_error("Wavefunction filename not set");
    }

    occ::log::info("Loading wavefunction from: {}", m_config.wavefunction_filename);
    auto wfn = occ::qm::Wavefunction::load(m_config.wavefunction_filename);
    return run(wfn);
}

DMADriver::DMAOutput DMADriver::run(const occ::qm::Wavefunction& wfn) {
    // Create a copy of the wavefunction that we can modify
    occ::qm::Wavefunction oriented_wfn = wfn;

    DMAOutput output;

    // Accumulate every transform applied to the wavefunction so the frame the
    // results are reported in can be stated exactly. Composing
    // x -> R2*(R1*x + t1) + t2 gives (R2*R1, R2*t1 + t2).
    auto apply = [&](const occ::Mat3 &rotation, const occ::Vec3 &translation) {
        output.rotation = rotation * output.rotation;
        output.translation = rotation * output.translation + translation;
        oriented_wfn.apply_transformation(rotation, translation);
    };

    // User-supplied rigid-body transform, for reusing a wavefunction computed
    // on a symmetry-equivalent copy of this fragment. Applied first, so the
    // axis method below sees the molecule where the user placed it.
    const bool has_wfn_rotation = !m_config.wfn_rotation.isIdentity(1e-12);
    const bool has_wfn_translation = !m_config.wfn_translation.isZero(1e-12);
    if (has_wfn_rotation || has_wfn_translation) {
        occ::log::info("Applying supplied transformation to wavefunction");
        occ::log::debug("Rotation\n{}", format_matrix(m_config.wfn_rotation));
        occ::log::debug("Translation (Angstrom) [{:.6f}, {:.6f}, {:.6f}]",
                        m_config.wfn_translation.x(), m_config.wfn_translation.y(),
                        m_config.wfn_translation.z());
        apply(m_config.wfn_rotation,
              m_config.wfn_translation * occ::units::ANGSTROM_TO_BOHR);
    }

    // Multipoles for CSP programs are expected in the principal axis frame, so
    // default to MOI rather than silently reporting a lab-frame result. Done
    // here, not in the CLI, so library callers behave the same way.
    std::string axis_method = m_config.axis_method;
    if (!m_config.csp_input_filename.empty() && axis_method == "none") {
        axis_method = "moi";
        occ::log::info("CSP output requested: orienting molecule on principal axes");
    }
    output.axis_method = axis_method;
    output.axis_atoms = m_config.axis_atoms;

    // Apply molecular orientation if requested
    if (axis_method != "none") {
        occ::log::info("Applying molecular orientation using {} method", axis_method);

        occ::core::MolecularAxisCalculator axis_calc(oriented_wfn.positions(),
                                                     oriented_wfn.atomic_numbers());
        occ::core::AxisMethod method = string_to_axis_method(axis_method);

        // Calculate molecular axes
        occ::core::MolecularAxisResult axis_result;
        if (method == occ::core::AxisMethod::Neighcrys) {
            std::vector<int> atoms_to_use = m_config.axis_atoms;
            if (atoms_to_use.empty()) {
                // Default to first 3 atoms if no axis atoms specified
                if (oriented_wfn.positions().cols() < 3) {
                    throw std::runtime_error("Neighcrys axis method requires at least 3 atoms");
                }
                atoms_to_use = {0, 1, 2};
                occ::log::info("Using default axis atoms: 0, 1, 2");
            } else {
                occ::log::info("Using specified axis atoms: {}, {}, {}",
                              atoms_to_use[0], atoms_to_use[1], atoms_to_use[2]);
            }
            axis_result = axis_calc.calculate_axes(method, atoms_to_use);
            output.axis_atoms = atoms_to_use;
        } else {
            axis_result = axis_calc.calculate_axes(method);
            if (method == occ::core::AxisMethod::PCA) {
                occ::log::info("Using PCA-based molecular axes");
            } else if (method == occ::core::AxisMethod::MOI) {
                occ::log::info("Using moment of inertia-based molecular axes");
            }
            if (!axis_result.axis_atoms.empty())
                output.axis_atoms = axis_result.axis_atoms;
        }

        // Centring on the centre of mass means x -> A*(x - com), so the
        // translation is -A*com; -com alone leaves an offset of (A*com - com).
        const occ::Vec3 translation = -(axis_result.axes * axis_result.center_of_mass);
        apply(axis_result.axes, translation);

        occ::log::debug("Applied rotation matrix (det = {}):\n{}",
                        axis_result.determinant, format_matrix(axis_result.axes));
        occ::log::debug("Applied translation: [{:.6f}, {:.6f}, {:.6f}]",
                        translation.x(), translation.y(), translation.z());
    }

    // Write the geometry DMA is about to analyse. Requesting this without an
    // axis frame is not an error -- it is still the analysed geometry.
    if (m_config.write_oriented_xyz || !m_config.oriented_xyz_filename.empty()) {
        std::string xyz_filename = m_config.oriented_xyz_filename.empty() ?
                                 "oriented.xyz" : m_config.oriented_xyz_filename;
        occ::core::MolecularAxisCalculator::write_oriented_xyz(
            xyz_filename, oriented_wfn.positions(), oriented_wfn.atomic_numbers());
        if (axis_method == "none")
            occ::log::info("Wrote geometry to: {} (axis method is 'none', so it "
                           "is the input geometry unchanged)", xyz_filename);
        else
            occ::log::info("Wrote oriented molecule to: {}", xyz_filename);
    }

    // Write neighcrys axis file if requested
    if (m_config.write_axis_file || !m_config.axis_filename.empty()) {
        std::string axis_filename = m_config.axis_filename.empty() ?
                                  "molecule.mols" : m_config.axis_filename;

        std::vector<int> axis_atoms_for_file = output.axis_atoms;
        if (axis_atoms_for_file.empty() && oriented_wfn.positions().cols() >= 3) {
            axis_atoms_for_file = {0, 1, 2};
        }

        occ::core::MolecularAxisCalculator oriented_axis_calc(
            oriented_wfn.positions(), oriented_wfn.atomic_numbers());
        auto axis_info = oriented_axis_calc.generate_neighcrys_info(axis_atoms_for_file);
        occ::core::MolecularAxisCalculator::write_neighcrys_axis_file(axis_filename, axis_info);
        occ::log::info("Wrote neighcrys axis file to: {}", axis_filename);
    }

    occ::dma::DMACalculator calc(oriented_wfn);
    calc.update_settings(m_config.settings);

    // Per-element overrides. These are applied after update_settings(), which
    // is what applies the max_rank clamp, so clamp explicitly here.
    for (const auto& [element, radius] : m_config.atom_radii) {
        int atomic_number = occ::core::Element(element).atomic_number();
        calc.set_radius_for_element(atomic_number, radius);
        occ::log::debug("Setting radius for {} to {:.3f} Angstrom", element, radius);
    }

    for (const auto& [element, limit] : m_config.atom_limits) {
        int atomic_number = occ::core::Element(element).atomic_number();
        int effective = std::min(limit, m_config.settings.max_rank);
        if (effective != limit) {
            occ::log::info("Rank limit for {} reduced from {} to {} "
                           "(--max-rank is {})", element, limit, effective,
                           m_config.settings.max_rank);
        }
        calc.set_limit_for_element(atomic_number, effective);
        occ::log::debug("Setting max rank for {} to {}", element, effective);
    }

    // Set default H settings if not specified
    if (m_config.atom_radii.find("H") == m_config.atom_radii.end()) {
        calc.set_radius_for_element(1, 0.35);
    }
    if (m_config.atom_limits.find("H") == m_config.atom_limits.end()) {
        calc.set_limit_for_element(1, std::min(1, m_config.settings.max_rank));
    }

    occ::log::debug("Running DMA calculation with max_rank={}, big_exponent={}",
                    m_config.settings.max_rank, m_config.settings.big_exponent);

    output.result = calc.compute_multipoles();
    output.sites = calc.sites();
    // Must come from this calculator: it shifts the site multipoles along the
    // site vectors, so both have to be in the same frame.
    output.total = calc.compute_total_multipoles(output.result);

    if (m_config.write_punch && !m_config.punch_filename.empty()) {
        write_punch_file(m_config.punch_filename, output.result, output.sites);
        occ::log::info("Wrote punch file to: {}", m_config.punch_filename);
    }

    if (!m_config.json_filename.empty()) {
        write_json(m_config.json_filename, m_config, output);
        occ::log::info("Wrote JSON results to: {}", m_config.json_filename);
    }

    return output;
}

std::string DMADriver::generate_punch_file(const occ::dma::DMAResult& result,
                                          const occ::dma::DMASites& sites) {
    std::stringstream ss;

    ss << fmt::format("! Distributed multipoles from occ dma\n");
    ss << fmt::format("! Max rank: {}\n", result.max_rank);
    ss << fmt::format("\n");
    ss << fmt::format("Units angstrom\n\n");

    // Write individual site multipoles
    for (int i = 0; i < result.multipoles.size(); i++) {
        const auto& m = result.multipoles[i];
        const auto pos = sites.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;

        ss << fmt::format("{:<8s} {:12.8f} {:12.8f} {:12.8f}\n",
                         sites.name[i], pos.x(), pos.y(), pos.z());
        ss << fmt::format("Rank {}\n", m.max_rank);

        // Write multipoles in order: Q00, Q10, Q11c, Q11s, Q20, Q21c, Q21s, Q22c, Q22s, etc.
        int idx = 0;
        for (int rank = 0; rank <= m.max_rank; rank++) {
            int num_components = 2 * rank + 1;
            for (int comp = 0; comp < num_components; comp++) {
                ss << fmt::format(" {:16.10f}", m.q(idx++));
                if ((comp + 1) % 3 == 0 || comp == num_components - 1) {
                    ss << "\n";
                }
            }
        }
        ss << "\n";
    }

    return ss.str();
}

void DMADriver::write_punch_file(const std::string& filename,
                                const occ::dma::DMAResult& result,
                                const occ::dma::DMASites& sites) {
    std::ofstream punch(filename);
    if (!punch.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open punch file: {}", filename));
    }

    punch << generate_punch_file(result, sites);
    punch.close();
}

namespace {

// Multipoles in the punch file's component order (Q00; Q10, Q11c, Q11s; ...),
// truncated to the rank actually present. Mult always allocates 121 components
// regardless of rank, so slicing on num_components() is required.
std::vector<double> multipole_components(const occ::dma::Mult &m) {
    const int n = m.num_components();
    return std::vector<double>(m.q.data(), m.q.data() + n);
}

} // namespace

std::string DMADriver::generate_json(const DMAConfig &config,
                                     const DMAOutput &output) {
    nlohmann::json j;
    j["program"] = "occ";
    j["kind"] = "dma";
    j["schema_version"] = 1;

    j["settings"] = {
        {"max_rank", config.settings.max_rank},
        {"big_exponent", config.settings.big_exponent},
        {"include_nuclei", config.settings.include_nuclei},
        {"axis_method_requested", config.axis_method},
        {"axis_method_applied", output.axis_method},
        {"axis_atoms", output.axis_atoms},
    };

    j["units"] = {{"position", "angstrom"}, {"multipole", "atomic"}};

    // Row-major, matching --wfn-rotation's input order.
    std::vector<double> rotation(9);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            rotation[3 * r + c] = output.rotation(r, c);
    const occ::Vec3 translation = output.translation * occ::units::BOHR_TO_ANGSTROM;
    j["transform"] = {
        {"rotation", rotation},
        {"translation", {translation.x(), translation.y(), translation.z()}},
    };

    const auto &sites = output.sites;
    nlohmann::json site_array = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(output.result.multipoles.size()); i++) {
        const auto &m = output.result.multipoles[i];
        const occ::Vec3 pos = sites.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;
        const int atom_index = sites.atom_indices(i);
        int atomic_number = 0;
        if (atom_index >= 0 && atom_index < static_cast<int>(sites.atoms.size()))
            atomic_number = sites.atoms[atom_index].atomic_number;

        site_array.push_back({
            {"name", sites.name[i]},
            {"atomic_number", atomic_number},
            {"atom_index", atom_index},
            {"position", {pos.x(), pos.y(), pos.z()}},
            // Effective values: defaults, then per-element overrides, then the
            // max_rank clamp. This is what the calculation actually used.
            {"radius", sites.radii(i) * occ::units::BOHR_TO_ANGSTROM},
            {"limit", sites.limits(i)},
            {"rank", m.max_rank},
            {"multipoles", multipole_components(m)},
        });
    }
    j["sites"] = site_array;

    j["total"] = {
        {"origin", {0.0, 0.0, 0.0}},
        {"rank", output.total.max_rank},
        {"multipoles", multipole_components(output.total)},
    };

    return j.dump(2) + "\n";
}

void DMADriver::write_json(const std::string &filename, const DMAConfig &config,
                           const DMAOutput &output) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(
            fmt::format("Failed to open JSON output file: {}", filename));
    }
    file << generate_json(config, output);
}

} // namespace occ::driver
