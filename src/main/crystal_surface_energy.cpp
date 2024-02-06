#include <occ/core/log.h>
#include <occ/crystal/surface.h>
#include <occ/io/crystal_json.h>
#include <occ/io/gmf.h>
#include <occ/main/crystal_surface_energy.h>

namespace occ::main {

using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;

CrystalSurfaceEnergies calculate_crystal_surface_energies(
    const std::string &filename, const Crystal &crystal,
    const CrystalDimers &uc_dimers, int max_number_of_surfaces, int sign) {
    CrystalSurfaceEnergies result{crystal, {}, {}};

    crystal::CrystalSurfaceGenerationParameters params;
    io::GMFWriter gmf(crystal);
    gmf.set_title("created by occ-cg");
    gmf.set_name(filename);
    params.d_min = 0.1;
    params.unique = true;
    auto surfaces = crystal::generate_surfaces(crystal, params);
    log::debug("Top {} surfaces", max_number_of_surfaces);
    int number_of_surfaces = 0;
    constexpr double tolerance{1e-5};

    // find unique positions to consider
    Mat3N unique_positions(3, crystal.unit_cell_molecules().size());
    log::debug("Unique positions to check: {}", unique_positions.cols());
    {
        int i = 0;
        for (const auto &mol : crystal.unit_cell_molecules()) {
            unique_positions.col(i) = mol.centroid();
            i++;
        }
    }
    constexpr double KJ_PER_MOL_TO_J_PER_M2 = 0.16604390671;

    for (auto &surf : surfaces) {
        const auto hkl = surf.hkl();
        log::debug("{:-^72s}",
                   fmt::format("  {} {} {} surface  ", hkl.h, hkl.k, hkl.l));
        surf.print();
        auto cuts = surf.possible_cuts(unique_positions);
        log::debug("{} unique cuts", cuts.size());
        double min_shift = 0.0;
        double min_energy = std::numeric_limits<double>::max();
        bool found_valid_cut = false;

        for (const double &cut : cuts) {
            log::debug("\nCut @ {:.6f} * depth", cut);
            auto surface_cut_result =
                surf.count_crystal_dimers_cut_by_surface(uc_dimers, cut);
            size_t num_molecules = surface_cut_result.molecules.size();
            FacetEnergies f{hkl, cut,
                            surface_cut_result.unique_counts_above(uc_dimers),
                            0.0, surf.area()};
            log::debug("Num molecules = {} ({} in crystal uc)", num_molecules,
                       uc_dimers.molecule_neighbors.size());
            double surface_energy_a{0.0}, surface_energy_b{0.0};
            {
                surface_energy_a = surface_cut_result.total_above(uc_dimers);
                log::debug("Surface energy (A) (kJ/mol) = {:12.3f}",
                           surface_energy_a);
                surface_energy_a = sign *
                                   (surface_energy_a * 0.5 / surf.area()) *
                                   KJ_PER_MOL_TO_J_PER_M2;
                log::debug("Surface energy (A) (J/m^2)  = {:12.6f}",
                           surface_energy_a);

                f.energy = surface_energy_a;
            }

            result.facets.push_back(f);

            {
                surface_energy_b = surface_cut_result.total_below(uc_dimers);
                log::debug("Surface energy (B) (kJ/mol) = {:12.3f}",
                           surface_energy_b);
                surface_energy_b = sign *
                                   (surface_energy_b * 0.5 / surf.area()) *
                                   KJ_PER_MOL_TO_J_PER_M2;
                log::debug("Surface energy (B) (J/m^2)  = {:12.6f}",
                           surface_energy_b);
            }

            {
                double bulk_energy = surface_cut_result.total_bulk(uc_dimers);
                double slab_energy = surface_cut_result.total_slab(uc_dimers);
                log::debug("Bulk energy (kJ/mol)        = {:12.3f}",
                           bulk_energy);
                log::debug("Slab energy (kJ/mol)        = {:12.3f}",
                           slab_energy);
                double surface_energy = sign * 0.5 * KJ_PER_MOL_TO_J_PER_M2 *
                                        (bulk_energy - slab_energy) /
                                        (surf.area() * 2);
                if (std::abs(surface_energy - surface_energy_a) > tolerance ||
                    std::abs(surface_energy - surface_energy_b) > tolerance) {
		    const auto &dipole = surf.dipole();
                    log::warn(
                        "Discrepency in surface energies for ({} {} {}) cut {:.3f}",
			hkl.h, hkl.k, hkl.l, cut
		    );
		    log::warn("bulk - slab yields different results to cut above or below");
		    log::warn("Surface dipole: D = ({:.3f}, {:.3f}, {:.3f})", dipole(0), dipole(1), dipole(2));
                }

                log::debug("Surface energy (S) (J/m^2)  = {:12.6f}",
                           surface_energy);

                if ((surface_energy > 0.0) && (surface_energy < min_energy)) {
                    min_shift = cut;
                    min_energy = surface_energy;
                    found_valid_cut = true;
                } else if (surface_energy <= 0.0) {
                    log::warn("Invalid surface energy encountered: "
                              "surface ({}, {}, {}), e = {:12.6f}",
                              hkl.h, hkl.k, hkl.l, surface_energy);
                }
            }
        }

        if (found_valid_cut) {
            log::info("({} {} {}) cut = {:6.3f}, energy = {:9.6f}", hkl.h,
                      hkl.k, hkl.l, min_shift, min_energy);
            io::GMFWriter::Facet facet{hkl, min_shift, 1, 1, min_energy};
            gmf.add_facet(facet);
        } else {
            log::warn("No valid (energy > 0) cuts for surface {} {} {}", hkl.h,
                      hkl.k, hkl.l);
        }

        number_of_surfaces++;
        if (number_of_surfaces >= max_number_of_surfaces)
            break;
    }
    gmf.write(filename);
    return result;
}

void to_json(nlohmann::json &j, const FacetEnergies &facet) {
    j["hkl"] = {facet.hkl.h, facet.hkl.k, facet.hkl.l};
    j["offset"] = facet.offset;
    j["area"] = facet.area;
    j["energy"] = facet.energy;
    j["interaction_energy_counts"] = facet.interaction_energy_counts;
}

void to_json(nlohmann::json &j,
             const CrystalSurfaceEnergies &surface_energies) {
    j["crystal"] = surface_energies.crystal;
    nlohmann::json facets;
    for (const auto &facet : surface_energies.facets) {
        facets.push_back(facet);
    }
    j["facets"] = facets;
}
} // namespace occ::main
