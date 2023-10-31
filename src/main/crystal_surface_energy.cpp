#include <fmt/core.h>
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

    occ::crystal::CrystalSurfaceGenerationParameters params;
    occ::io::GMFWriter gmf(crystal);
    gmf.set_title("created by occ-cg");
    gmf.set_name(filename);
    params.d_min = 0.1;
    params.unique = true;
    auto surfaces = occ::crystal::generate_surfaces(crystal, params);
    fmt::print("Top {} surfaces\n", max_number_of_surfaces);
    int number_of_surfaces = 0;

    // find unique positions to consider
    occ::Mat3N unique_positions(3, crystal.unit_cell_molecules().size());
    fmt::print("Unique positions to check: {}\n", unique_positions.cols());
    {
        int i = 0;
        for (const auto &mol : crystal.unit_cell_molecules()) {
            unique_positions.col(i) = mol.centroid();
            i++;
        }
    }
    const double kjmol_jm2_fac = 0.16604390671;

    for (auto &surf : surfaces) {
        fmt::print("{:=^72s}\n", "  Surface  ");
        surf.print();
        auto cuts = surf.possible_cuts(unique_positions);
        fmt::print("{} unique cuts\n", cuts.size());
        double min_shift = 0.0;
        double min_energy = std::numeric_limits<double>::max();

        for (const double &cut : cuts) {
            fmt::print("cut offset = {:.6f} * depth\n", cut);
            auto surface_cut_result =
                surf.count_crystal_dimers_cut_by_surface(uc_dimers, cut);
            size_t num_molecules = surface_cut_result.molecules.size();
            FacetEnergies f{surf.hkl(), cut,
                            surface_cut_result.unique_counts_above(uc_dimers),
                            0.0, surf.area()};
            fmt::print("Num molecules = {} ({} in crystal uc)\n", num_molecules,
                       uc_dimers.molecule_neighbors.size());
            {
                double surface_energy_a =
                    surface_cut_result.total_above(uc_dimers);
                fmt::print("Surface energy (A) (kJ/mol) = {:12.3f}\n",
                           surface_energy_a);
                surface_energy_a = sign *
                                   (surface_energy_a * 0.5 / surf.area()) *
                                   kjmol_jm2_fac;
                fmt::print("Surface energy (A) (J/m^2)  = {:12.6f}\n",
                           surface_energy_a);

                f.energy = surface_energy_a;
            }

            result.facets.push_back(f);

            {
                double surface_energy_b =
                    surface_cut_result.total_below(uc_dimers);
                fmt::print("Surface energy (B) (kJ/mol) = {:12.3f}\n",
                           surface_energy_b);
                surface_energy_b = sign *
                                   (surface_energy_b * 0.5 / surf.area()) *
                                   kjmol_jm2_fac;
                fmt::print("Surface energy (B) (J/m^2)  = {:12.6f}\n",
                           surface_energy_b);
            }

            {
                double bulk_energy = surface_cut_result.total_bulk(uc_dimers);
                double slab_energy = surface_cut_result.total_slab(uc_dimers);
                fmt::print("Bulk energy (kJ/mol)        = {:12.3f}\n",
                           bulk_energy);
                fmt::print("Slab energy (kJ/mol)        = {:12.3f}\n",
                           slab_energy);
                double surface_energy = sign * 0.5 * kjmol_jm2_fac *
                                        (bulk_energy - slab_energy) /
                                        (surf.area() * 2);
                fmt::print("Surface energy (S) (J/m^2)  = {:12.6f}\n",
                           surface_energy);
                if (surface_energy < min_energy) {
                    min_shift = cut;
                    min_energy = surface_energy;
                }
            }
        }

        occ::io::GMFWriter::Facet facet{surf.hkl(), min_shift, 1, 1,
                                        min_energy};
        gmf.add_facet(facet);

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
