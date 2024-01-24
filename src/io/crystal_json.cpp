#include <occ/io/crystal_json.h>
#include <occ/io/eigen_json.h>
#include <occ/io/core_json.h>

namespace occ::crystal {

void to_json(nlohmann::json &j, const Crystal &crystal) {
    j["asymmetric unit"] = crystal.asymmetric_unit();
    j["space group"] = crystal.space_group();
    j["unit cell"] = crystal.unit_cell();
    j["unit cell atoms"] = crystal.unit_cell_atoms();
    const auto &connectivity = crystal.unit_cell_connectivity();
    nlohmann::json connections;
    nlohmann::json edges;
    connections["number of edges"] = connectivity.num_edges();
    for (const auto &[edge_descriptor, edge] : connectivity.edges()) {
        nlohmann::json e;
        e["distance"] = edge.dist;
        e["source"] = edge.source;
        e["target"] = edge.target;
        e["source asym"] = edge.source_asym_idx;
        e["target asym"] = edge.target_asym_idx;
        nlohmann::json shift;
        shift.push_back(edge.h);
        shift.push_back(edge.k);
        shift.push_back(edge.l);
        e["shift"] = shift;
        edges.push_back(e);
    }
    connections["edges"] = edges;

    j["unit cell connectivity"] = connections;
}

void to_json(nlohmann::json &j, const SpaceGroup &sg) {
    j["symbol"] = sg.symbol();
    j["short name"] = sg.short_name();
    j["number"] = sg.number();
    nlohmann::json symops;
    for (const auto &symop : sg.symmetry_operations()) {
        symops.push_back(symop);
    }
    j["symmetry_operations"] = symops;
}

void to_json(nlohmann::json &j, const SymmetryOperation &symop) {
    j["seitz"] = symop.seitz();
    j["integer_code"] = symop.to_int();
    j["string_code"] = symop.to_string();
}

void to_json(nlohmann::json &j, const AsymmetricUnit &asym) {
    j["site count"] = asym.atomic_numbers.rows();
    j["labels"] = asym.labels;
    j["atomic numbers"] = asym.atomic_numbers.transpose();
    j["positions"] = asym.positions;
    if (asym.occupations.rows() > 0)
        j["occupations"] = asym.occupations.transpose();
    if (asym.charges.rows() > 0)
        j["charges"] = asym.charges.transpose();
}

void to_json(nlohmann::json &j, const UnitCell &uc) {
    j["direct_matrix"] = uc.direct();
    j["reciprocal_matrix"] = uc.reciprocal();
}

void to_json(nlohmann::json &j, const CrystalAtomRegion &region) {
    j["site count"] = region.size();
    j["fractional positions"] = region.frac_pos;
    j["cartesian positions"] = region.cart_pos;
    j["asymmetric atom index"] = region.asym_idx.transpose();
    j["atomic numbers"] = region.atomic_numbers.transpose();
    j["symmetry operation"] = region.symop.transpose();
}


void to_json(nlohmann::json &j, const CrystalDimers &dimers) {
    j["generation radius"] = dimers.radius;

    nlohmann::json unique_dimers;
    for(const auto &dimer: dimers.unique_dimers) {
	nlohmann::json d = dimer;
	unique_dimers.push_back(d);
    }

    nlohmann::json neighbors_json;
    for(const auto &neighbors: dimers.molecule_neighbors) {
	nlohmann::json molecule_neighbors;
	nlohmann::json molecule_neighbor_idxs;
        for(const auto &[dimer, idx]: neighbors) {
	    molecule_neighbors.push_back(dimer);
	    molecule_neighbor_idxs.push_back(idx);
	}

	nlohmann::json neighbor{
	    {"dimers", molecule_neighbors},
	    {"unique dimer index", molecule_neighbor_idxs}
	};
	neighbors_json.push_back(neighbor);
    }
    
    j["neighbors"] = neighbors_json;

}

} // namespace occ::crystal
