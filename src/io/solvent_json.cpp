#include <occ/io/solvent_json.h>

namespace occ::solvent {

void to_json(nlohmann::json &j, const SMDSolventParameters &params) {
    j["refractive index 293K"] = params.refractive_index_293K;
    j["refractive index 298K"] = params.refractive_index_293K;
    j["acidity"] = params.acidity;
    j["basicity"] = params.basicity;
    j["gamma"] = params.gamma;
    j["dielectric"] = params.dielectric;
    j["aromaticity"] = params.aromaticity;
    j["electronegative halogenicity"] = params.electronegative_halogenicity;
    j["is water"] = params.is_water;
}
    
void from_json(const nlohmann::json &j, SMDSolventParameters &params) {
    j.at("refractive index 293K").get_to(params.refractive_index_293K);
    j.at("refractive index 298K").get_to(params.refractive_index_293K);
    j.at("acidity").get_to(params.acidity);
    j.at("basicity").get_to(params.basicity);
    j.at("gamma").get_to(params.gamma);
    j.at("dielectric").get_to(params.dielectric);
    j.at("aromaticity").get_to(params.aromaticity);
    j.at("electronegative halogenicity").get_to(params.electronegative_halogenicity);
    j.at("is water").get_to(params.is_water);
}

}
