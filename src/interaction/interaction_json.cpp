#include <occ/interaction/interaction_json.h>

namespace occ::interaction {

void to_json(nlohmann::json &j, const CEEnergyComponents &c) {
  j["Coulomb"] = c.coulomb;
  j["Exchange"] = c.exchange;
  j["Repulsion"] = c.repulsion;
  j["Dispersion"] = c.dispersion;
  j["Polarization"] = c.polarization;
  j["Total"] = c.total;
}

void from_json(const nlohmann::json &j, CEEnergyComponents &c) {
  if (j.contains("Coulomb"))
    j.at("Coulomb").get_to(c.coulomb);
  if (j.contains("coulomb"))
    j.at("coulomb").get_to(c.coulomb);

  if (j.contains("Exchange"))
    j.at("Exchange").get_to(c.exchange);
  if (j.contains("exchange"))
    j.at("exchange").get_to(c.exchange);

  if (j.contains("Repulsion"))
    j.at("Repulsion").get_to(c.repulsion);
  if (j.contains("repulsion"))
    j.at("repulsion").get_to(c.repulsion);

  if (j.contains("Dispersion"))
    j.at("Dispersion").get_to(c.dispersion);
  if (j.contains("dispersion"))
    j.at("dispersion").get_to(c.dispersion);

  if (j.contains("Polarization"))
    j.at("Polarization").get_to(c.polarization);
  if (j.contains("polarization"))
    j.at("polarization").get_to(c.polarization);

  if (j.contains("Total"))
    j.at("Total").get_to(c.total);
  if (j.contains("total"))
    j.at("total").get_to(c.total);
}

} // namespace occ::interaction
