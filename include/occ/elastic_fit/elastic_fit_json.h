#pragma once
#include <occ/elastic_fit/elastic_fitting.h>
#include <nlohmann/json.hpp>
#include <string>

namespace occ::elastic_fit {

// JSON serialization
void to_json(nlohmann::json &j, const MoleculeInput &m);
void from_json(const nlohmann::json &j, MoleculeInput &m);

void to_json(nlohmann::json &j, const PairInput &p);
void from_json(const nlohmann::json &j, PairInput &p);

void to_json(nlohmann::json &j, const ElasticFitInput &input);
void from_json(const nlohmann::json &j, ElasticFitInput &input);

// File I/O
void write_elastic_fit_json(const std::string &filename,
                            const ElasticFitInput &input);
ElasticFitInput read_elastic_fit_json(const std::string &filename);

} // namespace occ::elastic_fit
