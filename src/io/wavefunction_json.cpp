#include <cctype>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/io/eigen_json.h>
#include <occ/io/json_basis.h>
#include <occ/io/wavefunction_json.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

void from_json(const nlohmann::json &J, occ::qm::MolecularOrbitals &mo) {
    bool valid_sk = occ::qm::get_spinorbital_kind_from_string(
        J.at("spinorbital kind"), mo.kind);
    if(! valid_sk) throw std::runtime_error("Found invalid spinorbital kind in JSON");

    occ::log::debug("JSON wavefunction spinorbital kind: {}", occ::qm::spinorbital_kind_to_string(mo.kind));

    J.at("alpha electrons").get_to(mo.n_alpha);
    J.at("beta electrons").get_to(mo.n_beta);
    J.at("atomic orbitals").get_to(mo.n_ao);
    J.at("orbital coefficients").get_to(mo.C);
    J.at("occupied orbital coefficients").get_to(mo.Cocc);
    J.at("density matrix").get_to(mo.D);
    J.at("orbital energies").get_to(mo.energies);
}

void to_json(nlohmann::json &J, const occ::qm::MolecularOrbitals &mo) {
    J["spinorbital kind"] = occ::qm::spinorbital_kind_to_string(mo.kind);
    J["alpha electrons"] = mo.n_alpha;
    J["beta electrons"] = mo.n_beta;
    J["atomic orbitals"] = mo.n_ao;
    J["orbital coefficients"] = mo.C;
    J["occupied orbital coefficients"] = mo.Cocc;
    J["density matrix"] = mo.D;
    J["orbital energies"] = mo.energies;
}

void from_json(const nlohmann::json &J, occ::qm::Energy &energy) {
    J.at("coulomb").get_to(energy.coulomb);
    J.at("exchange").get_to(energy.exchange);
    J.at("nuclear attraction").get_to(energy.nuclear_attraction);
    J.at("nuclear repulsion").get_to(energy.nuclear_repulsion);
    J.at("kinetic").get_to(energy.kinetic);
    J.at("core").get_to(energy.core);
    J.at("total").get_to(energy.total);
    J.at("effective core potential").get_to(energy.ecp);
}

void to_json(nlohmann::json &J, const occ::qm::Energy &energy) {
    J["coulomb"] = energy.coulomb;
    J["exchange"] = energy.exchange;
    J["nuclear attraction"] = energy.nuclear_attraction;
    J["nuclear repulsion"] = energy.nuclear_repulsion;
    J["kinetic"] = energy.kinetic;
    J["core"] = energy.core;
    J["total"] = energy.total;
    J["effective core potential"] = energy.ecp;
}

void to_json(nlohmann::json &J, const occ::qm::Shell &shell) {
    J["spherical"] = shell.kind == occ::qm::Shell::Kind::Spherical;
    J["l"] = shell.l;
    J["origin"] = shell.origin;
    J["exponents"] = shell.exponents;
    J["contraction coefficients"] = shell.contraction_coefficients;
    J["unnormalized contraction coefficients"] = shell.u_coefficients;
    if (shell.max_ln_coefficient.size() > 0) {
        J["max ln coefficient"] = shell.max_ln_coefficient;
    }
    if (shell.ecp_r_exponents.size() > 0) {
        J["ecp r exponents"] = shell.ecp_r_exponents;
    }
    J["extent"] = shell.extent;
}

void from_json(const nlohmann::json &J, occ::qm::Shell &shell) {
    if (J.at("spherical")) {
        shell.kind = occ::qm::Shell::Kind::Spherical;
    } else {
        shell.kind = occ::qm::Shell::Kind::Cartesian;
    }
    J.at("l").get_to(shell.l);
    J.at("origin").get_to(shell.origin);
    J.at("exponents").get_to(shell.exponents);
    J.at("contraction coefficients").get_to(shell.contraction_coefficients);
    J.at("unnormalized contraction coefficients").get_to(shell.u_coefficients);
    if (J.contains("max ln coefficient")) {
        J.at("max ln coefficient").get_to(shell.max_ln_coefficient);
    }
    if (J.contains("ecp r exponents")) {
        J.at("ecp r exponents").get_to(shell.ecp_r_exponents);
    }
    J.at("extent").get_to(shell.extent);
}

} // namespace occ::qm

namespace occ::core {
void from_json(const nlohmann::json &J, occ::core::Atom &atom) {
    J.at("n").get_to(atom.atomic_number);
    J.at("pos")[0].get_to(atom.x);
    J.at("pos")[1].get_to(atom.y);
    J.at("pos")[2].get_to(atom.z);
}

void to_json(nlohmann::json &J, const occ::core::Atom &atom) {
    J["n"] = atom.atomic_number;
    J["pos"] = {atom.x, atom.y, atom.z};
}
} // namespace occ::core

namespace occ::qm {
void from_json(const nlohmann::json &J, occ::qm::Wavefunction &wfn) {
    wfn.atoms.clear();
    J.at("electrons").get_to(wfn.num_electrons);
    J.at("frozen electrons").get_to(wfn.num_electrons);
    J.at("basis functions").get_to(wfn.nbf);

    J.at("molecular orbitals").get_to(wfn.mo);

    for (const auto &x : J.at("atoms")) {
        wfn.atoms.push_back(x);
    }
    occ::log::debug("Loaded atoms from json");

    std::vector<occ::qm::Shell> ao_shells;
    const auto &basis = J.at("orbital basis");
    for (const auto &x : basis.at("shells")) {
        ao_shells.push_back(x);
    }
    occ::log::debug("Loaded ao shells from json");

    std::vector<occ::qm::Shell> ecp_shells;
    std::vector<int> ecp_electrons;
    basis.at("ecp electrons").get_to(ecp_electrons);
    if (basis.contains("ecp shells")) {
        for (const auto &x : basis.at("ecp shells")) {
            ecp_shells.push_back(x);
        }
        occ::log::debug("Loaded ecp shells from json");
    }

    wfn.basis =
        occ::qm::AOBasis(wfn.atoms, ao_shells, basis.at("name"), ecp_shells);
    wfn.basis.set_ecp_electrons(ecp_electrons);

    if (J.contains("kinetic energy matrix")) {
        J.at("kinetic energy matrix").get_to(wfn.T);
    }

    if (J.contains("nuclear attraction matrix")) {
        J.at("nuclear attraction matrix").get_to(wfn.V);
    }

    if (J.contains("core hamiltonian matrix")) {
        J.at("core hamiltonian matrix").get_to(wfn.H);
    }

    if (J.contains("coulomb matrix")) {
        J.at("coulomb matrix").get_to(wfn.J);
    }

    if (J.contains("exchange matrix")) {
        J.at("exchange matrix").get_to(wfn.K);
    }

    if (J.contains("effective core potential matrix")) {
        J.at("effective core potential matrix").get_to(wfn.Vecp);
    }

    if (J.contains("energy")) {
        J.at("energy").get_to(wfn.energy);
        wfn.have_energies = true;
    }

    if (J.contains("xdm parameters")) {
        wfn.have_xdm_parameters = true;
        const auto &xdm = J.at("xdm parameters");
        xdm.at("polarizabilities").get_to(wfn.xdm_polarizabilities);
        xdm.at("moments").get_to(wfn.xdm_moments);
        xdm.at("volumes").get_to(wfn.xdm_volumes);
        xdm.at("free volumes").get_to(wfn.xdm_free_volumes);
        xdm.at("energy").get_to(wfn.xdm_energy);
    }
}

void to_json(nlohmann::json &J, const occ::qm::Wavefunction &wfn) {
    J["electrons"] = wfn.num_electrons;
    J["frozen electrons"] = wfn.num_electrons;
    J["basis functions"] = wfn.nbf;

    J["molecular orbitals"] = wfn.mo;

    J["atoms"] = wfn.atoms;

    nlohmann::json basis;
    basis["name"] = wfn.basis.name();
    basis["shells"] = wfn.basis.shells();
    basis["ecp electrons"] = wfn.basis.ecp_electrons();

    if (wfn.basis.ecp_shells().size() > 0) {
        basis["ecp shells"] = wfn.basis.ecp_shells();
    }

    J["orbital basis"] = basis;

    if (wfn.T.size() > 0) {
        J["kinetic energy matrix"] = wfn.T;
    }

    if (wfn.V.size() > 0) {
        J["nuclear attraction matrix"] = wfn.V;
    }

    if (wfn.H.size() > 0) {
        J["core hamiltonian matrix"] = wfn.H;
    }

    if (wfn.J.size() > 0) {
        J["coulomb matrix"] = wfn.J;
    }

    if (wfn.K.size() > 0) {
        J["exchange matrix"] = wfn.K;
    }

    if (wfn.Vecp.size() > 0) {
        J["effective core potential matrix"] = wfn.Vecp;
    }

    if (wfn.have_energies) {
        J["energy"] = wfn.energy;
    }

    if (wfn.have_xdm_parameters) {
        nlohmann::json xdm_params;
        xdm_params["polarizabilities"] = wfn.xdm_polarizabilities;
        xdm_params["moments"] = wfn.xdm_moments;
        xdm_params["volumes"] = wfn.xdm_volumes;
        xdm_params["free volumes"] = wfn.xdm_free_volumes;
        xdm_params["energy"] = wfn.xdm_energy;
        J["xdm parameters"] = xdm_params;
    }
}
} // namespace occ::qm

namespace occ::io {

JsonWavefunctionReader::JsonWavefunctionReader(const std::string &filename, JsonFormat fmt)
    : m_format(fmt), m_filename{filename} {
    occ::timing::start(occ::timing::category::io);
    std::ios_base::openmode mode = std::ios_base::out;
    switch(m_format) {
	case JsonFormat::JSON:
	    break;
	case JsonFormat::CBOR:    // fallthrough
	case JsonFormat::UBJSON:  // fallthrough
	case JsonFormat::MSGPACK: // fallthrough
	case JsonFormat::BSON:
	    mode |= std::ios_base::binary;
	    break;
    }
    std::ifstream file(filename, mode);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

JsonWavefunctionReader::JsonWavefunctionReader(std::istream &file) {

    occ::timing::start(occ::timing::category::io);
    parse(file);
    occ::timing::stop(occ::timing::category::io);
}

qm::Wavefunction JsonWavefunctionReader::wavefunction() const {
    return m_wavefunction;
}

void JsonWavefunctionReader::set_filename(const std::string &name) {
    m_filename = name;
}

void JsonWavefunctionReader::parse(std::istream &file) {
    nlohmann::json j;
    switch(m_format) {
	case JsonFormat::JSON:
	    file >> j;
	    break;
	case JsonFormat::UBJSON: {
	    std::vector<uint8_t> ubjson(std::istreambuf_iterator<char>(file), {});
	    j = nlohmann::json::from_ubjson(ubjson);
	    break;
        }
	case JsonFormat::CBOR: {
	    std::vector<uint8_t> cbor(std::istreambuf_iterator<char>(file), {});
	    j = nlohmann::json::from_cbor(cbor);
	    break;
	}
	case JsonFormat::MSGPACK: {
	    std::vector<uint8_t> msgpack(std::istreambuf_iterator<char>(file), {});
	    j = nlohmann::json::from_msgpack(msgpack);
	    break;
	}
	case JsonFormat::BSON: {
	    std::vector<uint8_t> bson(std::istreambuf_iterator<char>(file), {});
	    j = nlohmann::json::from_bson(bson);
	    break;
	}
    }
    j.get_to(m_wavefunction);
}

JsonWavefunctionWriter::JsonWavefunctionWriter() {}

void JsonWavefunctionWriter::set_shiftwidth(int shiftwidth) {
    m_shiftwidth = shiftwidth;
}

std::string
JsonWavefunctionWriter::to_string(const qm::Wavefunction &wfn) const {
    occ::timing::start(occ::timing::category::io);
    nlohmann::json j = wfn;
    std::string result = j.dump(m_shiftwidth);
    occ::timing::stop(occ::timing::category::io);
    return result;
}

void JsonWavefunctionWriter::write(const qm::Wavefunction &wfn,
                                   const std::string &filename) const {
    std::ios_base::openmode mode = std::ios_base::out;

    switch(m_format) {
	case JsonFormat::JSON:
	    break;
	case JsonFormat::CBOR:    // fallthrough
	case JsonFormat::UBJSON:  // fallthrough
	case JsonFormat::MSGPACK: // fallthrough
	case JsonFormat::BSON:
	    mode |= std::ios_base::binary;
	    break;
    }

    std::ofstream dest(filename, mode);
    occ::timing::start(occ::timing::category::io);
    nlohmann::json j = wfn;
    switch(m_format) {
	case JsonFormat::JSON:
	    dest << j.dump(m_shiftwidth);
	    break;
	case JsonFormat::UBJSON: {
	    std::vector<uint8_t> ubjson = nlohmann::json::to_ubjson(j);
	    dest.write(reinterpret_cast<const char*>(ubjson.data()), ubjson.size());
	    break;
        }
	case JsonFormat::CBOR: {
	    std::vector<uint8_t> cbor = nlohmann::json::to_cbor(j);
	    dest.write(reinterpret_cast<const char*>(cbor.data()), cbor.size());
	    break;
	}
	case JsonFormat::MSGPACK: {
	    std::vector<uint8_t> msgpack = nlohmann::json::to_msgpack(j);
	    dest.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size());
	    break;
	}
	case JsonFormat::BSON: {
	    std::vector<uint8_t> bson = nlohmann::json::to_bson(j);
	    dest.write(reinterpret_cast<const char*>(bson.data()), bson.size());
	    break;
	}

    }
    occ::timing::stop(occ::timing::category::io);
}

void JsonWavefunctionWriter::write(const qm::Wavefunction &wfn,
                                   std::ostream &dest) const {
    occ::timing::start(occ::timing::category::io);
    nlohmann::json j = wfn;
    dest << j.dump(m_shiftwidth);
    occ::timing::stop(occ::timing::category::io);
}

} // namespace occ::io
