#pragma once
#include <occ/core/util.h>
#include <occ/qm/spinorbital.h>

namespace occ::main {

enum class MethodKind {
    HF,
    DFT,
};


inline qm::SpinorbitalKind determine_spinorbital_kind(const std::string &name, int multiplicity, MethodKind method_kind) {
    auto lc = occ::util::to_lower_copy(name);
    switch(method_kind) {
	case MethodKind::HF: {
	    if(lc[0] == 'g') return qm::SpinorbitalKind::General;
	    else if(lc[0] == 'u' || multiplicity > 1) return qm::SpinorbitalKind::Unrestricted;
	    else return qm::SpinorbitalKind::Restricted;
	    break;
	}
	case MethodKind::DFT: {
	    if(lc[0] == 'u' || multiplicity > 1) return qm::SpinorbitalKind::Unrestricted;
	    else return qm::SpinorbitalKind::Restricted;
        }
    }
}

inline MethodKind method_kind_from_string(const std::string &name) {
    auto lc = occ::util::to_lower_copy(name);
    if(lc == "hf" || 
       lc == "rhf" || 
       lc == "uhf" || 
       lc == "ghf" || 
       lc == "scf" ||
       lc == "hartree-fock" ||
       lc == "hartree fock") {
	return MethodKind::HF;
    }
    return MethodKind::DFT;
};

}
