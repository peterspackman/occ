#pragma once

#include <gemmi/symmetry.hpp>
#include <map>
#include <occ/crystal/hkl.h>
#include <occ/crystal/symmetryoperation.h>
#include <string>
#include <tuple>
#include <vector>

namespace occ::crystal {

using occ::IVec;
using occ::Mat3N;

using SGData = gemmi::SpaceGroup;

class SpaceGroup {
  public:
    class ReciprocalAsymmetricUnit {
      public:
        ReciprocalAsymmetricUnit(const SGData *sg) : m_asu(sg) {}
        bool is_in(const HKL &hkl) const {
            return m_asu.is_in({hkl.h, hkl.k, hkl.l});
        }

      private:
        gemmi::ReciprocalAsu m_asu;
    };

    SpaceGroup(int);
    SpaceGroup(const std::string &);
    SpaceGroup(const std::vector<std::string> &);
    SpaceGroup(const std::vector<SymmetryOperation> &symops);

    int number() const;
    const std::string &symbol() const;
    const std::string &short_name() const;
    const std::vector<SymmetryOperation> &symmetry_operations() const;
    bool has_H_R_choice() const;
    std::pair<IVec, Mat3N> apply_all_symmetry_operations(const Mat3N &) const;

    inline auto reciprocal_asu() const {
        return ReciprocalAsymmetricUnit(m_sgdata);
    }

  private:
    void update_from_sgdata();
    int m_number{0};
    std::string m_symbol{"XX"};
    std::string m_short_name{"unknown"};
    const SGData *m_sgdata{nullptr};
    std::vector<SymmetryOperation> m_symops;
};

} // namespace occ::crystal
