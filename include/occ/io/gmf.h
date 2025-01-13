#pragma once

#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>

namespace occ::io {

class GMFWriter {

public:
  struct Facet {
    occ::crystal::HKL hkl;
    double shift{0.0};
    int region0{1};
    int region1{1};
    double surface{0.0};
    double attachment{0.0};
    double surface_relaxed{0.0};
    double attachment_relaxed{0.0};
    double gnorm{-1};
  };

private:
  std::string m_title;
  std::string m_name;
  occ::crystal::Crystal m_crystal;
  std::string m_morphology_kind{"unrelaxed equilibrium"};
  std::vector<Facet> m_facets;

public:
  GMFWriter(const occ::crystal::Crystal &);

  inline void set_title(const std::string &title) { m_title = title; }
  inline const auto &title() const { return m_title; }

  inline void set_name(const std::string &name) { m_name = name; }
  inline const auto &name() const { return m_name; }

  inline void set_crystal(const occ::crystal::Crystal &crystal) {
    m_crystal = crystal;
  }
  inline const auto &crystal() const { return m_crystal; }

  inline void set_morphology_kind(const std::string &morph) {
    m_morphology_kind = morph;
  }
  inline const auto &morphology_kind() const { return m_morphology_kind; }

  void add_facet(const Facet &facet) { m_facets.push_back(facet); }

  inline auto number_of_facets() const { return m_facets.size(); }

  bool write(const std::string &filename) const;
  bool write(std::ostream &output) const;
};

} // namespace occ::io
