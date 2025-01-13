#include <ankerl/unordered_dense.h>
#include <fmt/ostream.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/io/crystalgrower.h>
#include <string>

namespace occ::io::crystalgrower {
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;
using SymDimer = CrystalDimers::SymmetryRelatedDimer;

void sort_neighbors(CrystalDimers &dimers) {
  for (auto &neighbors : dimers.molecule_neighbors) {
    // sort only the non-unique dimers, by center of mass distance
    std::stable_sort(neighbors.begin(), neighbors.end(),
                     [](const SymDimer &left, const SymDimer &right) {
                       return left.dimer.centroid_distance() <
                              right.dimer.centroid_distance();
                     });
  }
}

StructureWriter::StructureWriter(const std::string &filename)
    : m_owned_destination(filename), m_dest(m_owned_destination) {}

StructureWriter::StructureWriter(std::ostream &stream) : m_dest(stream) {}

void StructureWriter::write(const Crystal &crystal,
                            const CrystalDimers &uc_dimers) {
  using occ::units::degrees;
  fmt::print(m_dest, "{}\n\n", "title");
  const auto &uc_molecules = crystal.unit_cell_molecules();
  fmt::print(m_dest, "{}\n\n", uc_molecules.size());

  const auto &uc_mols = crystal.unit_cell_molecules();
  CrystalDimers uc_dimers_copy = uc_dimers;
  sort_neighbors(uc_dimers_copy);

  const auto &neighbors = uc_dimers_copy.molecule_neighbors;
  size_t uc_idx = 0;
  for (const auto &mol : uc_molecules) {
    size_t num_neighbors = neighbors[uc_idx].size();
    fmt::print(m_dest, "{} {} (1,0) {}\n\n",
               core::chemical_formula(mol.elements()),
               mol.unit_cell_molecule_idx() + 1, num_neighbors);
    for (const auto &[n, unique_index] : neighbors[uc_idx]) {
      const auto &b = n.b();
      const auto uc_idx_b = b.unit_cell_molecule_idx() + 1;
      const auto uc_shift = b.cell_shift() - uc_mols[uc_idx_b - 1].cell_shift();
      fmt::print(m_dest, "{}({},{},{}) ", uc_idx_b, uc_shift.x(), uc_shift.y(),
                 uc_shift.z());
    }
    fmt::print(m_dest, "\n\nX 0[0]\n\n");

    uc_idx++;
  }

  const auto &uc = crystal.unit_cell();
  double a = uc.a(), b = uc.b(), c = uc.c();
  double alpha = degrees(uc.alpha()), beta = degrees(uc.beta()),
         gamma = degrees(uc.gamma());
  fmt::print(m_dest, "{:8.4f} {:8.4f} {:8.4f} P\n", a, b, c);
  fmt::print(m_dest, "{:7.3f} {:7.3f} {:7.3f}\n", alpha, beta, gamma);

  fmt::print(m_dest, "\nNon primitive data\n");
  fmt::print(m_dest, "{:8.4f} {:8.4f} {:8.4f}\n", a, b, c);
  fmt::print(m_dest, "{:7.3f} {:7.3f} {:7.3f}\n", alpha, beta, gamma);
  fmt::print(m_dest, "\n");

  for (const auto &mol : uc_molecules) {
    const auto &bonds = mol.bonds();
    const auto &elements = mol.elements();
    const auto &pos = crystal.to_fractional(mol.positions());
    fmt::print(m_dest, "{} {} {}/{}\n", mol.name(),
               mol.unit_cell_molecule_idx() + 1, mol.size(), bonds.size() / 2);
    for (size_t i = 0; i < mol.size(); i++) {
      fmt::print(m_dest, "{} {:s} {: 8.5f} {: 8.5f} {: 8.5f}\n", i + 1,
                 elements[i].symbol(), pos(0, i), pos(1, i), pos(2, i));
    }
    fmt::print(m_dest, "\n");
    for (const auto &b : bonds) {
      if (b.first < b.second)
        fmt::print(m_dest, " {} {}\n", b.first + 1, b.second + 1);
    }
    fmt::print(m_dest, "\n");
  }
}

NetWriter::NetWriter(const std::string &filename)
    : m_owned_destination(filename), m_dest(m_owned_destination) {}

NetWriter::NetWriter(std::ostream &stream) : m_dest(stream) {}

inline std::string get_asymmetric_dimer_name(const core::Dimer &n) {
  const auto &properties = n.properties();
  std::string name = "";
  const auto kv = properties.find("asymmetric_dimer");
  if (kv != properties.end()) {
    name = kv->second;
  }
  return name;
}

void NetWriter::write(const Crystal &crystal, const CrystalDimers &uc_dimers) {
  m_interaction_labels.clear();
  const auto &uc_molecules = crystal.unit_cell_molecules();
  CrystalDimers uc_dimers_copy = uc_dimers;
  sort_neighbors(uc_dimers_copy);
  const auto &neighbors = uc_dimers_copy.molecule_neighbors;

  auto dimer_labeller = crystal::SymmetryDimerLabeller(crystal);
  dimer_labeller.connection = "-";
  dimer_labeller.format.fmt_string = "{}";

  size_t uc_idx = 0;
  constexpr double max_de = 1e-4;

  ankerl::unordered_dense::map<std::string, std::vector<double>> f2e;

  for (const auto &mol : uc_molecules) {

    std::string formula_index = mol.name();
    std::vector<double> &unique_interaction_energies = f2e[formula_index];
    auto mol_neighbors = neighbors[uc_idx];
    for (const auto &[n, unique_index] : neighbors[uc_idx]) {
      const auto uc_shift = n.b().cell_shift();
      const double e_int = n.interaction_energy();

      auto match =
          std::find_if(unique_interaction_energies.begin(),
                       unique_interaction_energies.end(), [&e_int](double x) {
                         return std::abs(x - e_int) < max_de;
                       });
      size_t interaction_idx = 1;

      if (match == std::end(unique_interaction_energies)) {
        unique_interaction_energies.push_back(e_int);
        interaction_idx = unique_interaction_energies.size();
      } else {
        interaction_idx =
            1 + std::distance(unique_interaction_energies.begin(), match);
      }

      auto asym_dimer_filename = get_asymmetric_dimer_name(n);
      auto dimer_label = dimer_labeller(n);
      m_interaction_labels.insert(
          {dimer_label, fmt::format("{}-{}", formula_index, interaction_idx)});

      fmt::print(m_dest, "{}:[{}][{}] {} R={:.3f}\n", interaction_idx,
                 formula_index, dimer_label, asym_dimer_filename,
                 n.centroid_distance());
    }
    for (double e : unique_interaction_energies) {
      fmt::print(m_dest, "{:7.3f}\n", e * occ::units::KJ_TO_KCAL);
    }
    uc_idx++;
  }
}

} // namespace occ::io::crystalgrower
