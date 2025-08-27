#pragma once
#include <occ/crystal/crystal.h>
#include <gemmi/cifdoc.hpp>
#include <string>

namespace occ::io {

/**
 * \brief Writer class for crystallographic information files (CIF)
 *
 * This class provides functionality to write Crystal objects to CIF format
 * using the gemmi library. It's particularly useful for visualizing
 * normalized crystal structures or exporting structure data.
 */
class CifWriter {
public:
  /**
   * \brief Write a Crystal structure to a CIF file
   * 
   * \param filename Path to the output CIF file
   * \param crystal The Crystal object to write
   * \param title Optional title for the structure (defaults to chemical formula)
   */
  void write(const std::string& filename, 
             const occ::crystal::Crystal& crystal,
             const std::string& title = "");

  /**
   * \brief Convert Crystal to CIF string representation
   * 
   * \param crystal The Crystal object to convert
   * \param title Optional title for the structure (defaults to chemical formula)
   * \return CIF format string
   */
  std::string to_string(const occ::crystal::Crystal& crystal,
                        const std::string& title = "");

  /**
   * \brief Set precision for coordinate output
   * 
   * \param precision Number of decimal places for coordinates (default: 6)
   */
  void set_precision(int precision) { 
    m_precision = precision; 
  }

private:
  int m_precision = 6;

  /**
   * \brief Convert Crystal to gemmi CIF Document
   * 
   * \param crystal The Crystal object to convert
   * \param title Title for the structure
   * \return gemmi CIF Document
   */
  gemmi::cif::Document crystal_to_cif_document(const occ::crystal::Crystal& crystal,
                                               const std::string& title) const;
};

} // namespace occ::io