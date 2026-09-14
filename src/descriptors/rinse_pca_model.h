#pragma once

namespace occ::descriptors::impl {

/**
 * The PCA model behind rinse_hash(), fitted to descriptors of the Cambridge
 * Structural Database and taken verbatim from the `pca_components.json`
 * bundled with rinse-descriptor 2.0.0.
 *
 * Replacing it changes every hash, so only ever take a refitted model together
 * with the reference implementation it belongs to.
 */
extern const int pca_num_components;
extern const int pca_num_features;
/// Row major, `pca_num_components` rows of `pca_num_features`
extern const double pca_components[];
extern const double pca_mean[];

} // namespace occ::descriptors::impl
