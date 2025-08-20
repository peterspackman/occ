/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <occ/opt/pseudoinverse.h>
#include <occ/core/log.h>

namespace occ::opt {


Mat pseudoinverse(Eigen::Ref<const Mat> A) {
  // Pseudoinverse implementation
  Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Vec& singular_values = svd.singularValues();
  const Mat& U = svd.matrixU();
  const Mat& V = svd.matrixV();
  
  const double thre = 1e3;
  const double thre_log = 1e8;
  
  Vec D = singular_values;
  int n = D.size() - 1; // Default to keep all singular values
  
  // Compute gaps: gaps = D[:-1] / D[1:]
  if (D.size() > 1) {
    Vec gaps(D.size() - 1);
    for (int i = 0; i < gaps.size(); i++) {
      gaps(i) = D(i) / D(i + 1);
    }
    
    // Find first gap above threshold: n = flatnonzero(gaps > thre)[0]
    for (int i = 0; i < gaps.size(); i++) {
      if (gaps(i) > thre) {
        n = i;
        double gap = gaps(n);
        if (gap < thre_log) {
          occ::log::warn("Pseudoinverse gap of only: {:.1e}", gap);
        }
        break;
      }
    }
  }
  
  // Zero small values and invert: D[n+1:] = 0; D[:n+1] = 1/D[:n+1]
  Vec D_inv = Vec::Zero(D.size());
  for (int i = 0; i <= n; i++) {
    D_inv(i) = 1.0 / D(i);
  }
  
  // Reconstruct: return U * diag(D) * V
  return U * D_inv.asDiagonal() * V.transpose();
}



}
