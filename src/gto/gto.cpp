#include <occ/core/constants.h>
#include <occ/core/log.h>
#include <occ/gto/gto.h>

namespace occ::gto {

double common_fac(int l, bool spherical) {
  switch (l) {
  case 0:
    return 0.282094791773878143;
  case 1:
    return 0.488602511902919921;
  default:
    return spherical ? std::sqrt((2 * l + 1) / (4 * M_PI)) : 1.0;
  }
}

Vec evaluate_decay_cutoff(const qm::AOBasis &basis) {
  occ::log::trace("Evaluating decay cutoff (threshold = 1e-12) for AOBasis");
  occ::timing::start(occ::timing::category::gto);
  size_t natoms = basis.atoms().size();
  auto shell2bf = basis.first_bf();
  auto atom2shell = basis.atom_to_shell();
  size_t max_shell_size = basis.max_shell_size();
  const double max_radius = 30.0;
  const double threshold = 1e-12;
  const double increment = 0.1;
  const double min_radius = 5.0;
  const int npts = static_cast<int>((max_radius - min_radius) / increment) + 1;
  Mat3N pts = Mat3N::Zero(3, npts);
  pts.row(0).array() = 1.0;
  pts.row(0).array() *= Vec::LinSpaced(npts, min_radius, max_radius).array();
  Mat values(npts, max_shell_size);
  Vec result(basis.size());
  for (size_t i = 0; i < natoms; i++) {
    for (const auto &shell_idx : atom2shell[i]) {
      const auto &sh = basis[shell_idx];
      pts.colwise() += sh.origin;
      occ::timing::start(occ::timing::category::gto_shell);
      double *output = values.data();
      const double *xyz = pts.data();
      long int xyz_stride = 3;
      const double *coeffs = sh.contraction_coefficients.data();
      const double *alpha = sh.exponents.data();
      const double *center = sh.origin.data();
      int L = sh.l;
      bool spherical = (sh.kind == qm::Shell::Kind::Spherical);
      double fac = common_fac(L, spherical);
      int order = spherical ? GG_SPHERICAL_CCA : GG_CARTESIAN_CCA;
      gg_collocation(L, npts, xyz, xyz_stride, sh.num_primitives(), coeffs,
                     alpha, center, order, output);
      values.array() *= fac;
      for (int i = 0; i < values.rows(); i++) {
        result(shell_idx) = i * increment + min_radius;
        double v = values.block(i, 0, 1, sh.size())
                       .cwiseAbs()
                       .lpNorm<Eigen::Infinity>();
        if (v < threshold)
          break;
      }
      occ::timing::stop(occ::timing::category::gto_shell);
      pts.colwise() -= sh.origin;
    }
  }
  occ::timing::stop(occ::timing::category::gto);
  return result;
}

void evaluate_basis(const qm::AOBasis &basis, const occ::Mat &grid_pts,
                    GTOValues &gto_values, int max_derivative) {
  occ::timing::start(occ::timing::category::gto);
  size_t nbf = basis.nbf();
  size_t npts = grid_pts.cols();
  size_t natoms = basis.atoms().size();
  gto_values.reserve(nbf, npts, max_derivative);
  gto_values.set_zero();
  auto shell2bf = basis.first_bf();
  auto atom2shell = basis.atom_to_shell();
  // change this if we allow mixed integral kinds
  bool spherical = (basis.kind() == qm::Shell::Kind::Spherical);
  int order = spherical ? GG_SPHERICAL_CCA : GG_CARTESIAN_CCA;
  for (size_t i = 0; i < natoms; i++) {
    for (const auto &shell_idx : atom2shell[i]) {
      occ::timing::start(occ::timing::category::gto_shell);
      size_t bf = shell2bf[shell_idx];
      double *output = gto_values.phi.col(bf).data();
      const double *xyz = grid_pts.data();
      long int xyz_stride = 3;
      const auto &sh = basis[shell_idx];
      const double *coeffs = sh.contraction_coefficients.data();
      const double *alpha = sh.exponents.data();
      const double *center = sh.origin.data();
      int L = sh.l;
      double fac = common_fac(L, spherical);

      if (max_derivative == 0) {
        gg_collocation(L, npts, xyz, xyz_stride, sh.num_primitives(), coeffs,
                       alpha, center, order, output);
        gto_values.phi.block(0, bf, npts, sh.size()) *= fac;
      } else if (max_derivative == 1) {
        double *x_out = gto_values.phi_x.col(bf).data();
        double *y_out = gto_values.phi_y.col(bf).data();
        double *z_out = gto_values.phi_z.col(bf).data();
        gg_collocation_deriv1(L, npts, xyz, xyz_stride, sh.num_primitives(),
                              coeffs, alpha, center, order, output, x_out,
                              y_out, z_out);
        gto_values.phi.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_x.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_y.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_z.block(0, bf, npts, sh.size()) *= fac;
      } else if (max_derivative == 2) {
        double *x_out = gto_values.phi_x.col(bf).data();
        double *y_out = gto_values.phi_y.col(bf).data();
        double *z_out = gto_values.phi_z.col(bf).data();
        double *xx_out = gto_values.phi_xx.col(bf).data();
        double *xy_out = gto_values.phi_xy.col(bf).data();
        double *xz_out = gto_values.phi_xz.col(bf).data();
        double *yy_out = gto_values.phi_yy.col(bf).data();
        double *yz_out = gto_values.phi_yz.col(bf).data();
        double *zz_out = gto_values.phi_zz.col(bf).data();
        gg_collocation_deriv2(L, npts, xyz, xyz_stride, sh.num_primitives(),
                              coeffs, alpha, center, order, output, x_out,
                              y_out, z_out, xx_out, xy_out, xz_out, yy_out,
                              yz_out, zz_out);
        gto_values.phi.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_x.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_y.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_z.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_xx.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_xy.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_xz.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_yy.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_yz.block(0, bf, npts, sh.size()) *= fac;
        gto_values.phi_zz.block(0, bf, npts, sh.size()) *= fac;
      }
      occ::timing::stop(occ::timing::category::gto_shell);
    }
  }
  occ::timing::stop(occ::timing::category::gto);
}

Mat cartesian_to_spherical_transformation_matrix(int l) {
  Mat result = Mat::Zero(num_subshells(false, l), num_subshells(true, l));
  switch (l) {
  case 0: {
    result(0, 0) = 1.0;
    break;
  }
  case 1: {
    result <<
        // py
        0,
        1, 0,
        // pz
        0, 0, 1,
        // px
        1, 0, 0;
    break;
  }
  case 2: {
    // dxy
    result << 0, 1.092548430592079070, 0, 0, 0, 0,
        // dyz
        0, 0, 0, 0, 1.092548430592079070, 0,
        // dz2
        -0.315391565252520002, 0, 0, -0.315391565252520002, 0,
        0.630783130505040012,
        // dxz
        0, 0, 1.092548430592079070, 0, 0, 0,
        // dy2
        0.546274215296039535, 0, 0, -0.546274215296039535, 0, 0;

    break;
  }
  case 3: {
    result <<
        // f-3 ~ fyx2
        0,
        1.770130769779930531, 0, 0, 0, 0, -0.590043589926643510, 0, 0, 0,
        // f-2 ~ fxyz
        0, 0, 0, 0, 2.890611442640554055, 0, 0, 0, 0, 0,
        // f-1 ~ fyz2
        0, -0.457045799464465739, 0, 0, 0, 0, -0.457045799464465739, 0,
        1.828183197857862944, 0,
        // f0 ~ fz3
        0, 0, -1.119528997770346170, 0, 0, 0, 0, -1.119528997770346170, 0,
        0.746352665180230782,
        // f1 ~ fxz2
        -0.457045799464465739, 0, 0, -0.457045799464465739, 0,
        1.828183197857862944, 0, 0, 0, 0,
        // f2 ~ fzx2
        0, 0, 1.445305721320277020, 0, 0, 0, 0, -1.445305721320277020, 0, 0,
        // f3 ~ fx3
        0.590043589926643510, 0, 0, -1.770130769779930530, 0, 0, 0, 0, 0, 0;

    break;
  }
  case 4: {
    result <<
        // g-4 ~ gyx3
        0,
        2.503342941796704538, 0, 0, 0, 0, -2.503342941796704530, 0, 0, 0, 0, 0,
        0, 0, 0,
        // g-3 ~ gx2yz
        0, 0, 0, 0, 5.310392309339791593, 0, 0, 0, 0, 0, 0,
        -1.770130769779930530, 0, 0, 0,
        // g-2 ~ gxyz2
        0, -0.946174695757560014, 0, 0, 0, 0, -0.946174695757560014, 0,
        5.677048174545360108, 0, 0, 0, 0, 0, 0,
        // g-1 ~ gyz3
        0, 0, 0, 0, -2.007139630671867500, 0, 0, 0, 0, 0, 0,
        -2.007139630671867500, 0, 2.676186174229156671, 0,
        // g0 ~ gz4
        0.317356640745612911, 0, 0, 0.634713281491225822, 0,
        -2.538853125964903290, 0, 0, 0, 0, 0.317356640745612911, 0,
        -2.538853125964903290, 0, 0.846284375321634430,
        // g1 ~ gxz3
        0, 0, -2.007139630671867500, 0, 0, 0, 0, -2.007139630671867500, 0,
        2.676186174229156671, 0, 0, 0, 0, 0,
        // g2 ~ gx2z2
        -0.473087347878780002, 0, 0, 0, 0, 2.838524087272680054, 0, 0, 0, 0,
        0.473087347878780009, 0, -2.838524087272680050, 0, 0,
        // g3 ~ gzx3
        0, 0, 1.770130769779930531, 0, 0, 0, 0, -5.310392309339791590, 0, 0, 0,
        0, 0, 0, 0,
        // g4 ~ gy4
        0.625835735449176134, 0, 0, -3.755014412695056800, 0, 0, 0, 0, 0, 0,
        0.625835735449176134, 0, 0, 0, 0;
    break;
  }
  case 5: {
    result(0, 1) = 1.169267933366856683;
    result(0, 6) = -1.5309310892394863114;
    result(0, 15) = 0.7015607600201140098;
    result(1, 4) = 1.1180339887498948482;
    result(1, 11) = -1.1180339887498948482;
    result(2, 1) = -0.52291251658379721749;
    result(2, 6) = -0.22821773229381921394;
    result(2, 8) = 1.2247448713915890491;
    result(2, 15) = 0.52291251658379721749;
    result(2, 17) = -0.91287092917527685576;
    result(3, 4) = -0.6454972243679028142;
    result(3, 11) = -0.6454972243679028142;
    result(3, 13) = 1.2909944487358056284;
    result(4, 1) = 0.16137430609197570355;
    result(4, 6) = 0.21128856368212914438;
    result(4, 8) = -0.56694670951384084082;
    result(4, 15) = 0.48412291827592711065;
    result(4, 17) = -1.2677313820927748663;
    result(4, 19) = 1.2909944487358056284;
    result(5, 2) = 0.625;
    result(5, 7) = 0.36596252735569994226;
    result(5, 9) = -1.0910894511799619063;
    result(5, 16) = 0.625;
    result(5, 18) = -1.0910894511799619063;
    result(5, 20) = 1.0;
    result(6, 0) = 0.48412291827592711065;
    result(6, 3) = 0.21128856368212914438;
    result(6, 5) = -1.2677313820927748663;
    result(6, 10) = 0.16137430609197570355;
    result(6, 12) = -0.56694670951384084082;
    result(6, 14) = 1.2909944487358056284;
    result(7, 2) = -0.85391256382996653193;
    result(7, 9) = 1.1180339887498948482;
    result(7, 16) = 0.85391256382996653193;
    result(7, 18) = -1.1180339887498948482;
    result(8, 0) = -0.52291251658379721749;
    result(8, 3) = 0.22821773229381921394;
    result(8, 5) = 0.91287092917527685576;
    result(8, 10) = 0.52291251658379721749;
    result(8, 12) = -1.2247448713915890491;
    result(9, 2) = 0.73950997288745200532;
    result(9, 7) = -1.2990381056766579701;
    result(9, 16) = 0.73950997288745200532;
    result(10, 0) = 0.7015607600201140098;
    result(10, 3) = -1.5309310892394863114;
    result(10, 10) = 1.169267933366856683;
    break;
  }
  case 6: {
    result(0, 1) = 1.2151388809514737933;
    result(0, 6) = -1.9764235376052370825;
    result(0, 15) = 1.2151388809514737933;
    result(1, 4) = 1.169267933366856683;
    result(1, 11) = -1.5309310892394863114;
    result(1, 22) = 0.7015607600201140098;
    result(2, 1) = -0.59829302641309923139;
    result(2, 8) = 1.3055824196677337863;
    result(2, 15) = 0.59829302641309923139;
    result(2, 17) = -1.3055824196677337863;
    result(3, 4) = -0.81924646641122153043;
    result(3, 11) = -0.35754847096709711829;
    result(3, 13) = 1.4301938838683884732;
    result(3, 22) = 0.81924646641122153043;
    result(3, 24) = -1.0660035817780521715;
    result(4, 1) = 0.27308215547040717681;
    result(4, 6) = 0.26650089544451304287;
    result(4, 8) = -0.95346258924559231545;
    result(4, 15) = 0.27308215547040717681;
    result(4, 17) = -0.95346258924559231545;
    result(4, 19) = 1.4564381625088382763;
    result(5, 4) = 0.28785386654489893242;
    result(5, 11) = 0.37688918072220452831;
    result(5, 13) = -0.75377836144440905662;
    result(5, 22) = 0.86356159963469679725;
    result(5, 24) = -1.6854996561581052156;
    result(5, 26) = 1.3816985594155148756;
    result(6, 0) = -0.3125;
    result(6, 3) = -0.16319780245846672329;
    result(6, 5) = 0.97918681475080033975;
    result(6, 10) = -0.16319780245846672329;
    result(6, 12) = 0.57335309036732873772;
    result(6, 14) = -1.3055824196677337863;
    result(6, 21) = -0.3125;
    result(6, 23) = 0.97918681475080033975;
    result(6, 25) = -1.3055824196677337863;
    result(6, 27) = 1.0;
    result(7, 2) = 0.86356159963469679725;
    result(7, 7) = 0.37688918072220452831;
    result(7, 9) = -1.6854996561581052156;
    result(7, 16) = 0.28785386654489893242;
    result(7, 18) = -0.75377836144440905662;
    result(7, 20) = 1.3816985594155148756;
    result(8, 0) = 0.45285552331841995543;
    result(8, 3) = 0.078832027985861408788;
    result(8, 5) = -1.2613124477737825406;
    result(8, 10) = -0.078832027985861408788;
    result(8, 14) = 1.2613124477737825406;
    result(8, 21) = -0.45285552331841995543;
    result(8, 23) = 1.2613124477737825406;
    result(8, 25) = -1.2613124477737825406;
    result(9, 2) = -0.81924646641122153043;
    result(9, 7) = 0.35754847096709711829;
    result(9, 9) = 1.0660035817780521715;
    result(9, 16) = 0.81924646641122153043;
    result(9, 18) = -1.4301938838683884732;
    result(10, 0) = -0.49607837082461073572;
    result(10, 3) = 0.43178079981734839863;
    result(10, 5) = 0.86356159963469679725;
    result(10, 10) = 0.43178079981734839863;
    result(10, 12) = -1.5169496905422946941;
    result(10, 21) = -0.49607837082461073572;
    result(10, 23) = 0.86356159963469679725;
    result(11, 2) = 0.7015607600201140098;
    result(11, 7) = -1.5309310892394863114;
    result(11, 16) = 1.169267933366856683;
    result(12, 0) = 0.67169328938139615748;
    result(12, 3) = -1.7539019000502850245;
    result(12, 10) = 1.7539019000502850245;
    result(12, 21) = -0.67169328938139615748;
    break;
  }
  case 7: {
    result(0, 1) = 1.2566230789301937693;
    result(0, 6) = -2.4456993503903949804;
    result(0, 15) = 1.96875;
    result(0, 28) = -0.64725984928774934788;
    result(1, 4) = 1.2151388809514737933;
    result(1, 11) = -1.9764235376052370825;
    result(1, 22) = 1.2151388809514737933;
    result(2, 1) = -0.65864945955866621126;
    result(2, 6) = 0.25637895441948968451;
    result(2, 8) = 1.3758738481975177632;
    result(2, 15) = 0.61914323168888299344;
    result(2, 17) = -1.8014417303072302517;
    result(2, 28) = -0.47495887979908323849;
    result(2, 30) = 0.82552430891851065792;
    result(3, 4) = -0.95323336395336381126;
    result(3, 13) = 1.5504341823651057024;
    result(3, 22) = 0.95323336395336381126;
    result(3, 24) = -1.5504341823651057024;
    result(4, 1) = 0.35746251148251142922;
    result(4, 6) = 0.23190348980538452414;
    result(4, 8) = -1.2445247218182462513;
    result(4, 15) = 0.062226236090912312563;
    result(4, 17) = -0.54315511828342602619;
    result(4, 19) = 1.6593662957576616683;
    result(4, 28) = -0.42961647140211000062;
    result(4, 30) = 1.2445247218182462513;
    result(4, 32) = -1.2368186122953841287;
    result(5, 4) = 0.50807509012231371428;
    result(5, 11) = 0.49583051751369852316;
    result(5, 13) = -1.3222147133698627284;
    result(5, 22) = 0.50807509012231371428;
    result(5, 24) = -1.3222147133698627284;
    result(5, 26) = 1.6258402883914038857;
    result(6, 1) = -0.1146561540164598136;
    result(6, 6) = -0.13388954226515238921;
    result(6, 8) = 0.47901778876993906273;
    result(6, 15) = -0.17963167078872714852;
    result(6, 17) = 0.62718150750531807803;
    result(6, 19) = -0.95803557753987812546;
    result(6, 28) = -0.41339864235384227977;
    result(6, 30) = 1.4370533663098171882;
    result(6, 32) = -2.1422326762424382273;
    result(6, 34) = 1.4675987714106856141;
    result(7, 2) = -0.60670333962134435221;
    result(7, 7) = -0.31684048566533184861;
    result(7, 9) = 1.4169537279434593918;
    result(7, 16) = -0.31684048566533184861;
    result(7, 18) = 0.82968314787883083417;
    result(7, 20) = -1.5208343311935928733;
    result(7, 29) = -0.60670333962134435221;
    result(7, 31) = 1.4169537279434593918;
    result(7, 33) = -1.5208343311935928733;
    result(7, 35) = 1.0;
    result(8, 0) = -0.41339864235384227977;
    result(8, 3) = -0.17963167078872714852;
    result(8, 5) = 1.4370533663098171882;
    result(8, 10) = -0.13388954226515238921;
    result(8, 12) = 0.62718150750531807803;
    result(8, 14) = -2.1422326762424382273;
    result(8, 21) = -0.1146561540164598136;
    result(8, 23) = 0.47901778876993906273;
    result(8, 25) = -0.95803557753987812546;
    result(8, 27) = 1.4675987714106856141;
    result(9, 2) = 0.84254721963085980365;
    result(9, 7) = 0.14666864502533059662;
    result(9, 9) = -1.7491256557036030854;
    result(9, 16) = -0.14666864502533059662;
    result(9, 20) = 1.4080189922431737275;
    result(9, 29) = -0.84254721963085980365;
    result(9, 31) = 1.7491256557036030854;
    result(9, 33) = -1.4080189922431737275;
    result(10, 0) = 0.42961647140211000062;
    result(10, 3) = -0.062226236090912312563;
    result(10, 5) = -1.2445247218182462513;
    result(10, 10) = -0.23190348980538452414;
    result(10, 12) = 0.54315511828342602619;
    result(10, 14) = 1.2368186122953841287;
    result(10, 21) = -0.35746251148251142922;
    result(10, 23) = 1.2445247218182462513;
    result(10, 25) = -1.6593662957576616683;
    result(11, 2) = -0.79037935147039945351;
    result(11, 7) = 0.6879369240987588816;
    result(11, 9) = 1.025515817677958738;
    result(11, 16) = 0.6879369240987588816;
    result(11, 18) = -1.8014417303072302517;
    result(11, 29) = -0.79037935147039945351;
    result(11, 31) = 1.025515817677958738;
    result(12, 0) = -0.47495887979908323849;
    result(12, 3) = 0.61914323168888299344;
    result(12, 5) = 0.82552430891851065792;
    result(12, 10) = 0.25637895441948968451;
    result(12, 12) = -1.8014417303072302517;
    result(12, 21) = -0.65864945955866621126;
    result(12, 23) = 1.3758738481975177632;
    result(13, 2) = 0.67169328938139615748;
    result(13, 7) = -1.7539019000502850245;
    result(13, 16) = 1.7539019000502850245;
    result(13, 29) = -0.67169328938139615748;
    result(14, 0) = 0.64725984928774934788;
    result(14, 3) = -1.96875;
    result(14, 10) = 2.4456993503903949804;
    result(14, 21) = -1.2566230789301937693;
    break;
  }
  default:
    throw "not implemented";
  }
  return result;
}

Mat spherical_to_cartesian_transformation_matrix(int l) {
  throw std::runtime_error(
      "Not implemented: spherical_to_cartesian_transformation matrix "
      "normalisation is wrong");
  using occ::util::factorial;
  Mat c = cartesian_to_spherical_transformation_matrix(l);
  Mat S = Mat::Zero(num_subshells(true, l), num_subshells(true, l));

  size_t i = 0;
  auto lambda = [&](int lx1, int ly1, int lz1, int L1) {
    size_t j = 0;
    auto lambda2 = [&](int lx2, int ly2, int lz2, int L2) {
      int lxsum = lx1 + lx2;
      int lysum = ly1 + ly2;
      int lzsum = lz1 + lz2;
      if (lxsum % 2 || lysum % 2 || lzsum % 2) {
        j++;
        return;
      }
      double lhs = factorial(lxsum) / factorial(lxsum / 2) * factorial(lysum) /
                   factorial(lysum / 2) * factorial(lzsum) /
                   factorial(lzsum / 2);
      double rhs =
          std::sqrt(factorial(lx1) / factorial(2 * lx1) * factorial(ly1) /
                    factorial(2 * ly1) * factorial(lz1) / factorial(2 * lz1) *
                    factorial(lx2) / factorial(2 * lx2) * factorial(ly2) /
                    factorial(2 * ly2) * factorial(lz2) / factorial(2 * lz2));
      S(i, j) = lhs * rhs;
      if (L1 > 1) {
        S(i, j) /= common_fac(L1, true);
      }
      if (L2 > 1) {
        S(i, j) /= common_fac(L2, true);
      }
      j++;
    };
    iterate_over_shell<true>(lambda2, l);
    i++;
  };
  iterate_over_shell<true>(lambda, l);
  return S * c.transpose();
}
} // namespace occ::gto
