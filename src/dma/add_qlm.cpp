#include <cmath>
#include <occ/dma/add_qlm.h>

namespace occ::dma {

namespace {
constexpr double rt2 = 1.41421356237309515;
constexpr double rt3 = 1.73205080756887719;
constexpr double rt5 = 2.23606797749978981;
constexpr double rt7 = 2.64575131106459072;
constexpr double rt11 = 3.31662479035539981;
constexpr double rt13 = 3.60555127546398912;
constexpr double rt17 = 4.12310562561766059;
constexpr double rt19 = 4.35889894354067398;
} // namespace

inline void addqlm_0(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {
  //  Monopole term
  out.Q00() += f * gx(0) * gy(0) * gz(0);
}

inline void addqlm_1(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {
  //  Dipole terms
  out.Q10() += f * (gx(0) * gy(0) * gz(1));
  out.Q11c() += f * (gx(1) * gy(0) * gz(0));
  out.Q11s() += f * (gx(0) * gy(1) * gz(0));
}

inline void addqlm_2(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {

  //  Quadrupole terms
  out.Q20() += f *
               (2.0 * gx(0) * gy(0) * gz(2) - gx(0) * gy(2) * gz(0) -
                gx(2) * gy(0) * gz(0)) /
               2.0;
  out.Q21c() += f * rt3 * (gx(1) * gy(0) * gz(1));
  out.Q21s() += f * rt3 * (gx(0) * gy(1) * gz(1));
  out.Q22c() +=
      f * rt3 * (-gx(0) * gy(2) * gz(0) + gx(2) * gy(0) * gz(0)) / 2.0;
  out.Q22s() += f * rt3 * (2.0 * gx(1) * gy(1) * gz(0)) / 2.0;
}

inline void addqlm_3(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {
  //  Octopole terms
  out.Q30() += f *
               (2.0 * gx(0) * gy(0) * gz(3) - 3.0 * gx(0) * gy(2) * gz(1) -
                3.0 * gx(2) * gy(0) * gz(1)) /
               2.0;
  out.Q31c() += f * rt2 * rt3 *
                (4.0 * gx(1) * gy(0) * gz(2) - gx(1) * gy(2) * gz(0) -
                 gx(3) * gy(0) * gz(0)) /
                4.0;
  out.Q31s() += f * rt2 * rt3 *
                (4.0 * gx(0) * gy(1) * gz(2) - gx(0) * gy(3) * gz(0) -
                 gx(2) * gy(1) * gz(0)) /
                4.0;
  out.Q32c() +=
      f * rt3 * rt5 * (-gx(0) * gy(2) * gz(1) + gx(2) * gy(0) * gz(1)) / 2.0;
  out.Q32s() += f * rt3 * rt5 * (2.0 * gx(1) * gy(1) * gz(1)) / 2.0;
  out.Q33c() += f * rt2 * rt5 *
                (-3.0 * gx(1) * gy(2) * gz(0) + gx(3) * gy(0) * gz(0)) / 4.0;
  out.Q33s() += f * rt2 * rt5 *
                (-gx(0) * gy(3) * gz(0) + 3.0 * gx(2) * gy(1) * gz(0)) / 4.0;
}

inline void addqlm_4(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {

  //  Hexadecapole terms
  out.Q40() += f *
               (8.0 * gx(0) * gy(0) * gz(4) - 24.0 * gx(0) * gy(2) * gz(2) -
                24.0 * gx(2) * gy(0) * gz(2) + 3.0 * gx(0) * gy(4) * gz(0) +
                6.0 * gx(2) * gy(2) * gz(0) + 3.0 * gx(4) * gy(0) * gz(0)) /
               8.0;
  out.Q41c() += f * rt2 * rt5 *
                (4.0 * gx(1) * gy(0) * gz(3) - 3.0 * gx(1) * gy(2) * gz(1) -
                 3.0 * gx(3) * gy(0) * gz(1)) /
                4.0;
  out.Q41s() += f * rt2 * rt5 *
                (4.0 * gx(0) * gy(1) * gz(3) - 3.0 * gx(0) * gy(3) * gz(1) -
                 3.0 * gx(2) * gy(1) * gz(1)) /
                4.0;
  out.Q42c() += f * rt5 *
                (-6.0 * gx(0) * gy(2) * gz(2) + 6.0 * gx(2) * gy(0) * gz(2) +
                 gx(0) * gy(4) * gz(0) - gx(4) * gy(0) * gz(0)) /
                4.0;
  out.Q42s() += f * rt5 *
                (12.0 * gx(1) * gy(1) * gz(2) - 2.0 * gx(1) * gy(3) * gz(0) -
                 2.0 * gx(3) * gy(1) * gz(0)) /
                4.0;
  out.Q43c() += f * rt2 * rt5 * rt7 *
                (-3.0 * gx(1) * gy(2) * gz(1) + gx(3) * gy(0) * gz(1)) / 4.0;
  out.Q43s() += f * rt2 * rt5 * rt7 *
                (-gx(0) * gy(3) * gz(1) + 3.0 * gx(2) * gy(1) * gz(1)) / 4.0;
  out.Q44c() += f * rt5 * rt7 *
                (gx(0) * gy(4) * gz(0) - 6.0 * gx(2) * gy(2) * gz(0) +
                 gx(4) * gy(0) * gz(0)) /
                8.0;
  out.Q44s() += f * rt5 * rt7 *
                (-4.0 * gx(1) * gy(3) * gz(0) + 4.0 * gx(3) * gy(1) * gz(0)) /
                8.0;
}

inline void addqlm_5(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {

  out.Q50() += f *
               (8.0 * gx(0) * gy(0) * gz(5) - 40.0 * gx(0) * gy(2) * gz(3) -
                40.0 * gx(2) * gy(0) * gz(3) + 15.0 * gx(0) * gy(4) * gz(1) +
                30.0 * gx(2) * gy(2) * gz(1) + 15.0 * gx(4) * gy(0) * gz(1)) /
               8.0;
  out.Q51c() += f * rt3 * rt5 *
                (8.0 * gx(1) * gy(0) * gz(4) - 12.0 * gx(1) * gy(2) * gz(2) -
                 12.0 * gx(3) * gy(0) * gz(2) + gx(1) * gy(4) * gz(0) +
                 2.0 * gx(3) * gy(2) * gz(0) + gx(5) * gy(0) * gz(0)) /
                8.0;
  out.Q51s() += f * rt3 * rt5 *
                (8.0 * gx(0) * gy(1) * gz(4) - 12.0 * gx(0) * gy(3) * gz(2) -
                 12.0 * gx(2) * gy(1) * gz(2) + gx(0) * gy(5) * gz(0) +
                 2.0 * gx(2) * gy(3) * gz(0) + gx(4) * gy(1) * gz(0)) /
                8.0;
  out.Q52c() += f * rt3 * rt5 * rt7 *
                (-2.0 * gx(0) * gy(2) * gz(3) + 2.0 * gx(2) * gy(0) * gz(3) +
                 gx(0) * gy(4) * gz(1) - gx(4) * gy(0) * gz(1)) /
                4.0;
  out.Q52s() += f * rt3 * rt5 * rt7 *
                (4.0 * gx(1) * gy(1) * gz(3) - 2.0 * gx(1) * gy(3) * gz(1) -
                 2.0 * gx(3) * gy(1) * gz(1)) /
                4.0;
  out.Q53c() += f * rt2 * rt5 * rt7 *
                (-24.0 * gx(1) * gy(2) * gz(2) + 8.0 * gx(3) * gy(0) * gz(2) +
                 3.0 * gx(1) * gy(4) * gz(0) + 2.0 * gx(3) * gy(2) * gz(0) -
                 gx(5) * gy(0) * gz(0)) /
                16.0;
  out.Q53s() += f * rt2 * rt5 * rt7 *
                (-8.0 * gx(0) * gy(3) * gz(2) + 24.0 * gx(2) * gy(1) * gz(2) +
                 gx(0) * gy(5) * gz(0) - 2.0 * gx(2) * gy(3) * gz(0) -
                 3.0 * gx(4) * gy(1) * gz(0)) /
                16.0;
  out.Q54c() += f * rt5 * rt7 *
                (3.0 * gx(0) * gy(4) * gz(1) - 18.0 * gx(2) * gy(2) * gz(1) +
                 3.0 * gx(4) * gy(0) * gz(1)) /
                8.0;
  out.Q54s() += f * rt5 * rt7 *
                (-12.0 * gx(1) * gy(3) * gz(1) + 12.0 * gx(3) * gy(1) * gz(1)) /
                8.0;
  out.Q55c() += f * rt2 * rt7 *
                (15.0 * gx(1) * gy(4) * gz(0) - 30.0 * gx(3) * gy(2) * gz(0) +
                 3.0 * gx(5) * gy(0) * gz(0)) /
                16.0;
  out.Q55s() += f * rt2 * rt7 *
                (3.0 * gx(0) * gy(5) * gz(0) - 30.0 * gx(2) * gy(3) * gz(0) +
                 15.0 * gx(4) * gy(1) * gz(0)) /
                16.0;
}

inline void addqlm_6(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {

  out.Q60() += f *
               (16.0 * gx(0) * gy(0) * gz(6) - 120.0 * gx(0) * gy(2) * gz(4) -
                120.0 * gx(2) * gy(0) * gz(4) + 90.0 * gx(0) * gy(4) * gz(2) +
                180.0 * gx(2) * gy(2) * gz(2) + 90.0 * gx(4) * gy(0) * gz(2) -
                5.0 * gx(0) * gy(6) * gz(0) - 15.0 * gx(2) * gy(4) * gz(0) -
                15.0 * gx(4) * gy(2) * gz(0) - 5.0 * gx(6) * gy(0) * gz(0)) /
               16.0;
  out.Q61c() += f * rt3 * rt7 *
                (8.0 * gx(1) * gy(0) * gz(5) - 20.0 * gx(1) * gy(2) * gz(3) -
                 20.0 * gx(3) * gy(0) * gz(3) + 5.0 * gx(1) * gy(4) * gz(1) +
                 10.0 * gx(3) * gy(2) * gz(1) + 5.0 * gx(5) * gy(0) * gz(1)) /
                8.0;
  out.Q61s() += f * rt3 * rt7 *
                (8.0 * gx(0) * gy(1) * gz(5) - 20.0 * gx(0) * gy(3) * gz(3) -
                 20.0 * gx(2) * gy(1) * gz(3) + 5.0 * gx(0) * gy(5) * gz(1) +
                 10.0 * gx(2) * gy(3) * gz(1) + 5.0 * gx(4) * gy(1) * gz(1)) /
                8.0;
  out.Q62c() += f * rt2 * rt3 * rt5 * rt7 *
                (-16.0 * gx(0) * gy(2) * gz(4) + 16.0 * gx(2) * gy(0) * gz(4) +
                 16.0 * gx(0) * gy(4) * gz(2) - 16.0 * gx(4) * gy(0) * gz(2) -
                 gx(0) * gy(6) * gz(0) - gx(2) * gy(4) * gz(0) +
                 gx(4) * gy(2) * gz(0) + gx(6) * gy(0) * gz(0)) /
                32.0;
  out.Q62s() += f * rt2 * rt3 * rt5 * rt7 *
                (32.0 * gx(1) * gy(1) * gz(4) - 32.0 * gx(1) * gy(3) * gz(2) -
                 32.0 * gx(3) * gy(1) * gz(2) + 2.0 * gx(1) * gy(5) * gz(0) +
                 4.0 * gx(3) * gy(3) * gz(0) + 2.0 * gx(5) * gy(1) * gz(0)) /
                32.0;
  out.Q63c() += f * rt2 * rt3 * rt5 * rt7 *
                (-24.0 * gx(1) * gy(2) * gz(3) + 8.0 * gx(3) * gy(0) * gz(3) +
                 9.0 * gx(1) * gy(4) * gz(1) + 6.0 * gx(3) * gy(2) * gz(1) -
                 3.0 * gx(5) * gy(0) * gz(1)) /
                16.0;
  out.Q63s() += f * rt2 * rt3 * rt5 * rt7 *
                (-8.0 * gx(0) * gy(3) * gz(3) + 24.0 * gx(2) * gy(1) * gz(3) +
                 3.0 * gx(0) * gy(5) * gz(1) - 6.0 * gx(2) * gy(3) * gz(1) -
                 9.0 * gx(4) * gy(1) * gz(1)) /
                16.0;
  out.Q64c() += f * rt7 *
                (30.0 * gx(0) * gy(4) * gz(2) - 180.0 * gx(2) * gy(2) * gz(2) +
                 30.0 * gx(4) * gy(0) * gz(2) - 3.0 * gx(0) * gy(6) * gz(0) +
                 15.0 * gx(2) * gy(4) * gz(0) + 15.0 * gx(4) * gy(2) * gz(0) -
                 3.0 * gx(6) * gy(0) * gz(0)) /
                16.0;
  out.Q64s() +=
      f * rt7 *
      (-120.0 * gx(1) * gy(3) * gz(2) + 120.0 * gx(3) * gy(1) * gz(2) +
       12.0 * gx(1) * gy(5) * gz(0) - 12.0 * gx(5) * gy(1) * gz(0)) /
      16.0;
  out.Q65c() += f * rt2 * rt7 * rt11 *
                (15.0 * gx(1) * gy(4) * gz(1) - 30.0 * gx(3) * gy(2) * gz(1) +
                 3.0 * gx(5) * gy(0) * gz(1)) /
                16.0;
  out.Q65s() += f * rt2 * rt7 * rt11 *
                (3.0 * gx(0) * gy(5) * gz(1) - 30.0 * gx(2) * gy(3) * gz(1) +
                 15.0 * gx(4) * gy(1) * gz(1)) /
                16.0;
  out.Q66s() += f * rt2 * rt3 * rt7 * rt11 *
                (6.0 * gx(1) * gy(5) * gz(0) - 20.0 * gx(3) * gy(3) * gz(0) +
                 6.0 * gx(5) * gy(1) * gz(0)) /
                32.0;
  out.Q66c() += f * rt2 * rt3 * rt7 * rt11 *
                (-gx(0) * gy(6) * gz(0) + 15.0 * gx(2) * gy(4) * gz(0) -
                 15.0 * gx(4) * gy(2) * gz(0) + gx(6) * gy(0) * gz(0)) /
                32.0;
}

inline void addqlm_7(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {

  out.Q70() += f *
               (16.0 * gx(0) * gy(0) * gz(7) - 168.0 * gx(0) * gy(2) * gz(5) -
                168.0 * gx(2) * gy(0) * gz(5) + 210.0 * gx(0) * gy(4) * gz(3) +
                420.0 * gx(2) * gy(2) * gz(3) + 210.0 * gx(4) * gy(0) * gz(3) -
                35.0 * gx(0) * gy(6) * gz(1) - 105.0 * gx(2) * gy(4) * gz(1) -
                105.0 * gx(4) * gy(2) * gz(1) - 35.0 * gx(6) * gy(0) * gz(1)) /
               16.0;
  out.Q71c() += f * rt7 *
                (64.0 * gx(1) * gy(0) * gz(6) - 240.0 * gx(1) * gy(2) * gz(4) -
                 240.0 * gx(3) * gy(0) * gz(4) + 120.0 * gx(1) * gy(4) * gz(2) +
                 240.0 * gx(3) * gy(2) * gz(2) + 120.0 * gx(5) * gy(0) * gz(2) -
                 5.0 * gx(1) * gy(6) * gz(0) - 15.0 * gx(3) * gy(4) * gz(0) -
                 15.0 * gx(5) * gy(2) * gz(0) - 5.0 * gx(7) * gy(0) * gz(0)) /
                32.0;
  out.Q71s() += f * rt7 *
                (64.0 * gx(0) * gy(1) * gz(6) - 240.0 * gx(0) * gy(3) * gz(4) -
                 240.0 * gx(2) * gy(1) * gz(4) + 120.0 * gx(0) * gy(5) * gz(2) +
                 240.0 * gx(2) * gy(3) * gz(2) + 120.0 * gx(4) * gy(1) * gz(2) -
                 5.0 * gx(0) * gy(7) * gz(0) - 15.0 * gx(2) * gy(5) * gz(0) -
                 15.0 * gx(4) * gy(3) * gz(0) - 5.0 * gx(6) * gy(1) * gz(0)) /
                32.0;
  out.Q72c() += f * rt2 * rt3 * rt7 *
                (-48.0 * gx(0) * gy(2) * gz(5) + 48.0 * gx(2) * gy(0) * gz(5) +
                 80.0 * gx(0) * gy(4) * gz(3) - 80.0 * gx(4) * gy(0) * gz(3) -
                 15.0 * gx(0) * gy(6) * gz(1) - 15.0 * gx(2) * gy(4) * gz(1) +
                 15.0 * gx(4) * gy(2) * gz(1) + 15.0 * gx(6) * gy(0) * gz(1)) /
                32.0;
  out.Q72s() += f * rt2 * rt3 * rt7 *
                (96.0 * gx(1) * gy(1) * gz(5) - 160.0 * gx(1) * gy(3) * gz(3) -
                 160.0 * gx(3) * gy(1) * gz(3) + 30.0 * gx(1) * gy(5) * gz(1) +
                 60.0 * gx(3) * gy(3) * gz(1) + 30.0 * gx(5) * gy(1) * gz(1)) /
                32.0;
  out.Q73c() += f * rt3 * rt7 *
                (-240.0 * gx(1) * gy(2) * gz(4) + 80.0 * gx(3) * gy(0) * gz(4) +
                 180.0 * gx(1) * gy(4) * gz(2) + 120.0 * gx(3) * gy(2) * gz(2) -
                 60.0 * gx(5) * gy(0) * gz(2) - 9.0 * gx(1) * gy(6) * gz(0) -
                 15.0 * gx(3) * gy(4) * gz(0) - 3.0 * gx(5) * gy(2) * gz(0) +
                 3.0 * gx(7) * gy(0) * gz(0)) /
                32.0;
  out.Q73s() += f * rt3 * rt7 *
                (-80.0 * gx(0) * gy(3) * gz(4) + 240.0 * gx(2) * gy(1) * gz(4) +
                 60.0 * gx(0) * gy(5) * gz(2) - 120.0 * gx(2) * gy(3) * gz(2) -
                 180.0 * gx(4) * gy(1) * gz(2) - 3.0 * gx(0) * gy(7) * gz(0) +
                 3.0 * gx(2) * gy(5) * gz(0) + 15.0 * gx(4) * gy(3) * gz(0) +
                 9.0 * gx(6) * gy(1) * gz(0)) /
                32.0;
  out.Q74c() += f * rt3 * rt7 * rt11 *
                (10.0 * gx(0) * gy(4) * gz(3) - 60.0 * gx(2) * gy(2) * gz(3) +
                 10.0 * gx(4) * gy(0) * gz(3) - 3.0 * gx(0) * gy(6) * gz(1) +
                 15.0 * gx(2) * gy(4) * gz(1) + 15.0 * gx(4) * gy(2) * gz(1) -
                 3.0 * gx(6) * gy(0) * gz(1)) /
                16.0;
  out.Q74s() += f * rt3 * rt7 * rt11 *
                (-40.0 * gx(1) * gy(3) * gz(3) + 40.0 * gx(3) * gy(1) * gz(3) +
                 12.0 * gx(1) * gy(5) * gz(1) - 12.0 * gx(5) * gy(1) * gz(1)) /
                16.0;
  out.Q75c() += f * rt3 * rt7 * rt11 *
                (60.0 * gx(1) * gy(4) * gz(2) - 120.0 * gx(3) * gy(2) * gz(2) +
                 12.0 * gx(5) * gy(0) * gz(2) - 5.0 * gx(1) * gy(6) * gz(0) +
                 5.0 * gx(3) * gy(4) * gz(0) + 9.0 * gx(5) * gy(2) * gz(0) -
                 gx(7) * gy(0) * gz(0)) /
                32.0;
  out.Q75s() += f * rt3 * rt7 * rt11 *
                (12.0 * gx(0) * gy(5) * gz(2) - 120.0 * gx(2) * gy(3) * gz(2) +
                 60.0 * gx(4) * gy(1) * gz(2) - gx(0) * gy(7) * gz(0) +
                 9.0 * gx(2) * gy(5) * gz(0) + 5.0 * gx(4) * gy(3) * gz(0) -
                 5.0 * gx(6) * gy(1) * gz(0)) /
                32.0;
  out.Q76c() += f * rt2 * rt3 * rt7 * rt11 * rt13 *
                (-gx(0) * gy(6) * gz(1) + 15.0 * gx(2) * gy(4) * gz(1) -
                 15.0 * gx(4) * gy(2) * gz(1) + gx(6) * gy(0) * gz(1)) /
                32.0;
  out.Q76s() += f * rt2 * rt3 * rt7 * rt11 * rt13 *
                (6.0 * gx(1) * gy(5) * gz(1) - 20.0 * gx(3) * gy(3) * gz(1) +
                 6.0 * gx(5) * gy(1) * gz(1)) /
                32.0;
  out.Q77c() += f * rt3 * rt11 * rt13 *
                (-7.0 * gx(1) * gy(6) * gz(0) + 35.0 * gx(3) * gy(4) * gz(0) -
                 21.0 * gx(5) * gy(2) * gz(0) + gx(7) * gy(0) * gz(0)) /
                32.0;
  out.Q77s() += f * rt3 * rt11 * rt13 *
                (-gx(0) * gy(7) * gz(0) + 21.0 * gx(2) * gy(5) * gz(0) -
                 35.0 * gx(4) * gy(3) * gz(0) + 7.0 * gx(6) * gy(1) * gz(0)) /
                32.0;
}

inline void addqlm_8(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {
  out.Q80() +=
      f *
      (128.0 * gx(0) * gy(0) * gz(8) - 1792.0 * gx(0) * gy(2) * gz(6) -
       1792.0 * gx(2) * gy(0) * gz(6) + 3360.0 * gx(0) * gy(4) * gz(4) +
       6720.0 * gx(2) * gy(2) * gz(4) + 3360.0 * gx(4) * gy(0) * gz(4) -
       1120.0 * gx(0) * gy(6) * gz(2) - 3360.0 * gx(2) * gy(4) * gz(2) -
       3360.0 * gx(4) * gy(2) * gz(2) - 1120.0 * gx(6) * gy(0) * gz(2) +
       35.0 * gx(0) * gy(8) * gz(0) + 140.0 * gx(2) * gy(6) * gz(0) +
       210.0 * gx(4) * gy(4) * gz(0) + 140.0 * gx(6) * gy(2) * gz(0) +
       35.0 * gx(8) * gy(0) * gz(0)) /
      128.0;
  out.Q81c() +=
      f *
      (192.0 * gx(1) * gy(0) * gz(7) - 1008.0 * gx(1) * gy(2) * gz(5) -
       1008.0 * gx(3) * gy(0) * gz(5) + 840.0 * gx(1) * gy(4) * gz(3) +
       1680.0 * gx(3) * gy(2) * gz(3) + 840.0 * gx(5) * gy(0) * gz(3) -
       105.0 * gx(1) * gy(6) * gz(1) - 315.0 * gx(3) * gy(4) * gz(1) -
       315.0 * gx(5) * gy(2) * gz(1) - 105.0 * gx(7) * gy(0) * gz(1)) /
      32.0;
  out.Q81s() +=
      f *
      (192.0 * gx(0) * gy(1) * gz(7) - 1008.0 * gx(0) * gy(3) * gz(5) -
       1008.0 * gx(2) * gy(1) * gz(5) + 840.0 * gx(0) * gy(5) * gz(3) +
       1680.0 * gx(2) * gy(3) * gz(3) + 840.0 * gx(4) * gy(1) * gz(3) -
       105.0 * gx(0) * gy(7) * gz(1) - 315.0 * gx(2) * gy(5) * gz(1) -
       315.0 * gx(4) * gy(3) * gz(1) - 105.0 * gx(6) * gy(1) * gz(1)) /
      32.0;
  out.Q82c() += f * rt2 * rt5 * rt7 *
                (-96.0 * gx(0) * gy(2) * gz(6) + 96.0 * gx(2) * gy(0) * gz(6) +
                 240.0 * gx(0) * gy(4) * gz(4) - 240.0 * gx(4) * gy(0) * gz(4) -
                 90.0 * gx(0) * gy(6) * gz(2) - 90.0 * gx(2) * gy(4) * gz(2) +
                 90.0 * gx(4) * gy(2) * gz(2) + 90.0 * gx(6) * gy(0) * gz(2) +
                 3.0 * gx(0) * gy(8) * gz(0) + 6.0 * gx(2) * gy(6) * gz(0) -
                 6.0 * gx(6) * gy(2) * gz(0) - 3.0 * gx(8) * gy(0) * gz(0)) /
                64.0;
  out.Q82s() += f * rt2 * rt5 * rt7 *
                (192.0 * gx(1) * gy(1) * gz(6) - 480.0 * gx(1) * gy(3) * gz(4) -
                 480.0 * gx(3) * gy(1) * gz(4) + 180.0 * gx(1) * gy(5) * gz(2) +
                 360.0 * gx(3) * gy(3) * gz(2) + 180.0 * gx(5) * gy(1) * gz(2) -
                 6.0 * gx(1) * gy(7) * gz(0) - 18.0 * gx(3) * gy(5) * gz(0) -
                 18.0 * gx(5) * gy(3) * gz(0) - 6.0 * gx(7) * gy(1) * gz(0)) /
                64.0;
  out.Q83c() += f * rt3 * rt5 * rt7 * rt11 *
                (-48.0 * gx(1) * gy(2) * gz(5) + 16.0 * gx(3) * gy(0) * gz(5) +
                 60.0 * gx(1) * gy(4) * gz(3) + 40.0 * gx(3) * gy(2) * gz(3) -
                 20.0 * gx(5) * gy(0) * gz(3) - 9.0 * gx(1) * gy(6) * gz(1) -
                 15.0 * gx(3) * gy(4) * gz(1) - 3.0 * gx(5) * gy(2) * gz(1) +
                 3.0 * gx(7) * gy(0) * gz(1)) /
                32.0;
  out.Q83s() += f * rt3 * rt5 * rt7 * rt11 *
                (-16.0 * gx(0) * gy(3) * gz(5) + 48.0 * gx(2) * gy(1) * gz(5) +
                 20.0 * gx(0) * gy(5) * gz(3) - 40.0 * gx(2) * gy(3) * gz(3) -
                 60.0 * gx(4) * gy(1) * gz(3) - 3.0 * gx(0) * gy(7) * gz(1) +
                 3.0 * gx(2) * gy(5) * gz(1) + 15.0 * gx(4) * gy(3) * gz(1) +
                 9.0 * gx(6) * gy(1) * gz(1)) /
                32.0;
  out.Q84c() += f * rt7 * rt11 *
                (120.0 * gx(0) * gy(4) * gz(4) - 720.0 * gx(2) * gy(2) * gz(4) +
                 120.0 * gx(4) * gy(0) * gz(4) - 72.0 * gx(0) * gy(6) * gz(2) +
                 360.0 * gx(2) * gy(4) * gz(2) + 360.0 * gx(4) * gy(2) * gz(2) -
                 72.0 * gx(6) * gy(0) * gz(2) + 3.0 * gx(0) * gy(8) * gz(0) -
                 12.0 * gx(2) * gy(6) * gz(0) - 30.0 * gx(4) * gy(4) * gz(0) -
                 12.0 * gx(6) * gy(2) * gz(0) + 3.0 * gx(8) * gy(0) * gz(0)) /
                64.0;
  out.Q84s() +=
      f * rt7 * rt11 *
      (-480.0 * gx(1) * gy(3) * gz(4) + 480.0 * gx(3) * gy(1) * gz(4) +
       288.0 * gx(1) * gy(5) * gz(2) - 288.0 * gx(5) * gy(1) * gz(2) -
       12.0 * gx(1) * gy(7) * gz(0) - 12.0 * gx(3) * gy(5) * gz(0) +
       12.0 * gx(5) * gy(3) * gz(0) + 12.0 * gx(7) * gy(1) * gz(0)) /
      64.0;
  out.Q85c() += f * rt7 * rt11 * rt13 *
                (60.0 * gx(1) * gy(4) * gz(3) - 120.0 * gx(3) * gy(2) * gz(3) +
                 12.0 * gx(5) * gy(0) * gz(3) - 15.0 * gx(1) * gy(6) * gz(1) +
                 15.0 * gx(3) * gy(4) * gz(1) + 27.0 * gx(5) * gy(2) * gz(1) -
                 3.0 * gx(7) * gy(0) * gz(1)) /
                32.0;
  out.Q85s() += f * rt7 * rt11 * rt13 *
                (12.0 * gx(0) * gy(5) * gz(3) - 120.0 * gx(2) * gy(3) * gz(3) +
                 60.0 * gx(4) * gy(1) * gz(3) - 3.0 * gx(0) * gy(7) * gz(1) +
                 27.0 * gx(2) * gy(5) * gz(1) + 15.0 * gx(4) * gy(3) * gz(1) -
                 15.0 * gx(6) * gy(1) * gz(1)) /
                32.0;
  out.Q86c() += f * rt2 * rt3 * rt11 * rt13 *
                (-14.0 * gx(0) * gy(6) * gz(2) + 210.0 * gx(2) * gy(4) * gz(2) -
                 210.0 * gx(4) * gy(2) * gz(2) + 14.0 * gx(6) * gy(0) * gz(2) +
                 gx(0) * gy(8) * gz(0) - 14.0 * gx(2) * gy(6) * gz(0) +
                 14.0 * gx(6) * gy(2) * gz(0) - gx(8) * gy(0) * gz(0)) /
                64.0;
  out.Q86s() += f * rt2 * rt3 * rt11 * rt13 *
                (84.0 * gx(1) * gy(5) * gz(2) - 280.0 * gx(3) * gy(3) * gz(2) +
                 84.0 * gx(5) * gy(1) * gz(2) - 6.0 * gx(1) * gy(7) * gz(0) +
                 14.0 * gx(3) * gy(5) * gz(0) + 14.0 * gx(5) * gy(3) * gz(0) -
                 6.0 * gx(7) * gy(1) * gz(0)) /
                64.0;
  out.Q87c() += f * rt5 * rt11 * rt13 *
                (-21.0 * gx(1) * gy(6) * gz(1) + 105.0 * gx(3) * gy(4) * gz(1) -
                 63.0 * gx(5) * gy(2) * gz(1) + 3.0 * gx(7) * gy(0) * gz(1)) /
                32.0;
  out.Q87s() += f * rt5 * rt11 * rt13 *
                (-3.0 * gx(0) * gy(7) * gz(1) + 63.0 * gx(2) * gy(5) * gz(1) -
                 105.0 * gx(4) * gy(3) * gz(1) + 21.0 * gx(6) * gy(1) * gz(1)) /
                32.0;
  out.Q88c() += f * rt5 * rt11 * rt13 *
                (3.0 * gx(0) * gy(8) * gz(0) - 84.0 * gx(2) * gy(6) * gz(0) +
                 210.0 * gx(4) * gy(4) * gz(0) - 84.0 * gx(6) * gy(2) * gz(0) +
                 3.0 * gx(8) * gy(0) * gz(0)) /
                128.0;
  out.Q88s() += f * rt5 * rt11 * rt13 *
                (-24.0 * gx(1) * gy(7) * gz(0) + 168.0 * gx(3) * gy(5) * gz(0) -
                 168.0 * gx(5) * gy(3) * gz(0) + 24.0 * gx(7) * gy(1) * gz(0)) /
                128.0;
}

inline void addqlm_9(double f, Eigen::Ref<const Vec> gx,
                     Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                     Mult &out) {

  out.Q90() +=
      f *
      (128.0 * gx(0) * gy(0) * gz(9) - 2304.0 * gx(0) * gy(2) * gz(7) -
       2304.0 * gx(2) * gy(0) * gz(7) + 6048.0 * gx(0) * gy(4) * gz(5) +
       12096.0 * gx(2) * gy(2) * gz(5) + 6048.0 * gx(4) * gy(0) * gz(5) -
       3360.0 * gx(0) * gy(6) * gz(3) - 10080.0 * gx(2) * gy(4) * gz(3) -
       10080.0 * gx(4) * gy(2) * gz(3) - 3360.0 * gx(6) * gy(0) * gz(3) +
       315.0 * gx(0) * gy(8) * gz(1) + 1260.0 * gx(2) * gy(6) * gz(1) +
       1890.0 * gx(4) * gy(4) * gz(1) + 1260.0 * gx(6) * gy(2) * gz(1) +
       315.0 * gx(8) * gy(0) * gz(1)) /
      128.0;
  out.Q91c() +=
      f * rt5 *
      (384.0 * gx(1) * gy(0) * gz(8) - 2688.0 * gx(1) * gy(2) * gz(6) -
       2688.0 * gx(3) * gy(0) * gz(6) + 3360.0 * gx(1) * gy(4) * gz(4) +
       6720.0 * gx(3) * gy(2) * gz(4) + 3360.0 * gx(5) * gy(0) * gz(4) -
       840.0 * gx(1) * gy(6) * gz(2) - 2520.0 * gx(3) * gy(4) * gz(2) -
       2520.0 * gx(5) * gy(2) * gz(2) - 840.0 * gx(7) * gy(0) * gz(2) +
       21.0 * gx(1) * gy(8) * gz(0) + 84.0 * gx(3) * gy(6) * gz(0) +
       126.0 * gx(5) * gy(4) * gz(0) + 84.0 * gx(7) * gy(2) * gz(0) +
       21.0 * gx(9) * gy(0) * gz(0)) /
      128.0;
  out.Q91s() +=
      f * rt5 *
      (384.0 * gx(0) * gy(1) * gz(8) - 2688.0 * gx(0) * gy(3) * gz(6) -
       2688.0 * gx(2) * gy(1) * gz(6) + 3360.0 * gx(0) * gy(5) * gz(4) +
       6720.0 * gx(2) * gy(3) * gz(4) + 3360.0 * gx(4) * gy(1) * gz(4) -
       840.0 * gx(0) * gy(7) * gz(2) - 2520.0 * gx(2) * gy(5) * gz(2) -
       2520.0 * gx(4) * gy(3) * gz(2) - 840.0 * gx(6) * gy(1) * gz(2) +
       21.0 * gx(0) * gy(9) * gz(0) + 84.0 * gx(2) * gy(7) * gz(0) +
       126.0 * gx(4) * gy(5) * gz(0) + 84.0 * gx(6) * gy(3) * gz(0) +
       21.0 * gx(8) * gy(1) * gz(0)) /
      128.0;
  out.Q92c() += f * rt2 * rt5 * rt11 *
                (-96.0 * gx(0) * gy(2) * gz(7) + 96.0 * gx(2) * gy(0) * gz(7) +
                 336.0 * gx(0) * gy(4) * gz(5) - 336.0 * gx(4) * gy(0) * gz(5) -
                 210.0 * gx(0) * gy(6) * gz(3) - 210.0 * gx(2) * gy(4) * gz(3) +
                 210.0 * gx(4) * gy(2) * gz(3) + 210.0 * gx(6) * gy(0) * gz(3) +
                 21.0 * gx(0) * gy(8) * gz(1) + 42.0 * gx(2) * gy(6) * gz(1) -
                 42.0 * gx(6) * gy(2) * gz(1) - 21.0 * gx(8) * gy(0) * gz(1)) /
                64.0;
  out.Q92s() += f * rt2 * rt5 * rt11 *
                (192.0 * gx(1) * gy(1) * gz(7) - 672.0 * gx(1) * gy(3) * gz(5) -
                 672.0 * gx(3) * gy(1) * gz(5) + 420.0 * gx(1) * gy(5) * gz(3) +
                 840.0 * gx(3) * gy(3) * gz(3) + 420.0 * gx(5) * gy(1) * gz(3) -
                 42.0 * gx(1) * gy(7) * gz(1) - 126.0 * gx(3) * gy(5) * gz(1) -
                 126.0 * gx(5) * gy(3) * gz(1) - 42.0 * gx(7) * gy(1) * gz(1)) /
                64.0;
  out.Q93c() += f * rt2 * rt3 * rt5 * rt7 * rt11 *
                (-192.0 * gx(1) * gy(2) * gz(6) + 64.0 * gx(3) * gy(0) * gz(6) +
                 360.0 * gx(1) * gy(4) * gz(4) + 240.0 * gx(3) * gy(2) * gz(4) -
                 120.0 * gx(5) * gy(0) * gz(4) - 108.0 * gx(1) * gy(6) * gz(2) -
                 180.0 * gx(3) * gy(4) * gz(2) - 36.0 * gx(5) * gy(2) * gz(2) +
                 36.0 * gx(7) * gy(0) * gz(2) + 3.0 * gx(1) * gy(8) * gz(0) +
                 8.0 * gx(3) * gy(6) * gz(0) + 6.0 * gx(5) * gy(4) * gz(0) -
                 gx(9) * gy(0) * gz(0)) /
                128.0;
  out.Q93s() += f * rt2 * rt3 * rt5 * rt7 * rt11 *
                (-64.0 * gx(0) * gy(3) * gz(6) + 192.0 * gx(2) * gy(1) * gz(6) +
                 120.0 * gx(0) * gy(5) * gz(4) - 240.0 * gx(2) * gy(3) * gz(4) -
                 360.0 * gx(4) * gy(1) * gz(4) - 36.0 * gx(0) * gy(7) * gz(2) +
                 36.0 * gx(2) * gy(5) * gz(2) + 180.0 * gx(4) * gy(3) * gz(2) +
                 108.0 * gx(6) * gy(1) * gz(2) + gx(0) * gy(9) * gz(0) -
                 6.0 * gx(4) * gy(5) * gz(0) - 8.0 * gx(6) * gy(3) * gz(0) -
                 3.0 * gx(8) * gy(1) * gz(0)) /
                128.0;
  out.Q94c() += f * rt5 * rt7 * rt11 * rt13 *
                (24.0 * gx(0) * gy(4) * gz(5) - 144.0 * gx(2) * gy(2) * gz(5) +
                 24.0 * gx(4) * gy(0) * gz(5) - 24.0 * gx(0) * gy(6) * gz(3) +
                 120.0 * gx(2) * gy(4) * gz(3) + 120.0 * gx(4) * gy(2) * gz(3) -
                 24.0 * gx(6) * gy(0) * gz(3) + 3.0 * gx(0) * gy(8) * gz(1) -
                 12.0 * gx(2) * gy(6) * gz(1) - 30.0 * gx(4) * gy(4) * gz(1) -
                 12.0 * gx(6) * gy(2) * gz(1) + 3.0 * gx(8) * gy(0) * gz(1)) /
                64.0;
  out.Q94s() += f * rt5 * rt7 * rt11 * rt13 *
                (-96.0 * gx(1) * gy(3) * gz(5) + 96.0 * gx(3) * gy(1) * gz(5) +
                 96.0 * gx(1) * gy(5) * gz(3) - 96.0 * gx(5) * gy(1) * gz(3) -
                 12.0 * gx(1) * gy(7) * gz(1) - 12.0 * gx(3) * gy(5) * gz(1) +
                 12.0 * gx(5) * gy(3) * gz(1) + 12.0 * gx(7) * gy(1) * gz(1)) /
                64.0;
  out.Q95c() +=
      f * rt2 * rt11 * rt13 *
      (840.0 * gx(1) * gy(4) * gz(4) - 1680.0 * gx(3) * gy(2) * gz(4) +
       168.0 * gx(5) * gy(0) * gz(4) - 420.0 * gx(1) * gy(6) * gz(2) +
       420.0 * gx(3) * gy(4) * gz(2) + 756.0 * gx(5) * gy(2) * gz(2) -
       84.0 * gx(7) * gy(0) * gz(2) + 15.0 * gx(1) * gy(8) * gz(0) -
       42.0 * gx(5) * gy(4) * gz(0) - 24.0 * gx(7) * gy(2) * gz(0) +
       3.0 * gx(9) * gy(0) * gz(0)) /
      128.0;
  out.Q95s() +=
      f * rt2 * rt11 * rt13 *
      (168.0 * gx(0) * gy(5) * gz(4) - 1680.0 * gx(2) * gy(3) * gz(4) +
       840.0 * gx(4) * gy(1) * gz(4) - 84.0 * gx(0) * gy(7) * gz(2) +
       756.0 * gx(2) * gy(5) * gz(2) + 420.0 * gx(4) * gy(3) * gz(2) -
       420.0 * gx(6) * gy(1) * gz(2) + 3.0 * gx(0) * gy(9) * gz(0) -
       24.0 * gx(2) * gy(7) * gz(0) - 42.0 * gx(4) * gy(5) * gz(0) +
       15.0 * gx(8) * gy(1) * gz(0)) /
      128.0;
  out.Q96c() += f * rt2 * rt3 * rt5 * rt11 * rt13 *
                (-14.0 * gx(0) * gy(6) * gz(3) + 210.0 * gx(2) * gy(4) * gz(3) -
                 210.0 * gx(4) * gy(2) * gz(3) + 14.0 * gx(6) * gy(0) * gz(3) +
                 3.0 * gx(0) * gy(8) * gz(1) - 42.0 * gx(2) * gy(6) * gz(1) +
                 42.0 * gx(6) * gy(2) * gz(1) - 3.0 * gx(8) * gy(0) * gz(1)) /
                64.0;
  out.Q96s() += f * rt2 * rt3 * rt5 * rt11 * rt13 *
                (84.0 * gx(1) * gy(5) * gz(3) - 280.0 * gx(3) * gy(3) * gz(3) +
                 84.0 * gx(5) * gy(1) * gz(3) - 18.0 * gx(1) * gy(7) * gz(1) +
                 42.0 * gx(3) * gy(5) * gz(1) + 42.0 * gx(5) * gy(3) * gz(1) -
                 18.0 * gx(7) * gy(1) * gz(1)) /
                64.0;
  out.Q97c() +=
      f * rt2 * rt5 * rt11 * rt13 *
      (-336.0 * gx(1) * gy(6) * gz(2) + 1680.0 * gx(3) * gy(4) * gz(2) -
       1008.0 * gx(5) * gy(2) * gz(2) + 48.0 * gx(7) * gy(0) * gz(2) +
       21.0 * gx(1) * gy(8) * gz(0) - 84.0 * gx(3) * gy(6) * gz(0) -
       42.0 * gx(5) * gy(4) * gz(0) + 60.0 * gx(7) * gy(2) * gz(0) -
       3.0 * gx(9) * gy(0) * gz(0)) /
      256.0;
  out.Q97s() +=
      f * rt2 * rt5 * rt11 * rt13 *
      (-48.0 * gx(0) * gy(7) * gz(2) + 1008.0 * gx(2) * gy(5) * gz(2) -
       1680.0 * gx(4) * gy(3) * gz(2) + 336.0 * gx(6) * gy(1) * gz(2) +
       3.0 * gx(0) * gy(9) * gz(0) - 60.0 * gx(2) * gy(7) * gz(0) +
       42.0 * gx(4) * gy(5) * gz(0) + 84.0 * gx(6) * gy(3) * gz(0) -
       21.0 * gx(8) * gy(1) * gz(0)) /
      256.0;
  out.Q98c() += f * rt5 * rt11 * rt13 * rt17 *
                (3.0 * gx(0) * gy(8) * gz(1) - 84.0 * gx(2) * gy(6) * gz(1) +
                 210.0 * gx(4) * gy(4) * gz(1) - 84.0 * gx(6) * gy(2) * gz(1) +
                 3.0 * gx(8) * gy(0) * gz(1)) /
                128.0;
  out.Q98s() += f * rt5 * rt11 * rt13 * rt17 *
                (-24.0 * gx(1) * gy(7) * gz(1) + 168.0 * gx(3) * gy(5) * gz(1) -
                 168.0 * gx(5) * gy(3) * gz(1) + 24.0 * gx(7) * gy(1) * gz(1)) /
                128.0;
  out.Q99c() += f * rt2 * rt5 * rt11 * rt13 * rt17 *
                (9.0 * gx(1) * gy(8) * gz(0) - 84.0 * gx(3) * gy(6) * gz(0) +
                 126.0 * gx(5) * gy(4) * gz(0) - 36.0 * gx(7) * gy(2) * gz(0) +
                 gx(9) * gy(0) * gz(0)) /
                256.0;
  out.Q99s() += f * rt2 * rt5 * rt11 * rt13 * rt17 *
                (gx(0) * gy(9) * gz(0) - 36.0 * gx(2) * gy(7) * gz(0) +
                 126.0 * gx(4) * gy(5) * gz(0) - 84.0 * gx(6) * gy(3) * gz(0) +
                 9.0 * gx(8) * gy(1) * gz(0)) /
                256.0;
}

inline void addqlm_10(double f, Eigen::Ref<const Vec> gx,
                      Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
                      Mult &out) {
  out.QA0() +=
      f *
      (256.0 * gx(0) * gy(0) * gz(10) - 5760.0 * gx(0) * gy(2) * gz(8) -
       5760.0 * gx(2) * gy(0) * gz(8) + 20160.0 * gx(0) * gy(4) * gz(6) +
       40320.0 * gx(2) * gy(2) * gz(6) + 20160.0 * gx(4) * gy(0) * gz(6) -
       16800.0 * gx(0) * gy(6) * gz(4) - 50400.0 * gx(2) * gy(4) * gz(4) -
       50400.0 * gx(4) * gy(2) * gz(4) - 16800.0 * gx(6) * gy(0) * gz(4) +
       3150.0 * gx(0) * gy(8) * gz(2) + 12600.0 * gx(2) * gy(6) * gz(2) +
       18900.0 * gx(4) * gy(4) * gz(2) + 12600.0 * gx(6) * gy(2) * gz(2) +
       3150.0 * gx(8) * gy(0) * gz(2) - 63.0 * gx(0) * gy(10) * gz(0) -
       315.0 * gx(2) * gy(8) * gz(0) - 630.0 * gx(4) * gy(6) * gz(0) -
       630.0 * gx(6) * gy(4) * gz(0) - 315.0 * gx(8) * gy(2) * gz(0) -
       63.0 * gx(10) * gy(0) * gz(0)) /
      256.0;

  out.QA1c() +=
      f * rt5 * rt11 *
      (128.0 * gx(1) * gy(0) * gz(9) - 1152.0 * gx(1) * gy(2) * gz(7) -
       1152.0 * gx(3) * gy(0) * gz(7) + 2016.0 * gx(1) * gy(4) * gz(5) +
       4032.0 * gx(3) * gy(2) * gz(5) + 2016.0 * gx(5) * gy(0) * gz(5) -
       840.0 * gx(1) * gy(6) * gz(3) - 2520.0 * gx(3) * gy(4) * gz(3) -
       2520.0 * gx(5) * gy(2) * gz(3) - 840.0 * gx(7) * gy(0) * gz(3) +
       63.0 * gx(1) * gy(8) * gz(1) + 252.0 * gx(3) * gy(6) * gz(1) +
       378.0 * gx(5) * gy(4) * gz(1) + 252.0 * gx(7) * gy(2) * gz(1) +
       63.0 * gx(9) * gy(0) * gz(1)) /
      128.0;
  out.QA1s() +=
      f * rt5 * rt11 *
      (128.0 * gx(0) * gy(1) * gz(9) - 1152.0 * gx(0) * gy(3) * gz(7) -
       1152.0 * gx(2) * gy(1) * gz(7) + 2016.0 * gx(0) * gy(5) * gz(5) +
       4032.0 * gx(2) * gy(3) * gz(5) + 2016.0 * gx(4) * gy(1) * gz(5) -
       840.0 * gx(0) * gy(7) * gz(3) - 2520.0 * gx(2) * gy(5) * gz(3) -
       2520.0 * gx(4) * gy(3) * gz(3) - 840.0 * gx(6) * gy(1) * gz(3) +
       63.0 * gx(0) * gy(9) * gz(1) + 252.0 * gx(2) * gy(7) * gz(1) +
       378.0 * gx(4) * gy(5) * gz(1) + 252.0 * gx(6) * gy(3) * gz(1) +
       63.0 * gx(8) * gy(1) * gz(1)) /
      128.0;
  out.QA2c() +=
      f * rt3 * rt5 * rt11 *
      (-384.0 * gx(0) * gy(2) * gz(8) + 384.0 * gx(2) * gy(0) * gz(8) +
       1792.0 * gx(0) * gy(4) * gz(6) - 1792.0 * gx(4) * gy(0) * gz(6) -
       1680.0 * gx(0) * gy(6) * gz(4) - 1680.0 * gx(2) * gy(4) * gz(4) +
       1680.0 * gx(4) * gy(2) * gz(4) + 1680.0 * gx(6) * gy(0) * gz(4) +
       336.0 * gx(0) * gy(8) * gz(2) + 672.0 * gx(2) * gy(6) * gz(2) -
       672.0 * gx(6) * gy(2) * gz(2) - 336.0 * gx(8) * gy(0) * gz(2) -
       7.0 * gx(0) * gy(10) * gz(0) - 21.0 * gx(2) * gy(8) * gz(0) -
       14.0 * gx(4) * gy(6) * gz(0) + 14.0 * gx(6) * gy(4) * gz(0) +
       21.0 * gx(8) * gy(2) * gz(0) + 7.0 * gx(10) * gy(0) * gz(0)) /
      256.0;
  out.QA2s() +=
      f * rt3 * rt5 * rt11 *
      (768.0 * gx(1) * gy(1) * gz(8) - 3584.0 * gx(1) * gy(3) * gz(6) -
       3584.0 * gx(3) * gy(1) * gz(6) + 3360.0 * gx(1) * gy(5) * gz(4) +
       6720.0 * gx(3) * gy(3) * gz(4) + 3360.0 * gx(5) * gy(1) * gz(4) -
       672.0 * gx(1) * gy(7) * gz(2) - 2016.0 * gx(3) * gy(5) * gz(2) -
       2016.0 * gx(5) * gy(3) * gz(2) - 672.0 * gx(7) * gy(1) * gz(2) +
       14.0 * gx(1) * gy(9) * gz(0) + 56.0 * gx(3) * gy(7) * gz(0) +
       84.0 * gx(5) * gy(5) * gz(0) + 56.0 * gx(7) * gy(3) * gz(0) +
       14.0 * gx(9) * gy(1) * gz(0)) /
      256.0;
  out.QA3c() += f * rt2 * rt3 * rt5 * rt11 * rt13 *
                (-192.0 * gx(1) * gy(2) * gz(7) + 64.0 * gx(3) * gy(0) * gz(7) +
                 504.0 * gx(1) * gy(4) * gz(5) + 336.0 * gx(3) * gy(2) * gz(5) -
                 168.0 * gx(5) * gy(0) * gz(5) - 252.0 * gx(1) * gy(6) * gz(3) -
                 420.0 * gx(3) * gy(4) * gz(3) - 84.0 * gx(5) * gy(2) * gz(3) +
                 84.0 * gx(7) * gy(0) * gz(3) + 21.0 * gx(1) * gy(8) * gz(1) +
                 56.0 * gx(3) * gy(6) * gz(1) + 42.0 * gx(5) * gy(4) * gz(1) -
                 7.0 * gx(9) * gy(0) * gz(1)) /
                128.0;
  out.QA3s() += f * rt2 * rt3 * rt5 * rt11 * rt13 *
                (-64.0 * gx(0) * gy(3) * gz(7) + 192.0 * gx(2) * gy(1) * gz(7) +
                 168.0 * gx(0) * gy(5) * gz(5) - 336.0 * gx(2) * gy(3) * gz(5) -
                 504.0 * gx(4) * gy(1) * gz(5) - 84.0 * gx(0) * gy(7) * gz(3) +
                 84.0 * gx(2) * gy(5) * gz(3) + 420.0 * gx(4) * gy(3) * gz(3) +
                 252.0 * gx(6) * gy(1) * gz(3) + 7.0 * gx(0) * gy(9) * gz(1) -
                 42.0 * gx(4) * gy(5) * gz(1) - 56.0 * gx(6) * gy(3) * gz(1) -
                 21.0 * gx(8) * gy(1) * gz(1)) /
                128.0;
  out.QA4c() += f * rt3 * rt5 * rt11 * rt13 *
                (112.0 * gx(0) * gy(4) * gz(6) - 672.0 * gx(2) * gy(2) * gz(6) +
                 112.0 * gx(4) * gy(0) * gz(6) - 168.0 * gx(0) * gy(6) * gz(4) +
                 840.0 * gx(2) * gy(4) * gz(4) + 840.0 * gx(4) * gy(2) * gz(4) -
                 168.0 * gx(6) * gy(0) * gz(4) + 42.0 * gx(0) * gy(8) * gz(2) -
                 168.0 * gx(2) * gy(6) * gz(2) - 420.0 * gx(4) * gy(4) * gz(2) -
                 168.0 * gx(6) * gy(2) * gz(2) + 42.0 * gx(8) * gy(0) * gz(2) -
                 gx(0) * gy(10) * gz(0) + 3.0 * gx(2) * gy(8) * gz(0) +
                 14.0 * gx(4) * gy(6) * gz(0) + 14.0 * gx(6) * gy(4) * gz(0) +
                 3.0 * gx(8) * gy(2) * gz(0) - gx(10) * gy(0) * gz(0)) /
                128.0;
  out.QA4s() +=
      f * rt3 * rt5 * rt11 * rt13 *
      (-448.0 * gx(1) * gy(3) * gz(6) + 448.0 * gx(3) * gy(1) * gz(6) +
       672.0 * gx(1) * gy(5) * gz(4) - 672.0 * gx(5) * gy(1) * gz(4) -
       168.0 * gx(1) * gy(7) * gz(2) - 168.0 * gx(3) * gy(5) * gz(2) +
       168.0 * gx(5) * gy(3) * gz(2) + 168.0 * gx(7) * gy(1) * gz(2) +
       4.0 * gx(1) * gy(9) * gz(0) + 8.0 * gx(3) * gy(7) * gz(0) -
       8.0 * gx(7) * gy(3) * gz(0) - 4.0 * gx(9) * gy(1) * gz(0)) /
      128.0;
  out.QA5c() +=
      f * rt2 * rt3 * rt11 * rt13 *
      (840.0 * gx(1) * gy(4) * gz(5) - 1680.0 * gx(3) * gy(2) * gz(5) +
       168.0 * gx(5) * gy(0) * gz(5) - 700.0 * gx(1) * gy(6) * gz(3) +
       700.0 * gx(3) * gy(4) * gz(3) + 1260.0 * gx(5) * gy(2) * gz(3) -
       140.0 * gx(7) * gy(0) * gz(3) + 75.0 * gx(1) * gy(8) * gz(1) -
       210.0 * gx(5) * gy(4) * gz(1) - 120.0 * gx(7) * gy(2) * gz(1) +
       15.0 * gx(9) * gy(0) * gz(1)) /
      128.0;
  out.QA5s() +=
      f * rt2 * rt3 * rt11 * rt13 *
      (168.0 * gx(0) * gy(5) * gz(5) - 1680.0 * gx(2) * gy(3) * gz(5) +
       840.0 * gx(4) * gy(1) * gz(5) - 140.0 * gx(0) * gy(7) * gz(3) +
       1260.0 * gx(2) * gy(5) * gz(3) + 700.0 * gx(4) * gy(3) * gz(3) -
       700.0 * gx(6) * gy(1) * gz(3) + 15.0 * gx(0) * gy(9) * gz(1) -
       120.0 * gx(2) * gy(7) * gz(1) - 210.0 * gx(4) * gy(5) * gz(1) +
       75.0 * gx(8) * gy(1) * gz(1)) /
      128.0;
  out.QA6c() +=
      f * rt2 * rt3 * rt5 * rt11 * rt13 *
      (-224.0 * gx(0) * gy(6) * gz(4) + 3360.0 * gx(2) * gy(4) * gz(4) -
       3360.0 * gx(4) * gy(2) * gz(4) + 224.0 * gx(6) * gy(0) * gz(4) +
       96.0 * gx(0) * gy(8) * gz(2) - 1344.0 * gx(2) * gy(6) * gz(2) +
       1344.0 * gx(6) * gy(2) * gz(2) - 96.0 * gx(8) * gy(0) * gz(2) -
       3.0 * gx(0) * gy(10) * gz(0) + 39.0 * gx(2) * gy(8) * gz(0) +
       42.0 * gx(4) * gy(6) * gz(0) - 42.0 * gx(6) * gy(4) * gz(0) -
       39.0 * gx(8) * gy(2) * gz(0) + 3.0 * gx(10) * gy(0) * gz(0)) /
      512.0;
  out.QA6s() +=
      f * rt2 * rt3 * rt5 * rt11 * rt13 *
      (1344.0 * gx(1) * gy(5) * gz(4) - 4480.0 * gx(3) * gy(3) * gz(4) +
       1344.0 * gx(5) * gy(1) * gz(4) - 576.0 * gx(1) * gy(7) * gz(2) +
       1344.0 * gx(3) * gy(5) * gz(2) + 1344.0 * gx(5) * gy(3) * gz(2) -
       576.0 * gx(7) * gy(1) * gz(2) + 18.0 * gx(1) * gy(9) * gz(0) -
       24.0 * gx(3) * gy(7) * gz(0) - 84.0 * gx(5) * gy(5) * gz(0) -
       24.0 * gx(7) * gy(3) * gz(0) + 18.0 * gx(9) * gy(1) * gz(0)) /
      512.0;
  out.QA7c() += f * rt2 * rt3 * rt5 * rt11 * rt13 * rt17 *
                (-112.0 * gx(1) * gy(6) * gz(3) +
                 560.0 * gx(3) * gy(4) * gz(3) - 336.0 * gx(5) * gy(2) * gz(3) +
                 16.0 * gx(7) * gy(0) * gz(3) + 21.0 * gx(1) * gy(8) * gz(1) -
                 84.0 * gx(3) * gy(6) * gz(1) - 42.0 * gx(5) * gy(4) * gz(1) +
                 60.0 * gx(7) * gy(2) * gz(1) - 3.0 * gx(9) * gy(0) * gz(1)) /
                256.0;
  out.QA7s() += f * rt2 * rt3 * rt5 * rt11 * rt13 * rt17 *
                (-16.0 * gx(0) * gy(7) * gz(3) + 336.0 * gx(2) * gy(5) * gz(3) -
                 560.0 * gx(4) * gy(3) * gz(3) + 112.0 * gx(6) * gy(1) * gz(3) +
                 3.0 * gx(0) * gy(9) * gz(1) - 60.0 * gx(2) * gy(7) * gz(1) +
                 42.0 * gx(4) * gy(5) * gz(1) + 84.0 * gx(6) * gy(3) * gz(1) -
                 21.0 * gx(8) * gy(1) * gz(1)) /
                256.0;
  out.QA8c() += f * rt5 * rt11 * rt13 * rt17 *
                (18.0 * gx(0) * gy(8) * gz(2) - 504.0 * gx(2) * gy(6) * gz(2) +
                 1260.0 * gx(4) * gy(4) * gz(2) -
                 504.0 * gx(6) * gy(2) * gz(2) + 18.0 * gx(8) * gy(0) * gz(2) -
                 gx(0) * gy(10) * gz(0) + 27.0 * gx(2) * gy(8) * gz(0) -
                 42.0 * gx(4) * gy(6) * gz(0) - 42.0 * gx(6) * gy(4) * gz(0) +
                 27.0 * gx(8) * gy(2) * gz(0) - gx(10) * gy(0) * gz(0)) /
                256.0;
  out.QA8s() +=
      f * rt5 * rt11 * rt13 * rt17 *
      (-144.0 * gx(1) * gy(7) * gz(2) + 1008.0 * gx(3) * gy(5) * gz(2) -
       1008.0 * gx(5) * gy(3) * gz(2) + 144.0 * gx(7) * gy(1) * gz(2) +
       8.0 * gx(1) * gy(9) * gz(0) - 48.0 * gx(3) * gy(7) * gz(0) +
       48.0 * gx(7) * gy(3) * gz(0) - 8.0 * gx(9) * gy(1) * gz(0)) /
      256.0;
  out.QA9c() += f * rt2 * rt5 * rt11 * rt13 * rt17 * rt19 *
                (9.0 * gx(1) * gy(8) * gz(1) - 84.0 * gx(3) * gy(6) * gz(1) +
                 126.0 * gx(5) * gy(4) * gz(1) - 36.0 * gx(7) * gy(2) * gz(1) +
                 gx(9) * gy(0) * gz(1)) /
                256.0;
  out.QA9s() += f * rt2 * rt5 * rt11 * rt13 * rt17 * rt19 *
                (gx(0) * gy(9) * gz(1) - 36.0 * gx(2) * gy(7) * gz(1) +
                 126.0 * gx(4) * gy(5) * gz(1) - 84.0 * gx(6) * gy(3) * gz(1) +
                 9.0 * gx(8) * gy(1) * gz(1)) /
                256.0;
  out.QAAc() += f * rt2 * rt11 * rt13 * rt17 * rt19 *
                (-gx(0) * gy(10) * gz(0) + 45.0 * gx(2) * gy(8) * gz(0) -
                 210.0 * gx(4) * gy(6) * gz(0) + 210.0 * gx(6) * gy(4) * gz(0) -
                 45.0 * gx(8) * gy(2) * gz(0) + gx(10) * gy(0) * gz(0)) /
                512.0;
  out.QAAs() += f * rt2 * rt11 * rt13 * rt17 * rt19 *
                (10.0 * gx(1) * gy(9) * gz(0) - 120.0 * gx(3) * gy(7) * gz(0) +
                 252.0 * gx(5) * gy(5) * gz(0) - 120.0 * gx(7) * gy(3) * gz(0) +
                 10.0 * gx(9) * gy(1) * gz(0)) /
                512.0;
}

void addqlm(int l, int lmax, double f, Eigen::Ref<const Vec> gx,
            Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz, Mult &out) {

  int LL = std::min(l, lmax);

  addqlm_0(f, gx, gy, gz, out);
  if (LL == 0)
    return;

  addqlm_1(f, gx, gy, gz, out);
  if (LL == 1)
    return;

  addqlm_2(f, gx, gy, gz, out);
  if (LL == 2)
    return;

  addqlm_3(f, gx, gy, gz, out);
  if (LL == 3)
    return;

  addqlm_4(f, gx, gy, gz, out);
  if (LL == 4)
    return;

  addqlm_5(f, gx, gy, gz, out);
  if (LL == 5)
    return;

  addqlm_6(f, gx, gy, gz, out);
  if (LL == 6)
    return;

  addqlm_7(f, gx, gy, gz, out);
  if (LL == 7)
    return;

  addqlm_8(f, gx, gy, gz, out);
  if (LL == 8)
    return;

  addqlm_9(f, gx, gy, gz, out);
  if (LL == 9)
    return;

  addqlm_10(f, gx, gy, gz, out);
}

void addql0(int l, double f, Eigen::Ref<const Vec> gx, Eigen::Ref<const Vec> gy,
            Eigen::Ref<const Vec> gz, Mult &mult) {
  auto &q = mult.q;

  double xy0 = gx(0) * gy(0);
  q(0) += f * xy0 * gz(0);
  if (l == 0)
    return;

  q(1) += f * xy0 * gz(1);
  if (l == 1)
    return;

  double xy2 = gx(2) * gy(0) + gx(0) * gy(2);
  q(2) += f * (xy0 * gz(2) - 0.5 * xy2 * gz(0));
  if (l == 2)
    return;

  q(3) += f * (xy0 * gz(3) - 1.5 * xy2 * gz(1));
  if (l == 3)
    return;

  double xy4 = gx(4) * gy(0) + 2.0 * gx(2) * gy(2) + gx(0) * gy(4);
  q(4) +=
      0.125 * f * (8.0 * xy0 * gz(4) - 24.0 * xy2 * gz(2) + 3.0 * xy4 * gz(0));
  if (l == 4)
    return;

  q(5) +=
      0.125 * f * (8.0 * gz(5) * xy0 - 40.0 * gz(3) * xy2 + 15.0 * gz(1) * xy4);
  if (l == 5)
    return;

  double xy6 =
      gx(6) * gy(0) + 3.0 * gx(4) * gy(2) + 3.0 * gx(2) * gy(4) + gx(0) * gy(6);
  q(6) += 0.0625 * f *
          (16.0 * xy0 * gz(6) - 120.0 * xy2 * gz(4) + 90.0 * xy4 * gz(2) -
           5.0 * xy6 * gz(0));
  if (l == 6)
    return;

  q(7) += 0.0625 * f *
          (16.0 * xy0 * gz(7) - 168.0 * xy2 * gz(5) + 210.0 * xy4 * gz(3) -
           35.0 * xy6 * gz(1));
  if (l == 7)
    return;

  double xy8 = gx(8) * gy(0) + 4.0 * gx(6) * gy(2) + 6.0 * gx(4) * gy(4) +
               4.0 * gx(2) * gy(6) + gx(0) * gy(8);
  q(8) += 0.0078125 * f *
          (128.0 * xy0 * gz(8) - 1792.0 * xy2 * gz(6) + 3360.0 * xy4 * gz(4) -
           1120.0 * xy6 * gz(2) + 35.0 * xy8 * gz(0));
  if (l == 8)
    return;

  q(9) += 0.0078125 * f *
          (128.0 * xy0 * gz(9) - 2304.0 * xy2 * gz(7) + 6048.0 * xy4 * gz(5) -
           3360.0 * xy6 * gz(3) + 315.0 * xy8 * gz(1));
  if (l == 9)
    return;

  double xy10 = gx(10) * gy(0) + 5.0 * gx(8) * gy(2) + 10.0 * gx(6) * gy(4) +
                10.0 * gx(4) * gy(6) + 5.0 * gx(2) * gy(8) + gx(0) * gy(10);
  q(10) +=
      0.00390625 * f *
      (256.0 * xy0 * gz(10) - 5760.0 * xy2 * gz(8) + 20160.0 * xy4 * gz(6) -
       16800.0 * xy6 * gz(4) + 3150.0 * xy8 * gz(2) - 63.0 * xy10 * gz(0));

  if (l == 10)
    return;

  q(11) +=
      0.00390625 * f *
      (256.0 * xy0 * gz(11) - 7040.0 * xy2 * gz(9) + 31680.0 * xy4 * gz(7) -
       36960.0 * xy6 * gz(5) + 11550.0 * xy8 * gz(3) - 693.0 * xy10 * gz(1));
}

} // namespace occ::dma
