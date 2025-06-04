#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/macros.h>
#include <string>

namespace occ::dma {

struct Mult {
  Mult(int n);
  Mult();
  int max_rank{0};
  Vec q;

  inline int num_components() const { return (max_rank + 1) * (max_rank + 1); }

  std::string to_string(int lm) const;

  // Level 0
  OCC_ALWAYS_INLINE double &Q00() { return q(0); }
  OCC_ALWAYS_INLINE double &charge() { return q(0); }

  // Level 1
  OCC_ALWAYS_INLINE double &Q10() { return q(1); }
  OCC_ALWAYS_INLINE double &Q11c() { return q(2); }
  OCC_ALWAYS_INLINE double &Q11s() { return q(3); }

  // Level 2
  OCC_ALWAYS_INLINE double &Q20() { return q(4); }
  OCC_ALWAYS_INLINE double &Q21c() { return q(5); }
  OCC_ALWAYS_INLINE double &Q21s() { return q(6); }
  OCC_ALWAYS_INLINE double &Q22c() { return q(7); }
  OCC_ALWAYS_INLINE double &Q22s() { return q(8); }

  // Level 3
  OCC_ALWAYS_INLINE double &Q30() { return q(9); }
  OCC_ALWAYS_INLINE double &Q31c() { return q(10); }
  OCC_ALWAYS_INLINE double &Q31s() { return q(11); }
  OCC_ALWAYS_INLINE double &Q32c() { return q(12); }
  OCC_ALWAYS_INLINE double &Q32s() { return q(13); }
  OCC_ALWAYS_INLINE double &Q33c() { return q(14); }
  OCC_ALWAYS_INLINE double &Q33s() { return q(15); }

  // Level 4
  OCC_ALWAYS_INLINE double &Q40() { return q(16); }
  OCC_ALWAYS_INLINE double &Q41c() { return q(17); }
  OCC_ALWAYS_INLINE double &Q41s() { return q(18); }
  OCC_ALWAYS_INLINE double &Q42c() { return q(19); }
  OCC_ALWAYS_INLINE double &Q42s() { return q(20); }
  OCC_ALWAYS_INLINE double &Q43c() { return q(21); }
  OCC_ALWAYS_INLINE double &Q43s() { return q(22); }
  OCC_ALWAYS_INLINE double &Q44c() { return q(23); }
  OCC_ALWAYS_INLINE double &Q44s() { return q(24); }

  // Level 5
  OCC_ALWAYS_INLINE double &Q50() { return q(25); }
  OCC_ALWAYS_INLINE double &Q51c() { return q(26); }
  OCC_ALWAYS_INLINE double &Q51s() { return q(27); }
  OCC_ALWAYS_INLINE double &Q52c() { return q(28); }
  OCC_ALWAYS_INLINE double &Q52s() { return q(29); }
  OCC_ALWAYS_INLINE double &Q53c() { return q(30); }
  OCC_ALWAYS_INLINE double &Q53s() { return q(31); }
  OCC_ALWAYS_INLINE double &Q54c() { return q(32); }
  OCC_ALWAYS_INLINE double &Q54s() { return q(33); }
  OCC_ALWAYS_INLINE double &Q55c() { return q(34); }
  OCC_ALWAYS_INLINE double &Q55s() { return q(35); }

  // Level 6
  OCC_ALWAYS_INLINE double &Q60() { return q(36); }
  OCC_ALWAYS_INLINE double &Q61c() { return q(37); }
  OCC_ALWAYS_INLINE double &Q61s() { return q(38); }
  OCC_ALWAYS_INLINE double &Q62c() { return q(39); }
  OCC_ALWAYS_INLINE double &Q62s() { return q(40); }
  OCC_ALWAYS_INLINE double &Q63c() { return q(41); }
  OCC_ALWAYS_INLINE double &Q63s() { return q(42); }
  OCC_ALWAYS_INLINE double &Q64c() { return q(43); }
  OCC_ALWAYS_INLINE double &Q64s() { return q(44); }
  OCC_ALWAYS_INLINE double &Q65c() { return q(45); }
  OCC_ALWAYS_INLINE double &Q65s() { return q(46); }
  OCC_ALWAYS_INLINE double &Q66c() { return q(47); }
  OCC_ALWAYS_INLINE double &Q66s() { return q(48); }

  // Level 7
  OCC_ALWAYS_INLINE double &Q70() { return q(49); }
  OCC_ALWAYS_INLINE double &Q71c() { return q(50); }
  OCC_ALWAYS_INLINE double &Q71s() { return q(51); }
  OCC_ALWAYS_INLINE double &Q72c() { return q(52); }
  OCC_ALWAYS_INLINE double &Q72s() { return q(53); }
  OCC_ALWAYS_INLINE double &Q73c() { return q(54); }
  OCC_ALWAYS_INLINE double &Q73s() { return q(55); }
  OCC_ALWAYS_INLINE double &Q74c() { return q(56); }
  OCC_ALWAYS_INLINE double &Q74s() { return q(57); }
  OCC_ALWAYS_INLINE double &Q75c() { return q(58); }
  OCC_ALWAYS_INLINE double &Q75s() { return q(59); }
  OCC_ALWAYS_INLINE double &Q76c() { return q(60); }
  OCC_ALWAYS_INLINE double &Q76s() { return q(61); }
  OCC_ALWAYS_INLINE double &Q77c() { return q(62); }
  OCC_ALWAYS_INLINE double &Q77s() { return q(63); }

  // Level 8
  OCC_ALWAYS_INLINE double &Q80() { return q(64); }
  OCC_ALWAYS_INLINE double &Q81c() { return q(65); }
  OCC_ALWAYS_INLINE double &Q81s() { return q(66); }
  OCC_ALWAYS_INLINE double &Q82c() { return q(67); }
  OCC_ALWAYS_INLINE double &Q82s() { return q(68); }
  OCC_ALWAYS_INLINE double &Q83c() { return q(69); }
  OCC_ALWAYS_INLINE double &Q83s() { return q(70); }
  OCC_ALWAYS_INLINE double &Q84c() { return q(71); }
  OCC_ALWAYS_INLINE double &Q84s() { return q(72); }
  OCC_ALWAYS_INLINE double &Q85c() { return q(73); }
  OCC_ALWAYS_INLINE double &Q85s() { return q(74); }
  OCC_ALWAYS_INLINE double &Q86c() { return q(75); }
  OCC_ALWAYS_INLINE double &Q86s() { return q(76); }
  OCC_ALWAYS_INLINE double &Q87c() { return q(77); }
  OCC_ALWAYS_INLINE double &Q87s() { return q(78); }
  OCC_ALWAYS_INLINE double &Q88c() { return q(79); }
  OCC_ALWAYS_INLINE double &Q88s() { return q(80); }

  // Level 9
  OCC_ALWAYS_INLINE double &Q90() { return q(81); }
  OCC_ALWAYS_INLINE double &Q91c() { return q(82); }
  OCC_ALWAYS_INLINE double &Q91s() { return q(83); }
  OCC_ALWAYS_INLINE double &Q92c() { return q(84); }
  OCC_ALWAYS_INLINE double &Q92s() { return q(85); }
  OCC_ALWAYS_INLINE double &Q93c() { return q(86); }
  OCC_ALWAYS_INLINE double &Q93s() { return q(87); }
  OCC_ALWAYS_INLINE double &Q94c() { return q(88); }
  OCC_ALWAYS_INLINE double &Q94s() { return q(89); }
  OCC_ALWAYS_INLINE double &Q95c() { return q(90); }
  OCC_ALWAYS_INLINE double &Q95s() { return q(91); }
  OCC_ALWAYS_INLINE double &Q96c() { return q(92); }
  OCC_ALWAYS_INLINE double &Q96s() { return q(93); }
  OCC_ALWAYS_INLINE double &Q97c() { return q(94); }
  OCC_ALWAYS_INLINE double &Q97s() { return q(95); }
  OCC_ALWAYS_INLINE double &Q98c() { return q(96); }
  OCC_ALWAYS_INLINE double &Q98s() { return q(97); }
  OCC_ALWAYS_INLINE double &Q99c() { return q(98); }
  OCC_ALWAYS_INLINE double &Q99s() { return q(99); }

  // Level 10 (A = 10)
  OCC_ALWAYS_INLINE double &QA0() { return q(100); }
  OCC_ALWAYS_INLINE double &QA1c() { return q(101); }
  OCC_ALWAYS_INLINE double &QA1s() { return q(102); }
  OCC_ALWAYS_INLINE double &QA2c() { return q(103); }
  OCC_ALWAYS_INLINE double &QA2s() { return q(104); }
  OCC_ALWAYS_INLINE double &QA3c() { return q(105); }
  OCC_ALWAYS_INLINE double &QA3s() { return q(106); }
  OCC_ALWAYS_INLINE double &QA4c() { return q(107); }
  OCC_ALWAYS_INLINE double &QA4s() { return q(108); }
  OCC_ALWAYS_INLINE double &QA5c() { return q(109); }
  OCC_ALWAYS_INLINE double &QA5s() { return q(110); }
  OCC_ALWAYS_INLINE double &QA6c() { return q(111); }
  OCC_ALWAYS_INLINE double &QA6s() { return q(112); }
  OCC_ALWAYS_INLINE double &QA7c() { return q(113); }
  OCC_ALWAYS_INLINE double &QA7s() { return q(114); }
  OCC_ALWAYS_INLINE double &QA8c() { return q(115); }
  OCC_ALWAYS_INLINE double &QA8s() { return q(116); }
  OCC_ALWAYS_INLINE double &QA9c() { return q(117); }
  OCC_ALWAYS_INLINE double &QA9s() { return q(118); }
  OCC_ALWAYS_INLINE double &QAAc() { return q(119); }
  OCC_ALWAYS_INLINE double &QAAs() { return q(120); }

  // OCC_ALWAYS_INLINE const versions of all accessors
  OCC_ALWAYS_INLINE const double &Q00() const { return q(0); }
  OCC_ALWAYS_INLINE const double &charge() const { return q(0); }

  OCC_ALWAYS_INLINE const double &Q10() const { return q(1); }
  OCC_ALWAYS_INLINE const double &Q11c() const { return q(2); }
  OCC_ALWAYS_INLINE const double &Q11s() const { return q(3); }

  OCC_ALWAYS_INLINE const double &Q20() const { return q(4); }
  OCC_ALWAYS_INLINE const double &Q21c() const { return q(5); }
  OCC_ALWAYS_INLINE const double &Q21s() const { return q(6); }
  OCC_ALWAYS_INLINE const double &Q22c() const { return q(7); }
  OCC_ALWAYS_INLINE const double &Q22s() const { return q(8); }

  OCC_ALWAYS_INLINE const double &Q30() const { return q(9); }
  OCC_ALWAYS_INLINE const double &Q31c() const { return q(10); }
  OCC_ALWAYS_INLINE const double &Q31s() const { return q(11); }
  OCC_ALWAYS_INLINE const double &Q32c() const { return q(12); }
  OCC_ALWAYS_INLINE const double &Q32s() const { return q(13); }
  OCC_ALWAYS_INLINE const double &Q33c() const { return q(14); }
  OCC_ALWAYS_INLINE const double &Q33s() const { return q(15); }

  OCC_ALWAYS_INLINE const double &Q40() const { return q(16); }
  OCC_ALWAYS_INLINE const double &Q41c() const { return q(17); }
  OCC_ALWAYS_INLINE const double &Q41s() const { return q(18); }
  OCC_ALWAYS_INLINE const double &Q42c() const { return q(19); }
  OCC_ALWAYS_INLINE const double &Q42s() const { return q(20); }
  OCC_ALWAYS_INLINE const double &Q43c() const { return q(21); }
  OCC_ALWAYS_INLINE const double &Q43s() const { return q(22); }
  OCC_ALWAYS_INLINE const double &Q44c() const { return q(23); }
  OCC_ALWAYS_INLINE const double &Q44s() const { return q(24); }

  OCC_ALWAYS_INLINE const double &Q50() const { return q(25); }
  OCC_ALWAYS_INLINE const double &Q51c() const { return q(26); }
  OCC_ALWAYS_INLINE const double &Q51s() const { return q(27); }
  OCC_ALWAYS_INLINE const double &Q52c() const { return q(28); }
  OCC_ALWAYS_INLINE const double &Q52s() const { return q(29); }
  OCC_ALWAYS_INLINE const double &Q53c() const { return q(30); }
  OCC_ALWAYS_INLINE const double &Q53s() const { return q(31); }
  OCC_ALWAYS_INLINE const double &Q54c() const { return q(32); }
  OCC_ALWAYS_INLINE const double &Q54s() const { return q(33); }
  OCC_ALWAYS_INLINE const double &Q55c() const { return q(34); }
  OCC_ALWAYS_INLINE const double &Q55s() const { return q(35); }

  OCC_ALWAYS_INLINE const double &Q60() const { return q(36); }
  OCC_ALWAYS_INLINE const double &Q61c() const { return q(37); }
  OCC_ALWAYS_INLINE const double &Q61s() const { return q(38); }
  OCC_ALWAYS_INLINE const double &Q62c() const { return q(39); }
  OCC_ALWAYS_INLINE const double &Q62s() const { return q(40); }
  OCC_ALWAYS_INLINE const double &Q63c() const { return q(41); }
  OCC_ALWAYS_INLINE const double &Q63s() const { return q(42); }
  OCC_ALWAYS_INLINE const double &Q64c() const { return q(43); }
  OCC_ALWAYS_INLINE const double &Q64s() const { return q(44); }
  OCC_ALWAYS_INLINE const double &Q65c() const { return q(45); }
  OCC_ALWAYS_INLINE const double &Q65s() const { return q(46); }
  OCC_ALWAYS_INLINE const double &Q66c() const { return q(47); }
  OCC_ALWAYS_INLINE const double &Q66s() const { return q(48); }

  OCC_ALWAYS_INLINE const double &Q70() const { return q(49); }
  OCC_ALWAYS_INLINE const double &Q71c() const { return q(50); }
  OCC_ALWAYS_INLINE const double &Q71s() const { return q(51); }
  OCC_ALWAYS_INLINE const double &Q72c() const { return q(52); }
  OCC_ALWAYS_INLINE const double &Q72s() const { return q(53); }
  OCC_ALWAYS_INLINE const double &Q73c() const { return q(54); }
  OCC_ALWAYS_INLINE const double &Q73s() const { return q(55); }
  OCC_ALWAYS_INLINE const double &Q74c() const { return q(56); }
  OCC_ALWAYS_INLINE const double &Q74s() const { return q(57); }
  OCC_ALWAYS_INLINE const double &Q75c() const { return q(58); }
  OCC_ALWAYS_INLINE const double &Q75s() const { return q(59); }
  OCC_ALWAYS_INLINE const double &Q76c() const { return q(60); }
  OCC_ALWAYS_INLINE const double &Q76s() const { return q(61); }
  OCC_ALWAYS_INLINE const double &Q77c() const { return q(62); }
  OCC_ALWAYS_INLINE const double &Q77s() const { return q(63); }

  OCC_ALWAYS_INLINE const double &Q80() const { return q(64); }
  OCC_ALWAYS_INLINE const double &Q81c() const { return q(65); }
  OCC_ALWAYS_INLINE const double &Q81s() const { return q(66); }
  OCC_ALWAYS_INLINE const double &Q82c() const { return q(67); }
  OCC_ALWAYS_INLINE const double &Q82s() const { return q(68); }
  OCC_ALWAYS_INLINE const double &Q83c() const { return q(69); }
  OCC_ALWAYS_INLINE const double &Q83s() const { return q(70); }
  OCC_ALWAYS_INLINE const double &Q84c() const { return q(71); }
  OCC_ALWAYS_INLINE const double &Q84s() const { return q(72); }
  OCC_ALWAYS_INLINE const double &Q85c() const { return q(73); }
  OCC_ALWAYS_INLINE const double &Q85s() const { return q(74); }
  OCC_ALWAYS_INLINE const double &Q86c() const { return q(75); }
  OCC_ALWAYS_INLINE const double &Q86s() const { return q(76); }
  OCC_ALWAYS_INLINE const double &Q87c() const { return q(77); }
  OCC_ALWAYS_INLINE const double &Q87s() const { return q(78); }
  OCC_ALWAYS_INLINE const double &Q88c() const { return q(79); }
  OCC_ALWAYS_INLINE const double &Q88s() const { return q(80); }

  OCC_ALWAYS_INLINE const double &Q90() const { return q(81); }
  OCC_ALWAYS_INLINE const double &Q91c() const { return q(82); }
  OCC_ALWAYS_INLINE const double &Q91s() const { return q(83); }
  OCC_ALWAYS_INLINE const double &Q92c() const { return q(84); }
  OCC_ALWAYS_INLINE const double &Q92s() const { return q(85); }
  OCC_ALWAYS_INLINE const double &Q93c() const { return q(86); }
  OCC_ALWAYS_INLINE const double &Q93s() const { return q(87); }
  OCC_ALWAYS_INLINE const double &Q94c() const { return q(88); }
  OCC_ALWAYS_INLINE const double &Q94s() const { return q(89); }
  OCC_ALWAYS_INLINE const double &Q95c() const { return q(90); }
  OCC_ALWAYS_INLINE const double &Q95s() const { return q(91); }
  OCC_ALWAYS_INLINE const double &Q96c() const { return q(92); }
  OCC_ALWAYS_INLINE const double &Q96s() const { return q(93); }
  OCC_ALWAYS_INLINE const double &Q97c() const { return q(94); }
  OCC_ALWAYS_INLINE const double &Q97s() const { return q(95); }
  OCC_ALWAYS_INLINE const double &Q98c() const { return q(96); }
  OCC_ALWAYS_INLINE const double &Q98s() const { return q(97); }
  OCC_ALWAYS_INLINE const double &Q99c() const { return q(98); }
  OCC_ALWAYS_INLINE const double &Q99s() const { return q(99); }

  OCC_ALWAYS_INLINE const double &QA0() const { return q(100); }
  OCC_ALWAYS_INLINE const double &QA1c() const { return q(101); }
  OCC_ALWAYS_INLINE const double &QA1s() const { return q(102); }
  OCC_ALWAYS_INLINE const double &QA2c() const { return q(103); }
  OCC_ALWAYS_INLINE const double &QA2s() const { return q(104); }
  OCC_ALWAYS_INLINE const double &QA3c() const { return q(105); }
  OCC_ALWAYS_INLINE const double &QA3s() const { return q(106); }
  OCC_ALWAYS_INLINE const double &QA4c() const { return q(107); }
  OCC_ALWAYS_INLINE const double &QA4s() const { return q(108); }
  OCC_ALWAYS_INLINE const double &QA5c() const { return q(109); }
  OCC_ALWAYS_INLINE const double &QA5s() const { return q(110); }
  OCC_ALWAYS_INLINE const double &QA6c() const { return q(111); }
  OCC_ALWAYS_INLINE const double &QA6s() const { return q(112); }
  OCC_ALWAYS_INLINE const double &QA7c() const { return q(113); }
  OCC_ALWAYS_INLINE const double &QA7s() const { return q(114); }
  OCC_ALWAYS_INLINE const double &QA8c() const { return q(115); }
  OCC_ALWAYS_INLINE const double &QA8s() const { return q(116); }
  OCC_ALWAYS_INLINE const double &QA9c() const { return q(117); }
  OCC_ALWAYS_INLINE const double &QA9s() const { return q(118); }
  OCC_ALWAYS_INLINE const double &QAAc() const { return q(119); }
  OCC_ALWAYS_INLINE const double &QAAs() const { return q(120); }
};

} // namespace occ::dma
