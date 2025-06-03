#pragma once
#include <occ/core/linear_algebra.h>
#include <string>

namespace occ::dma {

struct Mult {
  Mult(int n) : q(Vec::Zero(n)) {}
  Mult() : q(Vec::Zero(121)) {}
  Vec q;


  std::string to_string(int lm) const;

  // Level 0
  double &Q00() { return q(0); }
  double &charge() { return q(0); }

  // Level 1
  double &Q10() { return q(1); }
  double &Q11c() { return q(2); }
  double &Q11s() { return q(3); }

  // Level 2
  double &Q20() { return q(4); }
  double &Q21c() { return q(5); }
  double &Q21s() { return q(6); }
  double &Q22c() { return q(7); }
  double &Q22s() { return q(8); }

  // Level 3
  double &Q30() { return q(9); }
  double &Q31c() { return q(10); }
  double &Q31s() { return q(11); }
  double &Q32c() { return q(12); }
  double &Q32s() { return q(13); }
  double &Q33c() { return q(14); }
  double &Q33s() { return q(15); }

  // Level 4
  double &Q40() { return q(16); }
  double &Q41c() { return q(17); }
  double &Q41s() { return q(18); }
  double &Q42c() { return q(19); }
  double &Q42s() { return q(20); }
  double &Q43c() { return q(21); }
  double &Q43s() { return q(22); }
  double &Q44c() { return q(23); }
  double &Q44s() { return q(24); }

  // Level 5
  double &Q50() { return q(25); }
  double &Q51c() { return q(26); }
  double &Q51s() { return q(27); }
  double &Q52c() { return q(28); }
  double &Q52s() { return q(29); }
  double &Q53c() { return q(30); }
  double &Q53s() { return q(31); }
  double &Q54c() { return q(32); }
  double &Q54s() { return q(33); }
  double &Q55c() { return q(34); }
  double &Q55s() { return q(35); }

  // Level 6
  double &Q60() { return q(36); }
  double &Q61c() { return q(37); }
  double &Q61s() { return q(38); }
  double &Q62c() { return q(39); }
  double &Q62s() { return q(40); }
  double &Q63c() { return q(41); }
  double &Q63s() { return q(42); }
  double &Q64c() { return q(43); }
  double &Q64s() { return q(44); }
  double &Q65c() { return q(45); }
  double &Q65s() { return q(46); }
  double &Q66c() { return q(47); }
  double &Q66s() { return q(48); }

  // Level 7
  double &Q70() { return q(49); }
  double &Q71c() { return q(50); }
  double &Q71s() { return q(51); }
  double &Q72c() { return q(52); }
  double &Q72s() { return q(53); }
  double &Q73c() { return q(54); }
  double &Q73s() { return q(55); }
  double &Q74c() { return q(56); }
  double &Q74s() { return q(57); }
  double &Q75c() { return q(58); }
  double &Q75s() { return q(59); }
  double &Q76c() { return q(60); }
  double &Q76s() { return q(61); }
  double &Q77c() { return q(62); }
  double &Q77s() { return q(63); }

  // Level 8
  double &Q80() { return q(64); }
  double &Q81c() { return q(65); }
  double &Q81s() { return q(66); }
  double &Q82c() { return q(67); }
  double &Q82s() { return q(68); }
  double &Q83c() { return q(69); }
  double &Q83s() { return q(70); }
  double &Q84c() { return q(71); }
  double &Q84s() { return q(72); }
  double &Q85c() { return q(73); }
  double &Q85s() { return q(74); }
  double &Q86c() { return q(75); }
  double &Q86s() { return q(76); }
  double &Q87c() { return q(77); }
  double &Q87s() { return q(78); }
  double &Q88c() { return q(79); }
  double &Q88s() { return q(80); }

  // Level 9
  double &Q90() { return q(81); }
  double &Q91c() { return q(82); }
  double &Q91s() { return q(83); }
  double &Q92c() { return q(84); }
  double &Q92s() { return q(85); }
  double &Q93c() { return q(86); }
  double &Q93s() { return q(87); }
  double &Q94c() { return q(88); }
  double &Q94s() { return q(89); }
  double &Q95c() { return q(90); }
  double &Q95s() { return q(91); }
  double &Q96c() { return q(92); }
  double &Q96s() { return q(93); }
  double &Q97c() { return q(94); }
  double &Q97s() { return q(95); }
  double &Q98c() { return q(96); }
  double &Q98s() { return q(97); }
  double &Q99c() { return q(98); }
  double &Q99s() { return q(99); }

  // Level 10 (A = 10)
  double &QA0() { return q(100); }
  double &QA1c() { return q(101); }
  double &QA1s() { return q(102); }
  double &QA2c() { return q(103); }
  double &QA2s() { return q(104); }
  double &QA3c() { return q(105); }
  double &QA3s() { return q(106); }
  double &QA4c() { return q(107); }
  double &QA4s() { return q(108); }
  double &QA5c() { return q(109); }
  double &QA5s() { return q(110); }
  double &QA6c() { return q(111); }
  double &QA6s() { return q(112); }
  double &QA7c() { return q(113); }
  double &QA7s() { return q(114); }
  double &QA8c() { return q(115); }
  double &QA8s() { return q(116); }
  double &QA9c() { return q(117); }
  double &QA9s() { return q(118); }
  double &QAAc() { return q(119); }
  double &QAAs() { return q(120); }

  // Const versions of all accessors
  const double &Q00() const { return q(0); }
  const double &charge() const { return q(0); }

  const double &Q10() const { return q(1); }
  const double &Q11c() const { return q(2); }
  const double &Q11s() const { return q(3); }

  const double &Q20() const { return q(4); }
  const double &Q21c() const { return q(5); }
  const double &Q21s() const { return q(6); }
  const double &Q22c() const { return q(7); }
  const double &Q22s() const { return q(8); }

  const double &Q30() const { return q(9); }
  const double &Q31c() const { return q(10); }
  const double &Q31s() const { return q(11); }
  const double &Q32c() const { return q(12); }
  const double &Q32s() const { return q(13); }
  const double &Q33c() const { return q(14); }
  const double &Q33s() const { return q(15); }

  const double &Q40() const { return q(16); }
  const double &Q41c() const { return q(17); }
  const double &Q41s() const { return q(18); }
  const double &Q42c() const { return q(19); }
  const double &Q42s() const { return q(20); }
  const double &Q43c() const { return q(21); }
  const double &Q43s() const { return q(22); }
  const double &Q44c() const { return q(23); }
  const double &Q44s() const { return q(24); }

  const double &Q50() const { return q(25); }
  const double &Q51c() const { return q(26); }
  const double &Q51s() const { return q(27); }
  const double &Q52c() const { return q(28); }
  const double &Q52s() const { return q(29); }
  const double &Q53c() const { return q(30); }
  const double &Q53s() const { return q(31); }
  const double &Q54c() const { return q(32); }
  const double &Q54s() const { return q(33); }
  const double &Q55c() const { return q(34); }
  const double &Q55s() const { return q(35); }

  const double &Q60() const { return q(36); }
  const double &Q61c() const { return q(37); }
  const double &Q61s() const { return q(38); }
  const double &Q62c() const { return q(39); }
  const double &Q62s() const { return q(40); }
  const double &Q63c() const { return q(41); }
  const double &Q63s() const { return q(42); }
  const double &Q64c() const { return q(43); }
  const double &Q64s() const { return q(44); }
  const double &Q65c() const { return q(45); }
  const double &Q65s() const { return q(46); }
  const double &Q66c() const { return q(47); }
  const double &Q66s() const { return q(48); }

  const double &Q70() const { return q(49); }
  const double &Q71c() const { return q(50); }
  const double &Q71s() const { return q(51); }
  const double &Q72c() const { return q(52); }
  const double &Q72s() const { return q(53); }
  const double &Q73c() const { return q(54); }
  const double &Q73s() const { return q(55); }
  const double &Q74c() const { return q(56); }
  const double &Q74s() const { return q(57); }
  const double &Q75c() const { return q(58); }
  const double &Q75s() const { return q(59); }
  const double &Q76c() const { return q(60); }
  const double &Q76s() const { return q(61); }
  const double &Q77c() const { return q(62); }
  const double &Q77s() const { return q(63); }

  const double &Q80() const { return q(64); }
  const double &Q81c() const { return q(65); }
  const double &Q81s() const { return q(66); }
  const double &Q82c() const { return q(67); }
  const double &Q82s() const { return q(68); }
  const double &Q83c() const { return q(69); }
  const double &Q83s() const { return q(70); }
  const double &Q84c() const { return q(71); }
  const double &Q84s() const { return q(72); }
  const double &Q85c() const { return q(73); }
  const double &Q85s() const { return q(74); }
  const double &Q86c() const { return q(75); }
  const double &Q86s() const { return q(76); }
  const double &Q87c() const { return q(77); }
  const double &Q87s() const { return q(78); }
  const double &Q88c() const { return q(79); }
  const double &Q88s() const { return q(80); }

  const double &Q90() const { return q(81); }
  const double &Q91c() const { return q(82); }
  const double &Q91s() const { return q(83); }
  const double &Q92c() const { return q(84); }
  const double &Q92s() const { return q(85); }
  const double &Q93c() const { return q(86); }
  const double &Q93s() const { return q(87); }
  const double &Q94c() const { return q(88); }
  const double &Q94s() const { return q(89); }
  const double &Q95c() const { return q(90); }
  const double &Q95s() const { return q(91); }
  const double &Q96c() const { return q(92); }
  const double &Q96s() const { return q(93); }
  const double &Q97c() const { return q(94); }
  const double &Q97s() const { return q(95); }
  const double &Q98c() const { return q(96); }
  const double &Q98s() const { return q(97); }
  const double &Q99c() const { return q(98); }
  const double &Q99s() const { return q(99); }

  const double &QA0() const { return q(100); }
  const double &QA1c() const { return q(101); }
  const double &QA1s() const { return q(102); }
  const double &QA2c() const { return q(103); }
  const double &QA2s() const { return q(104); }
  const double &QA3c() const { return q(105); }
  const double &QA3s() const { return q(106); }
  const double &QA4c() const { return q(107); }
  const double &QA4s() const { return q(108); }
  const double &QA5c() const { return q(109); }
  const double &QA5s() const { return q(110); }
  const double &QA6c() const { return q(111); }
  const double &QA6s() const { return q(112); }
  const double &QA7c() const { return q(113); }
  const double &QA7s() const { return q(114); }
  const double &QA8c() const { return q(115); }
  const double &QA8s() const { return q(116); }
  const double &QA9c() const { return q(117); }
  const double &QA9s() const { return q(118); }
  const double &QAAc() const { return q(119); }
  const double &QAAs() const { return q(120); }
};

} // namespace occ::dma
