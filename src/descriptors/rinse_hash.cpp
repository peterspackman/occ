#include "rinse_pca_model.h"
#include <array>
#include <fmt/core.h>
#include <occ/descriptors/rinse.h>
#include <stdexcept>
#include <string>

namespace occ::descriptors {

namespace {

// Proquint: alternating consonants and vowels, CVCVC, so a word carries
// 4 + 2 + 4 + 2 + 4 = 16 bits and is pronounceable.
constexpr std::string_view consonants = "bdfghjklmnprstvz";
constexpr std::string_view vowels = "aiou";
constexpr int bits_per_word = 16;

std::string encode_word(unsigned value) {
  return {consonants[(value >> 12) & 0xF], vowels[(value >> 10) & 0x3],
          consonants[(value >> 6) & 0xF], vowels[(value >> 4) & 0x3],
          consonants[value & 0xF]};
}

unsigned decode_word(std::string_view word) {
  if (word.size() != 5)
    throw std::invalid_argument(
        fmt::format("A proquint word is five characters, got '{}'", word));
  const auto index = [](std::string_view alphabet, char c, std::string_view w) {
    const auto pos = alphabet.find(c);
    if (pos == std::string_view::npos)
      throw std::invalid_argument(fmt::format(
          "'{}' is not a proquint word: '{}' is out of alphabet", w, c));
    return static_cast<unsigned>(pos);
  };
  return (index(consonants, word[0], word) << 12) |
         (index(vowels, word[1], word) << 10) |
         (index(consonants, word[2], word) << 6) |
         (index(vowels, word[3], word) << 4) | index(consonants, word[4], word);
}

// A hash is meant to be quoted in a paper and read aloud, so words that happen
// to spell something offensive are moved aside. Flipping progressively higher
// bits keeps the replacement deterministic and close to the original.
constexpr std::array<std::string_view, 8> blocked = {
    "fag", "nud", "jihad", "fuk", "hamas", "isis", "nazis", "putin"};

bool is_blocked(std::string_view word) {
  for (const auto sub : blocked)
    if (word.find(sub) != std::string_view::npos)
      return true;
  return false;
}

std::string sanitise(std::string word) {
  if (!is_blocked(word))
    return word;
  const unsigned value = decode_word(word);
  for (unsigned shift = 1; shift != 0; shift <<= 1) {
    word = encode_word((value ^ shift) & 0xFFFF);
    if (!is_blocked(word))
      break;
  }
  return word;
}

} // namespace

std::string rinse_hash(Eigen::Ref<const Vec> descriptor, int num_words) {
  const Eigen::Index dimension = descriptor.size();
  if (dimension == 0)
    throw std::invalid_argument("Cannot hash an empty descriptor");
  if (num_words < 1)
    throw std::invalid_argument(
        fmt::format("A hash needs at least one word, asked for {}", num_words));
  if (dimension != impl::pca_num_features)
    throw std::invalid_argument(fmt::format(
        "The bundled PCA model is for {}-element descriptors, got {}. The hash "
        "only applies to the default RINSE parameters.",
        impl::pca_num_features, dimension));

  const int num_bits = num_words * bits_per_word;
  if (num_bits > impl::pca_num_components)
    throw std::invalid_argument(fmt::format(
        "{} words needs {} principal components, the bundled model has {}",
        num_words, num_bits, impl::pca_num_components));

  const Eigen::Map<const Vec> mean(impl::pca_mean, dimension);
  const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
      components(impl::pca_components, impl::pca_num_components, dimension);

  const Vec projection = components.topRows(num_bits) * (descriptor - mean);

  std::string result;
  for (int w = 0; w < num_words; w++) {
    unsigned value = 0;
    for (int b = 0; b < bits_per_word; b++)
      value =
          (value << 1) | (projection(w * bits_per_word + b) > 0.0 ? 1u : 0u);
    if (w > 0)
      result += '-';
    result += sanitise(encode_word(value));
  }
  return result;
}

std::vector<bool> rinse_hash_to_bits(std::string_view hash) {
  std::vector<bool> result;
  size_t start = 0;
  while (start <= hash.size()) {
    const size_t end = std::min(hash.find('-', start), hash.size());
    const unsigned value = decode_word(hash.substr(start, end - start));
    for (int b = bits_per_word - 1; b >= 0; b--)
      result.push_back(((value >> b) & 1u) != 0u);
    start = end + 1;
  }
  return result;
}

} // namespace occ::descriptors
