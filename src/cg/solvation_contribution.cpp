#include <algorithm>
#include <occ/cg/solvation_contribution.h>
#include <stdexcept>

namespace occ::cg {

namespace {

const ContributionPair zero_pair{};

using ChannelMap = ankerl::unordered_dense::map<std::string, ContributionPair>;

void accumulate(ChannelMap &channels, std::string_view channel, double value,
                bool is_forward) {
  auto &pair = channels[std::string(channel)];
  if (is_forward)
    pair.forward += value;
  else
    pair.reverse += value;
}

const ContributionPair &lookup(const ChannelMap &channels,
                               std::string_view channel) {
  auto it = channels.find(std::string(channel));
  return (it == channels.end()) ? zero_pair : it->second;
}

std::vector<std::string> sorted_keys(const ChannelMap &channels) {
  std::vector<std::string> names;
  names.reserve(channels.size());
  for (const auto &[name, _] : channels)
    names.push_back(name);
  std::sort(names.begin(), names.end());
  return names;
}

/// Exchange every channel present in either map, so a channel written on only
/// one side of a pair still moves.
void exchange_maps(ChannelMap &a, ChannelMap &b) {
  for (const auto &[name, _] : b)
    a.try_emplace(name);
  for (auto &[name, pair] : a)
    pair.exchange_with(b[name]);
}

} // namespace

double ContributionPair::total() const { return forward + reverse; }

void ContributionPair::exchange_with(ContributionPair &other) {
  other.reverse = forward;
  reverse = other.forward;
}

void SolvationContribution::add_energy(std::string_view channel, double value,
                                       bool is_forward) {
  accumulate(m_energies, channel, value, is_forward);
}

void SolvationContribution::add_descriptor(std::string_view channel,
                                           double value, bool is_forward) {
  accumulate(m_descriptors, channel, value, is_forward);
}

const ContributionPair &
SolvationContribution::energy(std::string_view channel) const {
  return lookup(m_energies, channel);
}

const ContributionPair &
SolvationContribution::descriptor(std::string_view channel) const {
  return lookup(m_descriptors, channel);
}

std::vector<std::string> SolvationContribution::energy_channels() const {
  return sorted_keys(m_energies);
}

std::vector<std::string> SolvationContribution::descriptor_channels() const {
  return sorted_keys(m_descriptors);
}

double SolvationContribution::total_energy() const {
  double forward = 0.0, reverse = 0.0;
  for (const auto &[name, pair] : m_energies) {
    forward += pair.forward;
    reverse += pair.reverse;
  }
  const double total = forward + reverse;
  if (!m_antisymmetrize)
    return total;
  return total + 0.5 * (forward - reverse);
}

void SolvationContribution::exchange_with(SolvationContribution &other) {
  if (m_exchanged || other.m_exchanged) {
    throw std::runtime_error(
        "Attempting to exchange already processed contributions");
  }
  exchange_maps(m_energies, other.m_energies);
  exchange_maps(m_descriptors, other.m_descriptors);
  m_exchanged = true;
  other.m_exchanged = true;
}

} // namespace occ::cg
