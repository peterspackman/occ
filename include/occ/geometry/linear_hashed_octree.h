#pragma once
#include <ankerl/unordered_dense.h>
#include <deque>
#include <mutex>
#include <occ/3rdparty/concurrentqueue.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/geometry/morton.h>
#include <optional>
#include <thread>
#include <vector>

namespace occ::geometry {

template <typename N> struct LinearHashedOctree {
  using NodeMap = ankerl::unordered_dense::map<MIndex, N, MIndexHash>;
  NodeMap nodes;
  std::vector<MIndex> leaves;
  LinearHashedOctree() : nodes{}, leaves{} {}

  template <class R, class C>
  void build_st(R &should_refine, C &construct_node) {
    std::deque<MIndex> queue;
    queue.push_back(MIndex{});

    while (!queue.empty()) {
      auto key = queue.front();
      queue.pop_front();
      auto node = construct_node(key);

      if (should_refine(key, node)) {
        for (int i = 0; i < 8; ++i) {
          queue.push_back(key.child(i));
        }
      } else {
        leaves.push_back(key);
      }
      nodes[key] = node;
    }
  }

  template <class R, class C> void build(R &should_refine, C &construct_node) {
    std::mutex nodes_mutex;

    moodycamel::ConcurrentQueue<MIndex> queue;
    moodycamel::ConcurrentQueue<MIndex> leaves;

    queue.enqueue(MIndex{});

    auto worker_fn = [&](int thread_id) {
      moodycamel::ProducerToken producer_tok(queue);
      moodycamel::ProducerToken producer_tok_leaves(leaves);
      moodycamel::ConsumerToken consumer_tok(queue);
      std::array<MIndex, 8> children;
      while (true) {
        MIndex key;
        if (!queue.try_dequeue(consumer_tok, key))
          break;

        auto node = construct_node(key);

        if (should_refine(key, node)) {
          key.get_children(children);
          queue.enqueue_bulk(producer_tok, &children[0], 8);
        } else {
          leaves.enqueue(producer_tok_leaves, key);
        }

        // add to nodes
        {
          std::lock_guard<std::mutex> lock(
              nodes_mutex); // Synchronize access to nodes
          nodes[key] = node;
        }
      }
    };
    occ::parallel::parallel_do(worker_fn);
    occ::log::info("Octree build workers joined");

    // Copy from the temporary shared leaves vector to the class member
    {
      MIndex leaf;
      while (leaves.try_dequeue(leaf)) {
        this->leaves.push_back(leaf);
      }
    }
    occ::log::info("Octree build complete");
  }

  template <class V> void visit_leaves(V &visitor) const {
    for (const auto &key : leaves) {
      visitor(key);
    }
  }

  inline std::optional<N> get(MIndex key) const {
    auto search = nodes.find(key);
    if (search == nodes.end())
      return std::nullopt;
    return search->second;
  }
};

} // namespace occ::geometry
