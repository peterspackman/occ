#include <algorithm>
#include <occ/core/parallel.h>
#include <occ/qm/cc/einsum.h>
#include <occ/qm/cc/gemm.h>
#include <string>
#include <vector>

namespace occ::qm::cc::detail {

void permute(const double *in, const Eigen::Index *dims, const int rank,
             const int *perm, double *out) {
  // padded to rank 6 at the end; axis 0 of the output is the innermost loop,
  // a plain copy when it is also axis 0 of the input
  constexpr int R = 6;
  if (rank < 1 || rank > R)
    throw std::runtime_error("permute: rank must be 1..6");
  std::array<Eigen::Index, R> stride{}, od{}, st{};
  Eigen::Index s = 1;
  for (int k = 0; k < rank; k++) {
    stride[k] = s;
    s *= dims[k];
  }
  for (int k = 0; k < R; k++) {
    od[k] = k < rank ? dims[perm[k]] : 1;
    st[k] = k < rank ? stride[perm[k]] : 0;
  }
  const Eigen::Index inner = od[0];
  const size_t outer =
      static_cast<size_t>(od[1] * od[2] * od[3] * od[4] * od[5]);
  auto rows = [&](const size_t q0, const size_t q1) {
    for (size_t q = q0; q < q1; q++) {
      Eigen::Index rem = static_cast<Eigen::Index>(q), base = 0;
      for (int k = 1; k < R; k++) {
        base += (rem % od[k]) * st[k];
        rem /= od[k];
      }
      double *dst = out + static_cast<Eigen::Index>(q) * inner;
      const double *src = in + base;
      if (st[0] == 1)
        std::copy_n(src, inner, dst);
      else
        for (Eigen::Index x = 0; x < inner; x++)
          dst[x] = src[x * st[0]];
    }
  };
  if (static_cast<size_t>(inner) * outer <= (size_t(1) << 18)) {
    rows(0, outer);
    return;
  }
  const size_t nb = std::min<size_t>(
                   outer,
                   4 * static_cast<size_t>(occ::parallel::get_num_threads())),
               per = (outer + nb - 1) / nb;
  occ::parallel::parallel_for(size_t(0), nb, [&](const size_t b) {
    rows(b * per, std::min(outer, (b + 1) * per));
  });
}

namespace {

// How einsum lays out one contraction: A as [fa, con] (M x K), B as [con, fb]
// (K x N), C = A B as [fa, fb], then permuted to the output labels
struct Plan {
  std::vector<int> pa, pb, pc;   // permutations of A, B and of [fa, fb] -> out
  std::vector<Eigen::Index> dab; // dims of [fa, fb]
  Eigen::Index M = 1, K = 1, N = 1;
  bool a_transposed = false; // A already laid out as [con, fa] (K x M)
};

bool is_identity(const std::vector<int> &p) {
  for (size_t k = 0; k < p.size(); k++)
    if (p[k] != static_cast<int>(k))
      return false;
  return true;
}

Plan make_plan(std::string_view spec, const Eigen::Index *da, const int ra,
               const Eigen::Index *db, const int rb, const int rc) {
  const std::string s(spec);
  auto fail = [&](const char *why) {
    throw std::runtime_error("einsum '" + s + "': " + why);
  };
  const size_t comma = s.find(','), arrow = s.find("->");
  if (comma == std::string::npos || arrow == std::string::npos || arrow < comma)
    fail("expected 'ab,bc->ac'");
  const std::string la = s.substr(0, comma),
                    lb = s.substr(comma + 1, arrow - comma - 1),
                    lo = s.substr(arrow + 2);
  if (static_cast<int>(la.size()) != ra || static_cast<int>(lb.size()) != rb)
    fail("labels do not match the operand ranks");
  if (static_cast<int>(lo.size()) != rc)
    fail("labels do not match the result rank");
  auto has = [](const std::string &l, char c) {
    return l.find(c) != std::string::npos;
  };
  std::string fa, con, fb;
  for (const char c : la) {
    if (std::count(la.begin(), la.end(), c) > 1)
      fail("repeated label in an operand");
    (has(lb, c) && !has(lo, c) ? con : fa) += c;
  }
  for (const char c : lb) {
    if (std::count(lb.begin(), lb.end(), c) > 1)
      fail("repeated label in an operand");
    if (!has(con, c))
      fb += c;
  }
  for (const char c : fa)
    if (has(fb, c))
      fail("batch labels are not supported");
  for (const char c : lo)
    if (!has(fa, c) && !has(fb, c))
      fail("output label not in either operand");
  if (fa.size() + fb.size() != lo.size())
    fail("free labels must all appear in the output");

  Plan p;
  for (const char c : fa) {
    p.pa.push_back(static_cast<int>(la.find(c)));
    p.dab.push_back(da[la.find(c)]);
    p.M *= da[la.find(c)];
  }
  for (const char c : con) {
    if (da[la.find(c)] != db[lb.find(c)])
      fail("summed over unequal dimensions");
    p.pa.push_back(static_cast<int>(la.find(c)));
    p.pb.push_back(static_cast<int>(lb.find(c)));
    p.K *= da[la.find(c)];
  }
  for (const char c : fb) {
    p.pb.push_back(static_cast<int>(lb.find(c)));
    p.dab.push_back(db[lb.find(c)]);
    p.N *= db[lb.find(c)];
  }
  // A stored as [con, fa] is A^T of the matrix wanted: no permute needed
  if (!is_identity(p.pa)) {
    std::vector<int> con_first(p.pa.begin() + fa.size(), p.pa.end());
    con_first.insert(con_first.end(), p.pa.begin(), p.pa.begin() + fa.size());
    p.a_transposed = is_identity(con_first);
  }
  const std::string lab = fa + fb;
  for (const char c : lo)
    p.pc.push_back(static_cast<int>(lab.find(c)));
  return p;
}

} // namespace

void einsum_shape(std::string_view spec, const Eigen::Index *da, const int ra,
                  const Eigen::Index *db, const int rb, Eigen::Index *dc,
                  const int rc) {
  const Plan p = make_plan(spec, da, ra, db, rb, rc);
  for (int k = 0; k < rc; k++)
    dc[k] = p.dab[p.pc[k]];
}

void einsum_contract(std::string_view spec, const double *a,
                     const Eigen::Index *da, const int ra, const double *b,
                     const Eigen::Index *db, const int rb, double *out,
                     const int rc) {
  const Plan p = make_plan(spec, da, ra, db, rb, rc);
  std::vector<double> at, bt, ct;
  const double *ap = a, *bp = b;
  if (!is_identity(p.pa) && !p.a_transposed) {
    at.resize(static_cast<size_t>(p.M * p.K));
    permute(a, da, ra, p.pa.data(), at.data());
    ap = at.data();
  }
  if (!is_identity(p.pb)) {
    bt.resize(static_cast<size_t>(p.K * p.N));
    permute(b, db, rb, p.pb.data(), bt.data());
    bp = bt.data();
  }
  const bool direct = is_identity(p.pc);
  if (!direct)
    ct.resize(static_cast<size_t>(p.M * p.N));
  double *cp = direct ? out : ct.data();
  Eigen::Map<occ::Mat> C(cp, p.M, p.N);
  const Eigen::Map<const occ::Mat> B(bp, p.K, p.N);
  if (p.a_transposed)
    pgemm(C, Eigen::Map<const occ::Mat>(ap, p.K, p.M), B, false, true);
  else
    pgemm(C, Eigen::Map<const occ::Mat>(ap, p.M, p.K), B);
  if (!direct)
    permute(ct.data(), p.dab.data(), rc, p.pc.data(), out);
}

} // namespace occ::qm::cc::detail
