#pragma once
#include <chrono>
#include <string>

namespace occ::timing {

using duration_t = std::chrono::duration<double>;
using clock_t = std::chrono::high_resolution_clock;
using time_point_t = std::chrono::time_point<clock_t>;

enum category {
    ints1e,
    ints4c2e,
    ints3c2e,
    io,
    la,
    guess,
    mo,
    cube_evaluation,
    diis,
    grid_init,
    grid_points,
    dft_xc,
    dft,
    dft_nlc,
    xc_func_init,
    xc_func_end,
    rho,
    gto,
    fock,
    jkmat,
    jmat,
    df,
    gto_dist,
    gto_mask,
    gto_shell,
    gto_s,
    gto_p,
    gto_gen,
    ecp,
    assoc_legendre,
    fft,
    xdm,
    solvent,
    isosurface_function,
    isosurface_normals,
    isosurface_properties,
    mc_octree,
    mc_primal,
    mc_surface,
    global,
    _group_count
};

template <size_t count = 1> class StopWatch {
  public:
    StopWatch() {
        clear_all();
        set_now_overhead(0);
    }

    static time_point_t now() { return clock_t::now(); }

    void set_now_overhead(size_t ns) {
        m_overhead = std::chrono::nanoseconds(ns);
    }

    time_point_t start(size_t t = 0) {
        m_tstart[t] = now();
        return m_tstart[t];
    }

    duration_t stop(size_t t = 0) {
        const auto tstop = now();
        const duration_t result = (tstop - m_tstart[t]) - m_overhead;
        m_timers[t] += result;
        return result;
    }

    double read(size_t t = 0) const { return m_timers[t].count(); }

    void clear_all() {
        for (auto t = 0; t != ntimers; ++t) {
            m_timers[t] = duration_t::zero();
            m_tstart[t] = time_point_t();
        }
    }

  private:
    constexpr static auto ntimers = count;
    duration_t m_timers[ntimers];
    time_point_t m_tstart[ntimers];
    duration_t m_overhead;
};

time_point_t start(category cat);
duration_t stop(category cat);
double total(category cat);
void clear_all();

const char *category_name(category);
void print_timings();

} // namespace occ::timing
