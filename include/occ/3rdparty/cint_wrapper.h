#pragma once
#include <cstdlib>

namespace libcint {

extern "C" {

#include "cint.h"
#include "cint_funcs.h"

extern CACHE_SIZE_T int3c2e_sph(double *out, int *dims, int *shls, int *atm,
                                int natm, int *bas, int nbas, double *env,
                                CINTOpt *opt, double *cache);

extern CACHE_SIZE_T int3c2e_cart(double *out, int *dims, int *shls, int *atm,
                                 int natm, int *bas, int nbas, double *env,
                                 CINTOpt *opt, double *cache);

extern CACHE_SIZE_T int2c2e_sph(double *out, int *dims, int *shls, int *atm,
                                int natm, int *bas, int nbas, double *env,
                                CINTOpt *opt, double *cache);

extern CACHE_SIZE_T int2c2e_cart(double *out, int *dims, int *shls, int *atm,
                                 int natm, int *bas, int nbas, double *env,
                                 CINTOpt *opt, double *cache);

extern void int2c2e_optimizer(CINTOpt **opt, int *atm, int natm, int *bas,
                              int nbas, double *env);

extern void int3c2e_optimizer(CINTOpt **opt, int *atm, int natm, int *bas,
                              int nbas, double *env);
extern void int1e_r_optimizer(CINTOpt **opt, int *atm, int natm, int *bas,
                              int nbas, double *env);
extern void int1e_rr_optimizer(CINTOpt **opt, int *atm, int natm, int *bas,
                               int nbas, double *env);
extern void int1e_rrr_optimizer(CINTOpt **opt, int *atm, int natm, int *bas,
                                int nbas, double *env);
extern void int1e_rrrr_optimizer(CINTOpt **opt, int *atm, int natm, int *bas,
                                 int nbas, double *env);
}

static constexpr int environment_start_offset = PTR_ENV_START;
static constexpr int common_origin_offset = PTR_COMMON_ORIG;
static constexpr int exponent_cutoff_offset = PTR_EXPCUTOFF;
static constexpr int rinv_origin_offset = PTR_RINV_ORIG;
static constexpr int rinv_zeta_offset = PTR_RINV_ZETA;
static constexpr int range_omega_offset = PTR_RANGE_OMEGA;
static constexpr int f12_zeta_offset = PTR_F12_ZETA;
#define XXSTRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s
static constexpr const char *cint_version_string = XXSTRINGIFY(CINT_VERSION);
#undef XSTRINGIFY
#undef XXSTRINGIFY

} // namespace libcint
