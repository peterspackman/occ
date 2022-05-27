namespace libcint {

extern "C" {
#include "cint.h"
#include "cint_funcs.h"

CACHE_SIZE_T int3c2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, CINTOpt *opt,
                         double *cache);

CACHE_SIZE_T int3c2e_cart(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, CINTOpt *opt,
                          double *cache);

CACHE_SIZE_T int2c2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, CINTOpt *opt,
                         double *cache);

CACHE_SIZE_T int2c2e_cart(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, CINTOpt *opt,
                          double *cache);
}
} // namespace libcint
