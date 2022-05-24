namespace libcint {

extern "C" {
#include "cint.h"
#include "cint_funcs.h"

CACHE_SIZE_T int3c2e_sph(double *out, FINT *dims, FINT *shls, FINT *atm,
                         FINT natm, FINT *bas, FINT nbas, double *env,
                         CINTOpt *opt, double *cache);

CACHE_SIZE_T int3c2e_cart(double *out, FINT *dims, FINT *shls, FINT *atm,
                          FINT natm, FINT *bas, FINT nbas, double *env,
                          CINTOpt *opt, double *cache);
}
} // namespace libcint
