#include <occ/sht/clebsch.h>
#include <occ/core/util.h>

namespace occ::sht {

double clebsch(int j1, int m1, int j2, int m2, int j, int m) noexcept {
   // Calculation using Racah formula taken from "Angular Momentum",
   // D.M.Brink & G.R.Satchler, Oxford, 1968
   using occ::util::factorial;
   double res = 0.0;

   int j1nm1, jnj2pm1, j2pm2, jnj1nm2, j1pj2nj;
   int mink, maxk, iphase;
   double tmp;

   if(abs(m1) > j1) return res;
   if(abs(m2) > j2) return res;
   if(abs(m) > j) return res;
   if((j1 < 0) || (j2 < 0) || (j < 0)) return res;
   if(abs(j1 - j2) > j) return res;
   if(j > j1 + j2) return res;

   if ((m1 + m2) != m) return res;

   j1nm1 = (j1 - m1)/2;
   jnj2pm1 = (j - j2 + m1)/2;
   j2pm2 = (j2 + m2)/2;
   jnj1nm2 = (j - j1 - m2)/2;
   j1pj2nj = (j1 + j2 - j)/2;

   // check if evenness is valid i.e. j1 and m1 both even/odd
   if (!(((j1nm1 * 2) == (j1 - m1)) &&
	((j2pm2 * 2) == (j2 + m2)) &&
	((j1pj2nj * 2) == (j1 + j2 - j)))) {
       return res;
   }

   mink = std::max(std::max(-jnj2pm1, -jnj1nm2), 0);
   maxk = std::min(std::min(j1nm1, j2pm2), j1pj2nj);

   if ((mink/2)*2 != mink) {
       iphase = -1;
   }
   else {
       iphase = 1;
   }

   for(int k = mink; k <= maxk; k++) {
       tmp = (factorial(j1nm1 - k) * factorial(jnj2pm1 + k) * factorial(j2pm2 - k)
           * factorial(jnj1nm2 + k) * factorial(k) * factorial(j1pj2nj - k));
       res = res + iphase/tmp;
       iphase = - iphase;
   }

   if (mink > maxk) res = 1.0;

   tmp = std::sqrt(1.0 * factorial(j1pj2nj));
   tmp = tmp * std::sqrt(factorial((j1 + j - j2) / 2));
   tmp = tmp * std::sqrt(factorial((j2 + j - j1)/2));
   tmp = tmp / std::sqrt(factorial((j1 + j2 + j)/2 + 1));
   tmp = tmp * std::sqrt(1.0 * (j + 1));
   tmp = tmp * std::sqrt(factorial((j1 + m1)/2));
   tmp = tmp * std::sqrt(factorial(j1nm1));
   tmp = tmp * std::sqrt(factorial(j2pm2));
   tmp = tmp * std::sqrt(factorial((j2 - m2)/2));
   tmp = tmp * std::sqrt(factorial((j + m)/2));
   tmp = tmp * std::sqrt(factorial((j - m)/2));

   return res * tmp;
}

}
