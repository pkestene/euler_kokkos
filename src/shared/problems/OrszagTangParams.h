#ifndef ORSZAG_TANG_PARAMS_H_
#define ORSZAG_TANG_PARAMS_H_

#include <math.h>

#include "utils/config/ConfigMap.h"

namespace euler_kokkos
{

struct OrszagTangParams
{

  // transverse wave vector
  real_t kt;

  OrszagTangParams(ConfigMap & configMap) { kt = configMap.getFloat("OrszagTang", "kt", 0.0); }

}; // struct OrszagTangParams

} // namespace euler_kokkos

#endif // ORSZAG_TANG_PARAMS_H_
