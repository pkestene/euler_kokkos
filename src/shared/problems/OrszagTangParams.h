// SPDX-FileCopyrightText: 2025 euler_kokkos authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ORSZAG_TANG_PARAMS_H_
#define ORSZAG_TANG_PARAMS_H_

#include <math.h>

#include "utils/config/ConfigMap.h"

namespace euler_kokkos
{

struct OrszagTangParams
{
  enum VortexDir : int
  {
    X,
    Y,
    Z
  };

  // transverse wave vector
  real_t kt;
  int    vortex_dir;

  OrszagTangParams(ConfigMap & configMap)
  {
    kt = configMap.getFloat("OrszagTang", "kt", 0.0);
    vortex_dir = configMap.getInteger("OrszagTang", "vortex_dir", VortexDir::Z);
  }

}; // struct OrszagTangParams

} // namespace euler_kokkos

#endif // ORSZAG_TANG_PARAMS_H_
