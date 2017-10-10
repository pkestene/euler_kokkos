#ifndef HYDRO_STATE_H_
#define HYDRO_STATE_H_

#include "real_type.h"

constexpr int HYDRO_2D_NBVAR=4;
constexpr int HYDRO_3D_NBVAR=5;
constexpr int MHD_2D_NBVAR=8;
constexpr int MHD_3D_NBVAR=8;
constexpr int MHD_NBVAR=8;

using HydroState2d = Kokkos::Array<real_t,HYDRO_2D_NBVAR>;
using HydroState3d = Kokkos::Array<real_t,HYDRO_3D_NBVAR>;
using MHDState     = Kokkos::Array<real_t,MHD_NBVAR>;
using BField       = Kokkos::Array<real_t,3>;

#endif // HYDRO_STATE_H_
