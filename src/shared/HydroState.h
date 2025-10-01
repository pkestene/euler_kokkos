#ifndef HYDRO_STATE_H_
#define HYDRO_STATE_H_

#include "real_type.h"

namespace euler_kokkos
{

constexpr int HYDRO_2D_NBVAR = 4;
constexpr int HYDRO_3D_NBVAR = 5;
constexpr int MHD_2D_NBVAR = 8;
constexpr int MHD_3D_NBVAR = 8;
constexpr int MHD_NBVAR = 8;

template <size_t dim>
constexpr size_t
nb_vars_hydro()
{
  if constexpr (dim == 2)
  {
    return HYDRO_2D_NBVAR;
  }
  else if (dim == 3)
  {
    return HYDRO_3D_NBVAR;
  }
} // nb_vars_hydro

template <size_t dim>
using HydroState = Kokkos::Array<real_t, nb_vars_hydro<dim>()>;

using HydroState2d = Kokkos::Array<real_t, HYDRO_2D_NBVAR>;
using HydroState3d = Kokkos::Array<real_t, HYDRO_3D_NBVAR>;
using MHDState = Kokkos::Array<real_t, MHD_NBVAR>;
using BField = Kokkos::Array<real_t, 3>;

} // namespace euler_kokkos

#endif // HYDRO_STATE_H_
