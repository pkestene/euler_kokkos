/**
 * \file initRiemannConfig2d.h
 * \brief Implement initialization routine to solve a four quadrant 2D Riemann
 * problem.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * \author P. Kestener
 *
 */
#ifndef INIT_RIEMANN_CONFIG_2D_H_
#define INIT_RIEMANN_CONFIG_2D_H_

#include "shared/real_type.h"
#include "shared/enums.h"
#include "shared/HydroState.h"

namespace euler_kokkos
{

// =====================================================================
// =====================================================================
template <size_t dim>
constexpr uint32_t
nb_riemann_states()
{
  if constexpr (dim == 2)
  {
    return 4;
  }
  else if constexpr (dim == 3)
  {
    return 8;
  }
} // nb_riemann_states

// =====================================================================
// =====================================================================
// =====================================================================
template <size_t dim>
struct RiemannConfig
{
  static constexpr uint32_t NB_STATES = nb_riemann_states<dim>();

  using HydroState_t = HydroState<dim>;

  using HydroStates_t = Kokkos::Array<HydroState_t, nb_riemann_states<dim>()>;
};

// =====================================================================
// =====================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION void
primToCons(HydroState<dim> & U, real_t gamma0)
{

  real_t rho = U[ID];
  real_t p = U[IP];
  real_t u = U[IU];
  real_t v = U[IV];

  if constexpr (dim == 2)
  {
    U[IU] *= rho; // rho*u
    U[IV] *= rho; // rho*v

    U[IP] = p / (gamma0 - 1.0) + rho * (u * u + v * v) * 0.5;
  }
  else if constexpr (dim == 3)
  {
    real_t w = U[IW];
    U[IU] *= rho; // rho*u
    U[IV] *= rho; // rho*v
    U[IW] *= rho; // rho*w

    U[IP] = p / (gamma0 - 1.0) + rho * (u * u + v * v + w * w) * 0.5;
  }

} // primToCons

// =====================================================================
// =====================================================================
RiemannConfig<2>::HydroStates_t
getRiemannConfig2d(int numConfig);

// =====================================================================
// =====================================================================
RiemannConfig<3>::HydroStates_t
getRiemannConfig3d(int numConfig);

// =====================================================================
// =====================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
getRiemannConfig(int numConfig)
{

  if constexpr (dim == 2)
  {
    return getRiemannConfig2d(numConfig);
  }
  else if constexpr (dim == 3)
  {
    return getRiemannConfig3d(numConfig);
  }
} // getRiemannConfig

} // namespace euler_kokkos

#endif // INIT_RIEMANN_CONFIG_2D_H_
