// SPDX-FileCopyrightText: 2025 euler_kokkos authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file mhd_utils.h
 * \brief Small MHD related utilities common to CPU / GPU code.
 *
 * These utility functions (find_speed_fast, etc...) are directly
 * adapted from Fortran original code found in RAMSES/DUMSES.
 *
 * \date 23 March 2011
 * \author Pierre Kestener.
 *
 */
#ifndef MHD_UTILS_H_
#define MHD_UTILS_H_

#include "real_type.h"

namespace euler_kokkos
{

/**
 * enum used to identify one the four states around a cell edge.
 *
 * This is useful when computing emf (electromotive forces)
 */
enum class MHDEdgeLocation : uint8_t
{
  LB,
  RB,
  LT,
  RT
};

/**
 * max value out of 4
 */
KOKKOS_INLINE_FUNCTION
real_t
FMAX4(real_t a0, real_t a1, real_t a2, real_t a3)
{
  real_t returnVal = a0;
  returnVal = (a1 > returnVal) ? a1 : returnVal;
  returnVal = (a2 > returnVal) ? a2 : returnVal;
  returnVal = (a3 > returnVal) ? a3 : returnVal;

  return returnVal;
} // FMAX4

/**
 * min value out of 4
 */
KOKKOS_INLINE_FUNCTION
real_t
FMIN4(real_t a0, real_t a1, real_t a2, real_t a3)
{
  real_t returnVal = a0;
  returnVal = (a1 < returnVal) ? a1 : returnVal;
  returnVal = (a2 < returnVal) ? a2 : returnVal;
  returnVal = (a3 < returnVal) ? a3 : returnVal;

  return returnVal;
} // FMIN4

/**
 * max value out of 5
 */
KOKKOS_INLINE_FUNCTION
real_t
FMAX5(real_t a0, real_t a1, real_t a2, real_t a3, real_t a4)
{
  real_t returnVal = a0;
  returnVal = (a1 > returnVal) ? a1 : returnVal;
  returnVal = (a2 > returnVal) ? a2 : returnVal;
  returnVal = (a3 > returnVal) ? a3 : returnVal;
  returnVal = (a4 > returnVal) ? a4 : returnVal;

  return returnVal;
} // FMAX5

/**
 * Compute the fast magnetosonic velocity.
 *
 * IU is index to Vnormal
 * IA is index to Bnormal
 *
 * IV, IW are indexes to Vtransverse1, Vtransverse2,
 * IB, IC are indexes to Btransverse1, Btransverse2
 *
 */
template <ComponentIndex3D dir>
KOKKOS_INLINE_FUNCTION real_t
find_speed_fast(const MHDState & qvar, const HydroParams & params)
{

  const real_t & gamma0 = params.settings.gamma0;
  real_t         d, p, a, b, c, b2, c2, d2, cf;

  d = qvar[ID];
  p = qvar[IP];
  a = qvar[IA];
  b = qvar[IB];
  c = qvar[IC];

  b2 = a * a + b * b + c * c;
  c2 = gamma0 * p / d;
  d2 = 0.5 * (b2 / d + c2);
  if (dir == IX)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * a * a / d));

  if (dir == IT)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * b * b / d));

  if (dir == IZ)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * c * c / d));

  return cf;

} // find_speed_fast

/**
 * Compute the Alfven velocity.
 *
 * The structure of qvar is :
 * rho, pressure,
 * vnormal, vtransverse1, vtransverse2,
 * bnormal, btransverse1, btransverse2
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
find_speed_alfven(MHDState qvar)
{

  real_t d = qvar[ID];
  real_t a = qvar[IA];

  return sqrt(a * a / d);

} // find_speed_alfven

/**
 * Compute the Alfven velocity.
 *
 * Simpler interface.
 * \param[in] d density
 * \param[in] a normal magnetic field\
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
find_speed_alfven(real_t d, real_t a)
{

  return sqrt(a * a / d);

} // find_speed_alfven

/**
 * Compute the 1d mhd fluxes from the conservative.
 *
 * Only used in Riemann solver HLL (probably cartesian only
 * compatible, since gas pressure is included).
 *
 * variables. The structure of qvar is :
 * rho, pressure,
 * vnormal, vtransverse1, vtransverse2,
 * bnormal, btransverse1, btransverse2.
 *
 * @param[in]  qvar state vector (primitive variables)
 * @param[out] cvar state vector (conservative variables)
 * @param[out] ff flux vector
 *
 */
KOKKOS_INLINE_FUNCTION
void
find_mhd_flux(const MHDState & qvar, MHDState & cvar, MHDState & ff, const HydroParams & params)
{

  const real_t & gamma0 = params.settings.gamma0;

  // ISOTHERMAL
  const real_t & cIso = params.settings.cIso;
  real_t         p;
  if (cIso > 0)
  {
    // recompute pressure
    p = qvar[ID] * cIso * cIso;
  }
  else
  {
    p = qvar[IP];
  }
  // end ISOTHERMAL

  // local variables
  const real_t entho = ONE_F / (gamma0 - ONE_F);

  real_t d, u, v, w, a, b, c;
  d = qvar[ID];
  u = qvar[IU];
  v = qvar[IV];
  w = qvar[IW];
  a = qvar[IA];
  b = qvar[IB];
  c = qvar[IC];

  real_t ecin = 0.5 * (u * u + v * v + w * w) * d;
  real_t emag = 0.5 * (a * a + b * b + c * c);
  real_t etot = p * entho + ecin + emag;
  real_t ptot = p + emag;

  // compute conservative variables
  cvar[ID] = d;
  cvar[IP] = etot;
  cvar[IU] = d * u;
  cvar[IV] = d * v;
  cvar[IW] = d * w;
  cvar[IA] = a;
  cvar[IB] = b;
  cvar[IC] = c;

  // compute fluxes
  ff[ID] = d * u;
  ff[IP] = (etot + ptot) * u - a * (a * u + b * v + c * w);
  ff[IU] = d * u * u - a * a + ptot; /* *** WARNING pressure included *** */
  ff[IV] = d * u * v - a * b;
  ff[IW] = d * u * w - a * c;
  ff[IA] = 0.0;
  ff[IB] = b * u - a * v;
  ff[IC] = c * u - a * w;

} // find_mhd_flux

/**
 * Computes fast magnetosonic wave for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastMagSpeed array containing fast magnetosonic speed along
 * x, y, and z direction.
 *
 * \tparam NDIM if NDIM==2, only computes magnetosonic speed along x
 * and y.
 */
template <DimensionType NDIM>
KOKKOS_INLINE_FUNCTION void
fast_mhd_speed(const MHDState & qState, real_t (&fastMagSpeed)[3], const HydroParams & params)
{

  const real_t & gamma0 = params.settings.gamma0;

  const real_t & rho = qState[ID];
  const real_t & p = qState[IP];
  /*const real_t& vx  = qState[IU];
    const real_t& vy  = qState[IV];
    const real_t& vz  = qState[IW];*/
  const real_t & bx = qState[IA];
  const real_t & by = qState[IB];
  const real_t & bz = qState[IC];

  real_t mag_perp, alfv, vit_son, some_vit, some_vit2, delta, fast_speed;

  // compute fast magnetosonic speed along X
  mag_perp = (by * by + bz * bz) / rho; // bt ^2 / rho
  alfv = bx * bx / rho;                 // bx / sqrt(4pi*rho)
  vit_son = gamma0 * p / rho;           // sonic contribution :  gamma*P / rho

  some_vit = mag_perp + alfv + vit_son; // whatever direction,
  // always the same
  some_vit2 = some_vit * some_vit;

  delta = fmax(ZERO_F, some_vit2 - 4 * vit_son * alfv);

  fast_speed = 0.5 * (some_vit + sqrt(delta));
  fast_speed = sqrt(fast_speed);

  fastMagSpeed[IX] = fast_speed;

  // compute fast magnetosonic speed along Y
  mag_perp = (bx * bx + bz * bz) / rho;
  alfv = by * by / rho;

  delta = fmax(ZERO_F, some_vit2 - 4 * vit_son * alfv);

  fast_speed = 0.5 * (some_vit + sqrt(delta));
  fast_speed = sqrt(fast_speed);

  fastMagSpeed[IT] = fast_speed;

  // compute fast magnetosonic speed along Z
  if (NDIM == THREE_D)
  {
    mag_perp = (bx * bx + by * by) / rho;
    alfv = bz * bz / rho;

    delta = fmax(ZERO_F, some_vit2 - 4 * vit_son * alfv);

    fast_speed = 0.5 * (some_vit + sqrt(delta));
    fast_speed = sqrt(fast_speed);

    fastMagSpeed[IZ] = fast_speed;
  }

} // fast_mhd_speed

/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed array containing fastest information speed along
 * x, y, and z direction.
 *
 * Directional information speed being defined as :
 * directional fast magneto speed + fabs(velocity component)
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 * \tparam NDIM if NDIM==2, only computes information speed along x
 * and y.
 */
template <DimensionType NDIM>
KOKKOS_INLINE_FUNCTION void
find_speed_info(const MHDState qState, real_t (&fastInfoSpeed)[3], const HydroParams & params)
{

  const real_t & gamma0 = params.settings.gamma0;
  real_t         d, p, a, b, c, b2, c2, d2, cf;
  const real_t & u = qState[IU];
  const real_t & v = qState[IV];
  const real_t & w = qState[IW];

  d = qState[ID];
  p = qState[IP];
  a = qState[IA];
  b = qState[IB];
  c = qState[IC];

  /*
   * compute fastest info speed along X
   */

  // square norm of magnetic field
  b2 = a * a + b * b + c * c;

  // square speed of sound
  c2 = gamma0 * p / d;

  d2 = 0.5 * (b2 / d + c2);

  cf = sqrt(d2 + sqrt(d2 * d2 - c2 * a * a / d));

  fastInfoSpeed[IX] = cf + fabs(u);

  // compute fastest info speed along Y
  cf = sqrt(d2 + sqrt(d2 * d2 - c2 * b * b / d));

  fastInfoSpeed[IT] = cf + fabs(v);


  // compute fastest info speed along Z
  if (NDIM == THREE_D)
  {
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * c * c / d));

    fastInfoSpeed[IZ] = cf + fabs(w);
  } // end THREE_D

} // find_speed_info

/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed fastest information speed along x
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
find_speed_info(const MHDState & qState, const HydroParams & params)
{

  const real_t & gamma0 = params.settings.gamma0;
  real_t         d, p, a, b, c, b2, c2, d2, cf;
  const real_t & u = qState[IU];
  // const real_t& v = qState[IV];
  // const real_t& w = qState[IW];

  d = qState[ID];
  p = qState[IP];
  a = qState[IA];
  b = qState[IB];
  c = qState[IC];

  // compute fastest info speed along X
  b2 = a * a + b * b + c * c;
  c2 = gamma0 * p / d;
  d2 = 0.5 * (b2 / d + c2);
  cf = sqrt(d2 + sqrt(d2 * d2 - c2 * a * a / d));

  // return value
  return cf + fabs(u);

} // find_speed_info

} // namespace euler_kokkos

#endif // MHD_UTILS_H_
