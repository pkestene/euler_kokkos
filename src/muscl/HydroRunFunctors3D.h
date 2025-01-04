// SPDX-FileCopyrightText: 2025 euler_kokkos authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HYDRO_RUN_FUNCTORS_3D_H_
#define HYDRO_RUN_FUNCTORS_3D_H_

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"
#include "shared/RiemannSolvers.h"

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Compute time step satisfying CFL constraint.
   *
   * \param[in] params
   * \param[in] Udata
   */
  ComputeDtFunctor3D(HydroParams params, DataArray3d Udata)
    : HydroBaseFunctor3D(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, real_t & invDt)
  {
    ComputeDtFunctor3D  functor(params, Udata);
    Kokkos::Max<real_t> reducer(invDt);
    Kokkos::parallel_reduce("ComputeDtFunctor3D",
                            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                              { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                            functor,
                            reducer);
  }

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k, real_t & invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    if (k >= ghostWidth and k < ksize - ghostWidth and j >= ghostWidth and
        j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c = 0.0;
      real_t     vx, vy, vz;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, k, ID);
      uLoc[IP] = Udata(i, j, k, IP);
      uLoc[IU] = Udata(i, j, k, IU);
      uLoc[IV] = Udata(i, j, k, IV);
      uLoc[IW] = Udata(i, j, k, IW);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      vx = c + fabs(qLoc[IU]);
      vy = c + fabs(qLoc[IV]);
      vz = c + fabs(qLoc[IW]);

      invDt = fmax(invDt, vx / dx + vy / dy + vz / dz);
    }

  } // operator ()


  DataArray3d Udata;

}; // ComputeDtFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * A specialized functor to compute CFL dt constraint when gravity source
 * term is activated.
 */
class ComputeDtGravityFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Compute time step satisfying CFL constraint.
   *
   * \param[in] params
   * \param[in] Udata
   */
  ComputeDtGravityFunctor3D(HydroParams   params,
                            real_t        cfl,
                            VectorField3d gravity,
                            DataArray3d   Udata)
    : HydroBaseFunctor3D(params)
    , cfl(cfl)
    , gravity(gravity)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, real_t cfl, VectorField3d gravity, DataArray3d Udata, real_t & invDt)
  {
    ComputeDtGravityFunctor3D functor(params, cfl, gravity, Udata);
    Kokkos::Max<real_t>       reducer(invDt);
    Kokkos::parallel_reduce("ComputeDtGravityFunctor3D",
                            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                              { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                            functor,
                            reducer);
  }

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k, real_t & invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    // const int nbvar = params.nbvar;
    real_t dx = fmin(params.dx, params.dy);
    dx = fmin(dx, params.dz);

    if (k >= ghostWidth and k < ksize - ghostWidth and j >= ghostWidth and
        j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c = 0.0;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, k, ID);
      uLoc[IP] = Udata(i, j, k, IP);
      uLoc[IU] = Udata(i, j, k, IU);
      uLoc[IV] = Udata(i, j, k, IV);
      uLoc[IW] = Udata(i, j, k, IW);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      real_t velocity = 0.0;
      velocity += c + fabs(qLoc[IU]);
      velocity += c + fabs(qLoc[IV]);
      velocity += c + fabs(qLoc[IW]);

      /* Due to the gravitational acceleration, the CFL condition
       * can be written as
       * g dt^2 / (2 dx) + u dt / dx <= cfl
       * where u = sum(|v_i| + c_s) and g = sum(|g_i|)
       *
       * u / dx has to be corrected by a factor k / (sqrt(1 + 2k) - 1)
       * in order to satisfy the new CFL, where k = g dx cfl / u^2
       */
      double kk =
        fabs(gravity(i, j, k, IX)) + fabs(gravity(i, j, k, IT)) + fabs(gravity(i, j, k, IZ));

      kk *= cfl * dx / (velocity * velocity);

      /* prevent numerical errors due to very low gravity */
      kk = fmax(kk, 1e-4);

      velocity *= kk / (sqrt(1.0 + 2.0 * kk) - 1.0);

      invDt = fmax(invDt, velocity / dx);
    }

  } // operator ()

  real_t        cfl;
  VectorField3d gravity;
  DataArray3d   Udata;

}; // ComputeDtGravityFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Convert conservative variables to primitive ones using equation of state.
   *
   * \param[in] params
   * \param[in] Udata conservative variables
   * \param[out] Qdata primitive variables
   */
  ConvertToPrimitivesFunctor3D(HydroParams params, DataArray3d Udata, DataArray3d Qdata)
    : HydroBaseFunctor3D(params)
    , Udata(Udata)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, DataArray3d Qdata)
  {
    ConvertToPrimitivesFunctor3D functor(params, Udata, Qdata);
    Kokkos::parallel_for("ConvertToPrimitivesFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    // const int ghostWidth = params.ghostWidth;

    if (k >= 0 and k < ksize and j >= 0 and j < jsize and i >= 0 and i < isize)
    {

      HydroState uLoc; // conservative variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, k, ID);
      uLoc[IP] = Udata(i, j, k, IP);
      uLoc[IU] = Udata(i, j, k, IU);
      uLoc[IV] = Udata(i, j, k, IV);
      uLoc[IW] = Udata(i, j, k, IW);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);

      // copy q state in q global
      Qdata(i, j, k, ID) = qLoc[ID];
      Qdata(i, j, k, IP) = qLoc[IP];
      Qdata(i, j, k, IU) = qLoc[IU];
      Qdata(i, j, k, IV) = qLoc[IV];
      Qdata(i, j, k, IW) = qLoc[IW];
    }
  }

  DataArray3d Udata;
  DataArray3d Qdata;

}; // ConvertToPrimitivesFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Compute (all-in-one) reconstructed states on faces, then compute Riemann fluxes and store them.
   *
   * \note All-in-one here means the stencil of this operator is larger (need to
   * fetch data in neighbor of neighbor).
   *
   * \param[in] Qdata primitive variables (at cell center)
   * \param[out] FluxData_x flux coming from the left neighbor along X
   * \param[out] FluxData_y flux coming from the left neighbor along Y
   * \param[out] FluxData_z flux coming from the left neighbor along Z
   * \param[in] gravity_enabled boolean value to activate static gravity
   * \param[in] gravity is a vector field
   */
  ComputeAndStoreFluxesFunctor3D(HydroParams   params,
                                 DataArray3d   Qdata,
                                 DataArray3d   FluxData_x,
                                 DataArray3d   FluxData_y,
                                 DataArray3d   FluxData_z,
                                 real_t        dt,
                                 bool          gravity_enabled,
                                 VectorField3d gravity)
    : HydroBaseFunctor3D(params)
    , Qdata(Qdata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z)
    , dt(dt)
    , dtdx(dt / params.dx)
    , dtdy(dt / params.dy)
    , dtdz(dt / params.dz)
    , gravity_enabled(gravity_enabled)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray3d   Qdata,
        DataArray3d   FluxData_x,
        DataArray3d   FluxData_y,
        DataArray3d   FluxData_z,
        real_t        dt,
        bool          gravity_enabled,
        VectorField3d gravity)
  {
    ComputeAndStoreFluxesFunctor3D functor(
      params, Qdata, FluxData_x, FluxData_y, FluxData_z, dt, gravity_enabled, gravity);
    Kokkos::parallel_for("ComputeAndStoreFluxesFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth and k <= ksize - ghostWidth and j >= ghostWidth and
        j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;
      HydroState qNeighbors_4;
      HydroState qNeighbors_5;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqZ;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;
      HydroState dqZ_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;
      HydroState flux_z;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      qLoc[ID] = Qdata(i, j, k, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, k, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, k, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, k, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, k, ID);
      qNeighbors_4[ID] = Qdata(i, j, k + 1, ID);
      qNeighbors_5[ID] = Qdata(i, j, k - 1, ID);

      qLoc[IP] = Qdata(i, j, k, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, k, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, k, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, k, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, k, IP);
      qNeighbors_4[IP] = Qdata(i, j, k + 1, IP);
      qNeighbors_5[IP] = Qdata(i, j, k - 1, IP);

      qLoc[IU] = Qdata(i, j, k, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, k, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, k, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, k, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, k, IU);
      qNeighbors_4[IU] = Qdata(i, j, k + 1, IU);
      qNeighbors_5[IU] = Qdata(i, j, k - 1, IU);

      qLoc[IV] = Qdata(i, j, k, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, k, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, k, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, k, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, k, IV);
      qNeighbors_4[IV] = Qdata(i, j, k + 1, IV);
      qNeighbors_5[IV] = Qdata(i, j, k - 1, IV);

      qLoc[IW] = Qdata(i, j, k, IW);
      qNeighbors_0[IW] = Qdata(i + 1, j, k, IW);
      qNeighbors_1[IW] = Qdata(i - 1, j, k, IW);
      qNeighbors_2[IW] = Qdata(i, j + 1, k, IW);
      qNeighbors_3[IW] = Qdata(i, j - 1, k, IW);
      qNeighbors_4[IW] = Qdata(i, j, k + 1, IW);
      qNeighbors_5[IW] = Qdata(i, j, k - 1, IW);

      slope_unsplit_hydro_3d(qLoc,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX,
                             dqY,
                             dqZ);

      // slopes at left neighbor along X
      qLocNeighbor[ID] = Qdata(i - 1, j, k, ID);
      qNeighbors_0[ID] = Qdata(i, j, k, ID);
      qNeighbors_1[ID] = Qdata(i - 2, j, k, ID);
      qNeighbors_2[ID] = Qdata(i - 1, j + 1, k, ID);
      qNeighbors_3[ID] = Qdata(i - 1, j - 1, k, ID);
      qNeighbors_4[ID] = Qdata(i - 1, j, k + 1, ID);
      qNeighbors_5[ID] = Qdata(i - 1, j, k - 1, ID);

      qLocNeighbor[IP] = Qdata(i - 1, j, k, IP);
      qNeighbors_0[IP] = Qdata(i, j, k, IP);
      qNeighbors_1[IP] = Qdata(i - 2, j, k, IP);
      qNeighbors_2[IP] = Qdata(i - 1, j + 1, k, IP);
      qNeighbors_3[IP] = Qdata(i - 1, j - 1, k, IP);
      qNeighbors_4[IP] = Qdata(i - 1, j, k + 1, IP);
      qNeighbors_5[IP] = Qdata(i - 1, j, k - 1, IP);

      qLocNeighbor[IU] = Qdata(i - 1, j, k, IU);
      qNeighbors_0[IU] = Qdata(i, j, k, IU);
      qNeighbors_1[IU] = Qdata(i - 2, j, k, IU);
      qNeighbors_2[IU] = Qdata(i - 1, j + 1, k, IU);
      qNeighbors_3[IU] = Qdata(i - 1, j - 1, k, IU);
      qNeighbors_4[IU] = Qdata(i - 1, j, k + 1, IU);
      qNeighbors_5[IU] = Qdata(i - 1, j, k - 1, IU);

      qLocNeighbor[IV] = Qdata(i - 1, j, k, IV);
      qNeighbors_0[IV] = Qdata(i, j, k, IV);
      qNeighbors_1[IV] = Qdata(i - 2, j, k, IV);
      qNeighbors_2[IV] = Qdata(i - 1, j + 1, k, IV);
      qNeighbors_3[IV] = Qdata(i - 1, j - 1, k, IV);
      qNeighbors_4[IV] = Qdata(i - 1, j, k + 1, IV);
      qNeighbors_5[IV] = Qdata(i - 1, j, k - 1, IV);

      qLocNeighbor[IW] = Qdata(i - 1, j, k, IW);
      qNeighbors_0[IW] = Qdata(i, j, k, IW);
      qNeighbors_1[IW] = Qdata(i - 2, j, k, IW);
      qNeighbors_2[IW] = Qdata(i - 1, j + 1, k, IW);
      qNeighbors_3[IW] = Qdata(i - 1, j - 1, k, IW);
      qNeighbors_4[IW] = Qdata(i - 1, j, k + 1, IW);
      qNeighbors_5[IW] = Qdata(i - 1, j, k - 1, IW);

      slope_unsplit_hydro_3d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX_neighbor,
                             dqY_neighbor,
                             dqZ_neighbor);

      //
      // compute reconstructed states at left interface along X
      //

      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_XMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dqZ_neighbor, dtdx, dtdy, dtdz, FACE_XMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i - 1, j, k, IX);
        qleft[IV] += 0.5 * dt * gravity(i - 1, j, k, IT);
        qleft[IW] += 0.5 * dt * gravity(i - 1, j, k, IZ);

        qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
        qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
      }

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann_hydro(qleft, qright, qgdnv, flux_x, params);

      //
      // store fluxes X
      //
      FluxData_x(i, j, k, ID) = flux_x[ID] * dtdx;
      FluxData_x(i, j, k, IP) = flux_x[IP] * dtdx;
      FluxData_x(i, j, k, IU) = flux_x[IU] * dtdx;
      FluxData_x(i, j, k, IV) = flux_x[IV] * dtdx;
      FluxData_x(i, j, k, IW) = flux_x[IW] * dtdx;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      qLocNeighbor[ID] = Qdata(i, j - 1, k, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j - 1, k, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j - 1, k, ID);
      qNeighbors_2[ID] = Qdata(i, j, k, ID);
      qNeighbors_3[ID] = Qdata(i, j - 2, k, ID);
      qNeighbors_4[ID] = Qdata(i, j - 1, k + 1, ID);
      qNeighbors_5[ID] = Qdata(i, j - 1, k - 1, ID);

      qLocNeighbor[IP] = Qdata(i, j - 1, k, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j - 1, k, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j - 1, k, IP);
      qNeighbors_2[IP] = Qdata(i, j, k, IP);
      qNeighbors_3[IP] = Qdata(i, j - 2, k, IP);
      qNeighbors_4[IP] = Qdata(i, j - 1, k + 1, IP);
      qNeighbors_5[IP] = Qdata(i, j - 1, k - 1, IP);

      qLocNeighbor[IU] = Qdata(i, j - 1, k, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j - 1, k, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j - 1, k, IU);
      qNeighbors_2[IU] = Qdata(i, j, k, IU);
      qNeighbors_3[IU] = Qdata(i, j - 2, k, IU);
      qNeighbors_4[IU] = Qdata(i, j - 1, k + 1, IU);
      qNeighbors_5[IU] = Qdata(i, j - 1, k - 1, IU);

      qLocNeighbor[IV] = Qdata(i, j - 1, k, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j - 1, k, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j - 1, k, IV);
      qNeighbors_2[IV] = Qdata(i, j, k, IV);
      qNeighbors_3[IV] = Qdata(i, j - 2, k, IV);
      qNeighbors_4[IV] = Qdata(i, j - 1, k + 1, IV);
      qNeighbors_5[IV] = Qdata(i, j - 1, k - 1, IV);

      qLocNeighbor[IW] = Qdata(i, j - 1, k, IW);
      qNeighbors_0[IW] = Qdata(i + 1, j - 1, k, IW);
      qNeighbors_1[IW] = Qdata(i - 1, j - 1, k, IW);
      qNeighbors_2[IW] = Qdata(i, j, k, IW);
      qNeighbors_3[IW] = Qdata(i, j - 2, k, IW);
      qNeighbors_4[IW] = Qdata(i, j - 1, k + 1, IW);
      qNeighbors_5[IW] = Qdata(i, j - 1, k - 1, IW);

      slope_unsplit_hydro_3d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX_neighbor,
                             dqY_neighbor,
                             dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //

      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dqZ_neighbor, dtdx, dtdy, dtdz, FACE_YMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i, j - 1, k, IX);
        qleft[IV] += 0.5 * dt * gravity(i, j - 1, k, IT);
        qleft[IW] += 0.5 * dt * gravity(i, j - 1, k, IZ);

        qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
        qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
      }

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qright[IU]), &(qright[IV]));
      riemann_hydro(qleft, qright, qgdnv, flux_y, params);

      //
      // store fluxes Y
      //
      FluxData_y(i, j, k, ID) = flux_y[ID] * dtdy;
      FluxData_y(i, j, k, IP) = flux_y[IP] * dtdy;
      FluxData_y(i, j, k, IU) = flux_y[IV] * dtdy; //
      FluxData_y(i, j, k, IV) = flux_y[IU] * dtdy; //
      FluxData_y(i, j, k, IW) = flux_y[IW] * dtdy;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Z !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Z
      qLocNeighbor[ID] = Qdata(i, j, k - 1, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, k - 1, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, k - 1, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, k - 1, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, k - 1, ID);
      qNeighbors_4[ID] = Qdata(i, j, k, ID);
      qNeighbors_5[ID] = Qdata(i, j, k - 2, ID);

      qLocNeighbor[IP] = Qdata(i, j, k - 1, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, k - 1, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, k - 1, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, k - 1, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, k - 1, IP);
      qNeighbors_4[IP] = Qdata(i, j, k, IP);
      qNeighbors_5[IP] = Qdata(i, j, k - 2, IP);

      qLocNeighbor[IU] = Qdata(i, j, k - 1, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, k - 1, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, k - 1, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, k - 1, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, k - 1, IU);
      qNeighbors_4[IU] = Qdata(i, j, k, IU);
      qNeighbors_5[IU] = Qdata(i, j, k - 2, IU);

      qLocNeighbor[IV] = Qdata(i, j, k - 1, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, k - 1, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, k - 1, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, k - 1, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, k - 1, IV);
      qNeighbors_4[IV] = Qdata(i, j, k, IV);
      qNeighbors_5[IV] = Qdata(i, j, k - 2, IV);

      qLocNeighbor[IW] = Qdata(i, j, k - 1, IW);
      qNeighbors_0[IW] = Qdata(i + 1, j, k - 1, IW);
      qNeighbors_1[IW] = Qdata(i - 1, j, k - 1, IW);
      qNeighbors_2[IW] = Qdata(i, j + 1, k - 1, IW);
      qNeighbors_3[IW] = Qdata(i, j - 1, k - 1, IW);
      qNeighbors_4[IW] = Qdata(i, j, k, IW);
      qNeighbors_5[IW] = Qdata(i, j, k - 2, IW);

      slope_unsplit_hydro_3d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX_neighbor,
                             dqY_neighbor,
                             dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Z
      //

      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_ZMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dqZ_neighbor, dtdx, dtdy, dtdz, FACE_ZMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i, j, k - 1, IX);
        qleft[IV] += 0.5 * dt * gravity(i, j, k - 1, IT);
        qleft[IW] += 0.5 * dt * gravity(i, j, k - 1, IZ);

        qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
        qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
      }

      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      swapValues(&(qleft[IU]), &(qleft[IW]));
      swapValues(&(qright[IU]), &(qright[IW]));
      riemann_hydro(qleft, qright, qgdnv, flux_z, params);

      //
      // store fluxes Z
      //
      FluxData_z(i, j, k, ID) = flux_z[ID] * dtdz;
      FluxData_z(i, j, k, IP) = flux_z[IP] * dtdz;
      FluxData_z(i, j, k, IU) = flux_z[IW] * dtdz; //
      FluxData_z(i, j, k, IV) = flux_z[IV] * dtdz;
      FluxData_z(i, j, k, IW) = flux_z[IU] * dtdz; //

    } // end if

  } // end operator ()

  DataArray3d   Qdata;
  DataArray3d   FluxData_x;
  DataArray3d   FluxData_y;
  DataArray3d   FluxData_z;
  real_t        dt, dtdx, dtdy, dtdz;
  bool          gravity_enabled;
  VectorField3d gravity;

}; // ComputeAndStoreFluxesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Perform time update using the stored fluxes.
   *
   * \note this functor must be called after ComputeAndStoreFluxesFunctor2D
   *
   * \param[in,out] Udata
   * \param[in] FluxData_x flux coming from the left neighbor along X
   * \param[in] FluxData_y flux coming from the left neighbor along Y
   * \param[in] FluxData_z flux coming from the left neighbor along Z
   */
  UpdateFunctor3D(HydroParams params,
                  DataArray3d Udata,
                  DataArray3d FluxData_x,
                  DataArray3d FluxData_y,
                  DataArray3d FluxData_z)
    : HydroBaseFunctor3D(params)
    , Udata(Udata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Udata,
        DataArray3d FluxData_x,
        DataArray3d FluxData_y,
        DataArray3d FluxData_z)
  {
    UpdateFunctor3D functor(params, Udata, FluxData_x, FluxData_y, FluxData_z);
    Kokkos::parallel_for("UpdateFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth and k < ksize - ghostWidth and j >= ghostWidth and
        j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      Udata(i, j, k, ID) += FluxData_x(i, j, k, ID);
      Udata(i, j, k, IP) += FluxData_x(i, j, k, IP);
      Udata(i, j, k, IU) += FluxData_x(i, j, k, IU);
      Udata(i, j, k, IV) += FluxData_x(i, j, k, IV);
      Udata(i, j, k, IW) += FluxData_x(i, j, k, IW);

      Udata(i, j, k, ID) -= FluxData_x(i + 1, j, k, ID);
      Udata(i, j, k, IP) -= FluxData_x(i + 1, j, k, IP);
      Udata(i, j, k, IU) -= FluxData_x(i + 1, j, k, IU);
      Udata(i, j, k, IV) -= FluxData_x(i + 1, j, k, IV);
      Udata(i, j, k, IW) -= FluxData_x(i + 1, j, k, IW);

      Udata(i, j, k, ID) += FluxData_y(i, j, k, ID);
      Udata(i, j, k, IP) += FluxData_y(i, j, k, IP);
      Udata(i, j, k, IU) += FluxData_y(i, j, k, IU);
      Udata(i, j, k, IV) += FluxData_y(i, j, k, IV);
      Udata(i, j, k, IW) += FluxData_y(i, j, k, IW);

      Udata(i, j, k, ID) -= FluxData_y(i, j + 1, k, ID);
      Udata(i, j, k, IP) -= FluxData_y(i, j + 1, k, IP);
      Udata(i, j, k, IU) -= FluxData_y(i, j + 1, k, IU);
      Udata(i, j, k, IV) -= FluxData_y(i, j + 1, k, IV);
      Udata(i, j, k, IW) -= FluxData_y(i, j + 1, k, IW);

      Udata(i, j, k, ID) += FluxData_z(i, j, k, ID);
      Udata(i, j, k, IP) += FluxData_z(i, j, k, IP);
      Udata(i, j, k, IU) += FluxData_z(i, j, k, IU);
      Udata(i, j, k, IV) += FluxData_z(i, j, k, IV);
      Udata(i, j, k, IW) += FluxData_z(i, j, k, IW);

      Udata(i, j, k, ID) -= FluxData_z(i, j, k + 1, ID);
      Udata(i, j, k, IP) -= FluxData_z(i, j, k + 1, IP);
      Udata(i, j, k, IU) -= FluxData_z(i, j, k + 1, IU);
      Udata(i, j, k, IV) -= FluxData_z(i, j, k + 1, IV);
      Udata(i, j, k, IW) -= FluxData_z(i, j, k + 1, IW);

    } // end if

  } // end operator ()

  DataArray3d Udata;
  DataArray3d FluxData_x;
  DataArray3d FluxData_y;
  DataArray3d FluxData_z;

}; // UpdateFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class UpdateDirFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Perform time update using the stored fluxes along direction dir.
   *
   * \param[in,out] Udata
   * \param[in] FluxData flux coming from the left neighbor along direction dir
   *
   */
  UpdateDirFunctor3D(HydroParams params, DataArray3d Udata, DataArray3d FluxData)
    : HydroBaseFunctor3D(params)
    , Udata(Udata)
    , FluxData(FluxData){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, DataArray3d FluxData)
  {
    UpdateDirFunctor3D<dir> functor(params, Udata, FluxData);
    Kokkos::parallel_for("UpdateDirFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth and k < ksize - ghostWidth and j >= ghostWidth and
        j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      if (dir == XDIR)
      {

        Udata(i, j, k, ID) += FluxData(i, j, k, ID);
        Udata(i, j, k, IP) += FluxData(i, j, k, IP);
        Udata(i, j, k, IU) += FluxData(i, j, k, IU);
        Udata(i, j, k, IV) += FluxData(i, j, k, IV);
        Udata(i, j, k, IW) += FluxData(i, j, k, IW);

        Udata(i, j, k, ID) -= FluxData(i + 1, j, k, ID);
        Udata(i, j, k, IP) -= FluxData(i + 1, j, k, IP);
        Udata(i, j, k, IU) -= FluxData(i + 1, j, k, IU);
        Udata(i, j, k, IV) -= FluxData(i + 1, j, k, IV);
        Udata(i, j, k, IW) -= FluxData(i + 1, j, k, IW);
      }
      else if (dir == YDIR)
      {

        Udata(i, j, k, ID) += FluxData(i, j, k, ID);
        Udata(i, j, k, IP) += FluxData(i, j, k, IP);
        Udata(i, j, k, IU) += FluxData(i, j, k, IU);
        Udata(i, j, k, IV) += FluxData(i, j, k, IV);
        Udata(i, j, k, IW) += FluxData(i, j, k, IW);

        Udata(i, j, k, ID) -= FluxData(i, j + 1, k, ID);
        Udata(i, j, k, IP) -= FluxData(i, j + 1, k, IP);
        Udata(i, j, k, IU) -= FluxData(i, j + 1, k, IU);
        Udata(i, j, k, IV) -= FluxData(i, j + 1, k, IV);
        Udata(i, j, k, IW) -= FluxData(i, j + 1, k, IW);
      }
      else if (dir == ZDIR)
      {

        Udata(i, j, k, ID) += FluxData(i, j, k, ID);
        Udata(i, j, k, IP) += FluxData(i, j, k, IP);
        Udata(i, j, k, IU) += FluxData(i, j, k, IU);
        Udata(i, j, k, IV) += FluxData(i, j, k, IV);
        Udata(i, j, k, IW) += FluxData(i, j, k, IW);

        Udata(i, j, k, ID) -= FluxData(i, j, k + 1, ID);
        Udata(i, j, k, IP) -= FluxData(i, j, k + 1, IP);
        Udata(i, j, k, IU) -= FluxData(i, j, k + 1, IU);
        Udata(i, j, k, IV) -= FluxData(i, j, k + 1, IV);
        Udata(i, j, k, IW) -= FluxData(i, j, k + 1, IW);
      }

    } // end if

  } // end operator ()

  DataArray3d Udata;
  DataArray3d FluxData;

}; // UpdateDirFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Compute limited slopes.
   *
   * \param[in] Qdata primitive variables
   * \param[out] Slopes_x limited slopes along direction X
   * \param[out] Slopes_y limited slopes along direction Y
   * \param[out] Slopes_z limited slopes along direction Z
   */
  ComputeSlopesFunctor3D(HydroParams params,
                         DataArray3d Qdata,
                         DataArray3d Slopes_x,
                         DataArray3d Slopes_y,
                         DataArray3d Slopes_z)
    : HydroBaseFunctor3D(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Slopes_z(Slopes_z){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Qdata,
        DataArray3d Slopes_x,
        DataArray3d Slopes_y,
        DataArray3d Slopes_z)
  {
    ComputeSlopesFunctor3D functor(params, Qdata, Slopes_x, Slopes_y, Slopes_z);
    Kokkos::parallel_for("ComputeSlopesFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth - 1 and k <= ksize - ghostWidth and j >= ghostWidth - 1 and
        j <= jsize - ghostWidth and i >= ghostWidth - 1 and i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;
      HydroState qNeighbors_4;
      HydroState qNeighbors_5;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqZ;

      // get primitive variables state vector
      qLoc[ID] = Qdata(i, j, k, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, k, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, k, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, k, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, k, ID);
      qNeighbors_4[ID] = Qdata(i, j, k + 1, ID);
      qNeighbors_5[ID] = Qdata(i, j, k - 1, ID);

      qLoc[IP] = Qdata(i, j, k, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, k, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, k, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, k, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, k, IP);
      qNeighbors_4[IP] = Qdata(i, j, k + 1, IP);
      qNeighbors_5[IP] = Qdata(i, j, k - 1, IP);

      qLoc[IU] = Qdata(i, j, k, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, k, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, k, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, k, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, k, IU);
      qNeighbors_4[IU] = Qdata(i, j, k + 1, IU);
      qNeighbors_5[IU] = Qdata(i, j, k - 1, IU);

      qLoc[IV] = Qdata(i, j, k, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, k, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, k, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, k, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, k, IV);
      qNeighbors_4[IV] = Qdata(i, j, k + 1, IV);
      qNeighbors_5[IV] = Qdata(i, j, k - 1, IV);

      qLoc[IW] = Qdata(i, j, k, IW);
      qNeighbors_0[IW] = Qdata(i + 1, j, k, IW);
      qNeighbors_1[IW] = Qdata(i - 1, j, k, IW);
      qNeighbors_2[IW] = Qdata(i, j + 1, k, IW);
      qNeighbors_3[IW] = Qdata(i, j - 1, k, IW);
      qNeighbors_4[IW] = Qdata(i, j, k + 1, IW);
      qNeighbors_5[IW] = Qdata(i, j, k - 1, IW);

      slope_unsplit_hydro_3d(qLoc,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX,
                             dqY,
                             dqZ);

      // copy back slopes in global arrays
      Slopes_x(i, j, k, ID) = dqX[ID];
      Slopes_y(i, j, k, ID) = dqY[ID];
      Slopes_z(i, j, k, ID) = dqZ[ID];

      Slopes_x(i, j, k, IP) = dqX[IP];
      Slopes_y(i, j, k, IP) = dqY[IP];
      Slopes_z(i, j, k, IP) = dqZ[IP];

      Slopes_x(i, j, k, IU) = dqX[IU];
      Slopes_y(i, j, k, IU) = dqY[IU];
      Slopes_z(i, j, k, IU) = dqZ[IU];

      Slopes_x(i, j, k, IV) = dqX[IV];
      Slopes_y(i, j, k, IV) = dqY[IV];
      Slopes_z(i, j, k, IV) = dqZ[IV];

      Slopes_x(i, j, k, IW) = dqX[IW];
      Slopes_y(i, j, k, IW) = dqY[IW];
      Slopes_z(i, j, k, IW) = dqZ[IW];

    } // end if

  } // end operator ()

  DataArray3d Qdata;
  DataArray3d Slopes_x, Slopes_y, Slopes_z;

}; // ComputeSlopesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Compute reconstructed states on faces (not stored), and fluxes (stored).
   *
   * \param[in] Qdata primitive variables
   * \param[in] Slopes_x limited slopes along direction X
   * \param[in] Slopes_y limited slopes along direction Y
   * \param[in] Slopes_z limited slopes along direction Z
   * \param[out] Fluxes along direction dir
   *
   * \tparam dir direction along which fluxes are computed.
   */
  ComputeTraceAndFluxes_Functor3D(HydroParams   params,
                                  DataArray3d   Qdata,
                                  DataArray3d   Slopes_x,
                                  DataArray3d   Slopes_y,
                                  DataArray3d   Slopes_z,
                                  DataArray3d   Fluxes,
                                  real_t        dt,
                                  bool          gravity_enabled,
                                  VectorField3d gravity)
    : HydroBaseFunctor3D(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Slopes_z(Slopes_z)
    , Fluxes(Fluxes)
    , dt(dt)
    , dtdx(dt / params.dx)
    , dtdy(dt / params.dy)
    , dtdz(dt / params.dz)
    , gravity_enabled(gravity_enabled)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray3d   Qdata,
        DataArray3d   Slopes_x,
        DataArray3d   Slopes_y,
        DataArray3d   Slopes_z,
        DataArray3d   Fluxes,
        real_t        dt,
        bool          gravity_enabled,
        VectorField3d gravity)
  {
    ComputeTraceAndFluxes_Functor3D<dir> functor(
      params, Qdata, Slopes_x, Slopes_y, Slopes_z, Fluxes, dt, gravity_enabled, gravity);
    Kokkos::parallel_for("ComputeTraceAndFluxes_Functor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth and k <= ksize - ghostWidth and j >= ghostWidth and
        j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqZ;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;
      HydroState dqZ_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux;

      //
      // compute reconstructed states at left interface along X
      //
      qLoc[ID] = Qdata(i, j, k, ID);
      dqX[ID] = Slopes_x(i, j, k, ID);
      dqY[ID] = Slopes_y(i, j, k, ID);
      dqZ[ID] = Slopes_z(i, j, k, ID);

      qLoc[IP] = Qdata(i, j, k, IP);
      dqX[IP] = Slopes_x(i, j, k, IP);
      dqY[IP] = Slopes_y(i, j, k, IP);
      dqZ[IP] = Slopes_z(i, j, k, IP);

      qLoc[IU] = Qdata(i, j, k, IU);
      dqX[IU] = Slopes_x(i, j, k, IU);
      dqY[IU] = Slopes_y(i, j, k, IU);
      dqZ[IU] = Slopes_z(i, j, k, IU);

      qLoc[IV] = Qdata(i, j, k, IV);
      dqX[IV] = Slopes_x(i, j, k, IV);
      dqY[IV] = Slopes_y(i, j, k, IV);
      dqZ[IV] = Slopes_z(i, j, k, IV);

      qLoc[IW] = Qdata(i, j, k, IW);
      dqX[IW] = Slopes_x(i, j, k, IW);
      dqY[IW] = Slopes_y(i, j, k, IW);
      dqZ[IW] = Slopes_z(i, j, k, IW);

      if (dir == XDIR)
      {

        // left interface : right state
        trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_XMIN, qright);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
          qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
          qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
        }

        qLocNeighbor[ID] = Qdata(i - 1, j, k, ID);
        dqX_neighbor[ID] = Slopes_x(i - 1, j, k, ID);
        dqY_neighbor[ID] = Slopes_y(i - 1, j, k, ID);
        dqZ_neighbor[ID] = Slopes_z(i - 1, j, k, ID);

        qLocNeighbor[IP] = Qdata(i - 1, j, k, IP);
        dqX_neighbor[IP] = Slopes_x(i - 1, j, k, IP);
        dqY_neighbor[IP] = Slopes_y(i - 1, j, k, IP);
        dqZ_neighbor[IP] = Slopes_z(i - 1, j, k, IP);

        qLocNeighbor[IU] = Qdata(i - 1, j, k, IU);
        dqX_neighbor[IU] = Slopes_x(i - 1, j, k, IU);
        dqY_neighbor[IU] = Slopes_y(i - 1, j, k, IU);
        dqZ_neighbor[IU] = Slopes_z(i - 1, j, k, IU);

        qLocNeighbor[IV] = Qdata(i - 1, j, k, IV);
        dqX_neighbor[IV] = Slopes_x(i - 1, j, k, IV);
        dqY_neighbor[IV] = Slopes_y(i - 1, j, k, IV);
        dqZ_neighbor[IV] = Slopes_z(i - 1, j, k, IV);

        qLocNeighbor[IW] = Qdata(i - 1, j, k, IW);
        dqX_neighbor[IW] = Slopes_x(i - 1, j, k, IW);
        dqY_neighbor[IW] = Slopes_y(i - 1, j, k, IW);
        dqZ_neighbor[IW] = Slopes_z(i - 1, j, k, IW);

        // left interface : left state
        trace_unsplit_3d_along_dir(qLocNeighbor,
                                   dqX_neighbor,
                                   dqY_neighbor,
                                   dqZ_neighbor,
                                   dtdx,
                                   dtdy,
                                   dtdz,
                                   FACE_XMAX,
                                   qleft);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qleft[IU] += 0.5 * dt * gravity(i - 1, j, k, IX);
          qleft[IV] += 0.5 * dt * gravity(i - 1, j, k, IT);
          qleft[IW] += 0.5 * dt * gravity(i - 1, j, k, IZ);
        }

        // Solve Riemann problem at X-interfaces and compute X-fluxes
        riemann_hydro(qleft, qright, qgdnv, flux, params);

        //
        // store fluxes
        //
        Fluxes(i, j, k, ID) = flux[ID] * dtdx;
        Fluxes(i, j, k, IP) = flux[IP] * dtdx;
        Fluxes(i, j, k, IU) = flux[IU] * dtdx;
        Fluxes(i, j, k, IV) = flux[IV] * dtdx;
        Fluxes(i, j, k, IW) = flux[IW] * dtdx;
      }
      else if (dir == YDIR)
      {

        // left interface : right state
        trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_YMIN, qright);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
          qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
          qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
        }

        qLocNeighbor[ID] = Qdata(i, j - 1, k, ID);
        dqX_neighbor[ID] = Slopes_x(i, j - 1, k, ID);
        dqY_neighbor[ID] = Slopes_y(i, j - 1, k, ID);
        dqZ_neighbor[ID] = Slopes_z(i, j - 1, k, ID);

        qLocNeighbor[IP] = Qdata(i, j - 1, k, IP);
        dqX_neighbor[IP] = Slopes_x(i, j - 1, k, IP);
        dqY_neighbor[IP] = Slopes_y(i, j - 1, k, IP);
        dqZ_neighbor[IP] = Slopes_z(i, j - 1, k, IP);

        qLocNeighbor[IU] = Qdata(i, j - 1, k, IU);
        dqX_neighbor[IU] = Slopes_x(i, j - 1, k, IU);
        dqY_neighbor[IU] = Slopes_y(i, j - 1, k, IU);
        dqZ_neighbor[IU] = Slopes_z(i, j - 1, k, IU);

        qLocNeighbor[IV] = Qdata(i, j - 1, k, IV);
        dqX_neighbor[IV] = Slopes_x(i, j - 1, k, IV);
        dqY_neighbor[IV] = Slopes_y(i, j - 1, k, IV);
        dqZ_neighbor[IV] = Slopes_z(i, j - 1, k, IV);

        qLocNeighbor[IW] = Qdata(i, j - 1, k, IW);
        dqX_neighbor[IW] = Slopes_x(i, j - 1, k, IW);
        dqY_neighbor[IW] = Slopes_y(i, j - 1, k, IW);
        dqZ_neighbor[IW] = Slopes_z(i, j - 1, k, IW);

        // left interface : left state
        trace_unsplit_3d_along_dir(qLocNeighbor,
                                   dqX_neighbor,
                                   dqY_neighbor,
                                   dqZ_neighbor,
                                   dtdx,
                                   dtdy,
                                   dtdz,
                                   FACE_YMAX,
                                   qleft);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qleft[IU] += 0.5 * dt * gravity(i, j - 1, k, IX);
          qleft[IV] += 0.5 * dt * gravity(i, j - 1, k, IT);
          qleft[IW] += 0.5 * dt * gravity(i, j - 1, k, IZ);
        }

        // Solve Riemann problem at Y-interfaces and compute Y-fluxes
        swapValues(&(qleft[IU]), &(qleft[IV]));
        swapValues(&(qright[IU]), &(qright[IV]));
        riemann_hydro(qleft, qright, qgdnv, flux, params);

        //
        // update hydro array
        //
        Fluxes(i, j, k, ID) = flux[ID] * dtdy;
        Fluxes(i, j, k, IP) = flux[IP] * dtdy;
        Fluxes(i, j, k, IU) = flux[IV] * dtdy; // IU/IV swapped
        Fluxes(i, j, k, IV) = flux[IU] * dtdy; // IU/IV swapped
        Fluxes(i, j, k, IW) = flux[IW] * dtdy;
      }
      else if (dir == ZDIR)
      {

        // left interface : right state
        trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_ZMIN, qright);

        qLocNeighbor[ID] = Qdata(i, j, k - 1, ID);
        dqX_neighbor[ID] = Slopes_x(i, j, k - 1, ID);
        dqY_neighbor[ID] = Slopes_y(i, j, k - 1, ID);
        dqZ_neighbor[ID] = Slopes_z(i, j, k - 1, ID);

        qLocNeighbor[IP] = Qdata(i, j, k - 1, IP);
        dqX_neighbor[IP] = Slopes_x(i, j, k - 1, IP);
        dqY_neighbor[IP] = Slopes_y(i, j, k - 1, IP);
        dqZ_neighbor[IP] = Slopes_z(i, j, k - 1, IP);

        qLocNeighbor[IU] = Qdata(i, j, k - 1, IU);
        dqX_neighbor[IU] = Slopes_x(i, j, k - 1, IU);
        dqY_neighbor[IU] = Slopes_y(i, j, k - 1, IU);
        dqZ_neighbor[IU] = Slopes_z(i, j, k - 1, IU);

        qLocNeighbor[IV] = Qdata(i, j, k - 1, IV);
        dqX_neighbor[IV] = Slopes_x(i, j, k - 1, IV);
        dqY_neighbor[IV] = Slopes_y(i, j, k - 1, IV);
        dqZ_neighbor[IV] = Slopes_z(i, j, k - 1, IV);

        qLocNeighbor[IW] = Qdata(i, j, k - 1, IW);
        dqX_neighbor[IW] = Slopes_x(i, j, k - 1, IW);
        dqY_neighbor[IW] = Slopes_y(i, j, k - 1, IW);
        dqZ_neighbor[IW] = Slopes_z(i, j, k - 1, IW);

        // left interface : left state
        trace_unsplit_3d_along_dir(qLocNeighbor,
                                   dqX_neighbor,
                                   dqY_neighbor,
                                   dqZ_neighbor,
                                   dtdx,
                                   dtdy,
                                   dtdz,
                                   FACE_ZMAX,
                                   qleft);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qleft[IU] += 0.5 * dt * gravity(i, j, k - 1, IX);
          qleft[IV] += 0.5 * dt * gravity(i, j, k - 1, IT);
          qleft[IW] += 0.5 * dt * gravity(i, j, k - 1, IZ);
        }

        // Solve Riemann problem at Y-interfaces and compute Y-fluxes
        swapValues(&(qleft[IU]), &(qleft[IW]));
        swapValues(&(qright[IU]), &(qright[IW]));
        riemann_hydro(qleft, qright, qgdnv, flux, params);

        //
        // update hydro array
        //
        Fluxes(i, j, k, ID) = flux[ID] * dtdz;
        Fluxes(i, j, k, IP) = flux[IP] * dtdz;
        Fluxes(i, j, k, IU) = flux[IW] * dtdz; // IU/IW swapped
        Fluxes(i, j, k, IV) = flux[IV] * dtdz;
        Fluxes(i, j, k, IW) = flux[IU] * dtdz; // IU/IW swapped
      }

    } // end if

  } // end operator ()

  DataArray3d   Qdata;
  DataArray3d   Slopes_x, Slopes_y, Slopes_z;
  DataArray3d   Fluxes;
  real_t        dt, dtdx, dtdy, dtdz;
  bool          gravity_enabled;
  VectorField3d gravity;

}; // ComputeTraceAndFluxes_Functor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAllFluxesAndUpdateFunctor3D : public HydroBaseFunctor3D
{

public:
  ComputeAllFluxesAndUpdateFunctor3D(HydroParams   params,
                                     DataArray3d   Qdata,
                                     DataArray3d   Udata,
                                     real_t        dt,
                                     bool          gravity_enabled,
                                     VectorField3d gravity)
    : HydroBaseFunctor3D(params)
    , Qdata(Qdata)
    , Udata(Udata)
    , dt(dt)
    , dtdx(dt / params.dx)
    , dtdy(dt / params.dy)
    , dtdz(dt / params.dz)
    , gravity_enabled(gravity_enabled)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray3d   Qdata,
        DataArray3d   Udata,
        real_t        dt,
        bool          gravity_enabled,
        VectorField3d gravity)
  {
    ComputeAllFluxesAndUpdateFunctor3D functor(params, Qdata, Udata, dt, gravity_enabled, gravity);
    Kokkos::parallel_for("ComputeAllFluxesAndUpdateFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth and k <= ksize - ghostWidth and j >= ghostWidth and
        j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;
      HydroState qNeighbors_4;
      HydroState qNeighbors_5;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqZ;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;
      HydroState dqZ_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;
      HydroState flux_z;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      // clang-format off
      qLoc[ID] = Qdata(i, j, k, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j    , k    , ID);
      qNeighbors_1[ID] = Qdata(i - 1, j    , k    , ID);
      qNeighbors_2[ID] = Qdata(i    , j + 1, k    , ID);
      qNeighbors_3[ID] = Qdata(i    , j - 1, k    , ID);
      qNeighbors_4[ID] = Qdata(i    , j    , k + 1, ID);
      qNeighbors_5[ID] = Qdata(i    , j    , k - 1, ID);

      qLoc[IP] = Qdata(i, j, k, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j    , k    , IP);
      qNeighbors_1[IP] = Qdata(i - 1, j    , k    , IP);
      qNeighbors_2[IP] = Qdata(i    , j + 1, k    , IP);
      qNeighbors_3[IP] = Qdata(i    , j - 1, k    , IP);
      qNeighbors_4[IP] = Qdata(i    , j    , k + 1, IP);
      qNeighbors_5[IP] = Qdata(i    , j    , k - 1, IP);

      qLoc[IU] = Qdata(i, j, k, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j    , k    , IU);
      qNeighbors_1[IU] = Qdata(i - 1, j    , k    , IU);
      qNeighbors_2[IU] = Qdata(i    , j + 1, k    , IU);
      qNeighbors_3[IU] = Qdata(i    , j - 1, k    , IU);
      qNeighbors_4[IU] = Qdata(i    , j    , k + 1, IU);
      qNeighbors_5[IU] = Qdata(i    , j    , k - 1, IU);

      qLoc[IV] = Qdata(i, j, k, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j    , k    , IV);
      qNeighbors_1[IV] = Qdata(i - 1, j    , k    , IV);
      qNeighbors_2[IV] = Qdata(i    , j + 1, k    , IV);
      qNeighbors_3[IV] = Qdata(i    , j - 1, k    , IV);
      qNeighbors_4[IV] = Qdata(i    , j    , k + 1, IV);
      qNeighbors_5[IV] = Qdata(i    , j    , k - 1, IV);

      qLoc[IW] = Qdata(i, j, k, IW);
      qNeighbors_0[IW] = Qdata(i + 1, j    , k    , IW);
      qNeighbors_1[IW] = Qdata(i - 1, j    , k    , IW);
      qNeighbors_2[IW] = Qdata(i    , j + 1, k    , IW);
      qNeighbors_3[IW] = Qdata(i    , j - 1, k    , IW);
      qNeighbors_4[IW] = Qdata(i    , j    , k + 1, IW);
      qNeighbors_5[IW] = Qdata(i    , j    , k - 1, IW);
      // clang-format on

      slope_unsplit_hydro_3d(qLoc,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX,
                             dqY,
                             dqZ);

      // slopes at left neighbor along X
      // clang-format off
      qLocNeighbor[ID] = Qdata(i - 1, j    , k    , ID);
      qNeighbors_0[ID] = Qdata(i    , j    , k    , ID);
      qNeighbors_1[ID] = Qdata(i - 2, j    , k    , ID);
      qNeighbors_2[ID] = Qdata(i - 1, j + 1, k    , ID);
      qNeighbors_3[ID] = Qdata(i - 1, j - 1, k    , ID);
      qNeighbors_4[ID] = Qdata(i - 1, j    , k + 1, ID);
      qNeighbors_5[ID] = Qdata(i - 1, j    , k - 1, ID);

      qLocNeighbor[IP] = Qdata(i - 1, j    , k    , IP);
      qNeighbors_0[IP] = Qdata(i    , j    , k    , IP);
      qNeighbors_1[IP] = Qdata(i - 2, j    , k    , IP);
      qNeighbors_2[IP] = Qdata(i - 1, j + 1, k    , IP);
      qNeighbors_3[IP] = Qdata(i - 1, j - 1, k    , IP);
      qNeighbors_4[IP] = Qdata(i - 1, j    , k + 1, IP);
      qNeighbors_5[IP] = Qdata(i - 1, j    , k - 1, IP);

      qLocNeighbor[IU] = Qdata(i - 1, j    , k    , IU);
      qNeighbors_0[IU] = Qdata(i    , j    , k    , IU);
      qNeighbors_1[IU] = Qdata(i - 2, j    , k    , IU);
      qNeighbors_2[IU] = Qdata(i - 1, j + 1, k    , IU);
      qNeighbors_3[IU] = Qdata(i - 1, j - 1, k    , IU);
      qNeighbors_4[IU] = Qdata(i - 1, j, k + 1    , IU);
      qNeighbors_5[IU] = Qdata(i - 1, j, k - 1    , IU);

      qLocNeighbor[IV] = Qdata(i - 1, j    , k    , IV);
      qNeighbors_0[IV] = Qdata(i    , j    , k    , IV);
      qNeighbors_1[IV] = Qdata(i - 2, j    , k    , IV);
      qNeighbors_2[IV] = Qdata(i - 1, j + 1, k    , IV);
      qNeighbors_3[IV] = Qdata(i - 1, j - 1, k    , IV);
      qNeighbors_4[IV] = Qdata(i - 1, j    , k + 1, IV);
      qNeighbors_5[IV] = Qdata(i - 1, j    , k - 1, IV);

      qLocNeighbor[IW] = Qdata(i - 1, j    , k    , IW);
      qNeighbors_0[IW] = Qdata(i    , j    , k    , IW);
      qNeighbors_1[IW] = Qdata(i - 2, j    , k    , IW);
      qNeighbors_2[IW] = Qdata(i - 1, j + 1, k    , IW);
      qNeighbors_3[IW] = Qdata(i - 1, j - 1, k    , IW);
      qNeighbors_4[IW] = Qdata(i - 1, j    , k + 1, IW);
      qNeighbors_5[IW] = Qdata(i - 1, j    , k - 1, IW);
      // clang-format on

      slope_unsplit_hydro_3d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX_neighbor,
                             dqY_neighbor,
                             dqZ_neighbor);

      //
      // compute reconstructed states at left interface along X
      //

      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_XMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dqZ_neighbor, dtdx, dtdy, dtdz, FACE_XMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i - 1, j, k, IX);
        qleft[IV] += 0.5 * dt * gravity(i - 1, j, k, IT);
        qleft[IW] += 0.5 * dt * gravity(i - 1, j, k, IZ);

        qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
        qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
      }

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann_hydro(qleft, qright, qgdnv, flux_x, params);

      //
      // Update with fluxes along X
      //
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, k, ID), flux_x[ID] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IP), flux_x[IP] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IU), flux_x[IU] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IV), flux_x[IV] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IW), flux_x[IW] * dtdx);
      }

      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i - 1, j, k, ID), flux_x[ID] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IP), flux_x[IP] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IU), flux_x[IU] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IV), flux_x[IV] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IW), flux_x[IW] * dtdx);
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      // clang-format off
      qLocNeighbor[ID] = Qdata(i    , j - 1, k    , ID);
      qNeighbors_0[ID] = Qdata(i + 1, j - 1, k    , ID);
      qNeighbors_1[ID] = Qdata(i - 1, j - 1, k    , ID);
      qNeighbors_2[ID] = Qdata(i    , j    , k    , ID);
      qNeighbors_3[ID] = Qdata(i    , j - 2, k    , ID);
      qNeighbors_4[ID] = Qdata(i    , j - 1, k + 1, ID);
      qNeighbors_5[ID] = Qdata(i    , j - 1, k - 1, ID);

      qLocNeighbor[IP] = Qdata(i    , j - 1, k    , IP);
      qNeighbors_0[IP] = Qdata(i + 1, j - 1, k    , IP);
      qNeighbors_1[IP] = Qdata(i - 1, j - 1, k    , IP);
      qNeighbors_2[IP] = Qdata(i    , j    , k    , IP);
      qNeighbors_3[IP] = Qdata(i    , j - 2, k    , IP);
      qNeighbors_4[IP] = Qdata(i    , j - 1, k + 1, IP);
      qNeighbors_5[IP] = Qdata(i    , j - 1, k - 1, IP);

      qLocNeighbor[IU] = Qdata(i    , j - 1, k    , IU);
      qNeighbors_0[IU] = Qdata(i + 1, j - 1, k    , IU);
      qNeighbors_1[IU] = Qdata(i - 1, j - 1, k    , IU);
      qNeighbors_2[IU] = Qdata(i    , j    , k    , IU);
      qNeighbors_3[IU] = Qdata(i    , j - 2, k    , IU);
      qNeighbors_4[IU] = Qdata(i    , j - 1, k + 1, IU);
      qNeighbors_5[IU] = Qdata(i    , j - 1, k - 1, IU);

      qLocNeighbor[IV] = Qdata(i    , j - 1, k    , IV);
      qNeighbors_0[IV] = Qdata(i + 1, j - 1, k    , IV);
      qNeighbors_1[IV] = Qdata(i - 1, j - 1, k    , IV);
      qNeighbors_2[IV] = Qdata(i    , j    , k    , IV);
      qNeighbors_3[IV] = Qdata(i    , j - 2, k    , IV);
      qNeighbors_4[IV] = Qdata(i    , j - 1, k + 1, IV);
      qNeighbors_5[IV] = Qdata(i    , j - 1, k - 1, IV);

      qLocNeighbor[IW] = Qdata(i    , j - 1, k    , IW);
      qNeighbors_0[IW] = Qdata(i + 1, j - 1, k    , IW);
      qNeighbors_1[IW] = Qdata(i - 1, j - 1, k    , IW);
      qNeighbors_2[IW] = Qdata(i    , j    , k    , IW);
      qNeighbors_3[IW] = Qdata(i    , j - 2, k    , IW);
      qNeighbors_4[IW] = Qdata(i    , j - 1, k + 1, IW);
      qNeighbors_5[IW] = Qdata(i    , j - 1, k - 1, IW);
      // clang-format on

      slope_unsplit_hydro_3d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX_neighbor,
                             dqY_neighbor,
                             dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //

      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dqZ_neighbor, dtdx, dtdy, dtdz, FACE_YMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i, j - 1, k, IX);
        qleft[IV] += 0.5 * dt * gravity(i, j - 1, k, IT);
        qleft[IW] += 0.5 * dt * gravity(i, j - 1, k, IZ);

        qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
        qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
      }

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qright[IU]), &(qright[IV]));
      riemann_hydro(qleft, qright, qgdnv, flux_y, params);
      swapValues(&(flux_y[IU]), &(flux_y[IV]));

      //
      // update with fluxes Y
      //
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, k, ID), flux_y[ID] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IP), flux_y[IP] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IU), flux_y[IU] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IV), flux_y[IV] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IW), flux_y[IW] * dtdy);
      }
      if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j - 1, k, ID), flux_y[ID] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IP), flux_y[IP] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IU), flux_y[IU] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IV), flux_y[IV] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IW), flux_y[IW] * dtdy);
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Z !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Z
      // clang-format off
      qLocNeighbor[ID] = Qdata(i    , j    , k - 1, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j    , k - 1, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j    , k - 1, ID);
      qNeighbors_2[ID] = Qdata(i    , j + 1, k - 1, ID);
      qNeighbors_3[ID] = Qdata(i    , j - 1, k - 1, ID);
      qNeighbors_4[ID] = Qdata(i    , j    , k    , ID);
      qNeighbors_5[ID] = Qdata(i    , j    , k - 2, ID);

      qLocNeighbor[IP] = Qdata(i    , j    , k - 1, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j    , k - 1, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j    , k - 1, IP);
      qNeighbors_2[IP] = Qdata(i    , j + 1, k - 1, IP);
      qNeighbors_3[IP] = Qdata(i    , j - 1, k - 1, IP);
      qNeighbors_4[IP] = Qdata(i    , j    , k    , IP);
      qNeighbors_5[IP] = Qdata(i    , j    , k - 2, IP);

      qLocNeighbor[IU] = Qdata(i    , j    , k - 1, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j    , k - 1, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j    , k - 1, IU);
      qNeighbors_2[IU] = Qdata(i    , j + 1, k - 1, IU);
      qNeighbors_3[IU] = Qdata(i    , j - 1, k - 1, IU);
      qNeighbors_4[IU] = Qdata(i    , j    , k    , IU);
      qNeighbors_5[IU] = Qdata(i    , j    , k - 2, IU);

      qLocNeighbor[IV] = Qdata(i    , j    , k - 1, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j    , k - 1, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j    , k - 1, IV);
      qNeighbors_2[IV] = Qdata(i    , j + 1, k - 1, IV);
      qNeighbors_3[IV] = Qdata(i    , j - 1, k - 1, IV);
      qNeighbors_4[IV] = Qdata(i    , j    , k    , IV);
      qNeighbors_5[IV] = Qdata(i    , j    , k - 2, IV);

      qLocNeighbor[IW] = Qdata(i    , j    , k - 1, IW);
      qNeighbors_0[IW] = Qdata(i + 1, j    , k - 1, IW);
      qNeighbors_1[IW] = Qdata(i - 1, j    , k - 1, IW);
      qNeighbors_2[IW] = Qdata(i    , j + 1, k - 1, IW);
      qNeighbors_3[IW] = Qdata(i    , j - 1, k - 1, IW);
      qNeighbors_4[IW] = Qdata(i    , j    , k    , IW);
      qNeighbors_5[IW] = Qdata(i    , j    , k - 2, IW);
      // clang-format on

      slope_unsplit_hydro_3d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             qNeighbors_4,
                             qNeighbors_5,
                             dqX_neighbor,
                             dqY_neighbor,
                             dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Z
      //

      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc, dqX, dqY, dqZ, dtdx, dtdy, dtdz, FACE_ZMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dqZ_neighbor, dtdx, dtdy, dtdz, FACE_ZMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i, j, k - 1, IX);
        qleft[IV] += 0.5 * dt * gravity(i, j, k - 1, IT);
        qleft[IW] += 0.5 * dt * gravity(i, j, k - 1, IZ);

        qright[IU] += 0.5 * dt * gravity(i, j, k, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, k, IT);
        qright[IW] += 0.5 * dt * gravity(i, j, k, IZ);
      }

      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      swapValues(&(qleft[IU]), &(qleft[IW]));
      swapValues(&(qright[IU]), &(qright[IW]));
      riemann_hydro(qleft, qright, qgdnv, flux_z, params);
      swapValues(&(flux_z[IU]), &(flux_z[IW]));

      //
      // update with fluxes Z
      //
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, k, ID), flux_z[ID] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IP), flux_z[IP] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IU), flux_z[IU] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IV), flux_z[IV] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IW), flux_z[IW] * dtdz);
      }
      if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j, k - 1, ID), flux_z[ID] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IP), flux_z[IP] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IU), flux_z[IU] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IV), flux_z[IV] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IW), flux_z[IW] * dtdz);
      }

    } // end if

  } // end operator ()

  DataArray3d   Qdata;
  DataArray3d   Udata;
  real_t        dt, dtdx, dtdy, dtdz;
  bool          gravity_enabled;
  VectorField3d gravity;

}; // ComputeAllFluxesAndUpdateFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class GravitySourceTermFunctor3D : public HydroBaseFunctor3D
{

public:
  /**
   * Update with gravity source term.
   *
   * \param[in] Udata_in conservative variables at t(n)
   * \param[in,out] Udata_out conservative variables at t(n+1)
   * \param[in] gravity is a vector field
   */
  GravitySourceTermFunctor3D(HydroParams   params,
                             DataArray3d   Udata_in,
                             DataArray3d   Udata_out,
                             VectorField3d gravity,
                             real_t        dt)
    : HydroBaseFunctor3D(params)
    , Udata_in(Udata_in)
    , Udata_out(Udata_out)
    , gravity(gravity)
    , dt(dt){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray3d   Udata_in,
        DataArray3d   Udata_out,
        VectorField3d gravity,
        real_t        dt)
  {
    GravitySourceTermFunctor3D functor(params, Udata_in, Udata_out, gravity, dt);
    Kokkos::parallel_for("GravitySourceTermFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    if (k >= ghostWidth and k < ksize - ghostWidth and j >= ghostWidth and
        j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      real_t rhoOld = Udata_in(i, j, k, ID);
      real_t rhoNew = Udata_out(i, j, k, ID);

      real_t rhou = Udata_out(i, j, k, IU);
      real_t rhov = Udata_out(i, j, k, IV);
      real_t rhow = Udata_out(i, j, k, IW);

      // compute kinetic energy before updating momentum
      real_t ekin_old = 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rhoNew;

      // update momentum
      rhou += 0.5 * dt * gravity(i, j, k, IX) * (rhoOld + rhoNew);
      rhov += 0.5 * dt * gravity(i, j, k, IT) * (rhoOld + rhoNew);
      rhow += 0.5 * dt * gravity(i, j, k, IZ) * (rhoOld + rhoNew);

      Udata_out(i, j, k, IU) = rhou;
      Udata_out(i, j, k, IV) = rhov;
      Udata_out(i, j, k, IW) = rhow;

      // compute kinetic energy after updating momentum
      real_t ekin_new = 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rhoNew;

      // update total energy
      Udata_out(i, j, k, IE) += (ekin_new - ekin_old);
    }

  } // end operator ()

  DataArray3d   Udata_in, Udata_out;
  VectorField3d gravity;
  real_t        dt;

}; // GravitySourceTermFunctor3D

} // namespace muscl

} // namespace euler_kokkos

#endif // HYDRO_RUN_FUNCTORS_3D_H_
