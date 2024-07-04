#ifndef HYDRO_RUN_FUNCTORS_2D_H_
#define HYDRO_RUN_FUNCTORS_2D_H_

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor2D.h"
#include "shared/RiemannSolvers.h"

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Compute time step satisfying CFL constraint.
   *
   * \param[in] params
   * \param[in] Udata
   */
  ComputeDtFunctor2D(HydroParams params, DataArray2d Udata)
    : HydroBaseFunctor2D(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, real_t & invDt)
  {
    ComputeDtFunctor2D  functor(params, Udata);
    Kokkos::Max<real_t> reducer(invDt);
    Kokkos::parallel_reduce(
      "ComputeDtFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor,
      reducer);
  }

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, real_t & invDt) const
  {
    const int    isize = params.isize;
    const int    jsize = params.jsize;
    const int    ghostWidth = params.ghostWidth;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    if (j >= ghostWidth and j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c = 0.0;
      real_t     vx, vy;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, ID);
      uLoc[IP] = Udata(i, j, IP);
      uLoc[IU] = Udata(i, j, IU);
      uLoc[IV] = Udata(i, j, IV);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      vx = c + fabs(qLoc[IU]);
      vy = c + fabs(qLoc[IV]);

      invDt = fmax(invDt, vx / dx + vy / dy);
    }

  } // operator ()

  DataArray2d Udata;

}; // ComputeDtFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * A specialized functor to compute CFL dt constraint when gravity source
 * term is activated.
 */
class ComputeDtGravityFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Compute time step satisfying CFL constraint.
   *
   * \param[in] params
   * \param[in] Udata
   */
  ComputeDtGravityFunctor2D(HydroParams   params,
                            real_t        cfl,
                            VectorField2d gravity,
                            DataArray2d   Udata)
    : HydroBaseFunctor2D(params)
    , cfl(cfl)
    , gravity(gravity)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, real_t cfl, VectorField2d gravity, DataArray2d Udata, real_t & invDt)
  {
    ComputeDtGravityFunctor2D functor(params, cfl, gravity, Udata);
    Kokkos::Max<real_t>       reducer(invDt);
    Kokkos::parallel_reduce(
      "ComputeDtGravityFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor,
      reducer);
  }

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, real_t & invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    // const int nbvar = params.nbvar;
    const real_t dx = fmin(params.dx, params.dy);

    if (j >= ghostWidth and j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c = 0.0;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, ID);
      uLoc[IP] = Udata(i, j, IP);
      uLoc[IU] = Udata(i, j, IU);
      uLoc[IV] = Udata(i, j, IV);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      real_t velocity = 0.0;
      velocity += c + fabs(qLoc[IU]);
      velocity += c + fabs(qLoc[IV]);

      /* Due to the gravitational acceleration, the CFL condition
       * can be written as
       * g dt^2 / (2 dx) + u dt / dx <= cfl
       * where u = sum(|v_i| + c_s) and g = sum(|g_i|)
       *
       * u / dx has to be corrected by a factor k / (sqrt(1 + 2k) - 1)
       * in order to satisfy the new CFL, where k = g dx cfl / u^2
       */
      double k = fabs(gravity(i, j, IX)) + fabs(gravity(i, j, IY));

      k *= cfl * dx / (velocity * velocity);

      /* prevent numerical errors due to very low gravity */
      k = fmax(k, 1e-4);

      velocity *= k / (sqrt(1.0 + 2.0 * k) - 1.0);

      invDt = fmax(invDt, velocity / dx);
    }

  } // operator ()

  real_t        cfl;
  VectorField2d gravity;
  DataArray2d   Udata;

}; // ComputeDtGravityFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Convert conservative variables to primitive ones using equation of state.
   *
   * \param[in] params
   * \param[in] Udata conservative variables
   * \param[out] Qdata primitive variables
   */
  ConvertToPrimitivesFunctor2D(HydroParams params, DataArray2d Udata, DataArray2d Qdata)
    : HydroBaseFunctor2D(params)
    , Udata(Udata)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, DataArray2d Qdata)
  {
    ConvertToPrimitivesFunctor2D functor(params, Udata, Qdata);
    Kokkos::parallel_for(
      "ConvertToPrimitivesFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    // const int ghostWidth = params.ghostWidth;

    if (j >= 0 and j < jsize and i >= 0 and i < isize)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, ID);
      uLoc[IP] = Udata(i, j, IP);
      uLoc[IU] = Udata(i, j, IU);
      uLoc[IV] = Udata(i, j, IV);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);

      // copy q state in q global
      Qdata(i, j, ID) = qLoc[ID];
      Qdata(i, j, IP) = qLoc[IP];
      Qdata(i, j, IU) = qLoc[IU];
      Qdata(i, j, IV) = qLoc[IV];
    }
  }

  DataArray2d Udata;
  DataArray2d Qdata;

}; // ConvertToPrimitivesFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndUpdateFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Perform time update by computing Riemann fluxes at cell interfaces.
   *
   * \param[in] params
   * \param[in,out] Udata conservative variables
   * \param[in] Qm_x primitive variables reconstructed on face -X
   * \param[in] Qm_y primitive variables reconstructed on face -Y
   * \param[in] Qp_x primitive variables reconstructed on face +X
   * \param[in] Qp_y primitive variables reconstructed on face +Y
   */
  ComputeFluxesAndUpdateFunctor2D(HydroParams params,
                                  DataArray2d Udata,
                                  DataArray2d Qm_x,
                                  DataArray2d Qm_y,
                                  DataArray2d Qp_x,
                                  DataArray2d Qp_y,
                                  real_t      dtdx,
                                  real_t      dtdy)
    : HydroBaseFunctor2D(params)
    , Udata(Udata)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
    {

      HydroState qleft, qright;
      HydroState flux_x, flux_y;
      HydroState qgdnv;

      //
      // Solve Riemann problem at X-interfaces and compute
      // X-fluxes
      //
      qleft[ID] = Qm_x(i - 1, j, ID);
      qleft[IP] = Qm_x(i - 1, j, IP);
      qleft[IU] = Qm_x(i - 1, j, IU);
      qleft[IV] = Qm_x(i - 1, j, IV);

      qright[ID] = Qp_x(i, j, ID);
      qright[IP] = Qp_x(i, j, IP);
      qright[IU] = Qp_x(i, j, IU);
      qright[IV] = Qp_x(i, j, IV);

      // compute hydro flux_x
      // riemann_hllc(qleft,qright,qgdnv,flux_x);
      riemann_hydro(qleft, qright, qgdnv, flux_x, params);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      qleft[ID] = Qm_y(i, j - 1, ID);
      qleft[IP] = Qm_y(i, j - 1, IP);
      qleft[IU] = Qm_y(i, j - 1, IV); // watchout IU, IV permutation
      qleft[IV] = Qm_y(i, j - 1, IU); // watchout IU, IV permutation

      qright[ID] = Qp_y(i, j, ID);
      qright[IP] = Qp_y(i, j, IP);
      qright[IU] = Qp_y(i, j, IV); // watchout IU, IV permutation
      qright[IV] = Qp_y(i, j, IU); // watchout IU, IV permutation

      // compute hydro flux_y
      // riemann_hllc(qleft,qright,qgdnv,flux_y);
      riemann_hydro(qleft, qright, qgdnv, flux_y, params);

      //
      // update hydro array
      //
      Udata(i - 1, j, ID) += -flux_x[ID] * dtdx;
      Udata(i - 1, j, IP) += -flux_x[IP] * dtdx;
      Udata(i - 1, j, IU) += -flux_x[IU] * dtdx;
      Udata(i - 1, j, IV) += -flux_x[IV] * dtdx;

      Udata(i, j, ID) += flux_x[ID] * dtdx;
      Udata(i, j, IP) += flux_x[IP] * dtdx;
      Udata(i, j, IU) += flux_x[IU] * dtdx;
      Udata(i, j, IV) += flux_x[IV] * dtdx;

      Udata(i, j - 1, ID) += -flux_y[ID] * dtdy;
      Udata(i, j - 1, IP) += -flux_y[IP] * dtdy;
      Udata(i, j - 1, IU) += -flux_y[IV] * dtdy; // watchout IU and IV swapped
      Udata(i, j - 1, IV) += -flux_y[IU] * dtdy; // watchout IU and IV swapped

      Udata(i, j, ID) += flux_y[ID] * dtdy;
      Udata(i, j, IP) += flux_y[IP] * dtdy;
      Udata(i, j, IU) += flux_y[IV] * dtdy; // watchout IU and IV swapped
      Udata(i, j, IV) += flux_y[IU] * dtdy; // watchout IU and IV swapped
    }
  }

  DataArray2d Udata;
  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y;
  real_t      dtdx, dtdy;

}; // ComputeFluxesAndUpdateFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Compute (slope extrapolated) reconstructed states at face centers for all faces.
   *
   * \param[in] params
   * \param[in] Qdata primitive variables
   * \param[out] Qm_x primitive variables reconstructed at center face -X
   * \param[out] Qm_y primitive variables reconstructed at center face -Y
   * \param[out] Qp_x primitive variables reconstructed at center face +X
   * \param[out] Qp_y primitive variables reconstructed at center face +Y
   */
  ComputeTraceFunctor2D(HydroParams params,
                        DataArray2d Qdata,
                        DataArray2d Qm_x,
                        DataArray2d Qm_y,
                        DataArray2d Qp_x,
                        DataArray2d Qp_y,
                        real_t      dtdx,
                        real_t      dtdy)
    : HydroBaseFunctor2D(params)
    , Qdata(Qdata)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= 1 and j <= jsize - ghostWidth and i >= 1 and i <= isize - ghostWidth)
    {

      HydroState qLoc; // local primitive variables
      HydroState qPlusX;
      HydroState qMinusX;
      HydroState qPlusY;
      HydroState qMinusY;

      HydroState dqX;
      HydroState dqY;

      HydroState qmX;
      HydroState qmY;
      HydroState qpX;
      HydroState qpY;

      // get primitive variables state vector
      {
        qLoc[ID] = Qdata(i, j, ID);
        qPlusX[ID] = Qdata(i + 1, j, ID);
        qMinusX[ID] = Qdata(i - 1, j, ID);
        qPlusY[ID] = Qdata(i, j + 1, ID);
        qMinusY[ID] = Qdata(i, j - 1, ID);

        qLoc[IP] = Qdata(i, j, IP);
        qPlusX[IP] = Qdata(i + 1, j, IP);
        qMinusX[IP] = Qdata(i - 1, j, IP);
        qPlusY[IP] = Qdata(i, j + 1, IP);
        qMinusY[IP] = Qdata(i, j - 1, IP);

        qLoc[IU] = Qdata(i, j, IU);
        qPlusX[IU] = Qdata(i + 1, j, IU);
        qMinusX[IU] = Qdata(i - 1, j, IU);
        qPlusY[IU] = Qdata(i, j + 1, IU);
        qMinusY[IU] = Qdata(i, j - 1, IU);

        qLoc[IV] = Qdata(i, j, IV);
        qPlusX[IV] = Qdata(i + 1, j, IV);
        qMinusX[IV] = Qdata(i - 1, j, IV);
        qPlusY[IV] = Qdata(i, j + 1, IV);
        qMinusY[IV] = Qdata(i, j - 1, IV);

      } //

      // get hydro slopes dq
      slope_unsplit_hydro_2d(qLoc, qPlusX, qMinusX, qPlusY, qMinusY, dqX, dqY);

      // compute qm, qp
      trace_unsplit_hydro_2d(qLoc, dqX, dqY, dtdx, dtdy, qmX, qmY, qpX, qpY);

      // store qm, qp : only what is really needed
      Qm_x(i, j, ID) = qmX[ID];
      Qp_x(i, j, ID) = qpX[ID];
      Qm_y(i, j, ID) = qmY[ID];
      Qp_y(i, j, ID) = qpY[ID];

      Qm_x(i, j, IP) = qmX[IP];
      Qp_x(i, j, IP) = qpX[IP];
      Qm_y(i, j, IP) = qmY[IP];
      Qp_y(i, j, IP) = qpY[IP];

      Qm_x(i, j, IU) = qmX[IU];
      Qp_x(i, j, IU) = qpX[IU];
      Qm_y(i, j, IU) = qmY[IU];
      Qp_y(i, j, IU) = qpY[IU];

      Qm_x(i, j, IV) = qmX[IV];
      Qp_x(i, j, IV) = qpX[IV];
      Qm_y(i, j, IV) = qmY[IV];
      Qp_y(i, j, IV) = qpY[IV];
    }
  }

  DataArray2d Qdata;
  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y;
  real_t      dtdx, dtdy;

}; // ComputeTraceFunctor2D


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor2D : public HydroBaseFunctor2D
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
   * \param[in] gravity_enabled boolean value to activate static gravity
   * \param[in] gravity is a vector field
   */
  ComputeAndStoreFluxesFunctor2D(HydroParams   params,
                                 DataArray2d   Qdata,
                                 DataArray2d   FluxData_x,
                                 DataArray2d   FluxData_y,
                                 real_t        dt,
                                 bool          gravity_enabled,
                                 VectorField2d gravity)
    : HydroBaseFunctor2D(params)
    , Qdata(Qdata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , dt(dt)
    , dtdx(dt / params.dx)
    , dtdy(dt / params.dy)
    , gravity_enabled(gravity_enabled)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray2d   Qdata,
        DataArray2d   FluxData_x,
        DataArray2d   FluxData_y,
        real_t        dt,
        bool          gravity_enabled,
        VectorField2d gravity)
  {
    ComputeAndStoreFluxesFunctor2D functor(
      params, Qdata, FluxData_x, FluxData_y, dt, gravity_enabled, gravity);
    Kokkos::parallel_for(
      "ComputeAndStoreFluxesFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
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

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      // clang-format off
      qLoc[ID] = Qdata(i, j, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j    , ID);
      qNeighbors_1[ID] = Qdata(i - 1, j    , ID);
      qNeighbors_2[ID] = Qdata(i    , j + 1, ID);
      qNeighbors_3[ID] = Qdata(i    , j - 1, ID);

      qLoc[IP] = Qdata(i, j, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j    , IP);
      qNeighbors_1[IP] = Qdata(i - 1, j    , IP);
      qNeighbors_2[IP] = Qdata(i    , j + 1, IP);
      qNeighbors_3[IP] = Qdata(i    , j - 1, IP);

      qLoc[IU] = Qdata(i, j, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j    , IU);
      qNeighbors_1[IU] = Qdata(i - 1, j    , IU);
      qNeighbors_2[IU] = Qdata(i    , j + 1, IU);
      qNeighbors_3[IU] = Qdata(i    , j - 1, IU);

      qLoc[IV] = Qdata(i, j, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j    , IV);
      qNeighbors_1[IV] = Qdata(i - 1, j    , IV);
      qNeighbors_2[IV] = Qdata(i    , j + 1, IV);
      qNeighbors_3[IV] = Qdata(i    , j - 1, IV);
      // clang-format on

      slope_unsplit_hydro_2d(
        qLoc, qNeighbors_0, qNeighbors_1, qNeighbors_2, qNeighbors_3, dqX, dqY);

      // slopes at left neighbor along X
      // clang-format off
      qLocNeighbor[ID] = Qdata(i - 1, j    , ID);
      qNeighbors_0[ID] = Qdata(i    , j    , ID);
      qNeighbors_1[ID] = Qdata(i - 2, j    , ID);
      qNeighbors_2[ID] = Qdata(i - 1, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i - 1, j - 1, ID);

      qLocNeighbor[IP] = Qdata(i - 1, j    , IP);
      qNeighbors_0[IP] = Qdata(i    , j    , IP);
      qNeighbors_1[IP] = Qdata(i - 2, j    , IP);
      qNeighbors_2[IP] = Qdata(i - 1, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i - 1, j - 1, IP);

      qLocNeighbor[IU] = Qdata(i - 1, j    , IU);
      qNeighbors_0[IU] = Qdata(i    , j    , IU);
      qNeighbors_1[IU] = Qdata(i - 2, j    , IU);
      qNeighbors_2[IU] = Qdata(i - 1, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i - 1, j - 1, IU);

      qLocNeighbor[IV] = Qdata(i - 1, j    , IV);
      qNeighbors_0[IV] = Qdata(i    , j    , IV);
      qNeighbors_1[IV] = Qdata(i - 2, j    , IV);
      qNeighbors_2[IV] = Qdata(i - 1, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i - 1, j - 1, IV);
      // clang-format on

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along X
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_XMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i - 1, j, IX);
        qleft[IV] += 0.5 * dt * gravity(i - 1, j, IY);

        qright[IU] += 0.5 * dt * gravity(i, j, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, IY);
      }

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      // riemann_2d(qleft,qright,qgdnv,flux_x);
      riemann_hydro(qleft, qright, qgdnv, flux_x, params);

      //
      // store fluxes X
      //
      FluxData_x(i, j, ID) = flux_x[ID] * dtdx;
      FluxData_x(i, j, IP) = flux_x[IP] * dtdx;
      FluxData_x(i, j, IU) = flux_x[IU] * dtdx;
      FluxData_x(i, j, IV) = flux_x[IV] * dtdx;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      // clang-format off
      qLocNeighbor[ID] = Qdata(i    , j - 1, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j - 1, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j - 1, ID);
      qNeighbors_2[ID] = Qdata(i    , j    , ID);
      qNeighbors_3[ID] = Qdata(i    , j - 2, ID);

      qLocNeighbor[IP] = Qdata(i    , j - 1, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j - 1, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j - 1, IP);
      qNeighbors_2[IP] = Qdata(i    , j    , IP);
      qNeighbors_3[IP] = Qdata(i    , j - 2, IP);

      qLocNeighbor[IU] = Qdata(i    , j - 1, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j - 1, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j - 1, IU);
      qNeighbors_2[IU] = Qdata(i    , j    , IU);
      qNeighbors_3[IU] = Qdata(i    , j - 2, IU);

      qLocNeighbor[IV] = Qdata(i    , j - 1, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j - 1, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j - 1, IV);
      qNeighbors_2[IV] = Qdata(i    , j    , IV);
      qNeighbors_3[IV] = Qdata(i    , j - 2, IV);
      // clang-format on

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_YMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i, j - 1, IX);
        qleft[IV] += 0.5 * dt * gravity(i, j - 1, IY);

        qright[IU] += 0.5 * dt * gravity(i, j, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, IY);
      }

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qright[IU]), &(qright[IV]));
      // riemann_2d(qleft,qright,qgdnv,flux_y);
      riemann_hydro(qleft, qright, qgdnv, flux_y, params);

      //
      // store fluxes Y
      //
      FluxData_y(i, j, ID) = flux_y[ID] * dtdy;
      FluxData_y(i, j, IP) = flux_y[IP] * dtdy;
      FluxData_y(i, j, IU) = flux_y[IV] * dtdy; //
      FluxData_y(i, j, IV) = flux_y[IU] * dtdy; //

    } // end if

  } // end operator ()

  DataArray2d   Qdata;
  DataArray2d   FluxData_x;
  DataArray2d   FluxData_y;
  real_t        dt, dtdx, dtdy;
  bool          gravity_enabled;
  VectorField2d gravity;


}; // ComputeAndStoreFluxesFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor2D : public HydroBaseFunctor2D
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
   */
  UpdateFunctor2D(HydroParams params,
                  DataArray2d Udata,
                  DataArray2d FluxData_x,
                  DataArray2d FluxData_y)
    : HydroBaseFunctor2D(params)
    , Udata(Udata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, DataArray2d FluxData_x, DataArray2d FluxData_y)
  {
    UpdateFunctor2D functor(params, Udata, FluxData_x, FluxData_y);
    Kokkos::parallel_for(
      "UpdateFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      Udata(i, j, ID) += FluxData_x(i, j, ID);
      Udata(i, j, IP) += FluxData_x(i, j, IP);
      Udata(i, j, IU) += FluxData_x(i, j, IU);
      Udata(i, j, IV) += FluxData_x(i, j, IV);

      Udata(i, j, ID) -= FluxData_x(i + 1, j, ID);
      Udata(i, j, IP) -= FluxData_x(i + 1, j, IP);
      Udata(i, j, IU) -= FluxData_x(i + 1, j, IU);
      Udata(i, j, IV) -= FluxData_x(i + 1, j, IV);

      Udata(i, j, ID) += FluxData_y(i, j, ID);
      Udata(i, j, IP) += FluxData_y(i, j, IP);
      Udata(i, j, IU) += FluxData_y(i, j, IU);
      Udata(i, j, IV) += FluxData_y(i, j, IV);

      Udata(i, j, ID) -= FluxData_y(i, j + 1, ID);
      Udata(i, j, IP) -= FluxData_y(i, j + 1, IP);
      Udata(i, j, IU) -= FluxData_y(i, j + 1, IU);
      Udata(i, j, IV) -= FluxData_y(i, j + 1, IV);

    } // end if

  } // end operator ()

  DataArray2d Udata;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;

}; // UpdateFunctor2D


/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class UpdateDirFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Perform time update using the stored fluxes along direction dir.
   *
   * \param[in,out] Udata
   * \param[in] FluxData flux coming from the left neighbor along direction dir
   *
   */
  UpdateDirFunctor2D(HydroParams params, DataArray2d Udata, DataArray2d FluxData)
    : HydroBaseFunctor2D(params)
    , Udata(Udata)
    , FluxData(FluxData){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, DataArray2d FluxData)
  {
    UpdateDirFunctor2D<dir> functor(params, Udata, FluxData);
    Kokkos::parallel_for(
      "UpdateDirFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      if (dir == XDIR)
      {

        Udata(i, j, ID) += FluxData(i, j, ID);
        Udata(i, j, IP) += FluxData(i, j, IP);
        Udata(i, j, IU) += FluxData(i, j, IU);
        Udata(i, j, IV) += FluxData(i, j, IV);

        Udata(i, j, ID) -= FluxData(i + 1, j, ID);
        Udata(i, j, IP) -= FluxData(i + 1, j, IP);
        Udata(i, j, IU) -= FluxData(i + 1, j, IU);
        Udata(i, j, IV) -= FluxData(i + 1, j, IV);
      }
      else if (dir == YDIR)
      {

        Udata(i, j, ID) += FluxData(i, j, ID);
        Udata(i, j, IP) += FluxData(i, j, IP);
        Udata(i, j, IU) += FluxData(i, j, IU);
        Udata(i, j, IV) += FluxData(i, j, IV);

        Udata(i, j, ID) -= FluxData(i, j + 1, ID);
        Udata(i, j, IP) -= FluxData(i, j + 1, IP);
        Udata(i, j, IU) -= FluxData(i, j + 1, IU);
        Udata(i, j, IV) -= FluxData(i, j + 1, IV);
      }

    } // end if

  } // end operator ()

  DataArray2d Udata;
  DataArray2d FluxData;

}; // UpdateDirFunctor


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Compute limited slopes.
   *
   * \param[in] Qdata primitive variables
   * \param[out] Slopes_x limited slopes along direction X
   * \param[out] Slopes_y limited slopes along direction Y
   */
  ComputeSlopesFunctor2D(HydroParams params,
                         DataArray2d Qdata,
                         DataArray2d Slopes_x,
                         DataArray2d Slopes_y)
    : HydroBaseFunctor2D(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Qdata, DataArray2d Slopes_x, DataArray2d Slopes_y)
  {
    ComputeSlopesFunctor2D functor(params, Qdata, Slopes_x, Slopes_y);
    Kokkos::parallel_for(
      "ComputeSlopesFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth - 1 and j <= jsize - ghostWidth and i >= ghostWidth - 1 and
        i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;

      // Local slopes and neighbor slopes
      HydroState dqX{};
      HydroState dqY{};

      // get primitive variables state vector
      qLoc[ID] = Qdata(i, j, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, ID);

      qLoc[IP] = Qdata(i, j, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, IP);

      qLoc[IU] = Qdata(i, j, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, IU);

      qLoc[IV] = Qdata(i, j, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, IV);

      slope_unsplit_hydro_2d(
        qLoc, qNeighbors_0, qNeighbors_1, qNeighbors_2, qNeighbors_3, dqX, dqY);

      // copy back slopes in global arrays
      Slopes_x(i, j, ID) = dqX[ID];
      Slopes_y(i, j, ID) = dqY[ID];

      Slopes_x(i, j, IP) = dqX[IP];
      Slopes_y(i, j, IP) = dqY[IP];

      Slopes_x(i, j, IU) = dqX[IU];
      Slopes_y(i, j, IU) = dqY[IU];

      Slopes_x(i, j, IV) = dqX[IV];
      Slopes_y(i, j, IV) = dqY[IV];

    } // end if

  } // end operator ()

  DataArray2d Qdata;
  DataArray2d Slopes_x, Slopes_y;

}; // ComputeSlopesFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Compute reconstructed states on faces (not stored), and fluxes (stored).
   *
   * \param[in] Qdata primitive variables
   * \param[in] Slopes_x limited slopes along direction X
   * \param[in] Slopes_y limited slopes along direction Y
   * \param[out] Fluxes along direction dir
   *
   * \tparam dir direction along which fluxes are computed.
   */
  ComputeTraceAndFluxes_Functor2D(HydroParams   params,
                                  DataArray2d   Qdata,
                                  DataArray2d   Slopes_x,
                                  DataArray2d   Slopes_y,
                                  DataArray2d   Fluxes,
                                  real_t        dt,
                                  bool          gravity_enabled,
                                  VectorField2d gravity)
    : HydroBaseFunctor2D(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Fluxes(Fluxes)
    , dt(dt)
    , dtdx(dt / params.dx)
    , dtdy(dt / params.dy)
    , gravity_enabled(gravity_enabled)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray2d   Qdata,
        DataArray2d   Slopes_x,
        DataArray2d   Slopes_y,
        DataArray2d   Fluxes,
        real_t        dt,
        bool          gravity_enabled,
        VectorField2d gravity)
  {
    ComputeTraceAndFluxes_Functor2D<dir> functor(
      params, Qdata, Slopes_x, Slopes_y, Fluxes, dt, gravity_enabled, gravity);
    Kokkos::parallel_for(
      "ComputeTraceAndFluxes_Functor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux;

      //
      // compute reconstructed states at left interface along X
      //
      qLoc[ID] = Qdata(i, j, ID);
      dqX[ID] = Slopes_x(i, j, ID);
      dqY[ID] = Slopes_y(i, j, ID);

      qLoc[IP] = Qdata(i, j, IP);
      dqX[IP] = Slopes_x(i, j, IP);
      dqY[IP] = Slopes_y(i, j, IP);

      qLoc[IU] = Qdata(i, j, IU);
      dqX[IU] = Slopes_x(i, j, IU);
      dqY[IU] = Slopes_y(i, j, IU);

      qLoc[IV] = Qdata(i, j, IV);
      dqX[IV] = Slopes_x(i, j, IV);
      dqY[IV] = Slopes_y(i, j, IV);

      if (dir == XDIR)
      {

        // left interface : right state
        trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qright[IU] += 0.5 * dt * gravity(i, j, IX);
          qright[IV] += 0.5 * dt * gravity(i, j, IY);
        }

        qLocNeighbor[ID] = Qdata(i - 1, j, ID);
        dqX_neighbor[ID] = Slopes_x(i - 1, j, ID);
        dqY_neighbor[ID] = Slopes_y(i - 1, j, ID);

        qLocNeighbor[IP] = Qdata(i - 1, j, IP);
        dqX_neighbor[IP] = Slopes_x(i - 1, j, IP);
        dqY_neighbor[IP] = Slopes_y(i - 1, j, IP);

        qLocNeighbor[IU] = Qdata(i - 1, j, IU);
        dqX_neighbor[IU] = Slopes_x(i - 1, j, IU);
        dqY_neighbor[IU] = Slopes_y(i - 1, j, IU);

        qLocNeighbor[IV] = Qdata(i - 1, j, IV);
        dqX_neighbor[IV] = Slopes_x(i - 1, j, IV);
        dqY_neighbor[IV] = Slopes_y(i - 1, j, IV);

        // left interface : left state
        trace_unsplit_2d_along_dir(
          qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_XMAX, qleft);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qleft[IU] += 0.5 * dt * gravity(i - 1, j, IX);
          qleft[IV] += 0.5 * dt * gravity(i - 1, j, IY);
        }

        // Solve Riemann problem at X-interfaces and compute X-fluxes
        riemann_hydro(qleft, qright, qgdnv, flux, params);

        //
        // store fluxes
        //
        Fluxes(i, j, ID) = flux[ID] * dtdx;
        Fluxes(i, j, IP) = flux[IP] * dtdx;
        Fluxes(i, j, IU) = flux[IU] * dtdx;
        Fluxes(i, j, IV) = flux[IV] * dtdx;
      }
      else if (dir == YDIR)
      {

        // left interface : right state
        trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qright[IU] += 0.5 * dt * gravity(i, j, IX);
          qright[IV] += 0.5 * dt * gravity(i, j, IY);
        }

        qLocNeighbor[ID] = Qdata(i, j - 1, ID);
        dqX_neighbor[ID] = Slopes_x(i, j - 1, ID);
        dqY_neighbor[ID] = Slopes_y(i, j - 1, ID);

        qLocNeighbor[IP] = Qdata(i, j - 1, IP);
        dqX_neighbor[IP] = Slopes_x(i, j - 1, IP);
        dqY_neighbor[IP] = Slopes_y(i, j - 1, IP);

        qLocNeighbor[IU] = Qdata(i, j - 1, IU);
        dqX_neighbor[IU] = Slopes_x(i, j - 1, IU);
        dqY_neighbor[IU] = Slopes_y(i, j - 1, IU);

        qLocNeighbor[IV] = Qdata(i, j - 1, IV);
        dqX_neighbor[IV] = Slopes_x(i, j - 1, IV);
        dqY_neighbor[IV] = Slopes_y(i, j - 1, IV);

        // left interface : left state
        trace_unsplit_2d_along_dir(
          qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_YMAX, qleft);

        if (gravity_enabled)
        {
          // we need to modify input to flux computation with
          // gravity predictor (half time step)

          qleft[IU] += 0.5 * dt * gravity(i, j - 1, IX);
          qleft[IV] += 0.5 * dt * gravity(i, j - 1, IY);
        }

        // Solve Riemann problem at Y-interfaces and compute Y-fluxes
        swapValues(&(qleft[IU]), &(qleft[IV]));
        swapValues(&(qright[IU]), &(qright[IV]));
        riemann_hydro(qleft, qright, qgdnv, flux, params);

        //
        // update hydro array
        //
        Fluxes(i, j, ID) = flux[ID] * dtdy;
        Fluxes(i, j, IP) = flux[IP] * dtdy;
        Fluxes(i, j, IU) = flux[IV] * dtdy; // IU/IV swapped
        Fluxes(i, j, IV) = flux[IU] * dtdy; // IU/IV swapped
      }

    } // end if

  } // end operator ()

  DataArray2d   Qdata;
  DataArray2d   Slopes_x, Slopes_y;
  DataArray2d   Fluxes;
  real_t        dt, dtdx, dtdy;
  bool          gravity_enabled;
  VectorField2d gravity;

}; // ComputeTraceAndFluxes_Functor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAllFluxesAndUpdateFunctor2D : public HydroBaseFunctor2D
{

public:
  ComputeAllFluxesAndUpdateFunctor2D(HydroParams   params,
                                     DataArray2d   Qdata,
                                     DataArray2d   Udata,
                                     real_t        dt,
                                     bool          gravity_enabled,
                                     VectorField2d gravity)
    : HydroBaseFunctor2D(params)
    , Qdata(Qdata)
    , Udata(Udata)
    , dt(dt)
    , dtdx(dt / params.dx)
    , dtdy(dt / params.dy)
    , gravity_enabled(gravity_enabled)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray2d   Qdata,
        DataArray2d   Udata,
        real_t        dt,
        bool          gravity_enabled,
        VectorField2d gravity)
  {
    ComputeAllFluxesAndUpdateFunctor2D functor(params, Qdata, Udata, dt, gravity_enabled, gravity);
    Kokkos::parallel_for(
      "ComputeAllFluxesAndUpdateFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j <= jsize - ghostWidth and i >= ghostWidth and i <= isize - ghostWidth)
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

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      // clang-format off
      qLoc[ID] = Qdata(i, j, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j    , ID);
      qNeighbors_1[ID] = Qdata(i - 1, j    , ID);
      qNeighbors_2[ID] = Qdata(i    , j + 1, ID);
      qNeighbors_3[ID] = Qdata(i    , j - 1, ID);

      qLoc[IP] = Qdata(i, j, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j    , IP);
      qNeighbors_1[IP] = Qdata(i - 1, j    , IP);
      qNeighbors_2[IP] = Qdata(i    , j + 1, IP);
      qNeighbors_3[IP] = Qdata(i    , j - 1, IP);

      qLoc[IU] = Qdata(i, j, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j    , IU);
      qNeighbors_1[IU] = Qdata(i - 1, j    , IU);
      qNeighbors_2[IU] = Qdata(i    , j + 1, IU);
      qNeighbors_3[IU] = Qdata(i    , j - 1, IU);

      qLoc[IV] = Qdata(i, j, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j    , IV);
      qNeighbors_1[IV] = Qdata(i - 1, j    , IV);
      qNeighbors_2[IV] = Qdata(i    , j + 1, IV);
      qNeighbors_3[IV] = Qdata(i    , j - 1, IV);
      // clang-format on

      slope_unsplit_hydro_2d(
        qLoc, qNeighbors_0, qNeighbors_1, qNeighbors_2, qNeighbors_3, dqX, dqY);

      // slopes at left neighbor along X
      // clang-format off
      qLocNeighbor[ID] = Qdata(i - 1, j    , ID);
      qNeighbors_0[ID] = Qdata(i    , j    , ID);
      qNeighbors_1[ID] = Qdata(i - 2, j    , ID);
      qNeighbors_2[ID] = Qdata(i - 1, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i - 1, j - 1, ID);

      qLocNeighbor[IP] = Qdata(i - 1, j    , IP);
      qNeighbors_0[IP] = Qdata(i    , j    , IP);
      qNeighbors_1[IP] = Qdata(i - 2, j    , IP);
      qNeighbors_2[IP] = Qdata(i - 1, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i - 1, j - 1, IP);

      qLocNeighbor[IU] = Qdata(i - 1, j    , IU);
      qNeighbors_0[IU] = Qdata(i    , j    , IU);
      qNeighbors_1[IU] = Qdata(i - 2, j    , IU);
      qNeighbors_2[IU] = Qdata(i - 1, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i - 1, j - 1, IU);

      qLocNeighbor[IV] = Qdata(i - 1, j    , IV);
      qNeighbors_0[IV] = Qdata(i    , j    , IV);
      qNeighbors_1[IV] = Qdata(i - 2, j    , IV);
      qNeighbors_2[IV] = Qdata(i - 1, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i - 1, j - 1, IV);
      // clang-format on

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along X
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_XMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i - 1, j, IX);
        qleft[IV] += 0.5 * dt * gravity(i - 1, j, IY);

        qright[IU] += 0.5 * dt * gravity(i, j, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, IY);
      }

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann_hydro(qleft, qright, qgdnv, flux_x, params);

      //
      // Update with fluxes along X
      //
      if (j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, ID), flux_x[ID] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IP), flux_x[IP] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IU), flux_x[IU] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IV), flux_x[IV] * dtdx);
      }

      if (j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i - 1, j, ID), flux_x[ID] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IP), flux_x[IP] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IU), flux_x[IU] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IV), flux_x[IV] * dtdx);
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      // clang-format off
      qLocNeighbor[ID] = Qdata(i    , j - 1, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j - 1, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j - 1, ID);
      qNeighbors_2[ID] = Qdata(i    , j    , ID);
      qNeighbors_3[ID] = Qdata(i    , j - 2, ID);

      qLocNeighbor[IP] = Qdata(i    , j - 1, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j - 1, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j - 1, IP);
      qNeighbors_2[IP] = Qdata(i    , j    , IP);
      qNeighbors_3[IP] = Qdata(i    , j - 2, IP);

      qLocNeighbor[IU] = Qdata(i    , j - 1, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j - 1, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j - 1, IU);
      qNeighbors_2[IU] = Qdata(i    , j    , IU);
      qNeighbors_3[IU] = Qdata(i    , j - 2, IU);

      qLocNeighbor[IV] = Qdata(i    , j - 1, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j - 1, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j - 1, IV);
      qNeighbors_2[IV] = Qdata(i    , j    , IV);
      qNeighbors_3[IV] = Qdata(i    , j - 2, IV);
      // clang-format off

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_YMAX, qleft);

      if (gravity_enabled)
      {
        // we need to modify input to flux computation with
        // gravity predictor (half time step)

        qleft[IU] += 0.5 * dt * gravity(i, j - 1, IX);
        qleft[IV] += 0.5 * dt * gravity(i, j - 1, IY);

        qright[IU] += 0.5 * dt * gravity(i, j, IX);
        qright[IV] += 0.5 * dt * gravity(i, j, IY);
      }

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qright[IU]), &(qright[IV]));
      riemann_hydro(qleft, qright, qgdnv, flux_y, params);
      swapValues(&(flux_y[IU]), &(flux_y[IV]));

      //
      // store fluxes Y
      //
      if (j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, ID), flux_y[ID] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IP), flux_y[IP] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IU), flux_y[IU] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IV), flux_y[IV] * dtdy);
      }
      if (j > ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j - 1, ID), flux_y[ID] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IP), flux_y[IP] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IU), flux_y[IU] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IV), flux_y[IV] * dtdy);
      }

    } // end if

  } // end operator ()

  DataArray2d Qdata;
  DataArray2d Udata;
  real_t        dt, dtdx, dtdy;
  bool          gravity_enabled;
  VectorField2d gravity;

}; // ComputeAllFluxesAndUpdateFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class GravitySourceTermFunctor2D : public HydroBaseFunctor2D
{

public:
  /**
   * Update with gravity source term.
   *
   * \param[in] Udata_in conservative variables at t(n)
   * \param[in,out] Udata_out conservative variables at t(n+1)
   * \param[in] gravity is a vector field
   */
  GravitySourceTermFunctor2D(HydroParams   params,
                             DataArray2d   Udata_in,
                             DataArray2d   Udata_out,
                             VectorField2d gravity,
                             real_t        dt)
    : HydroBaseFunctor2D(params)
    , Udata_in(Udata_in)
    , Udata_out(Udata_out)
    , gravity(gravity)
    , dt(dt){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams   params,
        DataArray2d   Udata_in,
        DataArray2d   Udata_out,
        VectorField2d gravity,
        real_t        dt)
  {
    GravitySourceTermFunctor2D functor(params, Udata_in, Udata_out, gravity, dt);
    Kokkos::parallel_for(
      "GravitySourceTermFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth and j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      real_t rhoOld = Udata_in(i, j, ID);
      real_t rhoNew = fmax(params.settings.smallr, Udata_out(i, j, ID));

      real_t rhou = Udata_out(i, j, IU);
      real_t rhov = Udata_out(i, j, IV);

      // compute kinetic energy before updating momentum
      real_t ekin_old = 0.5 * (rhou * rhou + rhov * rhov) / rhoNew;

      // update momentum
      rhou += 0.5 * dt * gravity(i, j, IX) * (rhoOld + rhoNew);
      rhov += 0.5 * dt * gravity(i, j, IY) * (rhoOld + rhoNew);
      Udata_out(i, j, IU) = rhou;
      Udata_out(i, j, IV) = rhov;

      // compute kinetic energy after updating momentum
      real_t ekin_new = 0.5 * (rhou * rhou + rhov * rhov) / rhoNew;

      // update total energy
      Udata_out(i, j, IE) += (ekin_new - ekin_old);
    }

  } // end operator ()

  DataArray2d   Udata_in, Udata_out;
  VectorField2d gravity;
  real_t        dt;

}; // GravitySourceTermFunctor2D

} // namespace muscl

} // namespace euler_kokkos

#endif // HYDRO_RUN_FUNCTORS_2D_H_
