#ifndef MHD_RUN_FUNCTORS_2D_H_
#define MHD_RUN_FUNCTORS_2D_H_

#include "shared/kokkos_shared.h"
#include "MHDBaseFunctor2D.h"
#include "shared/RiemannSolvers_MHD.h"

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  ComputeDtFunctor2D_MHD(HydroParams params, DataArray2d Qdata)
    : MHDBaseFunctor2D(params)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Qdata, real_t & invDt)
  {
    ComputeDtFunctor2D_MHD functor(params, Qdata);
    Kokkos::Max<real_t>    reducer(invDt);
    Kokkos::parallel_reduce(
      "ComputeDtFunctor2D_MHD",
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

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth and
        i >= ghostWidth and i < isize - ghostWidth)
    // clang-format on
    {

      MHDState qLoc; // primitive    variables in current cell

      // get primitive variables in current cell
      qLoc[ID] = Qdata(i, j, ID);
      qLoc[IP] = Qdata(i, j, IP);
      qLoc[IU] = Qdata(i, j, IU);
      qLoc[IV] = Qdata(i, j, IV);
      qLoc[IW] = Qdata(i, j, IW);
      qLoc[IA] = Qdata(i, j, IA);
      qLoc[IB] = Qdata(i, j, IB);
      qLoc[IC] = Qdata(i, j, IC);

      // compute fastest information speeds
      real_t fastInfoSpeed[3];
      find_speed_info<TWO_D>(qLoc, fastInfoSpeed, params);

      real_t vx = fastInfoSpeed[IX];
      real_t vy = fastInfoSpeed[IY];

      invDt = fmax(invDt, vx / dx + vy / dy);
    }

  } // operator ()

  DataArray2d Qdata;

}; // ComputeDtFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  ConvertToPrimitivesFunctor2D_MHD(HydroParams params, DataArray2d Udata, DataArray2d Qdata)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, DataArray2d Qdata)
  {
    ConvertToPrimitivesFunctor2D_MHD functor(params, Udata, Qdata);
    Kokkos::parallel_for(
      "ConvertToPrimitivesFunctor2D_MHD",
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

    // magnetic field in neighbor cells
    real_t magFieldNeighbors[3];

    // clang-format off
    if (j >= 0 and j < jsize - 1 and
        i >= 0 and i < isize - 1)
    // clang-format on
    {

      MHDState uLoc; // conservative    variables in current cell
      MHDState qLoc; // primitive    variables in current cell
      real_t   c;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, ID);
      uLoc[IP] = Udata(i, j, IP);
      uLoc[IU] = Udata(i, j, IU);
      uLoc[IV] = Udata(i, j, IV);
      uLoc[IW] = Udata(i, j, IW);
      uLoc[IA] = Udata(i, j, IA);
      uLoc[IB] = Udata(i, j, IB);
      uLoc[IC] = Udata(i, j, IC);

      // get mag field in neighbor cells
      magFieldNeighbors[IX] = Udata(i + 1, j, IA);
      magFieldNeighbors[IY] = Udata(i, j + 1, IB);
      magFieldNeighbors[IZ] = 0.0;

      // get primitive variables in current cell
      constoprim_mhd(uLoc, magFieldNeighbors, c, qLoc);

      // copy q state in q global
      Qdata(i, j, ID) = qLoc[ID];
      Qdata(i, j, IP) = qLoc[IP];
      Qdata(i, j, IU) = qLoc[IU];
      Qdata(i, j, IV) = qLoc[IV];
      Qdata(i, j, IW) = qLoc[IW];
      Qdata(i, j, IA) = qLoc[IA];
      Qdata(i, j, IB) = qLoc[IB];
      Qdata(i, j, IC) = qLoc[IC];
    }
  }

  DataArray2d Udata;
  DataArray2d Qdata;

}; // ConvertToPrimitivesFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  /**
   * Compute limited slopes of primitives variables.
   *
   * Magnetic field slopes are computed for transverse component at cell center, and for normal
   * component at cell faces.
   *
   * \note Normal magnetic field component slopes are not limited.
   *
   * \param[in] Udata conservative variables
   * \param[in] Qdata primitive variables
   * \param[out] Slopes_x limited slopes along direction X
   * \param[out] Slopes_y limited slopes along direction Y
   */
  ComputeSlopesFunctor2D_MHD(HydroParams params,
                             DataArray2d Udata,
                             DataArray2d Qdata,
                             DataArray2d Slopes_x,
                             DataArray2d Slopes_y)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata,
        DataArray2d Qdata,
        DataArray2d Slopes_x,
        DataArray2d Slopes_y)
  {
    ComputeSlopesFunctor2D_MHD functor(params, Udata, Qdata, Slopes_x, Slopes_y);
    Kokkos::parallel_for(
      "ComputeSlopesFunctor2D_MHD",
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

    // clang-format off
    if (j >= 1 and j < jsize - 1 and
        i >= 1 and i < isize - 1)
    // clang-format on
    {
      MHDState qc, qm, qp;
      get_state(Qdata, i, j, qc);

      MHDState dq;

      // slopes along X
      get_state(Qdata, i - 1, j, qm);
      get_state(Qdata, i + 1, j, qp);
      slope_unsplit_hydro_2d(qc, qp, qm, dq);
      set_state(Slopes_x, i, j, dq);

      // modify slopes for normal magnetic field component using face centered value
      Slopes_x(i, j, IA) = Udata(i + 1, j, IA) - Udata(i, j, IA);

      // slopes along Y
      get_state(Qdata, i, j - 1, qm);
      get_state(Qdata, i, j + 1, qp);
      slope_unsplit_hydro_2d(qc, qp, qm, dq);
      set_state(Slopes_y, i, j, dq);

      // modify slopes for normal magnetic field component using face centered value
      Slopes_y(i, j, IB) = Udata(i, j + 1, IB) - Udata(i, j, IB);
    }
  } // end operator ()

  DataArray2d Udata, Qdata;
  DataArray2d Slopes_x, Slopes_y;

}; // ComputeSlopesFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeElecFieldFunctor2D : public MHDBaseFunctor2D
{

public:
  ComputeElecFieldFunctor2D(HydroParams params,
                            DataArray2d Udata,
                            DataArray2d Qdata,
                            DataArray2d ElecField)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , ElecField(ElecField){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, DataArray2d Qdata, DataArray2d ElecField)
  {
    ComputeElecFieldFunctor2D functor(params, Udata, Qdata, ElecField);
    Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;

    // clang-format off
    if (j > 0 and j < jsize  and
        i > 0 and i < isize)
    {

      // compute Ez

      // clang-format off
      const real_t u = ONE_FOURTH_F * (Qdata(i - 1, j - 1, IU) +
                                       Qdata(i - 1, j    , IU) +
                                       Qdata(i    , j - 1, IU) +
                                       Qdata(i    , j    , IU));

      const real_t v = ONE_FOURTH_F * (Qdata(i - 1, j - 1, IV) +
                                       Qdata(i - 1, j    , IV) +
                                       Qdata(i    , j - 1, IV) +
                                       Qdata(i    , j    , IV));

      const real_t A = HALF_F * (Udata(i    , j - 1, IA) + Udata(i, j, IA));
      const real_t B = HALF_F * (Udata(i - 1, j    , IB) + Udata(i, j, IB));
      // clang-format on

      ElecField(i, j, 0) = u * B - v * A;
    }
    // clang-format on

  } // operator ()

  DataArray2d Udata;
  DataArray2d Qdata;
  DataArray2d ElecField;

}; // ComputeElecFieldFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSourceFaceMagFunctor2D : public MHDBaseFunctor2D
{

public:
  ComputeSourceFaceMagFunctor2D(HydroParams params,
                                DataArray2d ElecField,
                                DataArray2d sFaceMag,
                                real_t      dtdx,
                                real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , ElecField(ElecField)
    , sFaceMag(sFaceMag)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d ElecField, DataArray2d sFaceMag, real_t dtdx, real_t dtdy)
  {
    ComputeSourceFaceMagFunctor2D functor(params, ElecField, sFaceMag, dtdx, dtdy);
    Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;

    // clang-format off
    if (j > 0 and j < jsize-1  and
        i > 0 and i < isize-1)
    {

      // sAL0 = +(ELR - ELL) * 0.5 * dtdy
      sFaceMag(i, j, IX) = (ElecField(i    , j + 1, 0) - ElecField(i, j, 0)) * 0.5 * dtdy;

      // sBL0 = -(ERL - ELL) * 0.5 * dtdx;
      sFaceMag(i, j, IY) = -(ElecField(i + 1, j    , 0) - ElecField(i, j, 0)) * 0.5 * dtdx;

    }
    // clang-format on

  } // operator ()

  DataArray2d ElecField;
  DataArray2d sFaceMag;
  real_t      dtdx, dtdy;

}; // ComputeSourceFaceMagFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndStoreFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  ComputeFluxesAndStoreFunctor2D_MHD(HydroParams params,
                                     DataArray2d Qm_x,
                                     DataArray2d Qm_y,
                                     DataArray2d Qp_x,
                                     DataArray2d Qp_y,
                                     DataArray2d Fluxes_x,
                                     DataArray2d Fluxes_y,
                                     real_t      dtdx,
                                     real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , Fluxes_x(Fluxes_x)
    , Fluxes_y(Fluxes_y)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Qm_x,
        DataArray2d Qm_y,
        DataArray2d Qp_x,
        DataArray2d Qp_y,
        DataArray2d Flux_x,
        DataArray2d Flux_y,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeFluxesAndStoreFunctor2D_MHD functor(
      params, Qm_x, Qm_y, Qp_x, Qp_y, Flux_x, Flux_y, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeFluxesAndStoreFunctor2D_MHD",
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

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    {

      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      get_state(Qm_x, i - 1, j, qleft);
      get_state(Qp_x, i    , j, qright);

      // compute hydro flux along X
      riemann_mhd(qleft, qright, flux, params);

      // store fluxes
      set_state(Fluxes_x, i, j, flux);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i    , j - 1, qleft);
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qleft[IA]), &(qleft[IB]));

      get_state(Qp_y, i    , j    , qright);
      swapValues(&(qright[IU]), &(qright[IV]));
      swapValues(&(qright[IA]), &(qright[IB]));

      // compute hydro flux along Y
      riemann_mhd(qleft, qright, flux, params);

      // store fluxes
      set_state(Fluxes_y, i, j, flux);
    }
    // clang-format on
  }

  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y;
  DataArray2d Fluxes_x, Fluxes_y;
  real_t      dtdx, dtdy;

}; // ComputeFluxesAndStoreFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndUpdateFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  ComputeFluxesAndUpdateFunctor2D_MHD(HydroParams params,
                                      DataArray2d Qm_x,
                                      DataArray2d Qm_y,
                                      DataArray2d Qp_x,
                                      DataArray2d Qp_y,
                                      DataArray2d Udata,
                                      real_t      dtdx,
                                      real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , Udata(Udata)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Qm_x,
        DataArray2d Qm_y,
        DataArray2d Qp_x,
        DataArray2d Qp_y,
        DataArray2d Udata,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeFluxesAndUpdateFunctor2D_MHD functor(params, Qm_x, Qm_y, Qp_x, Qp_y, Udata, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeFluxesAndUpdateFunctor2D_MHD",
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

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {

      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      get_state(Qm_x, i - 1, j, qleft);
      get_state(Qp_x, i, j, qright);

      // compute hydro flux along X
      riemann_mhd(qleft, qright, flux, params);

      //
      // Update with fluxes along X
      //
      if (j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, ID), flux[ID] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IP), flux[IP] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IU), flux[IU] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IV), flux[IV] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IW), flux[IW] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IC), flux[IC] * dtdx);
      }

      if (j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i - 1, j, ID), flux[ID] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IP), flux[IP] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IU), flux[IU] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IV), flux[IV] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IW), flux[IW] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IC), flux[IC] * dtdx);
      }

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i, j - 1, qleft);
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qleft[IA]), &(qleft[IB]));

      get_state(Qp_y, i, j, qright);
      swapValues(&(qright[IU]), &(qright[IV]));
      swapValues(&(qright[IA]), &(qright[IB]));

      // compute hydro flux along Y
      riemann_mhd(qleft, qright, flux, params);

      swapValues(&(flux[IU]), &(flux[IV]));
      swapValues(&(flux[IA]), &(flux[IB]));

      //
      // update with fluxes Y
      //
      if (j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, ID), flux[ID] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IP), flux[IP] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IU), flux[IU] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IV), flux[IV] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IW), flux[IW] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IC), flux[IC] * dtdy);
      }
      if (j > ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j - 1, ID), flux[ID] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IP), flux[IP] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IU), flux[IU] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IV), flux[IV] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IW), flux[IW] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IC), flux[IC] * dtdy);
      }
    }
  }

  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y, Udata;
  real_t      dtdx, dtdy;

}; // ComputeFluxesAndUpdateFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndStoreFunctor2D : public MHDBaseFunctor2D
{

public:
  ComputeEmfAndStoreFunctor2D(HydroParams     params,
                              DataArray2d     QEdge_RT,
                              DataArray2d     QEdge_RB,
                              DataArray2d     QEdge_LT,
                              DataArray2d     QEdge_LB,
                              DataArrayScalar Emf,
                              real_t          dtdx,
                              real_t          dtdy)
    : MHDBaseFunctor2D(params)
    , QEdge_RT(QEdge_RT)
    , QEdge_RB(QEdge_RB)
    , QEdge_LT(QEdge_LT)
    , QEdge_LB(QEdge_LB)
    , Emf(Emf)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams     params,
        DataArray2d     QEdge_RT,
        DataArray2d     QEdge_RB,
        DataArray2d     QEdge_LT,
        DataArray2d     QEdge_LB,
        DataArrayScalar Emf,
        real_t          dtdx,
        real_t          dtdy)
  {
    ComputeEmfAndStoreFunctor2D functor(
      params, QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB, Emf, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeEmfAndStoreFunctor2D",
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

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {

      // in 2D, we only need to compute emfZ
      MHDState qEdge_emfZ[4];

      // clang-format off
      get_state(QEdge_RT, i - 1, j - 1, qEdge_emfZ[IRT]);
      get_state(QEdge_RB, i - 1, j    , qEdge_emfZ[IRB]);
      get_state(QEdge_LT, i    , j - 1, qEdge_emfZ[ILT]);
      get_state(QEdge_LB, i    , j    , qEdge_emfZ[ILB]);
      // clang-format on

      // actually compute emfZ
      real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ, params);
      Emf(i, j) = emfZ;
    }
  }

  DataArray2d     QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  DataArrayScalar Emf;
  real_t          dtdx, dtdy;

}; // ComputeEmfAndStoreFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndUpdateFunctor2D : public MHDBaseFunctor2D
{

public:
  ComputeEmfAndUpdateFunctor2D(HydroParams params,
                               DataArray2d QEdge_RT,
                               DataArray2d QEdge_RB,
                               DataArray2d QEdge_LT,
                               DataArray2d QEdge_LB,
                               DataArray2d Udata,
                               real_t      dtdx,
                               real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , QEdge_RT(QEdge_RT)
    , QEdge_RB(QEdge_RB)
    , QEdge_LT(QEdge_LT)
    , QEdge_LB(QEdge_LB)
    , Udata(Udata)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d QEdge_RT,
        DataArray2d QEdge_RB,
        DataArray2d QEdge_LT,
        DataArray2d QEdge_LB,
        DataArray2d Udata,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeEmfAndUpdateFunctor2D functor(
      params, QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB, Udata, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeEmfAndUpdateFunctor2D",
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

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {

      // in 2D, we only need to compute emfZ
      MHDState qEdge_emfZ[4];

      // preparation for calling compute_emf (equivalent to cmp_mag_flx
      // in DUMSES)
      // in the following, the 2 first indexes in qEdge_emf array play
      // the same offset role as in the calling argument of cmp_mag_flx
      // in DUMSES (if you see what I mean ?!)

      // clang-format off
      get_state(QEdge_RT, i - 1, j - 1, qEdge_emfZ[IRT]);
      get_state(QEdge_RB, i - 1, j    , qEdge_emfZ[IRB]);
      get_state(QEdge_LT, i    , j - 1, qEdge_emfZ[ILT]);
      get_state(QEdge_LB, i    , j    , qEdge_emfZ[ILB]);
      // clang-format on

      // actually compute emfZ
      const real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ, params);

      // clang-format off
      Kokkos::atomic_sub(&Udata(i    , j    , IA), emfZ * dtdy);
      Kokkos::atomic_add(&Udata(i    , j    , IB), emfZ * dtdx);

      Kokkos::atomic_add(&Udata(i    , j - 1, IA), emfZ * dtdy);
      Kokkos::atomic_sub(&Udata(i - 1, j    , IB), emfZ * dtdx);
      // clang-format on
    }
  }

  DataArray2d QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB, Udata;
  real_t      dtdx, dtdy;

}; // ComputeEmfAndUpdateFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  ComputeTraceFunctor2D_MHD(HydroParams params,
                            DataArray2d Udata,
                            DataArray2d Qdata,
                            DataArray2d Qm_x,
                            DataArray2d Qm_y,
                            DataArray2d Qp_x,
                            DataArray2d Qp_y,
                            DataArray2d QEdge_RT,
                            DataArray2d QEdge_RB,
                            DataArray2d QEdge_LT,
                            DataArray2d QEdge_LB,
                            real_t      dtdx,
                            real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , QEdge_RT(QEdge_RT)
    , QEdge_RB(QEdge_RB)
    , QEdge_LT(QEdge_LT)
    , QEdge_LB(QEdge_LB)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata,
        DataArray2d Qdata,
        DataArray2d Qm_x,
        DataArray2d Qm_y,
        DataArray2d Qp_x,
        DataArray2d Qp_y,
        DataArray2d QEdge_RT,
        DataArray2d QEdge_RB,
        DataArray2d QEdge_LT,
        DataArray2d QEdge_LB,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeTraceFunctor2D_MHD functor(params,
                                      Udata,
                                      Qdata,
                                      Qm_x,
                                      Qm_y,
                                      Qp_x,
                                      Qp_y,
                                      QEdge_RT,
                                      QEdge_RB,
                                      QEdge_LT,
                                      QEdge_LB,
                                      dtdx,
                                      dtdy);
    Kokkos::parallel_for(
      "ComputeTraceFunctor2D_MHD",
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

    // clang-format off
    if (j >= ghostWidth - 2 and j < jsize - ghostWidth + 1 and
        i >= ghostWidth - 2 and i < isize - ghostWidth + 1)
    // clang-format on
    {

      MHDState qNb[3][3];
      BField   bfNb[4][4];

      MHDState qm[2];
      MHDState qp[2];

      MHDState qEdge[4];
      real_t   c = 0.0;

      // prepare qNb : q state in the 3-by-3 neighborhood
      // note that current cell (ii,jj) is in qNb[1][1]
      // also note that the effective stencil is 4-by-4 since
      // computation of primitive variable (q) requires mag
      // field on the right (see computePrimitives_MHD_2D)
      for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
        {
          get_state(Qdata, i + di - 1, j + dj - 1, qNb[di][dj]);
        }

      // prepare bfNb : bf (face centered mag field) in the
      // 4-by-4 neighborhood
      // note that current cell (ii,jj) is in bfNb[1][1]
      for (int di = 0; di < 4; di++)
        for (int dj = 0; dj < 4; dj++)
        {
          get_magField(Udata, i + di - 1, j + dj - 1, bfNb[di][dj]);
        }

      trace_unsplit_mhd_2d(qNb, bfNb, c, dtdx, dtdy, 0.0, qm, qp, qEdge);

      // store qm, qp : only what is really needed
      set_state(Qm_x, i, j, qm[0]);
      set_state(Qp_x, i, j, qp[0]);
      set_state(Qm_y, i, j, qm[1]);
      set_state(Qp_y, i, j, qp[1]);

      set_state(QEdge_RT, i, j, qEdge[IRT]);
      set_state(QEdge_RB, i, j, qEdge[IRB]);
      set_state(QEdge_LT, i, j, qEdge[ILT]);
      set_state(QEdge_LB, i, j, qEdge[ILB]);
    }
  }

  DataArray2d Udata, Qdata;
  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y;
  DataArray2d QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  real_t      dtdx, dtdy;

}; // ComputeTraceFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute cell-centered primitive data at t_{n+1/2} (half time step in
 * Muscl-Hancock).
 */
class ComputeUpdatedPrimVarFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  ComputeUpdatedPrimVarFunctor2D_MHD(HydroParams params,
                                     DataArray2d Udata,
                                     DataArray2d Qdata,
                                     DataArray2d Slopes_x,
                                     DataArray2d Slopes_y,
                                     DataArray2d Qdata2,
                                     real_t      dtdx,
                                     real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Qdata2(Qdata2)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata,
        DataArray2d Qdata,
        DataArray2d Slopes_x,
        DataArray2d Slopes_y,
        DataArray2d Qdata2,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeUpdatedPrimVarFunctor2D_MHD functor(
      params, Udata, Qdata, Slopes_x, Slopes_y, Qdata2, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeUpdatedPrimVarFunctor2D_MHD",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int    isize = params.isize;
    const int    jsize = params.jsize;
    const int    ghostWidth = params.ghostWidth;
    const real_t gamma = params.settings.gamma0;

    // clang-format off
    if (j >= ghostWidth - 2 and j < jsize - ghostWidth + 1 and
        i >= ghostWidth - 2 and i < isize - ghostWidth + 1)
    // clang-format on
    {
      // get primitive variable in current cell
      MHDState q;
      get_state(Qdata, i, j, q);

      MHDState dq[2];

      // retrieve hydro slopes along X
      get_state(Slopes_x, i, j, dq[IX]);

      // retrieve hydro slopes along Y
      get_state(Slopes_y, i, j, dq[IY]);

      // Cell centered values
      auto const & r = q[ID];
      auto const & p = q[IP];
      auto const & u = q[IU];
      auto const & v = q[IV];
      auto const & w = q[IW];
      auto const & A = q[IA];
      auto const & B = q[IB];
      auto const & C = q[IC];

      // Cell centered TVD slopes in X direction
      auto const & drx = dq[IX][ID];
      auto const & dpx = dq[IX][IP];
      auto const & dux = dq[IX][IU];
      auto const & dvx = dq[IX][IV];
      auto const & dwx = dq[IX][IW];
      auto const & dBx = dq[IX][IB];
      auto const & dCx = dq[IX][IC];

      // Cell centered TVD slopes in Y direction
      auto const & dry = dq[IY][ID];
      auto const & dpy = dq[IY][IP];
      auto const & duy = dq[IY][IU];
      auto const & dvy = dq[IY][IV];
      auto const & dwy = dq[IY][IW];
      auto const & dAy = dq[IY][IA];
      auto const & dCy = dq[IY][IC];

      auto const   db = compute_normal_mag_field_slopes(Udata, i, j);
      auto const & dAx = db[IX];
      auto const & dBy = db[IY];

      real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
      {

        sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
        su0 =
          (-u * dux - dpx / r - B * dBx / r - C * dCx / r) * dtdx + (-v * duy + B * dAy / r) * dtdy;
        sv0 =
          (-u * dvx + A * dBx / r) * dtdx + (-v * dvy - dpy / r - A * dAy / r - C * dCy / r) * dtdy;
        sw0 = (-u * dwx + A * dCx / r) * dtdx + (-v * dwy + B * dCy / r) * dtdy;
        sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
        sA0 = (u * dBy + B * duy - v * dAy - A * dvy) * dtdy;
        sB0 = (-u * dBx - B * dux + v * dAx + A * dvx) * dtdx;
        sC0 = (w * dAx + A * dwx - u * dCx - C * dux) * dtdx +
              (-v * dCy - C * dvy + w * dBy + B * dwy) * dtdy;
      }

      // Update in time the primitive variables (half time step)
      Qdata2(i, j, ID) = r + 0.5 * sr0;
      Qdata2(i, j, IU) = u + 0.5 * su0;
      Qdata2(i, j, IV) = v + 0.5 * sv0;
      Qdata2(i, j, IW) = w + 0.5 * sw0;
      Qdata2(i, j, IP) = p + 0.5 * sp0;
      Qdata2(i, j, IA) = A + 0.5 * sA0;
      Qdata2(i, j, IB) = B + 0.5 * sB0;
      Qdata2(i, j, IC) = C + 0.5 * sC0;
    }
  } // operator ()

  DataArray2d Udata, Qdata; // input
  DataArray2d Slopes_x, Slopes_y; // input
  DataArray2d Qdata2;       // output
  real_t      dtdx, dtdy;

}; // ComputeUpdatedPrimvarFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Reconstruct hydro state at cell face center, solve Riemann problems and update.
 *
 * This is done in a direction by direction manner.
 */
template <Direction dir>
class ComputeFluxAndUpdateAlongDirFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  //!
  //! \param[in] Udata_in is conservative variables at t_n
  //! \param[in] Udata_out is conservative variables at t_{n+1}
  //! \param[in] Qdata2 is primitive variables array at t_{n+1/2}
  //!
  ComputeFluxAndUpdateAlongDirFunctor2D_MHD(HydroParams params,
                                            DataArray2d Udata_in,
                                            DataArray2d Udata_out,
                                            DataArray2d Qdata2,
                                            DataArray2d Slopes_x,
                                            DataArray2d Slopes_y,
                                            DataArray2d sFaceMag,
                                            real_t      dtdx,
                                            real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Udata_in(Udata_in)
    , Udata_out(Udata_out)
    , Qdata2(Qdata2)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , sFaceMag(sFaceMag)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata_in,
        DataArray2d Udata_out,
        DataArray2d Qdata2,
        DataArray2d Slopes_x,
        DataArray2d Slopes_y,
        DataArray2d sFaceMag,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeFluxAndUpdateAlongDirFunctor2D_MHD<dir> functor(
      params, Udata_in, Udata_out, Qdata2, Slopes_x, Slopes_y, sFaceMag, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeFluxAndUpdateAlongDirFunctor2D_MHD",
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

    // const real_t gamma = params.settings.gamma0;
    const real_t smallR = params.settings.smallr;
    const real_t smallp = params.settings.smallp;

    constexpr auto delta_i = dir == DIR_X ? 1 : 0;
    constexpr auto delta_j = dir == DIR_Y ? 1 : 0;

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {
      // left and right reconstructed state (input for Riemann solver)
      MHDState qR, qL;

      // slopes in left and right cells
      MHDState dq, dqN;

      //
      // Right state at left interface (current cell)
      //

      // load hydro slopes along dir
      if constexpr (dir == DIR_X)
      {
        get_state(Slopes_x, i, j, dq);
      }
      else if constexpr (dir == DIR_Y)
      {
        get_state(Slopes_y, i, j, dq);
      }

      // get primitive variable in current cell at t_{n+1/2}
      MHDState q2;
      get_state(Qdata2, i, j, q2);

      qR[ID] = q2[ID] - 0.5 * dq[ID];
      qR[IU] = q2[IU] - 0.5 * dq[IU];
      qR[IV] = q2[IV] - 0.5 * dq[IV];
      qR[IW] = q2[IW] - 0.5 * dq[IW];
      qR[IP] = q2[IP] - 0.5 * dq[IP];
      qR[ID] = fmax(smallR, qR[ID]);
      qR[IP] = fmax(smallp * qR[ID], qR[IP]);
      if constexpr (dir == DIR_X)
      {
        const real_t AL = Udata_in(i, j, IA) + sFaceMag(i, j, IX);
        qR[IA] = AL;
        qR[IB] = q2[IB] - 0.5 * dq[IB];
        qR[IC] = q2[IC] - 0.5 * dq[IC];
      }
      else if constexpr (dir == DIR_Y)
      {
        const real_t BL = Udata_in(i, j, IB) + sFaceMag(i, j, IY);
        qR[IA] = q2[IA] - 0.5 * dq[IA];
        qR[IB] = BL;
        qR[IC] = q2[IC] - 0.5 * dq[IC];
      }

      //
      // Left state at right interface (neighbor cell)
      //

      // load hydro slopes along dir
      if constexpr (dir == DIR_X)
      {
        get_state(Slopes_x, i - delta_i, j - delta_j, dqN);
      }
      else if constexpr (dir == DIR_Y)
      {
        get_state(Slopes_y, i - delta_i, j - delta_j, dqN);
      }

      // get primitive variable in neighbor cell at t_{n+1/2}
      MHDState q2N;
      get_state(Qdata2, i - delta_i, j - delta_j, q2N);

      qL[ID] = q2N[ID] + 0.5 * dqN[ID];
      qL[IU] = q2N[IU] + 0.5 * dqN[IU];
      qL[IV] = q2N[IV] + 0.5 * dqN[IV];
      qL[IW] = q2N[IW] + 0.5 * dqN[IW];
      qL[IP] = q2N[IP] + 0.5 * dqN[IP];
      qL[ID] = fmax(smallR, qL[ID]);
      qL[IP] = fmax(smallp * qL[ID], qL[IP]);
      if constexpr (dir == DIR_X)
      {
        qL[IA] = qR[IA];
        qL[IB] = q2N[IB] + 0.5 * dqN[IB];
        qL[IC] = q2N[IC] + 0.5 * dqN[IC];
      }
      else if constexpr (dir == DIR_Y)
      {
        qL[IA] = q2N[IA] + 0.5 * dqN[IA];
        qL[IB] = qR[IB];
        qL[IC] = q2N[IC] + 0.5 * dqN[IC];
      }

      // now we are ready for computing hydro flux
      MHDState flux;

      if constexpr (dir == DIR_Y)
      {
        swapValues(&(qL[IU]), &(qL[IV]));
        swapValues(&(qL[IA]), &(qL[IB]));
        swapValues(&(qR[IU]), &(qR[IV]));
        swapValues(&(qR[IA]), &(qR[IB]));
      }
      riemann_mhd(qL, qR, flux, params);
      if constexpr (dir == DIR_Y)
      {
        swapValues(&(flux[IU]), &(flux[IV]));
        swapValues(&(flux[IA]), &(flux[IB]));
      }

      if constexpr (dir == DIR_X)
      {
        if (j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i, j, ID), flux[ID] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, IP), flux[IP] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, IU), flux[IU] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, IV), flux[IV] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, IW), flux[IW] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, IC), flux[IC] * dtdx);
        }

        if (j < jsize - ghostWidth and i > ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i - 1, j, ID), flux[ID] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, IP), flux[IP] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, IU), flux[IU] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, IV), flux[IV] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, IW), flux[IW] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, IC), flux[IC] * dtdx);
        }
      }
      else if constexpr (dir == DIR_Y)
      {
        if (j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i, j, ID), flux[ID] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, IP), flux[IP] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, IU), flux[IU] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, IV), flux[IV] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, IW), flux[IW] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, IC), flux[IC] * dtdy);
        }
        if (j > ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i, j - 1, ID), flux[ID] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, IP), flux[IP] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, IU), flux[IU] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, IV), flux[IV] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, IW), flux[IW] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, IC), flux[IC] * dtdy);
        }
      }
    }
  } // operator ()

  DataArray2d Udata_in, Udata_out;
  DataArray2d Qdata2;
  DataArray2d Slopes_x, Slopes_y;
  DataArray2d sFaceMag;
  real_t      dtdx, dtdy;

}; // ComputeFluxAndUpdateAlongDirFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Reconstruct hydrodynamics variables and magnetic field at edge center, compute Emf and update.
 */
class ReconstructEdgeComputeEmfAndUpdateFunctor2D : public MHDBaseFunctor2D
{

public:
  //!
  //! \param[in] Udata_in is conservative variables at t_n
  //! \param[in] Udata_out is conservative variables at t_{n+1}
  //! \param[in] Qdata2 is primitive variables array at t_{n+1/2}
  //!
  ReconstructEdgeComputeEmfAndUpdateFunctor2D(HydroParams params,
                                              DataArray2d Udata_in,
                                              DataArray2d Udata_out,
                                              DataArray2d Qdata2,
                                              DataArray2d Slopes_x,
                                              DataArray2d Slopes_y,
                                              DataArray2d sFaceMag,
                                              real_t      dtdx,
                                              real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Udata_in(Udata_in)
    , Udata_out(Udata_out)
    , Qdata2(Qdata2)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , sFaceMag(sFaceMag)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata_in,
        DataArray2d Udata_out,
        DataArray2d Qdata2,
        DataArray2d Slopes_x,
        DataArray2d Slopes_y,
        DataArray2d sFaceMag,
        real_t      dtdx,
        real_t      dtdy)
  {
    ReconstructEdgeComputeEmfAndUpdateFunctor2D functor(
      params, Udata_in, Udata_out, Qdata2, Slopes_x, Slopes_y, sFaceMag, dtdx, dtdy);
    Kokkos::parallel_for(
      "ReconstructEdgeComputeEmfAndUpdateFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  /**
   * Given a cell-center MHD state and limited slopes, reconstruct a MHD state at edge.
   *
   * \param[out] qEdge is the reconstructed MHD state at edge
   * \param[in] i0 x-coordinate (cell center) of the cell where the reconstruction takes place
   * \param[in] j0 y-coordinate (cell center) of the cell where the reconstruction takes place
   * \param[in] edge_loc is the type of edge reconstruction (from cell center)
   */
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_at_edge(MHDState &      qEdge,
                            uint32_t        i0,
                            uint32_t        j0,
                            MHDEdgeLocation edge_loc) const
  {

    // const real_t gamma = params.settings.gamma0;
    const real_t smallR = params.settings.smallr;
    const real_t smallp = params.settings.smallp;

    int sign_x = 1;
    int sign_y = 1;
    int ia = 0;
    int ja = 0;
    int ib = 0;
    int jb = 0;
    int sign_a = 1;
    int sign_b = 1;

    if (edge_loc == MHDEdgeLocation::LB)
    {
      ia = i0;
      ja = j0;
      ib = i0;
      jb = j0;
      sign_x = -1;
      sign_y = -1;
      sign_a = -1;
      sign_b = -1;
    }
    else if (edge_loc == MHDEdgeLocation::RT)
    {
      ia = i0 + 1;
      ja = j0;
      ib = i0;
      jb = j0 + 1;
      sign_x = 1;
      sign_y = 1;
      sign_a = 1;
      sign_b = 1;
    }
    else if (edge_loc == MHDEdgeLocation::RB)
    {
      ia = i0 + 1;
      ja = j0;
      ib = i0;
      jb = j0;
      sign_x = 1;
      sign_y = -1;
      sign_a = -1;
      sign_b = 1;
    }
    else if (edge_loc == MHDEdgeLocation::LT)
    {
      ia = i0;
      ja = j0;
      ib = i0;
      jb = j0 + 1;
      sign_x = -1;
      sign_y = 1;
      sign_a = 1;
      sign_b = -1;
    }

    const real_t A = Udata_in(ia, ja, IA) + sFaceMag(ia, ja, IX);
    const real_t dAy = compute_limited_slope<DIR_Y>(Udata_in, ia, ja, IA);

    const real_t B = Udata_in(ib, jb, IB) + sFaceMag(ib, jb, IY);
    const real_t dBx = compute_limited_slope<DIR_X>(Udata_in, ib, jb, IB);

    // get limited slopes
    MHDState dqX, dqY;
    get_state(Slopes_x, i0, j0, dqX);
    get_state(Slopes_y, i0, j0, dqY);

    // reconstructed state
    qEdge[ID] += 0.5 * (sign_x * dqX[ID] + sign_y * dqY[ID]);
    qEdge[IU] += 0.5 * (sign_x * dqX[IU] + sign_y * dqY[IU]);
    qEdge[IV] += 0.5 * (sign_x * dqX[IV] + sign_y * dqY[IV]);
    qEdge[IW] += 0.5 * (sign_x * dqX[IW] + sign_y * dqY[IW]);
    qEdge[IP] += 0.5 * (sign_x * dqX[IP] + sign_y * dqY[IP]);
    qEdge[IA] = A + 0.5 * (sign_a * dAy);
    qEdge[IB] = B + 0.5 * (sign_b * dBx);
    qEdge[IC] += 0.5 * (sign_x * dqX[IC] + sign_y * dqY[IC]);
    qEdge[ID] = fmax(smallR, qEdge[ID]);
    qEdge[IP] = fmax(smallp * qEdge[ID], qEdge[IP]);

  } // reconstruct_state_at_edge

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {
      // this drawing is a simple helper to reminder how reconstruction is done
      //
      // symbols LB, RB, LT, RT indicate location of reconstructed edge from the cell center.
      //
      //
      //  X locates the edge where hydrodynamics values must be reconstructed
      //    _________________________
      //   |           |            |
      //   |           |            |
      //   |     RB    |     LB     |
      //   |  (i-1,j)  |   (i,j)    |
      //   |          \| /          |
      //   |___________X____________|
      //   |          /| \          |
      //   |        /  |  \         |
      //   |     RT    |     LT     |
      //   | (i-1,j-1) |   (i,j-1)  |
      //   |           |            |
      //   |___________|____________|
      //


      // qLB is current cell, qLT, qRT and qRB are direct neighbors surrounding the lower left
      // edge MHDState qLB, qLT, qRB, qRT;
      MHDState   qEdge_emfZ[4];
      MHDState & qRT = qEdge_emfZ[IRT];
      MHDState & qLT = qEdge_emfZ[ILT];
      MHDState & qRB = qEdge_emfZ[IRB];
      MHDState & qLB = qEdge_emfZ[ILB];

      // get primitive variable in current cell and neighbors at t_{n+1/2}
      // clang-format off
      get_state(Qdata2, i    , j    , qLB);
      get_state(Qdata2, i    , j - 1, qLT);
      get_state(Qdata2, i - 1, j    , qRB);
      get_state(Qdata2, i - 1, j - 1, qRT);
      // clang-format on

      // LB at (i,j)
      {
        const auto i0 = i;
        const auto j0 = j;
        reconstruct_state_at_edge(qLB, i0, j0, MHDEdgeLocation::LB);
      }

      // RT (i-1, j-1)
      {
        const auto i0 = i - 1;
        const auto j0 = j - 1;
        reconstruct_state_at_edge(qRT, i0, j0, MHDEdgeLocation::RT);
      }

      // RB (i-1,j)
      {
        const auto i0 = i - 1;
        const auto j0 = j;
        reconstruct_state_at_edge(qRB, i0, j0, MHDEdgeLocation::RB);
      }

      // LT (i,j-1)
      {
        const auto i0 = i;
        const auto j0 = j - 1;
        reconstruct_state_at_edge(qLT, i0, j0, MHDEdgeLocation::LT);
      }

      const real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ, params);

      // clang-format off
      Kokkos::atomic_sub(&Udata_out(i    , j    , IA), emfZ * dtdy);
      Kokkos::atomic_add(&Udata_out(i    , j    , IB), emfZ * dtdx);

      Kokkos::atomic_add(&Udata_out(i    , j - 1, IA), emfZ * dtdy);
      Kokkos::atomic_sub(&Udata_out(i - 1, j    , IB), emfZ * dtdx);
      // clang-format on
    }
  } // operator ()

  DataArray2d Udata_in, Udata_out;
  DataArray2d Qdata2;
  DataArray2d Slopes_x, Slopes_y;
  DataArray2d sFaceMag;
  real_t      dtdx, dtdy;

}; // ReconstructEdgeComputeEmfAndUpdateFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor2D_MHD : public MHDBaseFunctor2D
{

public:
  UpdateFunctor2D_MHD(HydroParams params,
                      DataArray2d Udata,
                      DataArray2d FluxData_x,
                      DataArray2d FluxData_y,
                      real_t      dtdx,
                      real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata,
        DataArray2d FluxData_x,
        DataArray2d FluxData_y,
        real_t      dtdx,
        real_t      dtdy)
  {
    UpdateFunctor2D_MHD functor(params, Udata, FluxData_x, FluxData_y, dtdx, dtdy);
    Kokkos::parallel_for(
      "UpdateFunctor2D_MHD",
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

      MHDState udata;
      MHDState flux;
      get_state(Udata, i, j, udata);

      // add up contributions from all 4 faces

      get_state(FluxData_x, i, j, flux);
      udata[ID] += flux[ID] * dtdx;
      udata[IP] += flux[IP] * dtdx;
      udata[IU] += flux[IU] * dtdx;
      udata[IV] += flux[IV] * dtdx;
      udata[IW] += flux[IW] * dtdx;
      // udata[IA] +=  flux[IA]*dtdx;
      // udata[IB] +=  flux[IB]*dtdx;
      udata[IC] += flux[IC] * dtdx;

      get_state(FluxData_x, i + 1, j, flux);
      udata[ID] -= flux[ID] * dtdx;
      udata[IP] -= flux[IP] * dtdx;
      udata[IU] -= flux[IU] * dtdx;
      udata[IV] -= flux[IV] * dtdx;
      udata[IW] -= flux[IW] * dtdx;
      // udata[IA] -=  flux[IA]*dtdx;
      // udata[IB] -=  flux[IB]*dtdx;
      udata[IC] -= flux[IC] * dtdx;

      get_state(FluxData_y, i, j, flux);
      udata[ID] += flux[ID] * dtdy;
      udata[IP] += flux[IP] * dtdy;
      udata[IU] += flux[IV] * dtdy; //
      udata[IV] += flux[IU] * dtdy; //
      udata[IW] += flux[IW] * dtdy;
      // udata[IA] +=  flux[IA]*dtdy;
      // udata[IB] +=  flux[IB]*dtdy;
      udata[IC] += flux[IC] * dtdy;

      get_state(FluxData_y, i, j + 1, flux);
      udata[ID] -= flux[ID] * dtdy;
      udata[IP] -= flux[IP] * dtdy;
      udata[IU] -= flux[IV] * dtdy; //
      udata[IV] -= flux[IU] * dtdy; //
      udata[IW] -= flux[IW] * dtdy;
      // udata[IA] -=  flux[IA]*dtdy;
      // udata[IB] -=  flux[IB]*dtdy;
      udata[IC] -= flux[IC] * dtdy;

      // write back result in Udata
      set_state(Udata, i, j, udata);

    } // end if

  } // end operator ()

  DataArray2d Udata;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;
  real_t      dtdx, dtdy;

}; // UpdateFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateEmfFunctor2D : public MHDBaseFunctor2D
{

public:
  UpdateEmfFunctor2D(HydroParams     params,
                     DataArray2d     Udata,
                     DataArrayScalar Emf,
                     real_t          dtdx,
                     real_t          dtdy)
    : MHDBaseFunctor2D(params)
    , Udata(Udata)
    , Emf(Emf)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray2d Udata, DataArrayScalar Emf, real_t dtdx, real_t dtdy)
  {
    UpdateEmfFunctor2D functor(params, Udata, Emf, dtdx, dtdy);
    Kokkos::parallel_for("UpdateEmfFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    // clang-format off
    if (j >= ghostWidth and j < jsize - ghostWidth /*+1*/ and
        i >= ghostWidth and i < isize - ghostWidth /*+1*/)
    // clang-format on
    {

      // MHDState udata;
      // get_state(Udata, index, udata);

      // left-face B-field
      Udata(i, j, IA) += (Emf(i, j + 1) - Emf(i, j)) * dtdy;
      Udata(i, j, IB) -= (Emf(i + 1, j) - Emf(i, j)) * dtdx;
    }
  }

  DataArray2d     Udata;
  DataArrayScalar Emf;
  real_t          dtdx, dtdy;

}; // UpdateEmfFunctor2D


/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor2D_MHD : public MHDBaseFunctor2D
{

public:
  ComputeTraceAndFluxes_Functor2D_MHD(HydroParams params,
                                      DataArray2d Qdata,
                                      DataArray2d Slopes_x,
                                      DataArray2d Slopes_y,
                                      DataArray2d Fluxes,
                                      real_t      dtdx,
                                      real_t      dtdy)
    : MHDBaseFunctor2D(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Fluxes(Fluxes)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    // clang-format off
    if (j >= ghostWidth and j <= jsize - ghostWidth and
        i >= ghostWidth and i <= isize - ghostWidth)
    // clang-format on
    {

      // local primitive variables
      MHDState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      MHDState qLocNeighbor;

      // Local slopes and neighbor slopes
      MHDState dqX;
      MHDState dqY;
      MHDState dqX_neighbor;
      MHDState dqY_neighbor;

      // Local variables for Riemann problems solving
      MHDState qleft;
      MHDState qright;
      // MHDState qgdnv;
      MHDState flux;

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

        // Solve Riemann problem at X-interfaces and compute X-fluxes
        riemann_mhd(qleft, qright, flux, params);

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

        // Solve Riemann problem at Y-interfaces and compute Y-fluxes
        swapValues(&(qleft[IU]), &(qleft[IV]));
        swapValues(&(qright[IU]), &(qright[IV]));
        riemann_mhd(qleft, qright, flux, params);

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

  DataArray2d Qdata;
  DataArray2d Slopes_x, Slopes_y;
  DataArray2d Fluxes;
  real_t      dtdx, dtdy;

}; // ComputeTraceAndFluxes_Functor2D_MHD

} // namespace muscl
} // namespace euler_kokkos

#endif // MHD_RUN_FUNCTORS_2D_H_
