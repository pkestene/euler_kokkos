#ifndef MHD_RUN_FUNCTORS_3D_H_
#define MHD_RUN_FUNCTORS_3D_H_

#include "shared/kokkos_shared.h"
#include "MHDBaseFunctor3D.h"
#include "shared/RiemannSolvers_MHD.h"

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  ComputeDtFunctor3D_MHD(HydroParams params, DataArray3d Qdata)
    : MHDBaseFunctor3D(params)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, real_t & invDt)
  {
    ComputeDtFunctor3D_MHD functor(params, Udata);
    Kokkos::Max<real_t>    reducer(invDt);
    Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                              { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                            functor,
                            reducer);
  }

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k, real_t & invDt) const
  {
    const int    isize = params.isize;
    const int    jsize = params.jsize;
    const int    ksize = params.ksize;
    const int    ghostWidth = params.ghostWidth;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth and
        j >= ghostWidth and j < jsize - ghostWidth and
        i >= ghostWidth and i < isize - ghostWidth)
    // clang-format on
    {

      MHDState qLoc; // primitive    variables in current cell

      // get primitive variables in current cell
      qLoc[ID] = Qdata(i, j, k, ID);
      qLoc[IP] = Qdata(i, j, k, IP);
      qLoc[IU] = Qdata(i, j, k, IU);
      qLoc[IV] = Qdata(i, j, k, IV);
      qLoc[IW] = Qdata(i, j, k, IW);
      qLoc[IA] = Qdata(i, j, k, IA);
      qLoc[IB] = Qdata(i, j, k, IB);
      qLoc[IC] = Qdata(i, j, k, IC);

      // compute fastest information speeds
      real_t fastInfoSpeed[3];
      find_speed_info<THREE_D>(qLoc, fastInfoSpeed, params);

      real_t vx = fastInfoSpeed[IX];
      real_t vy = fastInfoSpeed[IY];
      real_t vz = fastInfoSpeed[IZ];

      invDt = fmax(invDt, vx / dx + vy / dy + vz / dz);
    }

  } // operator ()

  DataArray3d Qdata;

}; // ComputeDtFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  ConvertToPrimitivesFunctor3D_MHD(HydroParams params, DataArray3d Udata, DataArray3d Qdata)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, DataArray3d Qdata)
  {
    ConvertToPrimitivesFunctor3D_MHD functor(params, Udata, Qdata);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // magnetic field in neighbor cells
    real_t magFieldNeighbors[3];

    // clang-format off
    if (k >= 0 and k < ksize - 1 and
        j >= 0 and j < jsize - 1 and
        i >= 0 and i < isize - 1)
    // clang-format on
    {

      MHDState uLoc; // conservative variables in current cell
      MHDState qLoc; // primitive    variables in current cell
      real_t   c;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, k, ID);
      uLoc[IP] = Udata(i, j, k, IP);
      uLoc[IU] = Udata(i, j, k, IU);
      uLoc[IV] = Udata(i, j, k, IV);
      uLoc[IW] = Udata(i, j, k, IW);
      uLoc[IA] = Udata(i, j, k, IA);
      uLoc[IB] = Udata(i, j, k, IB);
      uLoc[IC] = Udata(i, j, k, IC);

      // get mag field in neighbor cells
      magFieldNeighbors[IX] = Udata(i + 1, j, k, IA);
      magFieldNeighbors[IY] = Udata(i, j + 1, k, IB);
      magFieldNeighbors[IZ] = Udata(i, j, k + 1, IC);

      // get primitive variables in current cell
      constoprim_mhd(uLoc, magFieldNeighbors, c, qLoc);

      // copy q state in q global
      Qdata(i, j, k, ID) = qLoc[ID];
      Qdata(i, j, k, IP) = qLoc[IP];
      Qdata(i, j, k, IU) = qLoc[IU];
      Qdata(i, j, k, IV) = qLoc[IV];
      Qdata(i, j, k, IW) = qLoc[IW];
      Qdata(i, j, k, IA) = qLoc[IA];
      Qdata(i, j, k, IB) = qLoc[IB];
      Qdata(i, j, k, IC) = qLoc[IC];
    }
  }

  DataArray3d Udata;
  DataArray3d Qdata;

}; // ConvertToPrimitivesFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  /**
   * Compute limited slopes of primitives variables at cell center.
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
   * \param[out] Slopes_z limited slopes along direction Z
   */
  ComputeSlopesFunctor3D_MHD(HydroParams params,
                             DataArray3d Udata,
                             DataArray3d Qdata,
                             DataArray3d Slopes_x,
                             DataArray3d Slopes_y,
                             DataArray3d Slopes_z)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Slopes_z(Slopes_z){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Udata,
        DataArray3d Qdata,
        DataArray3d Slopes_x,
        DataArray3d Slopes_y,
        DataArray3d Slopes_z)
  {
    ComputeSlopesFunctor3D_MHD functor(params, Udata, Qdata, Slopes_x, Slopes_y, Slopes_z);
    Kokkos::parallel_for("ComputeSlopesFunctor3D_MHD",
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

    // clang-format off
    if (k >= 1 and k < ksize - 1 and
        j >= 1 and j < jsize - 1 and
        i >= 1 and i < isize - 1)
    // clang-format on
    {
      MHDState qc, qm, qp;
      get_state(Qdata, i, j, k, qc);

      MHDState dq;

      //
      // slopes along X
      //
      get_state(Qdata, i - 1, j, k, qm);
      get_state(Qdata, i + 1, j, k, qp);
      slope_unsplit_hydro_3d(qc, qp, qm, dq);
      set_state(Slopes_x, i, j, k, dq);

      // modify slopes for normal magnetic field component using face centered value
      Slopes_x(i, j, k, IA) = Udata(i + 1, j, k, IA) - Udata(i, j, k, IA);

      //
      // slopes along Y
      //
      get_state(Qdata, i, j - 1, k, qm);
      get_state(Qdata, i, j + 1, k, qp);
      slope_unsplit_hydro_3d(qc, qp, qm, dq);
      set_state(Slopes_y, i, j, k, dq);

      // modify slopes for normal magnetic field component using face centered value
      Slopes_y(i, j, k, IB) = Udata(i, j + 1, k, IB) - Udata(i, j, k, IB);

      //
      // slopes along Z
      //
      get_state(Qdata, i, j, k - 1, qm);
      get_state(Qdata, i, j, k + 1, qp);
      slope_unsplit_hydro_3d(qc, qp, qm, dq);
      set_state(Slopes_z, i, j, k, dq);

      // modify slopes for normal magnetic field component using face centered value
      Slopes_z(i, j, k, IC) = Udata(i, j, k + 1, IC) - Udata(i, j, k, IC);
    }
  } // end operator ()

  DataArray3d Udata, Qdata;
  DataArray3d Slopes_x, Slopes_y, Slopes_z;

}; // ComputeSlopesFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeElecFieldFunctor3D : public MHDBaseFunctor3D
{

public:
  ComputeElecFieldFunctor3D(HydroParams params,
                            DataArray3d Udata,
                            DataArray3d Qdata,
                            DataArray3d ElecField)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , ElecField(ElecField){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, DataArray3d Qdata, DataArray3d ElecField)
  {
    ComputeElecFieldFunctor3D functor(params, Udata, Qdata, ElecField);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k > 0 and k < ksize - 1 and
        j > 0 and j < jsize - 1 and
        i > 0 and i < isize - 1)
    {

      real_t u, v, w, A, B, C;

      // compute Ex
      v = ONE_FOURTH_F * (Qdata(i, j - 1, k - 1, IV) + Qdata(i, j - 1, k, IV) +
                          Qdata(i, j    , k - 1, IV) + Qdata(i, j    , k, IV));

      w = ONE_FOURTH_F * (Qdata(i, j - 1, k - 1, IW) + Qdata(i, j - 1, k, IW) +
                          Qdata(i, j    , k - 1, IW) + Qdata(i, j    , k, IW));

      B = HALF_F * (Udata(i, j    , k - 1, IB) + Udata(i, j, k, IB));
      C = HALF_F * (Udata(i, j - 1, k    , IC) + Udata(i, j, k, IC));

      ElecField(i, j, k, IX) = v * C - w * B;

      // compute Ey
      u = ONE_FOURTH_F * (Qdata(i - 1, j, k - 1, IU) + Qdata(i - 1, j, k, IU) +
                          Qdata(i    , j, k - 1, IU) + Qdata(i    , j, k, IU));

      w = ONE_FOURTH_F * (Qdata(i - 1, j, k - 1, IW) + Qdata(i - 1, j, k, IW) +
                          Qdata(i    , j, k - 1, IW) + Qdata(i    , j, k, IW));

      A = HALF_F * (Udata(i    , j, k - 1, IA) + Udata(i, j, k, IA));
      C = HALF_F * (Udata(i - 1, j, k    , IC) + Udata(i, j, k, IC));

      ElecField(i, j, k, IY) = w * A - u * C;

      // compute Ez
      u = ONE_FOURTH_F * (Qdata(i - 1, j - 1, k, IU) + Qdata(i - 1, j, k, IU) +
                          Qdata(i    , j - 1, k, IU) + Qdata(i    , j, k, IU));

      v = ONE_FOURTH_F * (Qdata(i - 1, j - 1, k, IV) + Qdata(i - 1, j, k, IV) +
                          Qdata(i    , j - 1, k, IV) + Qdata(i    , j, k, IV));

      A = HALF_F * (Udata(i    , j - 1, k, IA) + Udata(i, j, k, IA));
      B = HALF_F * (Udata(i - 1, j    , k, IB) + Udata(i, j, k, IB));

      ElecField(i, j, k, IZ) = u * B - v * A;
    }
    // clang-format on

  } // operator ()

  DataArray3d Udata;
  DataArray3d Qdata;
  DataArray3d ElecField;

}; // ComputeElecFieldFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSourceFaceMagFunctor3D : public MHDBaseFunctor3D
{

public:
  ComputeSourceFaceMagFunctor3D(HydroParams params,
                                DataArray3d ElecField,
                                DataArray3d sFaceMag,
                                real_t      dtdx,
                                real_t      dtdy,
                                real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , ElecField(ElecField)
    , sFaceMag(sFaceMag)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d ElecField,
        DataArray3d sFaceMag,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ComputeSourceFaceMagFunctor3D functor(params, ElecField, sFaceMag, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k > 0 and k < ksize-1  and
        j > 0 and j < jsize-1  and
        i > 0 and i < isize-1)
    {

      // sAL0 = +(GLR - GLL) * dtdy * HALF_F - (FLR - FLL) * dtdz * HALF_F
      sFaceMag(i, j, k, IX) =
        (ElecField(i, j + 1, k    , IZ) - ElecField(i, j, k, IZ)) * HALF_F * dtdy -
        (ElecField(i, j    , k + 1, IY) - ElecField(i, j, k, IY)) * HALF_F * dtdz;

      // sBL0 = -(GRL - GLL) * dtdx * HALF_F + (ELR - ELL) * dtdz * HALF_F
      sFaceMag(i, j, k, IY) =
        -(ElecField(i + 1, j, k    , IZ) - ElecField(i, j, k, IZ)) * HALF_F * dtdx +
         (ElecField(i    , j, k + 1, IX) - ElecField(i, j, k, IX)) * HALF_F * dtdz;

      // sCL0 = +(FRL - FLL) * dtdx * HALF_F - (ERL - ELL) * dtdy * HALF_F
      sFaceMag(i, j, k, IZ) =
        (ElecField(i + 1, j    , k, IY) - ElecField(i, j, k, IY)) * HALF_F * dtdx -
        (ElecField(i    , j + 1, k, IX) - ElecField(i, j, k, IX)) * HALF_F * dtdy;
    }
    // clang-format on

  } // operator ()

  DataArray3d ElecField;
  DataArray3d sFaceMag;
  real_t      dtdx, dtdy, dtdz;

}; // ComputeSourceFaceMagFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeMagSlopesFunctor3D : public MHDBaseFunctor3D
{

public:
  ComputeMagSlopesFunctor3D(HydroParams      params,
                            DataArray3d      Udata,
                            DataArrayVector3 DeltaA,
                            DataArrayVector3 DeltaB,
                            DataArrayVector3 DeltaC)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , DeltaA(DeltaA)
    , DeltaB(DeltaB)
    , DeltaC(DeltaC){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams      params,
        DataArray3d      Udata,
        DataArrayVector3 DeltaA,
        DataArrayVector3 DeltaB,
        DataArrayVector3 DeltaC)
  {
    ComputeMagSlopesFunctor3D functor(params, Udata, DeltaA, DeltaB, DeltaC);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k > 0 and k < ksize - 1 and
        j > 0 and j < jsize - 1 and
        i > 0 and i < isize - 1)
    // clang-format on
    {

      real_t bfSlopes[15];
      real_t dbfSlopes[3][3];

      real_t(&dbfX)[3] = dbfSlopes[IX];
      real_t(&dbfY)[3] = dbfSlopes[IY];
      real_t(&dbfZ)[3] = dbfSlopes[IZ];

      // get magnetic slopes dbf

      // clang-format off
      bfSlopes[0] =  Udata(i    , j    , k    , IA);
      bfSlopes[1] =  Udata(i    , j + 1, k    , IA);
      bfSlopes[2] =  Udata(i    , j - 1, k    , IA);
      bfSlopes[3] =  Udata(i    , j    , k + 1, IA);
      bfSlopes[4] =  Udata(i    , j    , k - 1, IA);

      bfSlopes[5] =  Udata(i    , j    , k    , IB);
      bfSlopes[6] =  Udata(i + 1, j    , k    , IB);
      bfSlopes[7] =  Udata(i - 1, j    , k    , IB);
      bfSlopes[8] =  Udata(i    , j    , k + 1, IB);
      bfSlopes[9] =  Udata(i    , j    , k - 1, IB);

      bfSlopes[10] = Udata(i    , j    , k    , IC);
      bfSlopes[11] = Udata(i + 1, j    , k    , IC);
      bfSlopes[12] = Udata(i - 1, j    , k    , IC);
      bfSlopes[13] = Udata(i    , j + 1, k    , IC);
      bfSlopes[14] = Udata(i    , j - 1, k    , IC);
      // clang-format on

      // compute magnetic slopes
      slope_unsplit_mhd_3d(bfSlopes, dbfSlopes);

      // store magnetic slopes
      DeltaA(i, j, k, 0) = dbfX[IX];
      DeltaA(i, j, k, 1) = dbfY[IX];
      DeltaA(i, j, k, 2) = dbfZ[IX];

      DeltaB(i, j, k, 0) = dbfX[IY];
      DeltaB(i, j, k, 1) = dbfY[IY];
      DeltaB(i, j, k, 2) = dbfZ[IY];

      DeltaC(i, j, k, 0) = dbfX[IZ];
      DeltaC(i, j, k, 1) = dbfY[IZ];
      DeltaC(i, j, k, 2) = dbfZ[IZ];
    }

  } // operator ()
  DataArray3d      Udata;
  DataArrayVector3 DeltaA;
  DataArrayVector3 DeltaB;
  DataArrayVector3 DeltaC;

}; // class ComputeMagSlopesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  ComputeTraceFunctor3D_MHD(HydroParams      params,
                            DataArray3d      Udata,
                            DataArray3d      Qdata,
                            DataArrayVector3 DeltaA,
                            DataArrayVector3 DeltaB,
                            DataArrayVector3 DeltaC,
                            DataArrayVector3 ElecField,
                            DataArray3d      Qm_x,
                            DataArray3d      Qm_y,
                            DataArray3d      Qm_z,
                            DataArray3d      Qp_x,
                            DataArray3d      Qp_y,
                            DataArray3d      Qp_z,
                            DataArray3d      QEdge_RT,
                            DataArray3d      QEdge_RB,
                            DataArray3d      QEdge_LT,
                            DataArray3d      QEdge_LB,
                            DataArray3d      QEdge_RT2,
                            DataArray3d      QEdge_RB2,
                            DataArray3d      QEdge_LT2,
                            DataArray3d      QEdge_LB2,
                            DataArray3d      QEdge_RT3,
                            DataArray3d      QEdge_RB3,
                            DataArray3d      QEdge_LT3,
                            DataArray3d      QEdge_LB3,
                            real_t           dtdx,
                            real_t           dtdy,
                            real_t           dtdz)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , DeltaA(DeltaA)
    , DeltaB(DeltaB)
    , DeltaC(DeltaC)
    , ElecField(ElecField)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qm_z(Qm_z)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , Qp_z(Qp_z)
    , QEdge_RT(QEdge_RT)
    , QEdge_RB(QEdge_RB)
    , QEdge_LT(QEdge_LT)
    , QEdge_LB(QEdge_LB)
    , QEdge_RT2(QEdge_RT2)
    , QEdge_RB2(QEdge_RB2)
    , QEdge_LT2(QEdge_LT2)
    , QEdge_LB2(QEdge_LB2)
    , QEdge_RT3(QEdge_RT3)
    , QEdge_RB3(QEdge_RB3)
    , QEdge_LT3(QEdge_LT3)
    , QEdge_LB3(QEdge_LB3)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams      params,
        DataArray3d      Udata,
        DataArray3d      Qdata,
        DataArrayVector3 DeltaA,
        DataArrayVector3 DeltaB,
        DataArrayVector3 DeltaC,
        DataArray3d      ElecField,
        DataArray3d      Qm_x,
        DataArray3d      Qm_y,
        DataArray3d      Qm_z,
        DataArray3d      Qp_x,
        DataArray3d      Qp_y,
        DataArray3d      Qp_z,
        DataArray3d      QEdge_RT,
        DataArray3d      QEdge_RB,
        DataArray3d      QEdge_LT,
        DataArray3d      QEdge_LB,
        DataArray3d      QEdge_RT2,
        DataArray3d      QEdge_RB2,
        DataArray3d      QEdge_LT2,
        DataArray3d      QEdge_LB2,
        DataArray3d      QEdge_RT3,
        DataArray3d      QEdge_RB3,
        DataArray3d      QEdge_LT3,
        DataArray3d      QEdge_LB3,
        real_t           dtdx,
        real_t           dtdy,
        real_t           dtdz)
  {
    ComputeTraceFunctor3D_MHD functor(params,
                                      Udata,
                                      Qdata,
                                      DeltaA,
                                      DeltaB,
                                      DeltaC,
                                      ElecField,
                                      Qm_x,
                                      Qm_y,
                                      Qm_z,
                                      Qp_x,
                                      Qp_y,
                                      Qp_z,
                                      QEdge_RT,
                                      QEdge_RB,
                                      QEdge_LT,
                                      QEdge_LB,
                                      QEdge_RT2,
                                      QEdge_RB2,
                                      QEdge_LT2,
                                      QEdge_LB2,
                                      QEdge_RT3,
                                      QEdge_RB3,
                                      QEdge_LT3,
                                      QEdge_LB3,
                                      dtdx,
                                      dtdy,
                                      dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth - 2 and k < ksize - ghostWidth + 1 and
        j >= ghostWidth - 2 and j < jsize - ghostWidth + 1 and
        i >= ghostWidth - 2 and i < isize - ghostWidth + 1)
    // clang-format on
    {

      MHDState q;
      MHDState qPlusX, qMinusX, qPlusY, qMinusY, qPlusZ, qMinusZ;
      MHDState dq[3];

      real_t bfNb[6];
      real_t dbf[12];

      real_t elecFields[3][2][2];
      // alias to electric field components
      real_t(&Ex)[2][2] = elecFields[IX];
      real_t(&Ey)[2][2] = elecFields[IY];
      real_t(&Ez)[2][2] = elecFields[IZ];

      MHDState qm[THREE_D];
      MHDState qp[THREE_D];
      MHDState qEdge[4][3]; // array for qRT, qRB, qLT, qLB

      real_t xPos = params.xmin + params.dx / 2 + (i - ghostWidth) * params.dx;

      // clang-format off

      // get primitive variables state vector
      get_state(Qdata, i    , j    , k    , q);
      get_state(Qdata, i + 1, j    , k    , qPlusX);
      get_state(Qdata, i - 1, j    , k    , qMinusX);
      get_state(Qdata, i    , j + 1, k    , qPlusY);
      get_state(Qdata, i    , j - 1, k    , qMinusY);
      get_state(Qdata, i    , j    , k + 1, qPlusZ);
      get_state(Qdata, i    , j    , k - 1, qMinusZ);

      // get hydro slopes dq
      slope_unsplit_hydro_3d(q, qPlusX, qMinusX, qPlusY, qMinusY, qPlusZ, qMinusZ, dq);

      // get face-centered magnetic components
      bfNb[0] = Udata(i    , j    , k    , IA);
      bfNb[1] = Udata(i + 1, j    , k    , IA);
      bfNb[2] = Udata(i    , j    , k    , IB);
      bfNb[3] = Udata(i    , j + 1, k    , IB);
      bfNb[4] = Udata(i    , j    , k    , IC);
      bfNb[5] = Udata(i    , j    , k + 1, IC);

      // get dbf (transverse magnetic slopes)
      dbf[0] = DeltaA(i, j, k, IY);
      dbf[1] = DeltaA(i, j, k, IZ);
      dbf[2] = DeltaB(i, j, k, IX);
      dbf[3] = DeltaB(i, j, k, IZ);
      dbf[4] = DeltaC(i, j, k, IX);
      dbf[5] = DeltaC(i, j, k, IY);

      dbf[6]  = DeltaA(i + 1, j    , k    , IY);
      dbf[7]  = DeltaA(i + 1, j    , k    , IZ);
      dbf[8]  = DeltaB(i    , j + 1, k    , IX);
      dbf[9]  = DeltaB(i    , j + 1, k    , IZ);
      dbf[10] = DeltaC(i    , j    , k + 1, IX);
      dbf[11] = DeltaC(i    , j    , k + 1, IY);

      // get electric field components
      Ex[0][0] = ElecField(i    , j    , k    , IX); // ELL
      Ex[0][1] = ElecField(i    , j    , k + 1, IX); // ELR
      Ex[1][0] = ElecField(i    , j + 1, k    , IX); // ERL
      Ex[1][1] = ElecField(i    , j + 1, k + 1, IX); // ERR

      Ey[0][0] = ElecField(i    , j    , k    , IY); // FLL
      Ey[0][1] = ElecField(i    , j    , k + 1, IY); // FLR
      Ey[1][0] = ElecField(i + 1, j    , k    , IY); // FRL
      Ey[1][1] = ElecField(i + 1, j    , k + 1, IY); // FRR

      Ez[0][0] = ElecField(i    , j    , k    , IZ); // GLL
      Ez[0][1] = ElecField(i    , j + 1, k    , IZ); // GLR
      Ez[1][0] = ElecField(i + 1, j    , k    , IZ); // GRL
      Ez[1][1] = ElecField(i + 1, j + 1, k    , IZ); // GRR

      // clang-format on

      // compute qm, qp and qEdge
      trace_unsplit_mhd_3d_simpler(
        q, dq, bfNb, dbf, elecFields, dtdx, dtdy, dtdz, xPos, qm, qp, qEdge);

      // gravity predictor / modify velocity components
      // if (gravityEnabled) {

      // 	real_t grav_x = HALF_F * dt * h_gravity(i,j,k,IX);
      // 	real_t grav_y = HALF_F * dt * h_gravity(i,j,k,IY);
      // 	real_t grav_z = HALF_F * dt * h_gravity(i,j,k,IZ);

      // 	qm[0][IU] += grav_x; qm[0][IV] += grav_y; qm[0][IW] += grav_z;
      // 	qp[0][IU] += grav_x; qp[0][IV] += grav_y; qp[0][IW] += grav_z;

      // 	qm[1][IU] += grav_x; qm[1][IV] += grav_y; qm[1][IW] += grav_z;
      // 	qp[1][IU] += grav_x; qp[1][IV] += grav_y; qp[1][IW] += grav_z;

      // 	qm[2][IU] += grav_x; qm[2][IV] += grav_y; qm[2][IW] += grav_z;
      // 	qp[2][IU] += grav_x; qp[2][IV] += grav_y; qp[2][IW] += grav_z;

      // 	qEdge[IRT][0][IU] += grav_x;
      // 	qEdge[IRT][0][IV] += grav_y;
      // 	qEdge[IRT][0][IW] += grav_z;
      // 	qEdge[IRT][1][IU] += grav_x;
      // 	qEdge[IRT][1][IV] += grav_y;
      // 	qEdge[IRT][1][IW] += grav_z;
      // 	qEdge[IRT][2][IU] += grav_x;
      // 	qEdge[IRT][2][IV] += grav_y;
      // 	qEdge[IRT][2][IW] += grav_z;

      // 	qEdge[IRB][0][IU] += grav_x;
      // 	qEdge[IRB][0][IV] += grav_y;
      // 	qEdge[IRB][0][IW] += grav_z;
      // 	qEdge[IRB][1][IU] += grav_x;
      // 	qEdge[IRB][1][IV] += grav_y;
      // 	qEdge[IRB][1][IW] += grav_z;
      // 	qEdge[IRB][2][IU] += grav_x;
      // 	qEdge[IRB][2][IV] += grav_y;
      // 	qEdge[IRB][2][IW] += grav_z;

      // 	qEdge[ILT][0][IU] += grav_x;
      // 	qEdge[ILT][0][IV] += grav_y;
      // 	qEdge[ILT][0][IW] += grav_z;
      // 	qEdge[ILT][1][IU] += grav_x;
      // 	qEdge[ILT][1][IV] += grav_y;
      // 	qEdge[ILT][1][IW] += grav_z;
      // 	qEdge[ILT][2][IU] += grav_x;
      // 	qEdge[ILT][2][IV] += grav_y;
      // 	qEdge[ILT][2][IW] += grav_z;

      // 	qEdge[ILB][0][IU] += grav_x;
      // 	qEdge[ILB][0][IV] += grav_y;
      // 	qEdge[ILB][0][IW] += grav_z;
      // 	qEdge[ILB][1][IU] += grav_x;
      // 	qEdge[ILB][1][IV] += grav_y;
      // 	qEdge[ILB][1][IW] += grav_z;
      // 	qEdge[ILB][2][IU] += grav_x;
      // 	qEdge[ILB][2][IV] += grav_y;
      // 	qEdge[ILB][2][IW] += grav_z;

      // } // end gravity predictor

      // store qm, qp, qEdge : only what is really needed
      set_state(Qm_x, i, j, k, qm[0]);
      set_state(Qp_x, i, j, k, qp[0]);
      set_state(Qm_y, i, j, k, qm[1]);
      set_state(Qp_y, i, j, k, qp[1]);
      set_state(Qm_z, i, j, k, qm[2]);
      set_state(Qp_z, i, j, k, qp[2]);

      set_state(QEdge_RT, i, j, k, qEdge[IRT][0]);
      set_state(QEdge_RB, i, j, k, qEdge[IRB][0]);
      set_state(QEdge_LT, i, j, k, qEdge[ILT][0]);
      set_state(QEdge_LB, i, j, k, qEdge[ILB][0]);

      set_state(QEdge_RT2, i, j, k, qEdge[IRT][1]);
      set_state(QEdge_RB2, i, j, k, qEdge[IRB][1]);
      set_state(QEdge_LT2, i, j, k, qEdge[ILT][1]);
      set_state(QEdge_LB2, i, j, k, qEdge[ILB][1]);

      set_state(QEdge_RT3, i, j, k, qEdge[IRT][2]);
      set_state(QEdge_RB3, i, j, k, qEdge[IRB][2]);
      set_state(QEdge_LT3, i, j, k, qEdge[ILT][2]);
      set_state(QEdge_LB3, i, j, k, qEdge[ILB][2]);
    }

  } // operator ()

  DataArray3d      Udata, Qdata;
  DataArrayVector3 DeltaA, DeltaB, DeltaC;
  DataArray3d      ElecField;
  DataArray3d      Qm_x, Qm_y, Qm_z;
  DataArray3d      Qp_x, Qp_y, Qp_z;
  DataArray3d      QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  DataArray3d      QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d      QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  real_t           dtdx, dtdy, dtdz;

}; // class ComputeTraceFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute cell-centered primitive data at t_{n+1/2} (half time step in
 * Muscl-Hancock).
 */
class ComputeUpdatedPrimVarFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  ComputeUpdatedPrimVarFunctor3D_MHD(HydroParams params,
                                     DataArray3d Udata,
                                     DataArray3d Qdata,
                                     DataArray3d Slopes_x,
                                     DataArray3d Slopes_y,
                                     DataArray3d Slopes_z,
                                     DataArray3d Qdata2,
                                     real_t      dtdx,
                                     real_t      dtdy,
                                     real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Slopes_z(Slopes_z)
    , Qdata2(Qdata2)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Udata,
        DataArray3d Qdata,
        DataArray3d Slopes_x,
        DataArray3d Slopes_y,
        DataArray3d Slopes_z,
        DataArray3d Qdata2,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ComputeUpdatedPrimVarFunctor3D_MHD functor(
      params, Udata, Qdata, Slopes_x, Slopes_y, Slopes_z, Qdata2, dtdx, dtdy, dtdz);
    Kokkos::parallel_for("ComputeUpdatedPrimVarFunctor3D_MHD",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    const int    isize = params.isize;
    const int    jsize = params.jsize;
    const int    ksize = params.ksize;
    const int    ghostWidth = params.ghostWidth;
    const real_t gamma = params.settings.gamma0;

    // clang-format off
    if (k >= ghostWidth - 2 and k < ksize - ghostWidth + 1 and
        j >= ghostWidth - 2 and j < jsize - ghostWidth + 1 and
        i >= ghostWidth - 2 and i < isize - ghostWidth + 1)
    // clang-format on
    {
      // get primitive variable in current cell
      MHDState q;
      get_state(Qdata, i, j, k, q);

      MHDState dq[3];

      // retrieve hydro slopes along X
      get_state(Slopes_x, i, j, k, dq[IX]);

      // retrieve hydro slopes along Y
      get_state(Slopes_y, i, j, k, dq[IY]);

      // retrieve hydro slopes along Z
      get_state(Slopes_y, i, j, k, dq[IZ]);

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
      auto const & dCx = dq[IX][IC];
      auto const & dBx = dq[IX][IB];

      // Cell centered TVD slopes in Y direction
      auto const & dry = dq[IY][ID];
      auto const & dpy = dq[IY][IP];
      auto const & duy = dq[IY][IU];
      auto const & dvy = dq[IY][IV];
      auto const & dwy = dq[IY][IW];
      auto const & dCy = dq[IY][IC];
      auto const & dAy = dq[IY][IA];

      // Cell centered TVD slopes in Z direction
      auto const & drz = dq[IZ][ID];
      auto const & dpz = dq[IZ][IP];
      auto const & duz = dq[IZ][IU];
      auto const & dvz = dq[IZ][IV];
      auto const & dwz = dq[IZ][IW];
      auto const & dAz = dq[IZ][IA];
      auto const & dBz = dq[IZ][IB];

      auto const   db = compute_normal_mag_field_slopes(Udata, i, j, k);
      auto const & dAx = db[IX];
      auto const & dBy = db[IY];
      auto const & dCz = db[IZ];

      real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
      {

        sr0 =
          (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
        su0 = (-u * dux - (dpx + B * dBx + C * dCx) / r) * dtdx + (-v * duy + B * dAy / r) * dtdy +
              (-w * duz + C * dAz / r) * dtdz;
        sv0 = (-u * dvx + A * dBx / r) * dtdx + (-v * dvy - (dpy + A * dAy + C * dCy) / r) * dtdy +
              (-w * dvz + C * dBz / r) * dtdz;
        sw0 = (-u * dwx + A * dCx / r) * dtdx + (-v * dwy + B * dCy / r) * dtdy +
              (-w * dwz - (dpz + A * dAz + B * dBz) / r) * dtdz;
        sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy +
              (-w * dpz - dwz * gamma * p) * dtdz;
        sA0 = (u * dBy + B * duy - v * dAy - A * dvy) * dtdy +
              (u * dCz + C * duz - w * dAz - A * dwz) * dtdz;
        sB0 = (v * dAx + A * dvx - u * dBx - B * dux) * dtdx +
              (v * dCz + C * dvz - w * dBz - B * dwz) * dtdz;
        sC0 = (w * dAx + A * dwx - u * dCx - C * dux) * dtdx +
              (w * dBy + B * dwy - v * dCy - C * dvy) * dtdy;
      }

      // Update in time the primitive variables (half time step)
      Qdata2(i, j, k, ID) = r + 0.5 * sr0;
      Qdata2(i, j, k, IU) = u + 0.5 * su0;
      Qdata2(i, j, k, IV) = v + 0.5 * sv0;
      Qdata2(i, j, k, IW) = w + 0.5 * sw0;
      Qdata2(i, j, k, IP) = p + 0.5 * sp0;
      Qdata2(i, j, k, IA) = A + 0.5 * sA0;
      Qdata2(i, j, k, IB) = B + 0.5 * sB0;
      Qdata2(i, j, k, IC) = C + 0.5 * sC0;
    }
  } // operator ()

  DataArray3d Udata, Qdata; // input
  DataArray3d Slopes_x, Slopes_y, Slopes_z; // input
  DataArray3d Qdata2;       // output
  real_t      dtdx, dtdy, dtdz;

}; // ComputeUpdatedPrimvarFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Reconstruct hydro state at cell face center, solve Riemann problems and update.
 *
 * This is done in a direction by direction manner.
 */
template <Direction dir>
class ComputeFluxAndUpdateAlongDirFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  //!
  //! \param[in] Udata_in is conservative variables at t_n
  //! \param[in] Udata_out is conservative variables at t_{n+1}
  //! \param[in] Qdata is necessary to recompute limited slopes
  //! \param[in] Qdata2 is primitive variables array at t_{n+1/2}
  //!
  ComputeFluxAndUpdateAlongDirFunctor3D_MHD(HydroParams params,
                                            DataArray3d Udata_in,
                                            DataArray3d Udata_out,
                                            DataArray3d Qdata,
                                            DataArray3d Qdata2,
                                            DataArray3d Slopes_x,
                                            DataArray3d Slopes_y,
                                            DataArray3d Slopes_z,
                                            DataArray3d sFaceMag,
                                            real_t      dtdx,
                                            real_t      dtdy,
                                            real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , Udata_in(Udata_in)
    , Udata_out(Udata_out)
    , Qdata(Qdata)
    , Qdata2(Qdata2)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Slopes_z(Slopes_z)
    , sFaceMag(sFaceMag)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Udata_in,
        DataArray3d Udata_out,
        DataArray3d Qdata,
        DataArray3d Qdata2,
        DataArray3d Slopes_x,
        DataArray3d Slopes_y,
        DataArray3d Slopes_z,
        DataArray3d sFaceMag,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ComputeFluxAndUpdateAlongDirFunctor3D_MHD<dir> functor(params,
                                                           Udata_in,
                                                           Udata_out,
                                                           Qdata,
                                                           Qdata2,
                                                           Slopes_x,
                                                           Slopes_y,
                                                           Slopes_z,
                                                           sFaceMag,
                                                           dtdx,
                                                           dtdy,
                                                           dtdz);
    Kokkos::parallel_for("ComputeFluxAndUpdateAlongDirFunctor3D_MHD",
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

    // const real_t gamma = params.settings.gamma0;
    const real_t smallR = params.settings.smallr;
    const real_t smallp = params.settings.smallp;

    constexpr auto delta_i = dir == DIR_X ? 1 : 0;
    constexpr auto delta_j = dir == DIR_Y ? 1 : 0;
    constexpr auto delta_k = dir == DIR_Z ? 1 : 0;

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth + 1 and
        j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {
      MHDState dq, dqN;

      // cell-centered primitive variables in current cell, and left and right neighbor along dir
      MHDState q;
      get_state(Qdata, i, j, k, q);

      // cell-centered primitive variables in neighbor cell
      MHDState qN;
      get_state(Qdata, i - delta_i, j - delta_j, k - delta_k, qN);

      // left and right reconstructed state (input for Riemann solver)
      MHDState qR, qL;

      //
      // Right state at left interface (current cell)
      //

      // load hydro slopes along dir
      if constexpr (dir == DIR_X)
      {
        get_state(Slopes_x, i, j, k, dq);
      }
      else if constexpr (dir == DIR_Y)
      {
        get_state(Slopes_y, i, j, k, dq);
      }
      else if constexpr (dir == DIR_Z)
      {
        get_state(Slopes_z, i, j, k, dq);
      }

      // get primitive variable in current cell at t_{n+1/2}
      MHDState q2;
      get_state(Qdata2, i, j, k, q2);

      qR[ID] = q2[ID] - 0.5 * dq[ID];
      qR[IU] = q2[IU] - 0.5 * dq[IU];
      qR[IV] = q2[IV] - 0.5 * dq[IV];
      qR[IW] = q2[IW] - 0.5 * dq[IW];
      qR[IP] = q2[IP] - 0.5 * dq[IP];
      qR[ID] = fmax(smallR, qR[ID]);
      qR[IP] = fmax(smallp * qR[ID], qR[IP]);
      // clang-format off
      if constexpr (dir == DIR_X)
      {
        const real_t AL = Udata_in(i, j, k, IA) + sFaceMag(i, j, k, IX);
        qR[IA] = AL;
        qR[IB] = q2[IB] - 0.5 * dq[IB];
        qR[IC] = q2[IC] - 0.5 * dq[IC];
      }
      else if constexpr (dir == DIR_Y)
      {
        const real_t BL = Udata_in(i, j, k, IB) + sFaceMag(i, j, k, IY);
        qR[IA] = q2[IA] - 0.5 * dq[IA];
        qR[IB] = BL;
        qR[IC] = q2[IC] - 0.5 * dq[IC];
      }
      else if constexpr (dir == DIR_Z)
      {
        const real_t CL = Udata_in(i, j, k, IC) + sFaceMag(i, j, k, IZ);
        qR[IA] = q2[IA] - 0.5 * dq[IA];
        qR[IB] = q2[IB] - 0.5 * dq[IB];
        qR[IC] = CL;
      }
      // clang-format on

      //
      // Left state at right interface (neighbor cell)
      //

      // load hydro slopes along dir
      if constexpr (dir == DIR_X)
      {
        get_state(Slopes_x, i - delta_i, j - delta_j, k - delta_k, dqN);
      }
      else if constexpr (dir == DIR_Y)
      {
        get_state(Slopes_y, i - delta_i, j - delta_j, k - delta_k, dqN);
      }
      else if constexpr (dir == DIR_Z)
      {
        get_state(Slopes_z, i - delta_i, j - delta_j, k - delta_k, dqN);
      }

      // get primitive variable in neighbor cell at t_{n+1/2}
      MHDState q2N;
      get_state(Qdata2, i - delta_i, j - delta_j, k - delta_k, q2N);

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
      else if constexpr (dir == DIR_Z)
      {
        qL[IA] = q2N[IA] + 0.5 * dqN[IA];
        qL[IB] = q2N[IB] + 0.5 * dqN[IB];
        qL[IC] = qR[IC];
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
      else if constexpr (dir == DIR_Z)
      {
        swapValues(&(qL[IU]), &(qL[IW]));
        swapValues(&(qL[IA]), &(qL[IC]));
        swapValues(&(qR[IU]), &(qR[IW]));
        swapValues(&(qR[IA]), &(qR[IC]));
      }
      riemann_mhd(qL, qR, flux, params);
      if constexpr (dir == DIR_Y)
      {
        swapValues(&(flux[IU]), &(flux[IV]));
        swapValues(&(flux[IA]), &(flux[IB]));
      }
      else if constexpr (dir == DIR_Z)
      {
        swapValues(&(flux[IU]), &(flux[IW]));
        swapValues(&(flux[IA]), &(flux[IC]));
      }

      if constexpr (dir == DIR_X)
      {
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i, j, k, ID), flux[ID] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, k, IP), flux[IP] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, k, IU), flux[IU] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, k, IV), flux[IV] * dtdx);
          Kokkos::atomic_add(&Udata_out(i, j, k, IW), flux[IW] * dtdx);
        }

        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i - 1, j, k, ID), flux[ID] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, k, IP), flux[IP] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, k, IU), flux[IU] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, k, IV), flux[IV] * dtdx);
          Kokkos::atomic_sub(&Udata_out(i - 1, j, k, IW), flux[IW] * dtdx);
        }
      }
      else if constexpr (dir == DIR_Y)
      {
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i, j, k, ID), flux[ID] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, k, IP), flux[IP] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, k, IU), flux[IU] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, k, IV), flux[IV] * dtdy);
          Kokkos::atomic_add(&Udata_out(i, j, k, IW), flux[IW] * dtdy);
        }
        if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i, j - 1, k, ID), flux[ID] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, k, IP), flux[IP] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, k, IU), flux[IU] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, k, IV), flux[IV] * dtdy);
          Kokkos::atomic_sub(&Udata_out(i, j - 1, k, IW), flux[IW] * dtdy);
        }
      }
      else if constexpr (dir == DIR_Z)
      {
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i, j, k, ID), flux[ID] * dtdz);
          Kokkos::atomic_add(&Udata_out(i, j, k, IP), flux[IP] * dtdz);
          Kokkos::atomic_add(&Udata_out(i, j, k, IU), flux[IU] * dtdz);
          Kokkos::atomic_add(&Udata_out(i, j, k, IV), flux[IV] * dtdz);
          Kokkos::atomic_add(&Udata_out(i, j, k, IW), flux[IW] * dtdz);
        }
        if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i, j, k - 1, ID), flux[ID] * dtdz);
          Kokkos::atomic_sub(&Udata_out(i, j, k - 1, IP), flux[IP] * dtdz);
          Kokkos::atomic_sub(&Udata_out(i, j, k - 1, IU), flux[IU] * dtdz);
          Kokkos::atomic_sub(&Udata_out(i, j, k - 1, IV), flux[IV] * dtdz);
          Kokkos::atomic_sub(&Udata_out(i, j, k - 1, IW), flux[IW] * dtdz);
        }
      }
    }
  } // operator ()

  DataArray3d Udata_in, Udata_out;
  DataArray3d Qdata, Qdata2;
  DataArray3d Slopes_x, Slopes_y, Slopes_z;
  DataArray3d sFaceMag;
  real_t      dtdx, dtdy, dtdz;

}; // ComputeFluxAndUpdateAlongDirFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Reconstruct hydrodynamics variables and magnetic field at edge center, compute Emf and update.
 */
template <Direction dir>
class ReconstructEdgeComputeEmfAndUpdateFunctor3D : public MHDBaseFunctor3D
{

public:
  //!
  //! \param[in] Udata_in is conservative variables at t_n
  //! \param[in] Udata_out is conservative variables at t_{n+1}
  //! \param[in] Qdata is necessary to recompute limited slopes
  //! \param[in] Qdata2 is primitive variables array at t_{n+1/2}
  //!
  ReconstructEdgeComputeEmfAndUpdateFunctor3D(HydroParams params,
                                              DataArray3d Udata_in,
                                              DataArray3d Udata_out,
                                              DataArray3d Qdata,
                                              DataArray3d Qdata2,
                                              DataArray3d Slopes_x,
                                              DataArray3d Slopes_y,
                                              DataArray3d Slopes_z,
                                              DataArray3d sFaceMag,
                                              real_t      dtdx,
                                              real_t      dtdy,
                                              real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , Udata_in(Udata_in)
    , Udata_out(Udata_out)
    , Qdata(Qdata)
    , Qdata2(Qdata2)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Slopes_z(Slopes_z)
    , sFaceMag(sFaceMag)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Udata_in,
        DataArray3d Udata_out,
        DataArray3d Qdata,
        DataArray3d Qdata2,
        DataArray3d Slopes_x,
        DataArray3d Slopes_y,
        DataArray3d Slopes_z,
        DataArray3d sFaceMag,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ReconstructEdgeComputeEmfAndUpdateFunctor3D functor(params,
                                                        Udata_in,
                                                        Udata_out,
                                                        Qdata,
                                                        Qdata2,
                                                        Slopes_x,
                                                        Slopes_y,
                                                        Slopes_z,
                                                        sFaceMag,
                                                        dtdx,
                                                        dtdy,
                                                        dtdz);
    Kokkos::parallel_for("ReconstructEdgeComputeEmfAndUpdateFunctor3D",
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

    // const real_t gamma = params.settings.gamma0;
    const real_t smallR = params.settings.smallr;
    const real_t smallp = params.settings.smallp;

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth + 1 and
        j >= ghostWidth and j < jsize - ghostWidth + 1 and
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


      // qLB is current cell, qLT, qRT and qRB are direct neighbors surrounding the lower left edge
      // MHDState qLB, qLT, qRB, qRT;
      MHDState   qEdge_emf[4];
      MHDState & qRT = qEdge_emf[IRT];
      MHDState & qLT = qEdge_emf[ILT];
      MHDState & qRB = qEdge_emf[IRB];
      MHDState & qLB = qEdge_emf[ILB];

      // get primitive variable in current cell and neighbors at t_{n+1/2}
      // clang-format off
      if constexpr (dir == DIR_Z)
      {
        get_state(Qdata2, i    , j    , k    , qLB);
        get_state(Qdata2, i    , j - 1, k    , qLT);
        get_state(Qdata2, i - 1, j    , k    , qRB);
        get_state(Qdata2, i - 1, j - 1, k    , qRT);
      }
      else if constexpr (dir == DIR_Y)
      {
        get_state(Qdata2, i    , j    , k    , qLB);
        get_state(Qdata2, i    , j    , k - 1, qLT); // later LT and RB will swapped
        get_state(Qdata2, i - 1, j    , k    , qRB); // later LT and RB will swapped
        get_state(Qdata2, i - 1, j    , k - 1, qRT);
      }
      else if constexpr (dir == DIR_X)
      {
        get_state(Qdata2, i    , j    , k    , qLB);
        get_state(Qdata2, i    , j    , k - 1, qLT);
        get_state(Qdata2, i    , j - 1, k    , qRB);
        get_state(Qdata2, i    , j - 1, k - 1, qRT);
      }
      // clang-format on

      // reconstruct edge states using limited slopes
      MHDState dqX, dqY, dqZ;

      if constexpr (dir == DIR_Z)
      {
        // LB at (i,j,k)
        {
          const auto i0 = i;
          const auto j0 = j;
          const auto k0 = k;

          const real_t AL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IX);
          const real_t dALy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IA);

          const real_t BL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IY);
          const real_t dBLx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IB);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_y, i0, j0, k0, dqY);
          qLB[ID] += 0.5 * (-dqX[ID] - dqY[ID]);
          qLB[IU] += 0.5 * (-dqX[IU] - dqY[IU]);
          qLB[IV] += 0.5 * (-dqX[IV] - dqY[IV]);
          qLB[IW] += 0.5 * (-dqX[IW] - dqY[IW]);
          qLB[IP] += 0.5 * (-dqX[IP] - dqY[IP]);
          qLB[IA] = AL + 0.5 * (-dALy);
          qLB[IB] = BL + 0.5 * (-dBLx);
          qLB[IC] += 0.5 * (-dqX[IC] - dqY[IC]);
          qLB[ID] = fmax(smallR, qLB[ID]);
          qLB[IP] = fmax(smallp * qLB[ID], qLB[IP]);
        }

        // RT (i-1, j-1, k)
        {
          const auto i0 = i - 1;
          const auto j0 = j - 1;
          const auto k0 = k;

          const real_t AR =
            Udata_in(i0 + 1, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 1, j0 + 0, k0 + 0, IX);
          const real_t dARy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 1, j0 + 0, k0 + 0, IA);

          const real_t BR =
            Udata_in(i0 + 0, j0 + 1, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 1, k0 + 0, IY);
          const real_t dBRx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 1, k0 + 0, IB);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_y, i0, j0, k0, dqY);
          qRT[ID] += 0.5 * (+dqX[ID] + dqY[ID]);
          qRT[IU] += 0.5 * (+dqX[IU] + dqY[IU]);
          qRT[IV] += 0.5 * (+dqX[IV] + dqY[IV]);
          qRT[IW] += 0.5 * (+dqX[IW] + dqY[IW]);
          qRT[IP] += 0.5 * (+dqX[IP] + dqY[IP]);
          qRT[IA] = AR + 0.5 * (+dARy);
          qRT[IB] = BR + 0.5 * (+dBRx);
          qRT[IC] += 0.5 * (+dqX[IC] + dqY[IC]);
          qRT[ID] = fmax(smallR, qRT[ID]);
          qRT[IP] = fmax(smallp * qRT[ID], qRT[IP]);
        }

        // RB (i-1, j, k)
        {
          const auto i0 = i - 1;
          const auto j0 = j;
          const auto k0 = k;

          const real_t AR =
            Udata_in(i0 + 1, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 1, j0 + 0, k0 + 0, IX);
          const real_t dARy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 1, j0 + 0, k0 + 0, IA);

          const real_t BL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IY);
          const real_t dBLx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IB);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_y, i0, j0, k0, dqY);
          qRB[ID] += 0.5 * (+dqX[ID] - dqY[ID]);
          qRB[IU] += 0.5 * (+dqX[IU] - dqY[IU]);
          qRB[IV] += 0.5 * (+dqX[IV] - dqY[IV]);
          qRB[IW] += 0.5 * (+dqX[IW] - dqY[IW]);
          qRB[IP] += 0.5 * (+dqX[IP] - dqY[IP]);
          qRB[IA] = AR + 0.5 * (-dARy);
          qRB[IB] = BL + 0.5 * (+dBLx);
          qRB[IC] += 0.5 * (+dqX[IC] - dqY[IC]);
          qRB[ID] = fmax(smallR, qRB[ID]);
          qRB[IP] = fmax(smallp * qRB[ID], qRB[IP]);
        }

        // LT (i, j-1, k)
        {
          const auto i0 = i;
          const auto j0 = j - 1;
          const auto k0 = k;

          const real_t AL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IX);
          const real_t dALy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IA);

          const real_t BR =
            Udata_in(i0 + 0, j0 + 1, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 1, k0 + 0, IY);
          const real_t dBRx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 1, k0 + 0, IB);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_y, i0, j0, k0, dqY);
          qLT[ID] += 0.5 * (-dqX[ID] + dqY[ID]);
          qLT[IU] += 0.5 * (-dqX[IU] + dqY[IU]);
          qLT[IV] += 0.5 * (-dqX[IV] + dqY[IV]);
          qLT[IW] += 0.5 * (-dqX[IW] + dqY[IW]);
          qLT[IP] += 0.5 * (-dqX[IP] + dqY[IP]);
          qLT[IA] = AL + 0.5 * (+dALy);
          qLT[IB] = BR + 0.5 * (-dBRx);
          qLT[IC] += 0.5 * (-dqX[IC] + dqY[IC]);
          qLT[ID] = fmax(smallR, qLT[ID]);
          qLT[IP] = fmax(smallp * qLT[ID], qLT[IP]);
        }

        const real_t emfZ = compute_emf<EMFZ>(qEdge_emf, params);

        // clang-format off
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i    , j    , k     , IA), emfZ * dtdy);
          Kokkos::atomic_add(&Udata_out(i    , j    , k     , IB), emfZ * dtdx);
        }

        if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i    , j - 1, k     , IA), emfZ * dtdy);
        }
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i - 1, j    , k     , IB), emfZ * dtdx);
        }
        // clang-format on
      }
      else if constexpr (dir == DIR_Y)
      {
        // LB at (i,j,k)
        {
          const auto i0 = i;
          const auto j0 = j;
          const auto k0 = k;

          const real_t AL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IX);
          const real_t dALz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IA);

          const real_t CL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IZ);
          const real_t dCLx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IC);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qLB[ID] += 0.5 * (-dqX[ID] - dqZ[ID]);
          qLB[IU] += 0.5 * (-dqX[IU] - dqZ[IU]);
          qLB[IV] += 0.5 * (-dqX[IV] - dqZ[IV]);
          qLB[IW] += 0.5 * (-dqX[IW] - dqZ[IW]);
          qLB[IP] += 0.5 * (-dqX[IP] - dqZ[IP]);
          qLB[IA] = AL + 0.5 * (-dALz);
          qLB[IB] += 0.5 * (-dqX[IB] - dqZ[IB]);
          qLB[IC] = CL + 0.5 * (-dCLx);
          qLB[ID] = fmax(smallR, qLB[ID]);
          qLB[IP] = fmax(smallp * qLB[ID], qLB[IP]);
        }

        // RT (i-1, j, k-1)
        {
          const auto i0 = i - 1;
          const auto j0 = j;
          const auto k0 = k - 1;

          const real_t AR =
            Udata_in(i0 + 1, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 1, j0 + 0, k0 + 0, IX);
          const real_t dARz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 1, j0 + 0, k0 + 0, IA);

          const real_t CR =
            Udata_in(i0 + 0, j0 + 0, k0 + 1, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 1, IZ);
          const real_t dCRx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 0, k0 + 1, IC);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qRT[ID] += 0.5 * (+dqX[ID] + dqZ[ID]);
          qRT[IU] += 0.5 * (+dqX[IU] + dqZ[IU]);
          qRT[IV] += 0.5 * (+dqX[IV] + dqZ[IV]);
          qRT[IW] += 0.5 * (+dqX[IW] + dqZ[IW]);
          qRT[IP] += 0.5 * (+dqX[IP] + dqZ[IP]);
          qRT[IA] = AR + 0.5 * (+dARz);
          qRT[IB] += 0.5 * (+dqX[IB] + dqZ[IB]);
          qRT[IC] = CR + 0.5 * (+dCRx);
          qRT[ID] = fmax(smallR, qRT[ID]);
          qRT[IP] = fmax(smallp * qRT[ID], qRT[IP]);
        }

        // RB (i-1, j, k)
        {
          const auto i0 = i - 1;
          const auto j0 = j;
          const auto k0 = k;

          const real_t AR =
            Udata_in(i0 + 1, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 1, j0 + 0, k0 + 0, IX);
          const real_t dARz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 1, j0 + 0, k0 + 0, IA);

          const real_t CL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IZ);
          const real_t dCLx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IC);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qRB[ID] += 0.5 * (+dqX[ID] - dqZ[ID]);
          qRB[IU] += 0.5 * (+dqX[IU] - dqZ[IU]);
          qRB[IV] += 0.5 * (+dqX[IV] - dqZ[IV]);
          qRB[IW] += 0.5 * (+dqX[IW] - dqZ[IW]);
          qRB[IP] += 0.5 * (+dqX[IP] - dqZ[IP]);
          qRB[IA] = AR + 0.5 * (-dARz);
          qRB[IB] += 0.5 * (+dqX[IB] - dqZ[IB]);
          qRB[IC] = CL + 0.5 * (+dCLx);
          qRB[ID] = fmax(smallR, qRB[ID]);
          qRB[IP] = fmax(smallp * qRB[ID], qRB[IP]);
        }

        // LT (i, j, k-1)
        {
          const auto i0 = i;
          const auto j0 = j;
          const auto k0 = k - 1;

          const real_t AL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IA) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IX);
          const real_t dALz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IA);

          const real_t CR =
            Udata_in(i0 + 0, j0 + 0, k0 + 1, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 1, IZ);
          const real_t dCRx = compute_limited_slope<DIR_X>(Udata_in, i0 + 0, j0 + 0, k0 + 1, IC);

          get_state(Slopes_x, i0, j0, k0, dqX);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qLT[ID] += 0.5 * (-dqX[ID] + dqZ[ID]);
          qLT[IU] += 0.5 * (-dqX[IU] + dqZ[IU]);
          qLT[IV] += 0.5 * (-dqX[IV] + dqZ[IV]);
          qLT[IW] += 0.5 * (-dqX[IW] + dqZ[IW]);
          qLT[IP] += 0.5 * (-dqX[IP] + dqZ[IP]);
          qLT[IA] = AL + 0.5 * (+dALz);
          qLT[IB] += 0.5 * (-dqX[IB] + dqZ[IB]);
          qLT[IC] = CR + 0.5 * (-dCRx);
          qLT[ID] = fmax(smallR, qLT[ID]);
          qLT[IP] = fmax(smallp * qLT[ID], qLT[IP]);
        }

        // exchange RB and LT
        {
          swapValues(&(qRB[ID]), &(qLT[ID]));
          swapValues(&(qRB[IU]), &(qLT[IU]));
          swapValues(&(qRB[IV]), &(qLT[IV]));
          swapValues(&(qRB[IW]), &(qLT[IW]));
          swapValues(&(qRB[IP]), &(qLT[IP]));
          swapValues(&(qRB[IA]), &(qLT[IA]));
          swapValues(&(qRB[IB]), &(qLT[IB]));
          swapValues(&(qRB[IC]), &(qLT[IC]));
        }
        const real_t emfY = compute_emf<EMFY>(qEdge_emf, params);

        // clang-format off
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i    , j    , k     , IA), emfY * dtdz);
          Kokkos::atomic_sub(&Udata_out(i    , j    , k     , IC), emfY * dtdx);
        }
        if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i    , j    , k - 1 , IA), emfY * dtdz);
        }
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i - 1, j    , k     , IC), emfY * dtdx);
        }
        // clang-format on
      }
      else if constexpr (dir == DIR_X)
      {
        // LB at (i,j,k)
        {
          const auto i0 = i;
          const auto j0 = j;
          const auto k0 = k;

          const real_t BL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IY);
          const real_t dBLz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IB);

          const real_t CL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IZ);
          const real_t dCLy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IC);

          get_state(Slopes_y, i0, j0, k0, dqY);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qLB[ID] += 0.5 * (-dqY[ID] - dqZ[ID]);
          qLB[IU] += 0.5 * (-dqY[IU] - dqZ[IU]);
          qLB[IV] += 0.5 * (-dqY[IV] - dqZ[IV]);
          qLB[IW] += 0.5 * (-dqY[IW] - dqZ[IW]);
          qLB[IP] += 0.5 * (-dqY[IP] - dqZ[IP]);
          qLB[IA] += 0.5 * (-dqY[IA] - dqZ[IA]);
          qLB[IB] = BL + 0.5 * (-dBLz);
          qLB[IC] = CL + 0.5 * (-dCLy);
          qLB[ID] = fmax(smallR, qLB[ID]);
          qLB[IP] = fmax(smallp * qLB[ID], qLB[IP]);
        }

        // RT (i, j-1, k-1)
        {
          const auto i0 = i;
          const auto j0 = j - 1;
          const auto k0 = k - 1;

          const real_t BR =
            Udata_in(i0 + 0, j0 + 1, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 1, k0 + 0, IY);
          const real_t dBRz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 0, j0 + 1, k0 + 0, IB);

          const real_t CR =
            Udata_in(i0 + 0, j0 + 0, k0 + 1, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 1, IZ);
          const real_t dCRy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 0, j0 + 0, k0 + 1, IC);

          get_state(Slopes_y, i0, j0, k0, dqY);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qRT[ID] += 0.5 * (+dqY[ID] + dqZ[ID]);
          qRT[IU] += 0.5 * (+dqY[IU] + dqZ[IU]);
          qRT[IV] += 0.5 * (+dqY[IV] + dqZ[IV]);
          qRT[IW] += 0.5 * (+dqY[IW] + dqZ[IW]);
          qRT[IP] += 0.5 * (+dqY[IP] + dqZ[IP]);
          qRT[IA] += 0.5 * (+dqY[IA] + dqZ[IA]);
          qRT[IB] = BR + 0.5 * (+dBRz);
          qRT[IC] = CR + 0.5 * (+dCRy);
          qRT[ID] = fmax(smallR, qRT[ID]);
          qRT[IP] = fmax(smallp * qRT[ID], qRT[IP]);
        }

        // RB (i, j-1, k)
        {
          const auto i0 = i;
          const auto j0 = j - 1;
          const auto k0 = k;

          const real_t BR =
            Udata_in(i0 + 0, j0 + 1, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 1, k0 + 0, IY);
          const real_t dBRz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 0, j0 + 1, k0 + 0, IB);

          const real_t CL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IZ);
          const real_t dCLy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IC);

          get_state(Slopes_y, i0, j0, k0, dqY);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qRB[ID] += 0.5 * (+dqY[ID] - dqZ[ID]);
          qRB[IU] += 0.5 * (+dqY[IU] - dqZ[IU]);
          qRB[IV] += 0.5 * (+dqY[IV] - dqZ[IV]);
          qRB[IW] += 0.5 * (+dqY[IW] - dqZ[IW]);
          qRB[IP] += 0.5 * (+dqY[IP] - dqZ[IP]);
          qRB[IA] += 0.5 * (+dqY[IA] - dqZ[IA]);
          qRB[IB] = BR + 0.5 * (-dBRz);
          qRB[IC] = CL + 0.5 * (+dCLy);
          qRB[ID] = fmax(smallR, qRB[ID]);
          qRB[IP] = fmax(smallp * qRB[ID], qRB[IP]);
        }

        // LT (i, j, k-1)
        {
          const auto i0 = i;
          const auto j0 = j;
          const auto k0 = k - 1;

          const real_t BL =
            Udata_in(i0 + 0, j0 + 0, k0 + 0, IB) + sFaceMag(i0 + 0, j0 + 0, k0 + 0, IY);
          const real_t dBLz = compute_limited_slope<DIR_Z>(Udata_in, i0 + 0, j0 + 0, k0 + 0, IB);

          const real_t CR =
            Udata_in(i0 + 0, j0 + 0, k0 + 1, IC) + sFaceMag(i0 + 0, j0 + 0, k0 + 1, IZ);
          const real_t dCRy = compute_limited_slope<DIR_Y>(Udata_in, i0 + 0, j0 + 0, k0 + 1, IC);

          get_state(Slopes_y, i0, j0, k0, dqY);
          get_state(Slopes_z, i0, j0, k0, dqZ);
          qLT[ID] += 0.5 * (-dqY[ID] + dqZ[ID]);
          qLT[IU] += 0.5 * (-dqY[IU] + dqZ[IU]);
          qLT[IV] += 0.5 * (-dqY[IV] + dqZ[IV]);
          qLT[IW] += 0.5 * (-dqY[IW] + dqZ[IW]);
          qLT[IP] += 0.5 * (-dqY[IP] + dqZ[IP]);
          qLT[IA] += 0.5 * (-dqY[IA] + dqZ[IA]);
          qLT[IB] = BL + 0.5 * (+dBLz);
          qLT[IC] = CR + 0.5 * (-dCRy);
          qLT[ID] = fmax(smallR, qLT[ID]);
          qLT[IP] = fmax(smallp * qLT[ID], qLT[IP]);
        }

        const real_t emfX = compute_emf<EMFX>(qEdge_emf, params);

        // clang-format off
        if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i    , j    , k     , IB), emfX * dtdz);
          Kokkos::atomic_add(&Udata_out(i    , j    , k     , IC), emfX * dtdy);
        }
        if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_add(&Udata_out(i    , j    , k - 1 , IB), emfX * dtdz);
        }
        if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
        {
          Kokkos::atomic_sub(&Udata_out(i    , j - 1, k     , IC), emfX * dtdy);
        }
        // clang-format on
      }
    }
  } // operator ()

  DataArray3d Udata_in, Udata_out;
  DataArray3d Qdata, Qdata2;
  DataArray3d Slopes_x, Slopes_y, Slopes_z;
  DataArray3d sFaceMag;
  real_t      dtdx, dtdy, dtdz;

}; // ComputeEmfAndUpdateFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndStoreFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  ComputeFluxesAndStoreFunctor3D_MHD(HydroParams params,
                                     DataArray3d Qm_x,
                                     DataArray3d Qm_y,
                                     DataArray3d Qm_z,
                                     DataArray3d Qp_x,
                                     DataArray3d Qp_y,
                                     DataArray3d Qp_z,
                                     DataArray3d Fluxes_x,
                                     DataArray3d Fluxes_y,
                                     DataArray3d Fluxes_z,
                                     real_t      dtdx,
                                     real_t      dtdy,
                                     real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qm_z(Qm_z)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , Qp_z(Qp_z)
    , Fluxes_x(Fluxes_x)
    , Fluxes_y(Fluxes_y)
    , Fluxes_z(Fluxes_z)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Qm_x,
        DataArray3d Qm_y,
        DataArray3d Qm_z,
        DataArray3d Qp_x,
        DataArray3d Qp_y,
        DataArray3d Qp_z,
        DataArray3d Flux_x,
        DataArray3d Flux_y,
        DataArray3d Flux_z,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ComputeFluxesAndStoreFunctor3D_MHD functor(
      params, Qm_x, Qm_y, Qm_z, Qp_x, Qp_y, Qp_z, Flux_x, Flux_y, Flux_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth + 1 and
        j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {

      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      get_state(Qm_x, i - 1, j, k, qleft);
      get_state(Qp_x, i, j, k, qright);

      // compute hydro flux along X
      riemann_mhd(qleft, qright, flux, params);

      // store fluxes
      set_state(Fluxes_x, i, j, k, flux);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i, j - 1, k, qleft);
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qleft[IA]), &(qleft[IB]));

      get_state(Qp_y, i, j, k, qright);
      swapValues(&(qright[IU]), &(qright[IV]));
      swapValues(&(qright[IA]), &(qright[IB]));

      // compute hydro flux along Y
      riemann_mhd(qleft, qright, flux, params);

      // store fluxes
      set_state(Fluxes_y, i, j, k, flux);

      //
      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      //
      get_state(Qm_z, i, j, k - 1, qleft);
      swapValues(&(qleft[IU]), &(qleft[IW]));
      swapValues(&(qleft[IA]), &(qleft[IC]));

      get_state(Qp_z, i, j, k, qright);
      swapValues(&(qright[IU]), &(qright[IW]));
      swapValues(&(qright[IA]), &(qright[IC]));

      // compute hydro flux along Z
      riemann_mhd(qleft, qright, flux, params);

      // store fluxes
      set_state(Fluxes_z, i, j, k, flux);
    }
  }

  DataArray3d Qm_x, Qm_y, Qm_z;
  DataArray3d Qp_x, Qp_y, Qp_z;
  DataArray3d Fluxes_x, Fluxes_y, Fluxes_z;
  real_t      dtdx, dtdy, dtdz;

}; // ComputeFluxesAndStoreFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndUpdateFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  ComputeFluxesAndUpdateFunctor3D_MHD(HydroParams params,
                                      DataArray3d Qm_x,
                                      DataArray3d Qm_y,
                                      DataArray3d Qm_z,
                                      DataArray3d Qp_x,
                                      DataArray3d Qp_y,
                                      DataArray3d Qp_z,
                                      DataArray3d Udata,
                                      real_t      dtdx,
                                      real_t      dtdy,
                                      real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , Qm_x(Qm_x)
    , Qm_y(Qm_y)
    , Qm_z(Qm_z)
    , Qp_x(Qp_x)
    , Qp_y(Qp_y)
    , Qp_z(Qp_z)
    , Udata(Udata)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Qm_x,
        DataArray3d Qm_y,
        DataArray3d Qm_z,
        DataArray3d Qp_x,
        DataArray3d Qp_y,
        DataArray3d Qp_z,
        DataArray3d Udata,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ComputeFluxesAndUpdateFunctor3D_MHD functor(
      params, Qm_x, Qm_y, Qm_z, Qp_x, Qp_y, Qp_z, Udata, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth + 1 and
        j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    // clang-format on
    {

      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //

      // clang-format off
      get_state(Qm_x, i - 1, j, k, qleft);
      get_state(Qp_x, i    , j, k, qright);
      // clang-format on

      // compute hydro flux along X
      riemann_mhd(qleft, qright, flux, params);

      //
      // Update with fluxes along X
      //
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, k, ID), flux[ID] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IP), flux[IP] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IU), flux[IU] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IV), flux[IV] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, k, IW), flux[IW] * dtdx);
      }

      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i - 1, j, k, ID), flux[ID] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IP), flux[IP] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IU), flux[IU] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IV), flux[IV] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, k, IW), flux[IW] * dtdx);
      }

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i, j - 1, k, qleft);
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qleft[IA]), &(qleft[IB]));

      get_state(Qp_y, i, j, k, qright);
      swapValues(&(qright[IU]), &(qright[IV]));
      swapValues(&(qright[IA]), &(qright[IB]));

      // compute hydro flux along Y
      riemann_mhd(qleft, qright, flux, params);

      swapValues(&(flux[IU]), &(flux[IV]));
      swapValues(&(flux[IA]), &(flux[IB]));

      //
      // update with fluxes Y
      //
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, k, ID), flux[ID] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IP), flux[IP] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IU), flux[IU] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IV), flux[IV] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, k, IW), flux[IW] * dtdy);
      }
      if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j - 1, k, ID), flux[ID] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IP), flux[IP] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IU), flux[IU] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IV), flux[IV] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, k, IW), flux[IW] * dtdy);
      }

      //
      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      //
      get_state(Qm_z, i, j, k - 1, qleft);
      swapValues(&(qleft[IU]), &(qleft[IW]));
      swapValues(&(qleft[IA]), &(qleft[IC]));

      get_state(Qp_z, i, j, k, qright);
      swapValues(&(qright[IU]), &(qright[IW]));
      swapValues(&(qright[IA]), &(qright[IC]));

      // compute hydro flux along Z
      riemann_mhd(qleft, qright, flux, params);

      swapValues(&(flux[IU]), &(flux[IW]));
      swapValues(&(flux[IA]), &(flux[IC]));

      //
      // update with fluxes Z
      //
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, k, ID), flux[ID] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IP), flux[IP] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IU), flux[IU] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IV), flux[IV] * dtdz);
        Kokkos::atomic_add(&Udata(i, j, k, IW), flux[IW] * dtdz);
      }
      if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j, k - 1, ID), flux[ID] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IP), flux[IP] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IU), flux[IU] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IV), flux[IV] * dtdz);
        Kokkos::atomic_sub(&Udata(i, j, k - 1, IW), flux[IW] * dtdz);
      }
    }
  }

  DataArray3d Qm_x, Qm_y, Qm_z;
  DataArray3d Qp_x, Qp_y, Qp_z;
  DataArray3d Udata;
  real_t      dtdx, dtdy, dtdz;

}; // ComputeFluxesAndUpdateFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndStoreFunctor3D : public MHDBaseFunctor3D
{

public:
  ComputeEmfAndStoreFunctor3D(HydroParams      params,
                              DataArray3d      QEdge_RT,
                              DataArray3d      QEdge_RB,
                              DataArray3d      QEdge_LT,
                              DataArray3d      QEdge_LB,
                              DataArray3d      QEdge_RT2,
                              DataArray3d      QEdge_RB2,
                              DataArray3d      QEdge_LT2,
                              DataArray3d      QEdge_LB2,
                              DataArray3d      QEdge_RT3,
                              DataArray3d      QEdge_RB3,
                              DataArray3d      QEdge_LT3,
                              DataArray3d      QEdge_LB3,
                              DataArrayVector3 Emf,
                              real_t           dtdx,
                              real_t           dtdy,
                              real_t           dtdz)
    : MHDBaseFunctor3D(params)
    , QEdge_RT(QEdge_RT)
    , QEdge_RB(QEdge_RB)
    , QEdge_LT(QEdge_LT)
    , QEdge_LB(QEdge_LB)
    , QEdge_RT2(QEdge_RT2)
    , QEdge_RB2(QEdge_RB2)
    , QEdge_LT2(QEdge_LT2)
    , QEdge_LB2(QEdge_LB2)
    , QEdge_RT3(QEdge_RT3)
    , QEdge_RB3(QEdge_RB3)
    , QEdge_LT3(QEdge_LT3)
    , QEdge_LB3(QEdge_LB3)
    , Emf(Emf)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams      params,
        DataArray3d      QEdge_RT,
        DataArray3d      QEdge_RB,
        DataArray3d      QEdge_LT,
        DataArray3d      QEdge_LB,
        DataArray3d      QEdge_RT2,
        DataArray3d      QEdge_RB2,
        DataArray3d      QEdge_LT2,
        DataArray3d      QEdge_LB2,
        DataArray3d      QEdge_RT3,
        DataArray3d      QEdge_RB3,
        DataArray3d      QEdge_LT3,
        DataArray3d      QEdge_LB3,
        DataArrayVector3 Emf,
        real_t           dtdx,
        real_t           dtdy,
        real_t           dtdz)
  {
    ComputeEmfAndStoreFunctor3D functor(params,
                                        QEdge_RT,
                                        QEdge_RB,
                                        QEdge_LT,
                                        QEdge_LB,
                                        QEdge_RT2,
                                        QEdge_RB2,
                                        QEdge_LT2,
                                        QEdge_LB2,
                                        QEdge_RT3,
                                        QEdge_RB3,
                                        QEdge_LT3,
                                        QEdge_LB3,
                                        Emf,
                                        dtdx,
                                        dtdy,
                                        dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth + 1 and
        j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    {

      MHDState qEdge_emf[4];

      // preparation for calling compute_emf (equivalent to cmp_mag_flx
      // in DUMSES)
      // in the following, the 2 first indexes in qEdge_emf array play
      // the same offset role as in the calling argument of cmp_mag_flx
      // in DUMSES (if you see what I mean ?!)

      // actually compute emfZ
      get_state(QEdge_RT3, i - 1, j - 1, k, qEdge_emf[IRT]);
      get_state(QEdge_RB3, i - 1, j    , k, qEdge_emf[IRB]);
      get_state(QEdge_LT3, i    , j - 1, k, qEdge_emf[ILT]);
      get_state(QEdge_LB3, i    , j    , k, qEdge_emf[ILB]);

      Emf(i, j, k, I_EMFZ) = compute_emf<EMFZ>(qEdge_emf, params);

      // actually compute emfY (take care that RB and LT are
      // swapped !!!)
      get_state(QEdge_RT2, i - 1, j, k - 1, qEdge_emf[IRT]);
      get_state(QEdge_LT2, i    , j, k - 1, qEdge_emf[IRB]);
      get_state(QEdge_RB2, i - 1, j, k    , qEdge_emf[ILT]);
      get_state(QEdge_LB2, i    , j, k    , qEdge_emf[ILB]);

      Emf(i, j, k, I_EMFY) = compute_emf<EMFY>(qEdge_emf, params);

      // actually compute emfX
      get_state(QEdge_RT, i, j - 1, k - 1, qEdge_emf[IRT]);
      get_state(QEdge_RB, i, j - 1, k    , qEdge_emf[IRB]);
      get_state(QEdge_LT, i, j    , k - 1, qEdge_emf[ILT]);
      get_state(QEdge_LB, i, j    , k    , qEdge_emf[ILB]);

      Emf(i, j, k, I_EMFX) = compute_emf<EMFX>(qEdge_emf, params);
    }
    // clang-format on
  }

  DataArray3d      QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  DataArray3d      QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d      QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  DataArrayVector3 Emf;
  real_t           dtdx, dtdy, dtdz;

}; // ComputeEmfAndStoreFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndUpdateFunctor3D : public MHDBaseFunctor3D
{

public:
  ComputeEmfAndUpdateFunctor3D(HydroParams params,
                               DataArray3d QEdge_RT,
                               DataArray3d QEdge_RB,
                               DataArray3d QEdge_LT,
                               DataArray3d QEdge_LB,
                               DataArray3d QEdge_RT2,
                               DataArray3d QEdge_RB2,
                               DataArray3d QEdge_LT2,
                               DataArray3d QEdge_LB2,
                               DataArray3d QEdge_RT3,
                               DataArray3d QEdge_RB3,
                               DataArray3d QEdge_LT3,
                               DataArray3d QEdge_LB3,
                               DataArray3d Udata,
                               real_t      dtdx,
                               real_t      dtdy,
                               real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , QEdge_RT(QEdge_RT)
    , QEdge_RB(QEdge_RB)
    , QEdge_LT(QEdge_LT)
    , QEdge_LB(QEdge_LB)
    , QEdge_RT2(QEdge_RT2)
    , QEdge_RB2(QEdge_RB2)
    , QEdge_LT2(QEdge_LT2)
    , QEdge_LB2(QEdge_LB2)
    , QEdge_RT3(QEdge_RT3)
    , QEdge_RB3(QEdge_RB3)
    , QEdge_LT3(QEdge_LT3)
    , QEdge_LB3(QEdge_LB3)
    , Udata(Udata)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d QEdge_RT,
        DataArray3d QEdge_RB,
        DataArray3d QEdge_LT,
        DataArray3d QEdge_LB,
        DataArray3d QEdge_RT2,
        DataArray3d QEdge_RB2,
        DataArray3d QEdge_LT2,
        DataArray3d QEdge_LB2,
        DataArray3d QEdge_RT3,
        DataArray3d QEdge_RB3,
        DataArray3d QEdge_LT3,
        DataArray3d QEdge_LB3,
        DataArray3d Udata,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    ComputeEmfAndUpdateFunctor3D functor(params,
                                         QEdge_RT,
                                         QEdge_RB,
                                         QEdge_LT,
                                         QEdge_LB,
                                         QEdge_RT2,
                                         QEdge_RB2,
                                         QEdge_LT2,
                                         QEdge_LB2,
                                         QEdge_RT3,
                                         QEdge_RB3,
                                         QEdge_LT3,
                                         QEdge_LB3,
                                         Udata,
                                         dtdx,
                                         dtdy,
                                         dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth + 1 and
        j >= ghostWidth and j < jsize - ghostWidth + 1 and
        i >= ghostWidth and i < isize - ghostWidth + 1)
    {

      MHDState qEdge_emf[4];

      // preparation for calling compute_emf (equivalent to cmp_mag_flx
      // in DUMSES)
      // in the following, the 2 first indexes in qEdge_emf array play
      // the same offset role as in the calling argument of cmp_mag_flx
      // in DUMSES (if you see what I mean ?!)

      // actually compute emfZ
      get_state(QEdge_RT3, i - 1, j - 1, k, qEdge_emf[IRT]);
      get_state(QEdge_RB3, i - 1, j    , k, qEdge_emf[IRB]);
      get_state(QEdge_LT3, i    , j - 1, k, qEdge_emf[ILT]);
      get_state(QEdge_LB3, i    , j    , k, qEdge_emf[ILB]);

      const real_t emfZ = compute_emf<EMFZ>(qEdge_emf, params);

      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i    , j    , k     , IA), emfZ * dtdy);
        Kokkos::atomic_add(&Udata(i    , j    , k     , IB), emfZ * dtdx);
      }

      if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i    , j - 1, k     , IA), emfZ * dtdy);
      }
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i - 1, j    , k     , IB), emfZ * dtdx);
      }

      // actually compute emfY (take care that RB and LT are
      // swapped !!!)
      get_state(QEdge_RT2, i - 1, j, k - 1, qEdge_emf[IRT]);
      get_state(QEdge_LT2, i    , j, k - 1, qEdge_emf[IRB]);
      get_state(QEdge_RB2, i - 1, j, k    , qEdge_emf[ILT]);
      get_state(QEdge_LB2, i    , j, k    , qEdge_emf[ILB]);

      const real_t emfY = compute_emf<EMFY>(qEdge_emf, params);

      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i    , j    , k     , IA), emfY * dtdz);
        Kokkos::atomic_sub(&Udata(i    , j    , k     , IC), emfY * dtdx);
      }
      if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i    , j    , k - 1 , IA), emfY * dtdz);
      }
      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i - 1, j    , k     , IC), emfY * dtdx);
      }

      // actually compute emfX
      get_state(QEdge_RT, i, j - 1, k - 1, qEdge_emf[IRT]);
      get_state(QEdge_RB, i, j - 1, k    , qEdge_emf[IRB]);
      get_state(QEdge_LT, i, j    , k - 1, qEdge_emf[ILT]);
      get_state(QEdge_LB, i, j    , k    , qEdge_emf[ILB]);

      const real_t emfX = compute_emf<EMFX>(qEdge_emf, params);

      if (k < ksize - ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i    , j    , k     , IB), emfX * dtdz);
        Kokkos::atomic_add(&Udata(i    , j    , k     , IC), emfX * dtdy);
      }
      if (k > ghostWidth and j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i    , j    , k - 1 , IB), emfX * dtdz);
      }
      if (k < ksize - ghostWidth and j > ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i    , j - 1, k     , IC), emfX * dtdy);
      }
    }
    // clang-format on
  }

  DataArray3d QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  DataArray3d QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  DataArray3d Udata;
  real_t      dtdx, dtdy, dtdz;

}; // ComputeEmfAndUpdateFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  UpdateFunctor3D_MHD(HydroParams params,
                      DataArray3d Udata,
                      DataArray3d FluxData_x,
                      DataArray3d FluxData_y,
                      DataArray3d FluxData_z,
                      real_t      dtdx,
                      real_t      dtdy,
                      real_t      dtdz)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray3d Udata,
        DataArray3d FluxData_x,
        DataArray3d FluxData_y,
        DataArray3d FluxData_z,
        real_t      dtdx,
        real_t      dtdy,
        real_t      dtdz)
  {
    UpdateFunctor3D_MHD functor(
      params, Udata, FluxData_x, FluxData_y, FluxData_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth and
        j >= ghostWidth and j < jsize - ghostWidth and
        i >= ghostWidth and i < isize - ghostWidth)
    // clang-format on
    {

      MHDState udata;
      MHDState flux;
      get_state(Udata, i, j, k, udata);

      // add up contributions from all 6 faces

      get_state(FluxData_x, i, j, k, flux);
      udata[ID] += flux[ID] * dtdx;
      udata[IP] += flux[IP] * dtdx;
      udata[IU] += flux[IU] * dtdx;
      udata[IV] += flux[IV] * dtdx;
      udata[IW] += flux[IW] * dtdx;

      get_state(FluxData_x, i + 1, j, k, flux);
      udata[ID] -= flux[ID] * dtdx;
      udata[IP] -= flux[IP] * dtdx;
      udata[IU] -= flux[IU] * dtdx;
      udata[IV] -= flux[IV] * dtdx;
      udata[IW] -= flux[IW] * dtdx;

      get_state(FluxData_y, i, j, k, flux);
      udata[ID] += flux[ID] * dtdy;
      udata[IP] += flux[IP] * dtdy;
      udata[IU] += flux[IV] * dtdy; //
      udata[IV] += flux[IU] * dtdy; //
      udata[IW] += flux[IW] * dtdy;

      get_state(FluxData_y, i, j + 1, k, flux);
      udata[ID] -= flux[ID] * dtdy;
      udata[IP] -= flux[IP] * dtdy;
      udata[IU] -= flux[IV] * dtdy; //
      udata[IV] -= flux[IU] * dtdy; //
      udata[IW] -= flux[IW] * dtdy;

      get_state(FluxData_z, i, j, k, flux);
      udata[ID] += flux[ID] * dtdz;
      udata[IP] += flux[IP] * dtdz;
      udata[IU] += flux[IW] * dtdz; //
      udata[IV] += flux[IV] * dtdz;
      udata[IW] += flux[IU] * dtdz; //

      get_state(FluxData_z, i, j, k + 1, flux);
      udata[ID] -= flux[ID] * dtdz;
      udata[IP] -= flux[IP] * dtdz;
      udata[IU] -= flux[IW] * dtdz; //
      udata[IV] -= flux[IV] * dtdz;
      udata[IW] -= flux[IU] * dtdz; //

      // write back result in Udata
      set_state(Udata, i, j, k, udata);

    } // end if

  } // end operator ()

  DataArray3d Udata;
  DataArray3d FluxData_x, FluxData_y, FluxData_z;
  real_t      dtdx, dtdy, dtdz;

}; // UpdateFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateEmfFunctor3D : public MHDBaseFunctor3D
{

public:
  UpdateEmfFunctor3D(HydroParams      params,
                     DataArray3d      Udata,
                     DataArrayVector3 Emf,
                     real_t           dtdx,
                     real_t           dtdy,
                     real_t           dtdz)
    : MHDBaseFunctor3D(params)
    , Udata(Udata)
    , Emf(Emf)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams      params,
        DataArray3d      Udata,
        DataArrayVector3 Emf,
        real_t           dtdx,
        real_t           dtdy,
        real_t           dtdz)
  {
    UpdateEmfFunctor3D functor(params, Udata, Emf, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
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

    // clang-format off
    if (k >= ghostWidth and k < ksize - ghostWidth /*+ 1*/ and
        j >= ghostWidth and j < jsize - ghostWidth /*+ 1*/ and
        i >= ghostWidth and i < isize - ghostWidth /*+ 1*/)
    // clang-format on
    {

      MHDState udata;
      get_state(Udata, i, j, k, udata);

      // if (k < ksize - ghostWidth)
      {
        udata[IA] += (Emf(i, j + 1, k, I_EMFZ) - Emf(i, j, k, I_EMFZ)) * dtdy;

        udata[IB] -= (Emf(i + 1, j, k, I_EMFZ) - Emf(i, j, k, I_EMFZ)) * dtdx;
      }

      // update BX
      udata[IA] -= (Emf(i, j, k + 1, I_EMFY) - Emf(i, j, k, I_EMFY)) * dtdz;

      // update BY
      udata[IB] += (Emf(i, j, k + 1, I_EMFX) - Emf(i, j, k, I_EMFX)) * dtdz;

      // update BZ
      udata[IC] += (Emf(i + 1, j, k, I_EMFY) - Emf(i, j, k, I_EMFY)) * dtdx;

      udata[IC] -= (Emf(i, j + 1, k, I_EMFX) - Emf(i, j, k, I_EMFX)) * dtdy;

      Udata(i, j, k, IA) = udata[IA];
      Udata(i, j, k, IB) = udata[IB];
      Udata(i, j, k, IC) = udata[IC];
    }
  } // operator()

  DataArray3d      Udata;
  DataArrayVector3 Emf;
  real_t           dtdx, dtdy, dtdz;

}; // UpdateEmfFunctor3D

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_RUN_FUNCTORS_3D_H_
