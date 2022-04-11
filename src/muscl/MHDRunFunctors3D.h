#ifndef MHD_RUN_FUNCTORS_3D_H_
#define MHD_RUN_FUNCTORS_3D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "MHDBaseFunctor3D.h"
#include "shared/RiemannSolvers_MHD.h"

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

namespace euler_kokkos { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  ComputeDtFunctor3D_MHD(HydroParams params,
			 DataArray3d Qdata) :
    MHDBaseFunctor3D(params),
    Qdata(Qdata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
		    int nbCells,
                    real_t& invDt) {
    ComputeDtFunctor3D_MHD functor(params, Udata);
    Kokkos::parallel_reduce(nbCells, functor, invDt);
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN.
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &index, real_t &invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth &&
       j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {

      MHDState qLoc; // primitive    variables in current cell

      // get primitive variables in current cell
      qLoc[ID]  = Qdata(i,j,k,ID);
      qLoc[IP]  = Qdata(i,j,k,IP);
      qLoc[IU]  = Qdata(i,j,k,IU);
      qLoc[IV]  = Qdata(i,j,k,IV);
      qLoc[IW]  = Qdata(i,j,k,IW);
      qLoc[IBX] = Qdata(i,j,k,IBX);
      qLoc[IBY] = Qdata(i,j,k,IBY);
      qLoc[IBZ] = Qdata(i,j,k,IBZ);

      // compute fastest information speeds
      real_t fastInfoSpeed[3];
      find_speed_info<THREE_D>(qLoc, fastInfoSpeed, params);

      real_t vx = fastInfoSpeed[IX];
      real_t vy = fastInfoSpeed[IY];
      real_t vz = fastInfoSpeed[IZ];

      invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);

    }

  } // operator ()


  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  DataArray3d Qdata;

}; // ComputeDtFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  ConvertToPrimitivesFunctor3D_MHD(HydroParams params,
				   DataArray3d Udata,
				   DataArray3d Qdata) :
    MHDBaseFunctor3D(params), Udata(Udata), Qdata(Qdata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
                    DataArray3d Qdata,
		    int nbCells) {
    ConvertToPrimitivesFunctor3D_MHD functor(params, Udata, Qdata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // magnetic field in neighbor cells
    real_t magFieldNeighbors[3];

    if(k >= 0 && k < ksize-1  &&
       j >= 0 && j < jsize-1  &&
       i >= 0 && i < isize-1 ) {

      MHDState uLoc; // conservative    variables in current cell
      MHDState qLoc; // primitive    variables in current cell
      real_t c;

      // get local conservative variable
      uLoc[ID]  = Udata(i,j,k,ID);
      uLoc[IP]  = Udata(i,j,k,IP);
      uLoc[IU]  = Udata(i,j,k,IU);
      uLoc[IV]  = Udata(i,j,k,IV);
      uLoc[IW]  = Udata(i,j,k,IW);
      uLoc[IBX] = Udata(i,j,k,IBX);
      uLoc[IBY] = Udata(i,j,k,IBY);
      uLoc[IBZ] = Udata(i,j,k,IBZ);

      // get mag field in neighbor cells
      magFieldNeighbors[IX] = Udata(i+1,j  ,k  ,IBX);
      magFieldNeighbors[IY] = Udata(i  ,j+1,k  ,IBY);
      magFieldNeighbors[IZ] = Udata(i  ,j  ,k+1,IBZ);

      // get primitive variables in current cell
      constoprim_mhd(uLoc, magFieldNeighbors, c, qLoc);

      // copy q state in q global
      Qdata(i,j,k,ID)  = qLoc[ID];
      Qdata(i,j,k,IP)  = qLoc[IP];
      Qdata(i,j,k,IU)  = qLoc[IU];
      Qdata(i,j,k,IV)  = qLoc[IV];
      Qdata(i,j,k,IW)  = qLoc[IW];
      Qdata(i,j,k,IBX) = qLoc[IBX];
      Qdata(i,j,k,IBY) = qLoc[IBY];
      Qdata(i,j,k,IBZ) = qLoc[IBZ];

    }

  }

  DataArray3d Udata;
  DataArray3d Qdata;

}; // ConvertToPrimitivesFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeElecFieldFunctor3D : public MHDBaseFunctor3D {

public:

  ComputeElecFieldFunctor3D(HydroParams params,
			    DataArray3d Udata,
			    DataArray3d Qdata,
			    DataArrayVector3 ElecField) :
    MHDBaseFunctor3D(params),
    Udata(Udata), Qdata(Qdata), ElecField(ElecField) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
                    DataArray3d Qdata,
		    DataArrayVector3 ElecField,
		    int nbCells) {
    ComputeElecFieldFunctor3D functor(params, Udata, Qdata, ElecField);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if (k > 0 && k < ksize-1 &&
	j > 0 && j < jsize-1 &&
	i > 0 && i < isize-1) {

      real_t u, v, w, A, B, C;

      // compute Ex
      v = ONE_FOURTH_F * ( Qdata(i  ,j-1,k-1,IV) +
			   Qdata(i  ,j-1,k  ,IV) +
			   Qdata(i  ,j  ,k-1,IV) +
			   Qdata(i  ,j  ,k  ,IV) );

      w = ONE_FOURTH_F * ( Qdata(i  ,j-1,k-1,IW) +
			   Qdata(i  ,j-1,k  ,IW) +
			   Qdata(i  ,j  ,k-1,IW) +
			   Qdata(i  ,j  ,k  ,IW) );

      B = HALF_F  * ( Udata(i  ,j  ,k-1,IB) +
		      Udata(i  ,j  ,k  ,IB) );

      C = HALF_F  * ( Udata(i  ,j-1,k  ,IC) +
		      Udata(i  ,j  ,k  ,IC) );

      ElecField(i,j,k,IX) = v*C-w*B;

      // compute Ey
      u = ONE_FOURTH_F * ( Qdata   (i-1,j  ,k-1,IU) +
			   Qdata   (i-1,j  ,k  ,IU) +
			   Qdata   (i  ,j  ,k-1,IU) +
			   Qdata   (i  ,j  ,k  ,IU) );

      w = ONE_FOURTH_F * ( Qdata   (i-1,j  ,k-1,IW) +
			   Qdata   (i-1,j  ,k  ,IW) +
			   Qdata   (i  ,j  ,k-1,IW) +
			   Qdata   (i  ,j  ,k  ,IW) );

      A = HALF_F  * ( Udata(i  ,j  ,k-1,IA) +
		      Udata(i  ,j  ,k  ,IA) );

      C = HALF_F  * ( Udata(i-1,j  ,k  ,IC) +
		      Udata(i  ,j  ,k  ,IC) );

      ElecField(i,j,k,IY) = w*A-u*C;

      // compute Ez
      u = ONE_FOURTH_F * ( Qdata   (i-1,j-1,k  ,IU) +
			   Qdata   (i-1,j  ,k  ,IU) +
			   Qdata   (i  ,j-1,k  ,IU) +
			   Qdata   (i  ,j  ,k  ,IU) );

      v = ONE_FOURTH_F * ( Qdata   (i-1,j-1,k  ,IV) +
			   Qdata   (i-1,j  ,k  ,IV) +
			   Qdata   (i  ,j-1,k  ,IV) +
			   Qdata   (i  ,j  ,k  ,IV) );

      A = HALF_F  * ( Udata(i  ,j-1,k  ,IA) +
		      Udata(i  ,j  ,k  ,IA) );

      B = HALF_F  * ( Udata(i-1,j  ,k  ,IB) +
		      Udata(i  ,j  ,k  ,IB) );

      ElecField(i,j,k,IZ) = u*B-v*A;

    }
  } // operator ()

  DataArray3d Udata;
  DataArray3d Qdata;
  DataArrayVector3 ElecField;

}; // ComputeElecFieldFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeMagSlopesFunctor3D : public MHDBaseFunctor3D {

public:

  ComputeMagSlopesFunctor3D(HydroParams      params,
			    DataArray3d      Udata,
			    DataArrayVector3 DeltaA,
			    DataArrayVector3 DeltaB,
			    DataArrayVector3 DeltaC) :
    MHDBaseFunctor3D(params), Udata(Udata),
    DeltaA(DeltaA), DeltaB(DeltaB), DeltaC(DeltaC) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams      params,
                    DataArray3d      Udata,
		    DataArrayVector3 DeltaA,
		    DataArrayVector3 DeltaB,
		    DataArrayVector3 DeltaC,
		    int nbCells) {
    ComputeMagSlopesFunctor3D functor(params, Udata, DeltaA, DeltaB, DeltaC);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if (k > 0 && k < ksize-1 &&
	j > 0 && j < jsize-1 &&
	i > 0 && i < isize-1) {

      real_t bfSlopes[15];
      real_t dbfSlopes[3][3];

      real_t (&dbfX)[3] = dbfSlopes[IX];
      real_t (&dbfY)[3] = dbfSlopes[IY];
      real_t (&dbfZ)[3] = dbfSlopes[IZ];

      // get magnetic slopes dbf
      bfSlopes[0]  = Udata(i  ,j  ,k  , IA);
      bfSlopes[1]  = Udata(i  ,j+1,k  , IA);
      bfSlopes[2]  = Udata(i  ,j-1,k  , IA);
      bfSlopes[3]  = Udata(i  ,j  ,k+1, IA);
      bfSlopes[4]  = Udata(i  ,j  ,k-1, IA);

      bfSlopes[5]  = Udata(i  ,j  ,k  , IB);
      bfSlopes[6]  = Udata(i+1,j  ,k  , IB);
      bfSlopes[7]  = Udata(i-1,j  ,k  , IB);
      bfSlopes[8]  = Udata(i  ,j  ,k+1, IB);
      bfSlopes[9]  = Udata(i  ,j  ,k-1, IB);

      bfSlopes[10] = Udata(i  ,j  ,k  , IC);
      bfSlopes[11] = Udata(i+1,j  ,k  , IC);
      bfSlopes[12] = Udata(i-1,j  ,k  , IC);
      bfSlopes[13] = Udata(i  ,j+1,k  , IC);
      bfSlopes[14] = Udata(i  ,j-1,k  , IC);

      // compute magnetic slopes
      slope_unsplit_mhd_3d(bfSlopes, dbfSlopes);

      // store magnetic slopes
      DeltaA(i,j,k,0) = dbfX[IX];
      DeltaA(i,j,k,1) = dbfY[IX];
      DeltaA(i,j,k,2) = dbfZ[IX];

      DeltaB(i,j,k,0) = dbfX[IY];
      DeltaB(i,j,k,1) = dbfY[IY];
      DeltaB(i,j,k,2) = dbfZ[IY];

      DeltaC(i,j,k,0) = dbfX[IZ];
      DeltaC(i,j,k,1) = dbfY[IZ];
      DeltaC(i,j,k,2) = dbfZ[IZ];

    }

  } // operator ()
  DataArray3d Udata;
  DataArrayVector3 DeltaA;
  DataArrayVector3 DeltaB;
  DataArrayVector3 DeltaC;

}; // class ComputeMagSlopesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  ComputeTraceFunctor3D_MHD(HydroParams params,
			    DataArray3d Udata,
			    DataArray3d Qdata,
			    DataArrayVector3 DeltaA,
			    DataArrayVector3 DeltaB,
			    DataArrayVector3 DeltaC,
			    DataArrayVector3 ElecField,
			    DataArray3d Qm_x,
			    DataArray3d Qm_y,
			    DataArray3d Qm_z,
			    DataArray3d Qp_x,
			    DataArray3d Qp_y,
			    DataArray3d Qp_z,
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
			    real_t dtdx,
			    real_t dtdy,
			    real_t dtdz) :
    MHDBaseFunctor3D(params),
    Udata(Udata), Qdata(Qdata),
    DeltaA(DeltaA), DeltaB(DeltaB), DeltaC(DeltaC), ElecField(ElecField),
    Qm_x(Qm_x), Qm_y(Qm_y), Qm_z(Qm_z),
    Qp_x(Qp_x), Qp_y(Qp_y), Qp_z(Qp_z),
    QEdge_RT (QEdge_RT),  QEdge_RB (QEdge_RB),  QEdge_LT (QEdge_LT),  QEdge_LB (QEdge_LB),
    QEdge_RT2(QEdge_RT2), QEdge_RB2(QEdge_RB2), QEdge_LT2(QEdge_LT2), QEdge_LB2(QEdge_LB2),
    QEdge_RT3(QEdge_RT3), QEdge_RB3(QEdge_RB3), QEdge_LT3(QEdge_LT3), QEdge_LB3(QEdge_LB3),
    dtdx(dtdx), dtdy(dtdy), dtdz(dtdz) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray3d Udata,
		    DataArray3d Qdata,
		    DataArrayVector3 DeltaA,
		    DataArrayVector3 DeltaB,
		    DataArrayVector3 DeltaC,
		    DataArrayVector3 ElecField,
		    DataArray3d Qm_x,
		    DataArray3d Qm_y,
		    DataArray3d Qm_z,
		    DataArray3d Qp_x,
		    DataArray3d Qp_y,
		    DataArray3d Qp_z,
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
		    real_t dtdx,
		    real_t dtdy,
		    real_t dtdz,
		    int    nbCells)
  {
    ComputeTraceFunctor3D_MHD functor(params, Udata, Qdata,
				      DeltaA, DeltaB, DeltaC, ElecField,
				      Qm_x, Qm_y, Qm_z,
				      Qp_x, Qp_y, Qp_z,
				      QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB,
				      QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2,
				      QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3,
				      dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth-2 && k < ksize-ghostWidth+1 &&
       j >= ghostWidth-2 && j < jsize-ghostWidth+1 &&
       i >= ghostWidth-2 && i < isize-ghostWidth+1) {

      MHDState q;
      MHDState qPlusX, qMinusX, qPlusY, qMinusY, qPlusZ, qMinusZ;
      MHDState dq[3];

      real_t bfNb[6];
      real_t dbf[12];

      real_t elecFields[3][2][2];
      // alias to electric field components
      real_t (&Ex)[2][2] = elecFields[IX];
      real_t (&Ey)[2][2] = elecFields[IY];
      real_t (&Ez)[2][2] = elecFields[IZ];

      MHDState qm[THREE_D];
      MHDState qp[THREE_D];
      MHDState qEdge[4][3]; // array for qRT, qRB, qLT, qLB

      real_t xPos = params.xmin + params.dx/2 + (i-ghostWidth)*params.dx;

      // get primitive variables state vector
      get_state(Qdata, i  ,j  ,k  , q      );
      get_state(Qdata, i+1,j  ,k  , qPlusX );
      get_state(Qdata, i-1,j  ,k  , qMinusX);
      get_state(Qdata, i  ,j+1,k  , qPlusY );
      get_state(Qdata, i  ,j-1,k  , qMinusY);
      get_state(Qdata, i  ,j  ,k+1, qPlusZ );
      get_state(Qdata, i  ,j  ,k-1, qMinusZ);

      // get hydro slopes dq
      slope_unsplit_hydro_3d(q,
			     qPlusX, qMinusX,
			     qPlusY, qMinusY,
			     qPlusZ, qMinusZ,
			     dq);

      // get face-centered magnetic components
      bfNb[0] = Udata(i  ,j  ,k  , IA);
      bfNb[1] = Udata(i+1,j  ,k  , IA);
      bfNb[2] = Udata(i  ,j  ,k  , IB);
      bfNb[3] = Udata(i  ,j+1,k  , IB);
      bfNb[4] = Udata(i  ,j  ,k  , IC);
      bfNb[5] = Udata(i  ,j  ,k+1, IC);

      // get dbf (transverse magnetic slopes)
      dbf[0]  = DeltaA(i  ,j  ,k  , IY);
      dbf[1]  = DeltaA(i  ,j  ,k  , IZ);
      dbf[2]  = DeltaB(i  ,j  ,k  , IX);
      dbf[3]  = DeltaB(i  ,j  ,k  , IZ);
      dbf[4]  = DeltaC(i  ,j  ,k  , IX);
      dbf[5]  = DeltaC(i  ,j  ,k  , IY);

      dbf[6]  = DeltaA(i+1,j  ,k  , IY);
      dbf[7]  = DeltaA(i+1,j  ,k  , IZ);
      dbf[8]  = DeltaB(i  ,j+1,k  , IX);
      dbf[9]  = DeltaB(i  ,j+1,k  , IZ);
      dbf[10] = DeltaC(i  ,j  ,k+1, IX);
      dbf[11] = DeltaC(i  ,j  ,k+1, IY);

      // get electric field components
      Ex[0][0] = ElecField(i  ,j  ,k  , IX);
      Ex[0][1] = ElecField(i  ,j  ,k+1, IX);
      Ex[1][0] = ElecField(i  ,j+1,k  , IX);
      Ex[1][1] = ElecField(i  ,j+1,k+1, IX);

      Ey[0][0] = ElecField(i  ,j  ,k  , IY);
      Ey[0][1] = ElecField(i  ,j  ,k+1, IY);
      Ey[1][0] = ElecField(i+1,j  ,k  , IY);
      Ey[1][1] = ElecField(i+1,j  ,k+1, IY);

      Ez[0][0] = ElecField(i  ,j  ,k  , IZ);
      Ez[0][1] = ElecField(i  ,j+1,k  , IZ);
      Ez[1][0] = ElecField(i+1,j  ,k  , IZ);
      Ez[1][1] = ElecField(i+1,j+1,k  , IZ);

      // compute qm, qp and qEdge
      trace_unsplit_mhd_3d_simpler(q, dq, bfNb, dbf, elecFields,
				   dtdx, dtdy, dtdz, xPos,
				   qm, qp, qEdge);

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
      set_state(Qm_x, i,j,k, qm[0]);
      set_state(Qp_x, i,j,k, qp[0]);
      set_state(Qm_y, i,j,k, qm[1]);
      set_state(Qp_y, i,j,k, qp[1]);
      set_state(Qm_z, i,j,k, qm[2]);
      set_state(Qp_z, i,j,k, qp[2]);

      set_state(QEdge_RT , i,j,k, qEdge[IRT][0]);
      set_state(QEdge_RB , i,j,k, qEdge[IRB][0]);
      set_state(QEdge_LT , i,j,k, qEdge[ILT][0]);
      set_state(QEdge_LB , i,j,k, qEdge[ILB][0]);

      set_state(QEdge_RT2, i,j,k, qEdge[IRT][1]);
      set_state(QEdge_RB2, i,j,k, qEdge[IRB][1]);
      set_state(QEdge_LT2, i,j,k, qEdge[ILT][1]);
      set_state(QEdge_LB2, i,j,k, qEdge[ILB][1]);

      set_state(QEdge_RT3, i,j,k, qEdge[IRT][2]);
      set_state(QEdge_RB3, i,j,k, qEdge[IRB][2]);
      set_state(QEdge_LT3, i,j,k, qEdge[ILT][2]);
      set_state(QEdge_LB3, i,j,k, qEdge[ILB][2]);

    }

  } // operator ()

  DataArray3d Udata, Qdata;
  DataArrayVector3 DeltaA, DeltaB, DeltaC, ElecField;
  DataArray3d Qm_x, Qm_y, Qm_z;
  DataArray3d Qp_x, Qp_y, Qp_z;
  DataArray3d QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB;
  DataArray3d QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  real_t dtdx, dtdy, dtdz;

}; // class ComputeTraceFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndStoreFunctor3D_MHD : public MHDBaseFunctor3D {

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
				     real_t dtdx,
				     real_t dtdy,
				     real_t dtdz) :
    MHDBaseFunctor3D(params),
    Qm_x(Qm_x), Qm_y(Qm_y), Qm_z(Qm_z),
    Qp_x(Qp_x), Qp_y(Qp_y), Qp_z(Qp_z),
    Fluxes_x(Fluxes_x), Fluxes_y(Fluxes_y), Fluxes_z(Fluxes_z),
    dtdx(dtdx), dtdy(dtdy), dtdz(dtdz) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Qm_x,
                    DataArray3d Qm_y,
                    DataArray3d Qm_z,
                    DataArray3d Qp_x,
                    DataArray3d Qp_y,
                    DataArray3d Qp_z,
		    DataArray3d Flux_x,
		    DataArray3d Flux_y,
		    DataArray3d Flux_z,
		    real_t dtdx,
		    real_t dtdy,
		    real_t dtdz,
		    int    nbCells)
  {
    ComputeFluxesAndStoreFunctor3D_MHD functor(params,
					       Qm_x, Qm_y, Qm_z,
					       Qp_x, Qp_y, Qp_z,
					       Flux_x, Flux_y, Flux_z,
					       dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      get_state(Qm_x, i-1,j  ,k, qleft);
      get_state(Qp_x, i  ,j  ,k, qright);

      // compute hydro flux along X
      riemann_mhd(qleft,qright,flux,params);

      // store fluxes
      set_state(Fluxes_x, i, j, k, flux);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i,j-1,k, qleft);
      swapValues(&(qleft[IU])  ,&(qleft[IV]) );
      swapValues(&(qleft[IBX]) ,&(qleft[IBY]) );

      get_state(Qp_y, i,j,k, qright);
      swapValues(&(qright[IU])  ,&(qright[IV]) );
      swapValues(&(qright[IBX]) ,&(qright[IBY]) );

      // compute hydro flux along Y
      riemann_mhd(qleft,qright,flux,params);

      // store fluxes
      set_state(Fluxes_y, i,j,k, flux);

      //
      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      //
      get_state(Qm_z, i,j,k-1, qleft);
      swapValues(&(qleft[IU])  ,&(qleft[IW]) );
      swapValues(&(qleft[IBX]) ,&(qleft[IBZ]) );

      get_state(Qp_z, i,j,k, qright);
      swapValues(&(qright[IU])  ,&(qright[IW]) );
      swapValues(&(qright[IBX]) ,&(qright[IBZ]) );

      // compute hydro flux along Z
      riemann_mhd(qleft,qright,flux,params);

      // store fluxes
      set_state(Fluxes_z, i,j,k, flux);

    }

  }

  DataArray3d Qm_x, Qm_y, Qm_z;
  DataArray3d Qp_x, Qp_y, Qp_z;
  DataArray3d Fluxes_x, Fluxes_y, Fluxes_z;
  real_t dtdx, dtdy, dtdz;

}; // ComputeFluxesAndStoreFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndStoreFunctor3D : public MHDBaseFunctor3D {

public:

  ComputeEmfAndStoreFunctor3D(HydroParams params,
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
			      DataArrayVector3 Emf,
			      real_t dtdx,
			      real_t dtdy,
			      real_t dtdz) :
    MHDBaseFunctor3D(params),
    QEdge_RT(QEdge_RT),   QEdge_RB(QEdge_RB),   QEdge_LT(QEdge_LT),   QEdge_LB(QEdge_LB),
    QEdge_RT2(QEdge_RT2), QEdge_RB2(QEdge_RB2), QEdge_LT2(QEdge_LT2), QEdge_LB2(QEdge_LB2),
    QEdge_RT3(QEdge_RT3), QEdge_RB3(QEdge_RB3), QEdge_LT3(QEdge_LT3), QEdge_LB3(QEdge_LB3),
    Emf(Emf),
    dtdx(dtdx), dtdy(dtdy), dtdz(dtdz) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
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
		    DataArrayVector3 Emf,
		    real_t      dtdx,
		    real_t      dtdy,
		    real_t      dtdz,
		    int         nbCells)
  {
    ComputeEmfAndStoreFunctor3D functor(params,
					QEdge_RT , QEdge_RB , QEdge_LT , QEdge_LB ,
					QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2,
					QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3,
					Emf,
					dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      MHDState qEdge_emf[4];

      // preparation for calling compute_emf (equivalent to cmp_mag_flx
      // in DUMSES)
      // in the following, the 2 first indexes in qEdge_emf array play
      // the same offset role as in the calling argument of cmp_mag_flx
      // in DUMSES (if you see what I mean ?!)

      // actually compute emfZ
      get_state(QEdge_RT3, i-1,j-1,k  , qEdge_emf[IRT]);
      get_state(QEdge_RB3, i-1,j  ,k  , qEdge_emf[IRB]);
      get_state(QEdge_LT3, i  ,j-1,k  , qEdge_emf[ILT]);
      get_state(QEdge_LB3, i  ,j  ,k  , qEdge_emf[ILB]);

      Emf(i,j,k,I_EMFZ) = compute_emf<EMFZ>(qEdge_emf,params);

      // actually compute emfY (take care that RB and LT are
      // swapped !!!)
      get_state(QEdge_RT2, i-1,j  ,k-1, qEdge_emf[IRT]);
      get_state(QEdge_LT2, i  ,j  ,k-1, qEdge_emf[IRB]);
      get_state(QEdge_RB2, i-1,j  ,k  , qEdge_emf[ILT]);
      get_state(QEdge_LB2, i  ,j  ,k  , qEdge_emf[ILB]);

      Emf(i,j,k,I_EMFY) = compute_emf<EMFY>(qEdge_emf,params);

      // actually compute emfX
      get_state(QEdge_RT, i  ,j-1,k-1, qEdge_emf[IRT]);
      get_state(QEdge_RB, i  ,j-1,k  , qEdge_emf[IRB]);
      get_state(QEdge_LT, i  ,j  ,k-1, qEdge_emf[ILT]);
      get_state(QEdge_LB, i  ,j  ,k  , qEdge_emf[ILB]);

      Emf(i,j,k,I_EMFX) = compute_emf<EMFX>(qEdge_emf,params);
    }
  }

  DataArray3d QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB;
  DataArray3d QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  DataArrayVector3 Emf;
  real_t dtdx, dtdy, dtdz;

}; // ComputeEmfAndStoreFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  UpdateFunctor3D_MHD(HydroParams params,
		      DataArray3d Udata,
		      DataArray3d FluxData_x,
		      DataArray3d FluxData_y,
		      DataArray3d FluxData_z,
		      real_t dtdx,
		      real_t dtdy,
		      real_t dtdz) :
    MHDBaseFunctor3D(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z),
    dtdx(dtdx),
    dtdy(dtdy),
    dtdz(dtdz) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
		    DataArray3d FluxData_x,
		    DataArray3d FluxData_y,
		    DataArray3d FluxData_z,
		    real_t      dtdx,
		    real_t      dtdy,
		    real_t      dtdz,
		    int         nbCells)
  {
    UpdateFunctor3D_MHD functor(params, Udata,
				FluxData_x, FluxData_y, FluxData_z,
				dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      MHDState udata;
      MHDState flux;
      get_state(Udata, i,j,k, udata);

      // add up contributions from all 6 faces

      get_state(FluxData_x, i  ,j  ,k  , flux);
      udata[ID]  +=  flux[ID]*dtdx;
      udata[IP]  +=  flux[IP]*dtdx;
      udata[IU]  +=  flux[IU]*dtdx;
      udata[IV]  +=  flux[IV]*dtdx;
      udata[IW]  +=  flux[IW]*dtdx;

      get_state(FluxData_x, i+1,j  ,k  , flux);
      udata[ID]  -=  flux[ID]*dtdx;
      udata[IP]  -=  flux[IP]*dtdx;
      udata[IU]  -=  flux[IU]*dtdx;
      udata[IV]  -=  flux[IV]*dtdx;
      udata[IW]  -=  flux[IW]*dtdx;

      get_state(FluxData_y, i  ,j  ,k  , flux);
      udata[ID]  +=  flux[ID]*dtdy;
      udata[IP]  +=  flux[IP]*dtdy;
      udata[IU]  +=  flux[IV]*dtdy; //
      udata[IV]  +=  flux[IU]*dtdy; //
      udata[IW]  +=  flux[IW]*dtdy;

      get_state(FluxData_y, i  ,j+1,k  , flux);
      udata[ID]  -=  flux[ID]*dtdy;
      udata[IP]  -=  flux[IP]*dtdy;
      udata[IU]  -=  flux[IV]*dtdy; //
      udata[IV]  -=  flux[IU]*dtdy; //
      udata[IW]  -=  flux[IW]*dtdy;

      get_state(FluxData_z, i  ,j  ,k  , flux);
      udata[ID]  +=  flux[ID]*dtdy;
      udata[IP]  +=  flux[IP]*dtdy;
      udata[IU]  +=  flux[IW]*dtdy; //
      udata[IV]  +=  flux[IV]*dtdy;
      udata[IW]  +=  flux[IU]*dtdy; //

      get_state(FluxData_z, i  ,j  ,k+1, flux);
      udata[ID]  -=  flux[ID]*dtdz;
      udata[IP]  -=  flux[IP]*dtdz;
      udata[IU]  -=  flux[IW]*dtdz; //
      udata[IV]  -=  flux[IV]*dtdz;
      udata[IW]  -=  flux[IU]*dtdz; //

      // write back result in Udata
      set_state(Udata, i  ,j  ,k  , udata);

    } // end if

  } // end operator ()

  DataArray3d Udata;
  DataArray3d FluxData_x, FluxData_y, FluxData_z;
  real_t dtdx, dtdy, dtdz;

}; // UpdateFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateEmfFunctor3D : public MHDBaseFunctor3D {

public:

  UpdateEmfFunctor3D(HydroParams params,
		     DataArray3d Udata,
		     DataArrayVector3 Emf,
		     real_t dtdx,
		     real_t dtdy,
		     real_t dtdz) :
    MHDBaseFunctor3D(params),
    Udata(Udata),
    Emf(Emf),
    dtdx(dtdx),
    dtdy(dtdy),
    dtdz(dtdz) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
		    DataArrayVector3 Emf,
		    real_t      dtdx,
		    real_t      dtdy,
		    real_t      dtdz,
		    int         nbCells)
  {
    UpdateEmfFunctor3D functor(params, Udata, Emf,
			       dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize-ghostWidth+1  &&
       j >= ghostWidth && j < jsize-ghostWidth+1  &&
       i >= ghostWidth && i < isize-ghostWidth+1 ) {

      MHDState udata;
      get_state(Udata, i,j,k, udata);

      if (k<ksize-ghostWidth) {
	udata[IBX] += ( Emf(i  ,j+1, k,  I_EMFZ) -
			Emf(i,  j  , k,  I_EMFZ) ) * dtdy;

	udata[IBY] -= ( Emf(i+1,j  , k,  I_EMFZ) -
			Emf(i  ,j  , k,  I_EMFZ) ) * dtdx;

      }

      // update BX
      udata[IBX] -= ( Emf(i  ,j  ,k+1,  I_EMFY) -
		      Emf(i  ,j  ,k  ,  I_EMFY) ) * dtdz;

      // update BY
      udata[IBY] += ( Emf(i  ,j  ,k+1,  I_EMFX) -
		      Emf(i  ,j  ,k  ,  I_EMFX) ) * dtdz;

      // update BZ
      udata[IBZ] += ( Emf(i+1,j  ,k  ,  I_EMFY) -
		      Emf(i  ,j  ,k  ,  I_EMFY) ) * dtdx;

      udata[IBZ] -= ( Emf(i  ,j+1,k  ,  I_EMFX) -
		      Emf(i  ,j  ,k  ,  I_EMFX) ) * dtdy;

      Udata(i,j,k, IA) = udata[IBX];
      Udata(i,j,k, IB) = udata[IBY];
      Udata(i,j,k, IC) = udata[IBZ];

    }
  } // operator()

  DataArray3d Udata;
  DataArrayVector3 Emf;
  real_t dtdx, dtdy, dtdz;

}; // UpdateEmfFunctor3D

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_RUN_FUNCTORS_3D_H_
