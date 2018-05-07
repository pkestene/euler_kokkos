#ifndef HYDRO_RUN_FUNCTORS_3D_H_
#define HYDRO_RUN_FUNCTORS_3D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"
#include "shared/RiemannSolvers.h"

namespace euler_kokkos { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D : public HydroBaseFunctor3D {

public:
  
  ComputeDtFunctor3D(HydroParams params,
		     DataArray3d Udata) :
    HydroBaseFunctor3D(params),
    Udata(Udata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
		    int nbCells,
                    real_t& invDt)
  {
    ComputeDtFunctor3D functor(params, Udata);
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
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c=0.0;
      real_t vx, vy, vz;
      
      // get local conservative variable
      uLoc[ID] = Udata(i,j,k,ID);
      uLoc[IP] = Udata(i,j,k,IP);
      uLoc[IU] = Udata(i,j,k,IU);
      uLoc[IV] = Udata(i,j,k,IV);
      uLoc[IW] = Udata(i,j,k,IW);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      vx = c+FABS(qLoc[IU]);
      vy = c+FABS(qLoc[IV]);
      vz = c+FABS(qLoc[IW]);

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

  DataArray3d Udata;
  
}; // ComputeDtFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor3D : public HydroBaseFunctor3D {

public:

  ConvertToPrimitivesFunctor3D(HydroParams params,
			       DataArray3d Udata,
			       DataArray3d Qdata) :
    HydroBaseFunctor3D(params), Udata(Udata), Qdata(Qdata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
                    DataArray3d Qdata,
		    int nbCells)
  {
    ConvertToPrimitivesFunctor3D functor(params, Udata, Qdata);
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
    
    if(k >= 0 && k < ksize  &&
       j >= 0 && j < jsize  &&
       i >= 0 && i < isize ) {
      
      HydroState uLoc; // conservative variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc[ID] = Udata(i,j,k,ID);
      uLoc[IP] = Udata(i,j,k,IP);
      uLoc[IU] = Udata(i,j,k,IU);
      uLoc[IV] = Udata(i,j,k,IV);
      uLoc[IW] = Udata(i,j,k,IW);
      
      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);

      // copy q state in q global
      Qdata(i,j,k,ID) = qLoc[ID];
      Qdata(i,j,k,IP) = qLoc[IP];
      Qdata(i,j,k,IU) = qLoc[IU];
      Qdata(i,j,k,IV) = qLoc[IV];
      Qdata(i,j,k,IW) = qLoc[IW];
      
    }
    
  }
  
  DataArray3d Udata;
  DataArray3d Qdata;
    
}; // ConvertToPrimitivesFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor3D : public HydroBaseFunctor3D {

public:

  ComputeAndStoreFluxesFunctor3D(HydroParams params,
				 DataArray3d Qdata,
				 DataArray3d FluxData_x,
				 DataArray3d FluxData_y,
				 DataArray3d FluxData_z,
				 real_t dtdx,
				 real_t dtdy,
				 real_t dtdz) :
    HydroBaseFunctor3D(params),
    Qdata(Qdata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y), 
    FluxData_z(FluxData_z), 
    dtdx(dtdx),
    dtdy(dtdy),
    dtdz(dtdz) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Qdata,
		    DataArray3d FluxData_x,
		    DataArray3d FluxData_y,
		    DataArray3d FluxData_z,
		    real_t dtdx,
		    real_t dtdy,
		    real_t dtdz,
		    int    nbCells)
  {
    ComputeAndStoreFluxesFunctor3D functor(params, Qdata,
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

    if(k >= ghostWidth && k <= ksize-ghostWidth  &&
       j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {
      
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
      qLoc[ID]         = Qdata(i  ,j  ,k  , ID);
      qNeighbors_0[ID] = Qdata(i+1,j  ,k  , ID);
      qNeighbors_1[ID] = Qdata(i-1,j  ,k  , ID);
      qNeighbors_2[ID] = Qdata(i  ,j+1,k  , ID);
      qNeighbors_3[ID] = Qdata(i  ,j-1,k  , ID);
      qNeighbors_4[ID] = Qdata(i  ,j  ,k+1, ID);
      qNeighbors_5[ID] = Qdata(i  ,j  ,k-1, ID);
      
      qLoc[IP]         = Qdata(i  ,j  ,k  , IP);
      qNeighbors_0[IP] = Qdata(i+1,j  ,k  , IP);
      qNeighbors_1[IP] = Qdata(i-1,j  ,k  , IP);
      qNeighbors_2[IP] = Qdata(i  ,j+1,k  , IP);
      qNeighbors_3[IP] = Qdata(i  ,j-1,k  , IP);
      qNeighbors_4[IP] = Qdata(i  ,j  ,k+1, IP);
      qNeighbors_5[IP] = Qdata(i  ,j  ,k-1, IP);
      
      qLoc[IU]         = Qdata(i  ,j  ,k  , IU);
      qNeighbors_0[IU] = Qdata(i+1,j  ,k  , IU);
      qNeighbors_1[IU] = Qdata(i-1,j  ,k  , IU);
      qNeighbors_2[IU] = Qdata(i  ,j+1,k  , IU);
      qNeighbors_3[IU] = Qdata(i  ,j-1,k  , IU);
      qNeighbors_4[IU] = Qdata(i  ,j  ,k+1, IU);
      qNeighbors_5[IU] = Qdata(i  ,j  ,k-1, IU);
      
      qLoc[IV]         = Qdata(i  ,j  ,k  , IV);
      qNeighbors_0[IV] = Qdata(i+1,j  ,k  , IV);
      qNeighbors_1[IV] = Qdata(i-1,j  ,k  , IV);
      qNeighbors_2[IV] = Qdata(i  ,j+1,k  , IV);
      qNeighbors_3[IV] = Qdata(i  ,j-1,k  , IV);
      qNeighbors_4[IV] = Qdata(i  ,j  ,k+1, IV);
      qNeighbors_5[IV] = Qdata(i  ,j  ,k-1, IV);
      
      qLoc[IW]         = Qdata(i  ,j  ,k  , IW);
      qNeighbors_0[IW] = Qdata(i+1,j  ,k  , IW);
      qNeighbors_1[IW] = Qdata(i-1,j  ,k  , IW);
      qNeighbors_2[IW] = Qdata(i  ,j+1,k  , IW);
      qNeighbors_3[IW] = Qdata(i  ,j-1,k  , IW);
      qNeighbors_4[IW] = Qdata(i  ,j  ,k+1, IW);
      qNeighbors_5[IW] = Qdata(i  ,j  ,k-1, IW);
      
      slope_unsplit_hydro_3d(qLoc, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     qNeighbors_4, qNeighbors_5,
			     dqX, dqY, dqZ);
	
      // slopes at left neighbor along X
      qLocNeighbor[ID] = Qdata(i-1,j  ,k  , ID);
      qNeighbors_0[ID] = Qdata(i  ,j  ,k  , ID);
      qNeighbors_1[ID] = Qdata(i-2,j  ,k  , ID);
      qNeighbors_2[ID] = Qdata(i-1,j+1,k  , ID);
      qNeighbors_3[ID] = Qdata(i-1,j-1,k  , ID);
      qNeighbors_4[ID] = Qdata(i-1,j  ,k+1, ID);
      qNeighbors_5[ID] = Qdata(i-1,j  ,k-1, ID);
      
      qLocNeighbor[IP] = Qdata(i-1,j  ,k  , IP);
      qNeighbors_0[IP] = Qdata(i  ,j  ,k  , IP);
      qNeighbors_1[IP] = Qdata(i-2,j  ,k  , IP);
      qNeighbors_2[IP] = Qdata(i-1,j+1,k  , IP);
      qNeighbors_3[IP] = Qdata(i-1,j-1,k  , IP);
      qNeighbors_4[IP] = Qdata(i-1,j  ,k+1, IP);
      qNeighbors_5[IP] = Qdata(i-1,j  ,k-1, IP);
      
      qLocNeighbor[IU] = Qdata(i-1,j  ,k  , IU);
      qNeighbors_0[IU] = Qdata(i  ,j  ,k  , IU);
      qNeighbors_1[IU] = Qdata(i-2,j  ,k  , IU);
      qNeighbors_2[IU] = Qdata(i-1,j+1,k  , IU);
      qNeighbors_3[IU] = Qdata(i-1,j-1,k  , IU);
      qNeighbors_4[IU] = Qdata(i-1,j  ,k+1, IU);
      qNeighbors_5[IU] = Qdata(i-1,j  ,k-1, IU);

      qLocNeighbor[IV] = Qdata(i-1,j  ,k  , IV);
      qNeighbors_0[IV] = Qdata(i  ,j  ,k  , IV);
      qNeighbors_1[IV] = Qdata(i-2,j  ,k  , IV);
      qNeighbors_2[IV] = Qdata(i-1,j+1,k  , IV);
      qNeighbors_3[IV] = Qdata(i-1,j-1,k  , IV);
      qNeighbors_4[IV] = Qdata(i-1,j  ,k+1, IV);
      qNeighbors_5[IV] = Qdata(i-1,j  ,k-1, IV);

      qLocNeighbor[IW] = Qdata(i-1,j  ,k  , IW);
      qNeighbors_0[IW] = Qdata(i  ,j  ,k  , IW);
      qNeighbors_1[IW] = Qdata(i-2,j  ,k  , IW);
      qNeighbors_2[IW] = Qdata(i-1,j+1,k  , IW);
      qNeighbors_3[IW] = Qdata(i-1,j-1,k  , IW);
      qNeighbors_4[IW] = Qdata(i-1,j  ,k+1, IW);
      qNeighbors_5[IW] = Qdata(i-1,j  ,k-1, IW);

      slope_unsplit_hydro_3d(qLocNeighbor, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     qNeighbors_4, qNeighbors_5,
			     dqX_neighbor, dqY_neighbor, dqZ_neighbor);
      
      //
      // compute reconstructed states at left interface along X
      //
      
      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc,
				 dqX, dqY, dqZ,
				 dtdx, dtdy, dtdz,
				 FACE_XMIN, qright);
      
      // left interface : left state
      trace_unsplit_3d_along_dir(qLocNeighbor,
				 dqX_neighbor,dqY_neighbor,dqZ_neighbor,
				 dtdx, dtdy, dtdz,
				 FACE_XMAX, qleft);
      
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann_hydro(qleft,qright,qgdnv,flux_x,params);
	
      //
      // store fluxes X
      //
      FluxData_x(i  ,j  ,k  , ID) = flux_x[ID] * dtdx;
      FluxData_x(i  ,j  ,k  , IP) = flux_x[IP] * dtdx;
      FluxData_x(i  ,j  ,k  , IU) = flux_x[IU] * dtdx;
      FluxData_x(i  ,j  ,k  , IV) = flux_x[IV] * dtdx;
      FluxData_x(i  ,j  ,k  , IW) = flux_x[IW] * dtdx;
      
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      qLocNeighbor[ID] = Qdata(i  ,j-1,k  , ID);
      qNeighbors_0[ID] = Qdata(i+1,j-1,k  , ID);
      qNeighbors_1[ID] = Qdata(i-1,j-1,k  , ID);
      qNeighbors_2[ID] = Qdata(i  ,j  ,k  , ID);
      qNeighbors_3[ID] = Qdata(i  ,j-2,k  , ID);
      qNeighbors_4[ID] = Qdata(i  ,j-1,k+1, ID);
      qNeighbors_5[ID] = Qdata(i  ,j-1,k-1, ID);
      
      qLocNeighbor[IP] = Qdata(i  ,j-1,k  , IP);
      qNeighbors_0[IP] = Qdata(i+1,j-1,k  , IP);
      qNeighbors_1[IP] = Qdata(i-1,j-1,k  , IP);
      qNeighbors_2[IP] = Qdata(i  ,j  ,k  , IP);
      qNeighbors_3[IP] = Qdata(i  ,j-2,k  , IP);
      qNeighbors_4[IP] = Qdata(i  ,j-1,k+1, IP);
      qNeighbors_5[IP] = Qdata(i  ,j-1,k-1, IP);
      
      qLocNeighbor[IU] = Qdata(i  ,j-1,k  , IU);
      qNeighbors_0[IU] = Qdata(i+1,j-1,k  , IU);
      qNeighbors_1[IU] = Qdata(i-1,j-1,k  , IU);
      qNeighbors_2[IU] = Qdata(i  ,j  ,k  , IU);
      qNeighbors_3[IU] = Qdata(i  ,j-2,k  , IU);
      qNeighbors_4[IU] = Qdata(i  ,j-1,k+1, IU);
      qNeighbors_5[IU] = Qdata(i  ,j-1,k-1, IU);

      qLocNeighbor[IV] = Qdata(i  ,j-1,k  , IV);
      qNeighbors_0[IV] = Qdata(i+1,j-1,k  , IV);
      qNeighbors_1[IV] = Qdata(i-1,j-1,k  , IV);
      qNeighbors_2[IV] = Qdata(i  ,j  ,k  , IV);
      qNeighbors_3[IV] = Qdata(i  ,j-2,k  , IV);
      qNeighbors_4[IV] = Qdata(i  ,j-1,k+1, IV);
      qNeighbors_5[IV] = Qdata(i  ,j-1,k-1, IV);

      qLocNeighbor[IW] = Qdata(i  ,j-1,k  , IW);
      qNeighbors_0[IW] = Qdata(i+1,j-1,k  , IW);
      qNeighbors_1[IW] = Qdata(i-1,j-1,k  , IW);
      qNeighbors_2[IW] = Qdata(i  ,j  ,k  , IW);
      qNeighbors_3[IW] = Qdata(i  ,j-2,k  , IW);
      qNeighbors_4[IW] = Qdata(i  ,j-1,k+1, IW);
      qNeighbors_5[IW] = Qdata(i  ,j-1,k-1, IW);

      slope_unsplit_hydro_3d(qLocNeighbor, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     qNeighbors_4, qNeighbors_5,
			     dqX_neighbor, dqY_neighbor, dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //
	
      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc,
				 dqX, dqY, dqZ,
				 dtdx, dtdy, dtdz,
				 FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(qLocNeighbor,
				 dqX_neighbor,dqY_neighbor,dqZ_neighbor,
				 dtdx, dtdy, dtdz,
				 FACE_YMAX, qleft);

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]) ,&(qleft[IV]) );
      swapValues(&(qright[IU]),&(qright[IV]));
      riemann_hydro(qleft,qright,qgdnv,flux_y,params);

      //
      // store fluxes Y
      //
      FluxData_y(i  ,j  ,k  , ID) = flux_y[ID] * dtdy;
      FluxData_y(i  ,j  ,k  , IP) = flux_y[IP] * dtdy;
      FluxData_y(i  ,j  ,k  , IU) = flux_y[IU] * dtdy;
      FluxData_y(i  ,j  ,k  , IV) = flux_y[IV] * dtdy;
      FluxData_y(i  ,j  ,k  , IW) = flux_y[IW] * dtdy;
          
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Z !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Z
      qLocNeighbor[ID] = Qdata(i  ,j  ,k-1, ID);
      qNeighbors_0[ID] = Qdata(i+1,j  ,k-1, ID);
      qNeighbors_1[ID] = Qdata(i-1,j  ,k-1, ID);
      qNeighbors_2[ID] = Qdata(i  ,j+1,k-1, ID);
      qNeighbors_3[ID] = Qdata(i  ,j-1,k-1, ID);
      qNeighbors_4[ID] = Qdata(i  ,j  ,k  , ID);
      qNeighbors_5[ID] = Qdata(i  ,j  ,k-2, ID);
      
      qLocNeighbor[IP] = Qdata(i  ,j  ,k-1, IP);
      qNeighbors_0[IP] = Qdata(i+1,j  ,k-1, IP);
      qNeighbors_1[IP] = Qdata(i-1,j  ,k-1, IP);
      qNeighbors_2[IP] = Qdata(i  ,j+1,k-1, IP);
      qNeighbors_3[IP] = Qdata(i  ,j-1,k-1, IP);
      qNeighbors_4[IP] = Qdata(i  ,j  ,k  , IP);
      qNeighbors_5[IP] = Qdata(i  ,j  ,k-2, IP);
      
      qLocNeighbor[IU] = Qdata(i  ,j  ,k-1, IU);
      qNeighbors_0[IU] = Qdata(i+1,j  ,k-1, IU);
      qNeighbors_1[IU] = Qdata(i-1,j  ,k-1, IU);
      qNeighbors_2[IU] = Qdata(i  ,j+1,k-1, IU);
      qNeighbors_3[IU] = Qdata(i  ,j-1,k-1, IU);
      qNeighbors_4[IU] = Qdata(i  ,j  ,k  , IU);
      qNeighbors_5[IU] = Qdata(i  ,j  ,k-2, IU);

      qLocNeighbor[IV] = Qdata(i  ,j  ,k-1, IV);
      qNeighbors_0[IV] = Qdata(i+1,j  ,k-1, IV);
      qNeighbors_1[IV] = Qdata(i-1,j  ,k-1, IV);
      qNeighbors_2[IV] = Qdata(i  ,j+1,k-1, IV);
      qNeighbors_3[IV] = Qdata(i  ,j-1,k-1, IV);
      qNeighbors_4[IV] = Qdata(i  ,j  ,k  , IV);
      qNeighbors_5[IV] = Qdata(i  ,j  ,k-2, IV);

      qLocNeighbor[IW] = Qdata(i  ,j  ,k-1, IW);
      qNeighbors_0[IW] = Qdata(i+1,j  ,k-1, IW);
      qNeighbors_1[IW] = Qdata(i-1,j  ,k-1, IW);
      qNeighbors_2[IW] = Qdata(i  ,j+1,k-1, IW);
      qNeighbors_3[IW] = Qdata(i  ,j-1,k-1, IW);
      qNeighbors_4[IW] = Qdata(i  ,j  ,k  , IW);
      qNeighbors_5[IW] = Qdata(i  ,j  ,k-2, IW);
      
      slope_unsplit_hydro_3d(qLocNeighbor, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     qNeighbors_4, qNeighbors_5,
			     dqX_neighbor, dqY_neighbor, dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Z
      //
	
      // left interface : right state
      trace_unsplit_3d_along_dir(qLoc,
				 dqX, dqY, dqZ,
				 dtdx, dtdy, dtdz,
				 FACE_ZMIN, qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(qLocNeighbor,
				 dqX_neighbor,dqY_neighbor,dqZ_neighbor,
				 dtdx, dtdy, dtdz,
				 FACE_ZMAX, qleft);

      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      swapValues(&(qleft[IU]) ,&(qleft[IW]) );
      swapValues(&(qright[IU]),&(qright[IW]));
      riemann_hydro(qleft,qright,qgdnv,flux_z,params);

      //
      // store fluxes Z
      //
      FluxData_z(i  ,j  ,k  , ID) = flux_z[ID] * dtdz;
      FluxData_z(i  ,j  ,k  , IP) = flux_z[IP] * dtdz;
      FluxData_z(i  ,j  ,k  , IU) = flux_z[IU] * dtdz;
      FluxData_z(i  ,j  ,k  , IV) = flux_z[IV] * dtdz;
      FluxData_z(i  ,j  ,k  , IW) = flux_z[IW] * dtdz;
          
    } // end if
    
  } // end operator ()
  
  DataArray3d Qdata;
  DataArray3d FluxData_x;
  DataArray3d FluxData_y;
  DataArray3d FluxData_z;
  real_t dtdx, dtdy, dtdz;
  
}; // ComputeAndStoreFluxesFunctor3D
  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor3D : public HydroBaseFunctor3D {

public:

  UpdateFunctor3D(HydroParams params,
		  DataArray3d Udata,
		  DataArray3d FluxData_x,
		  DataArray3d FluxData_y,
		  DataArray3d FluxData_z) :
    HydroBaseFunctor3D(params),
    Udata(Udata), 
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
		    DataArray3d FluxData_x,
		    DataArray3d FluxData_y,
		    DataArray3d FluxData_z,
		    int nbCells)
  {
    UpdateFunctor3D functor(params, Udata, FluxData_x, FluxData_y, FluxData_z);
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

      Udata(i  ,j  ,k    , ID) +=  FluxData_x(i  ,j  ,k  , ID);
      Udata(i  ,j  ,k    , IP) +=  FluxData_x(i  ,j  ,k  , IP);
      Udata(i  ,j  ,k    , IU) +=  FluxData_x(i  ,j  ,k  , IU);
      Udata(i  ,j  ,k    , IV) +=  FluxData_x(i  ,j  ,k  , IV);
      Udata(i  ,j  ,k    , IW) +=  FluxData_x(i  ,j  ,k  , IW);

      Udata(i  ,j  ,k    , ID) -=  FluxData_x(i+1,j  ,k   , ID);
      Udata(i  ,j  ,k    , IP) -=  FluxData_x(i+1,j  ,k   , IP);
      Udata(i  ,j  ,k    , IU) -=  FluxData_x(i+1,j  ,k   , IU);
      Udata(i  ,j  ,k    , IV) -=  FluxData_x(i+1,j  ,k   , IV);
      Udata(i  ,j  ,k    , IW) -=  FluxData_x(i+1,j  ,k   , IW);
      
      Udata(i  ,j  ,k    , ID) +=  FluxData_y(i  ,j  ,k    , ID);
      Udata(i  ,j  ,k    , IP) +=  FluxData_y(i  ,j  ,k    , IP);
      Udata(i  ,j  ,k    , IU) +=  FluxData_y(i  ,j  ,k    , IV); //
      Udata(i  ,j  ,k    , IV) +=  FluxData_y(i  ,j  ,k    , IU); //
      Udata(i  ,j  ,k    , IW) +=  FluxData_y(i  ,j  ,k    , IW);
      
      Udata(i  ,j  ,k    , ID) -=  FluxData_y(i  ,j+1,k  , ID);
      Udata(i  ,j  ,k    , IP) -=  FluxData_y(i  ,j+1,k  , IP);
      Udata(i  ,j  ,k    , IU) -=  FluxData_y(i  ,j+1,k  , IV); //
      Udata(i  ,j  ,k    , IV) -=  FluxData_y(i  ,j+1,k  , IU); //
      Udata(i  ,j  ,k    , IW) -=  FluxData_y(i  ,j+1,k  , IW);

      Udata(i  ,j  ,k    , ID) +=  FluxData_z(i  ,j  ,k    , ID);
      Udata(i  ,j  ,k    , IP) +=  FluxData_z(i  ,j  ,k    , IP);
      Udata(i  ,j  ,k    , IU) +=  FluxData_z(i  ,j  ,k    , IW); //
      Udata(i  ,j  ,k    , IV) +=  FluxData_z(i  ,j  ,k    , IV);
      Udata(i  ,j  ,k    , IW) +=  FluxData_z(i  ,j  ,k    , IU); //
      
      Udata(i  ,j  ,k    , ID) -=  FluxData_z(i  ,j  ,k+1, ID);
      Udata(i  ,j  ,k    , IP) -=  FluxData_z(i  ,j  ,k+1, IP);
      Udata(i  ,j  ,k    , IU) -=  FluxData_z(i  ,j  ,k+1, IW); //
      Udata(i  ,j  ,k    , IV) -=  FluxData_z(i  ,j  ,k+1, IV);
      Udata(i  ,j  ,k    , IW) -=  FluxData_z(i  ,j  ,k+1, IU); //

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
class UpdateDirFunctor3D : public HydroBaseFunctor3D {

public:

  UpdateDirFunctor3D(HydroParams params,
		     DataArray3d Udata,
		     DataArray3d FluxData) :
    HydroBaseFunctor3D(params),
    Udata(Udata), 
    FluxData(FluxData) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Udata,
		    DataArray3d FluxData,
		    int nbCells)
  {
    UpdateDirFunctor3D<dir> functor(params, Udata, FluxData);
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

      if (dir == XDIR) {

	Udata(i  ,j  ,k  , ID) +=  FluxData(i  ,j  ,k  , ID);
	Udata(i  ,j  ,k  , IP) +=  FluxData(i  ,j  ,k  , IP);
	Udata(i  ,j  ,k  , IU) +=  FluxData(i  ,j  ,k  , IU);
	Udata(i  ,j  ,k  , IV) +=  FluxData(i  ,j  ,k  , IV);
	Udata(i  ,j  ,k  , IW) +=  FluxData(i  ,j  ,k  , IW);
	
	Udata(i  ,j  ,k  , ID) -=  FluxData(i+1,j  ,k   , ID);
	Udata(i  ,j  ,k  , IP) -=  FluxData(i+1,j  ,k   , IP);
	Udata(i  ,j  ,k  , IU) -=  FluxData(i+1,j  ,k   , IU);
	Udata(i  ,j  ,k  , IV) -=  FluxData(i+1,j  ,k   , IV);
	Udata(i  ,j  ,k  , IW) -=  FluxData(i+1,j  ,k   , IW);

      } else if (dir == YDIR) {

	Udata(i  ,j  ,k  , ID) +=  FluxData(i  ,j  ,k  , ID);
	Udata(i  ,j  ,k  , IP) +=  FluxData(i  ,j  ,k  , IP);
	Udata(i  ,j  ,k  , IU) +=  FluxData(i  ,j  ,k  , IU);
	Udata(i  ,j  ,k  , IV) +=  FluxData(i  ,j  ,k  , IV);
	Udata(i  ,j  ,k  , IW) +=  FluxData(i  ,j  ,k  , IW);
	
	Udata(i  ,j  ,k  , ID) -=  FluxData(i  ,j+1,k   , ID);
	Udata(i  ,j  ,k  , IP) -=  FluxData(i  ,j+1,k   , IP);
	Udata(i  ,j  ,k  , IU) -=  FluxData(i  ,j+1,k   , IU);
	Udata(i  ,j  ,k  , IV) -=  FluxData(i  ,j+1,k   , IV);
	Udata(i  ,j  ,k  , IW) -=  FluxData(i  ,j+1,k   , IW);

      } else if (dir == ZDIR) {

	Udata(i  ,j  ,k  , ID) +=  FluxData(i  ,j  ,k  , ID);
	Udata(i  ,j  ,k  , IP) +=  FluxData(i  ,j  ,k  , IP);
	Udata(i  ,j  ,k  , IU) +=  FluxData(i  ,j  ,k  , IU);
	Udata(i  ,j  ,k  , IV) +=  FluxData(i  ,j  ,k  , IV);
	Udata(i  ,j  ,k  , IW) +=  FluxData(i  ,j  ,k  , IW);
	
	Udata(i  ,j  ,k  , ID) -=  FluxData(i  ,j  ,k+1 , ID);
	Udata(i  ,j  ,k  , IP) -=  FluxData(i  ,j  ,k+1 , IP);
	Udata(i  ,j  ,k  , IU) -=  FluxData(i  ,j  ,k+1 , IU);
	Udata(i  ,j  ,k  , IV) -=  FluxData(i  ,j  ,k+1 , IV);
	Udata(i  ,j  ,k  , IW) -=  FluxData(i  ,j  ,k+1,  IW);

      }
      
    } // end if
    
  } // end operator ()
  
  DataArray3d Udata;
  DataArray3d FluxData;
  
}; // UpdateDirFunctor3D

    
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor3D : public HydroBaseFunctor3D {
  
public:
  
  ComputeSlopesFunctor3D(HydroParams params,
			 DataArray3d Qdata,
			 DataArray3d Slopes_x,
			 DataArray3d Slopes_y,
			 DataArray3d Slopes_z) :
    HydroBaseFunctor3D(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y), Slopes_z(Slopes_z) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Qdata,
		    DataArray3d Slopes_x,
		    DataArray3d Slopes_y,
		    DataArray3d Slopes_z,
		    int nbCells)
  {
    ComputeSlopesFunctor3D functor(params, Qdata, Slopes_x, Slopes_y, Slopes_z);
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

    if(k >= ghostWidth-1 && k <= ksize-ghostWidth  &&
       j >= ghostWidth-1 && j <= jsize-ghostWidth  &&
       i >= ghostWidth-1 && i <= isize-ghostWidth ) {

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
	qLoc[ID]         = Qdata(i  ,j  ,k   , ID);
	qNeighbors_0[ID] = Qdata(i+1,j  ,k   , ID);
	qNeighbors_1[ID] = Qdata(i-1,j  ,k   , ID);
	qNeighbors_2[ID] = Qdata(i  ,j+1,k   , ID);
	qNeighbors_3[ID] = Qdata(i  ,j-1,k   , ID);
	qNeighbors_4[ID] = Qdata(i  ,j  ,k+1 , ID);
	qNeighbors_5[ID] = Qdata(i  ,j  ,k-1 , ID);
	
	qLoc[IP]         = Qdata(i  ,j  ,k   , IP);
	qNeighbors_0[IP] = Qdata(i+1,j  ,k   , IP);
	qNeighbors_1[IP] = Qdata(i-1,j  ,k   , IP);
	qNeighbors_2[IP] = Qdata(i  ,j+1,k   , IP);
	qNeighbors_3[IP] = Qdata(i  ,j-1,k   , IP);
	qNeighbors_4[IP] = Qdata(i  ,j  ,k+1 , IP);
	qNeighbors_5[IP] = Qdata(i  ,j  ,k-1 , IP);
	
	qLoc[IU]         = Qdata(i  ,j  ,k   , IU);
	qNeighbors_0[IU] = Qdata(i+1,j  ,k   , IU);
	qNeighbors_1[IU] = Qdata(i-1,j  ,k   , IU);
	qNeighbors_2[IU] = Qdata(i  ,j+1,k   , IU);
	qNeighbors_3[IU] = Qdata(i  ,j-1,k   , IU);
	qNeighbors_4[IU] = Qdata(i  ,j  ,k+1 , IU);
	qNeighbors_5[IU] = Qdata(i  ,j  ,k-1 , IU);
	
	qLoc[IV]         = Qdata(i  ,j  ,k   , IV);
	qNeighbors_0[IV] = Qdata(i+1,j  ,k   , IV);
	qNeighbors_1[IV] = Qdata(i-1,j  ,k   , IV);
	qNeighbors_2[IV] = Qdata(i  ,j+1,k   , IV);
	qNeighbors_3[IV] = Qdata(i  ,j-1,k   , IV);
	qNeighbors_4[IV] = Qdata(i  ,j  ,k+1 , IV);
	qNeighbors_5[IV] = Qdata(i  ,j  ,k-1 , IV);
	
	qLoc[IW]         = Qdata(i  ,j  ,k   , IW);
	qNeighbors_0[IW] = Qdata(i+1,j  ,k   , IW);
	qNeighbors_1[IW] = Qdata(i-1,j  ,k   , IW);
	qNeighbors_2[IW] = Qdata(i  ,j+1,k   , IW);
	qNeighbors_3[IW] = Qdata(i  ,j-1,k   , IW);
	qNeighbors_4[IW] = Qdata(i  ,j  ,k+1 , IW);
	qNeighbors_5[IW] = Qdata(i  ,j  ,k-1 , IW);
	
	slope_unsplit_hydro_3d(qLoc, 
			       qNeighbors_0, qNeighbors_1, 
			       qNeighbors_2, qNeighbors_3,
			       qNeighbors_4, qNeighbors_5,
			       dqX, dqY, dqZ);
	
	// copy back slopes in global arrays
	Slopes_x(i,j,k, ID) = dqX[ID];
	Slopes_y(i,j,k, ID) = dqY[ID];
	Slopes_z(i,j,k, ID) = dqZ[ID];
	
	Slopes_x(i,j,k, IP) = dqX[IP];
	Slopes_y(i,j,k, IP) = dqY[IP];
	Slopes_z(i,j,k, IP) = dqZ[IP];
	
	Slopes_x(i,j,k, IU) = dqX[IU];
	Slopes_y(i,j,k, IU) = dqY[IU];
	Slopes_z(i,j,k, IU) = dqZ[IU];
	
	Slopes_x(i,j,k, IV) = dqX[IV];
	Slopes_y(i,j,k, IV) = dqY[IV];
	Slopes_z(i,j,k, IV) = dqZ[IV];

	Slopes_x(i,j,k, IW) = dqX[IW];
	Slopes_y(i,j,k, IW) = dqY[IW];
	Slopes_z(i,j,k, IW) = dqZ[IW];
      
    } // end if
    
  } // end operator ()
  
  DataArray3d Qdata;
  DataArray3d Slopes_x, Slopes_y, Slopes_z;
  
}; // ComputeSlopesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor3D : public HydroBaseFunctor3D {
  
public:
  
  ComputeTraceAndFluxes_Functor3D(HydroParams params,
				  DataArray3d Qdata,
				  DataArray3d Slopes_x,
				  DataArray3d Slopes_y,
				  DataArray3d Slopes_z,
				  DataArray3d Fluxes,
				  real_t    dtdx,
				  real_t    dtdy,
				  real_t    dtdz) :
    HydroBaseFunctor3D(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y), Slopes_z(Slopes_z),
    Fluxes(Fluxes),
    dtdx(dtdx), dtdy(dtdy), dtdz(dtdz) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray3d Qdata,
		    DataArray3d Slopes_x,
		    DataArray3d Slopes_y,
		    DataArray3d Slopes_z,
		    DataArray3d Fluxes,
		    real_t      dtdx,
		    real_t      dtdy,
		    real_t      dtdz,
		    int nbCells)
  {
    ComputeTraceAndFluxes_Functor3D<dir> functor(params, Qdata,
						 Slopes_x, Slopes_y, Slopes_z,
						 Fluxes,
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
    
    if(k >= ghostWidth && k <= ksize-ghostWidth  &&
       j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {

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
	qLoc[ID] = Qdata   (i,j,k, ID);
	dqX[ID]  = Slopes_x(i,j,k, ID);
	dqY[ID]  = Slopes_y(i,j,k, ID);
	dqZ[ID]  = Slopes_z(i,j,k, ID);
	
	qLoc[IP] = Qdata   (i,j,k, IP);
	dqX[IP]  = Slopes_x(i,j,k, IP);
	dqY[IP]  = Slopes_y(i,j,k, IP);
	dqZ[IP]  = Slopes_z(i,j,k, IP);
	
	qLoc[IU] = Qdata   (i,j,k, IU);
	dqX[IU]  = Slopes_x(i,j,k, IU);
	dqY[IU]  = Slopes_y(i,j,k, IU);
	dqZ[IU]  = Slopes_z(i,j,k, IU);

	qLoc[IV] = Qdata   (i,j,k, IV);
	dqX[IV]  = Slopes_x(i,j,k, IV);
	dqY[IV]  = Slopes_y(i,j,k, IV);
	dqZ[IV]  = Slopes_z(i,j,k, IV);

	qLoc[IW] = Qdata   (i,j,k, IW);
	dqX[IW]  = Slopes_x(i,j,k, IW);
	dqY[IW]  = Slopes_y(i,j,k, IW);
	dqZ[IW]  = Slopes_z(i,j,k, IW);

	if (dir == XDIR) {

	  // left interface : right state
	  trace_unsplit_3d_along_dir(qLoc,
				     dqX, dqY, dqZ,
				     dtdx, dtdy, dtdz,
				     FACE_XMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i-1,j  ,k  , ID);
	  dqX_neighbor[ID] = Slopes_x(i-1,j  ,k  , ID);
	  dqY_neighbor[ID] = Slopes_y(i-1,j  ,k  , ID);
	  dqZ_neighbor[ID] = Slopes_z(i-1,j  ,k  , ID);
	  
	  qLocNeighbor[IP] = Qdata   (i-1,j  ,k  , IP);
	  dqX_neighbor[IP] = Slopes_x(i-1,j  ,k  , IP);
	  dqY_neighbor[IP] = Slopes_y(i-1,j  ,k  , IP);
	  dqZ_neighbor[IP] = Slopes_z(i-1,j  ,k  , IP);
	  
	  qLocNeighbor[IU] = Qdata   (i-1,j  ,k  , IU);
	  dqX_neighbor[IU] = Slopes_x(i-1,j  ,k  , IU);
	  dqY_neighbor[IU] = Slopes_y(i-1,j  ,k  , IU);
	  dqZ_neighbor[IU] = Slopes_z(i-1,j  ,k  , IU);
	  
	  qLocNeighbor[IV] = Qdata   (i-1,j  ,k  , IV);
	  dqX_neighbor[IV] = Slopes_x(i-1,j  ,k  , IV);
	  dqY_neighbor[IV] = Slopes_y(i-1,j  ,k  , IV);
	  dqZ_neighbor[IV] = Slopes_z(i-1,j  ,k  , IV);
	  
	  qLocNeighbor[IW] = Qdata   (i-1,j  ,k  , IW);
	  dqX_neighbor[IW] = Slopes_x(i-1,j  ,k  , IW);
	  dqY_neighbor[IW] = Slopes_y(i-1,j  ,k  , IW);
	  dqZ_neighbor[IW] = Slopes_z(i-1,j  ,k  , IW);
	  
	  // left interface : left state
	  trace_unsplit_3d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,dqZ_neighbor,
				     dtdx, dtdy, dtdz,
				     FACE_XMAX, qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hydro(qleft,qright,qgdnv,flux,params);

	  //
	  // store fluxes
	  //	
	  Fluxes(i  ,j  ,k  , ID) =  flux[ID]*dtdx;
	  Fluxes(i  ,j  ,k  , IP) =  flux[IP]*dtdx;
	  Fluxes(i  ,j  ,k  , IU) =  flux[IU]*dtdx;
	  Fluxes(i  ,j  ,k  , IV) =  flux[IV]*dtdx;
	  Fluxes(i  ,j  ,k  , IW) =  flux[IW]*dtdx;

	} else if (dir == YDIR) {

	  // left interface : right state
	  trace_unsplit_3d_along_dir(qLoc,
				     dqX, dqY, dqZ,
				     dtdx, dtdy, dtdz,
				     FACE_YMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i  ,j-1,k  , ID);
	  dqX_neighbor[ID] = Slopes_x(i  ,j-1,k  , ID);
	  dqY_neighbor[ID] = Slopes_y(i  ,j-1,k  , ID);
	  dqZ_neighbor[ID] = Slopes_z(i  ,j-1,k  , ID);
	  
	  qLocNeighbor[IP] = Qdata   (i  ,j-1,k  , IP);
	  dqX_neighbor[IP] = Slopes_x(i  ,j-1,k  , IP);
	  dqY_neighbor[IP] = Slopes_y(i  ,j-1,k  , IP);
	  dqZ_neighbor[IP] = Slopes_z(i  ,j-1,k  , IP);
	  
	  qLocNeighbor[IU] = Qdata   (i  ,j-1,k  , IU);
	  dqX_neighbor[IU] = Slopes_x(i  ,j-1,k  , IU);
	  dqY_neighbor[IU] = Slopes_y(i  ,j-1,k  , IU);
	  dqZ_neighbor[IU] = Slopes_z(i  ,j-1,k  , IU);

	  qLocNeighbor[IV] = Qdata   (i  ,j-1,k  , IV);
	  dqX_neighbor[IV] = Slopes_x(i  ,j-1,k  , IV);
	  dqY_neighbor[IV] = Slopes_y(i  ,j-1,k  , IV);
	  dqZ_neighbor[IV] = Slopes_z(i  ,j-1,k  , IV);

	  qLocNeighbor[IW] = Qdata   (i  ,j-1,k  , IW);
	  dqX_neighbor[IW] = Slopes_x(i  ,j-1,k  , IW);
	  dqY_neighbor[IW] = Slopes_y(i  ,j-1,k  , IW);
	  dqZ_neighbor[IW] = Slopes_z(i  ,j-1,k  , IW);

	  // left interface : left state
	  trace_unsplit_3d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,dqZ_neighbor,
				     dtdx, dtdy, dtdz,
				     FACE_YMAX, qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	  swapValues(&(qright[IU]),&(qright[IV]));
	  riemann_hydro(qleft,qright,qgdnv,flux,params);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i  ,j  ,k  , ID) =  flux[ID]*dtdy;
	  Fluxes(i  ,j  ,k  , IP) =  flux[IP]*dtdy;
	  Fluxes(i  ,j  ,k  , IU) =  flux[IV]*dtdy; // IU/IV swapped
	  Fluxes(i  ,j  ,k  , IV) =  flux[IU]*dtdy; // IU/IV swapped
	  Fluxes(i  ,j  ,k  , IW) =  flux[IW]*dtdy;

	} else if (dir == ZDIR) {

	  // left interface : right state
	  trace_unsplit_3d_along_dir(qLoc,
				     dqX, dqY, dqZ,
				     dtdx, dtdy, dtdz,
				     FACE_ZMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i  ,j  ,k-1  , ID);
	  dqX_neighbor[ID] = Slopes_x(i  ,j  ,k-1  , ID);
	  dqY_neighbor[ID] = Slopes_y(i  ,j  ,k-1  , ID);
	  dqZ_neighbor[ID] = Slopes_z(i  ,j  ,k-1  , ID);
	  
	  qLocNeighbor[IP] = Qdata   (i  ,j  ,k-1  , IP);
	  dqX_neighbor[IP] = Slopes_x(i  ,j  ,k-1  , IP);
	  dqY_neighbor[IP] = Slopes_y(i  ,j  ,k-1  , IP);
	  dqZ_neighbor[IP] = Slopes_z(i  ,j  ,k-1  , IP);
	  
	  qLocNeighbor[IU] = Qdata   (i  ,j  ,k-1  , IU);
	  dqX_neighbor[IU] = Slopes_x(i  ,j  ,k-1  , IU);
	  dqY_neighbor[IU] = Slopes_y(i  ,j  ,k-1  , IU);
	  dqZ_neighbor[IU] = Slopes_z(i  ,j  ,k-1  , IU);

	  qLocNeighbor[IV] = Qdata   (i  ,j  ,k-1  , IV);
	  dqX_neighbor[IV] = Slopes_x(i  ,j  ,k-1  , IV);
	  dqY_neighbor[IV] = Slopes_y(i  ,j  ,k-1  , IV);
	  dqZ_neighbor[IV] = Slopes_z(i  ,j  ,k-1  , IV);

	  qLocNeighbor[IW] = Qdata   (i  ,j  ,k-1  , IW);
	  dqX_neighbor[IW] = Slopes_x(i  ,j  ,k-1  , IW);
	  dqY_neighbor[IW] = Slopes_y(i  ,j  ,k-1  , IW);
	  dqZ_neighbor[IW] = Slopes_z(i  ,j  ,k-1  , IW);

	  // left interface : left state
	  trace_unsplit_3d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,dqZ_neighbor,
				     dtdx, dtdy, dtdz,
				     FACE_ZMAX, qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft[IU]) ,&(qleft[IW]) );
	  swapValues(&(qright[IU]),&(qright[IW]));
	  riemann_hydro(qleft,qright,qgdnv,flux,params);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i  ,j  ,k  , ID) =  flux[ID]*dtdz;
	  Fluxes(i  ,j  ,k  , IP) =  flux[IP]*dtdz;
	  Fluxes(i  ,j  ,k  , IU) =  flux[IW]*dtdz; // IU/IW swapped
	  Fluxes(i  ,j  ,k  , IV) =  flux[IV]*dtdz;
	  Fluxes(i  ,j  ,k  , IW) =  flux[IU]*dtdz; // IU/IW swapped

	}
	      
    } // end if
    
  } // end operator ()
  
  DataArray3d Qdata;
  DataArray3d Slopes_x, Slopes_y, Slopes_z;
  DataArray3d Fluxes;
  real_t dtdx, dtdy, dtdz;
  
}; // ComputeTraceAndFluxes_Functor3D

} // namespace muscl

} // namespace euler_kokkos

#endif // HYDRO_RUN_FUNCTORS_3D_H_

