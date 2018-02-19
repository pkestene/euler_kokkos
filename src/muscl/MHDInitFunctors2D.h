#ifndef MHD_INIT_FUNCTORS_2D_H_
#define MHD_INIT_FUNCTORS_2D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "MHDBaseFunctor2D.h"

// init conditions
#include "shared/BlastParams.h"

#include "muscl/OrszagTangInit.h"

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

namespace euler_kokkos { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor2D_MHD : public MHDBaseFunctor2D {

public:
  InitImplodeFunctor2D_MHD(HydroParams params,
			   DataArray2d Udata) :
    MHDBaseFunctor2D(params), Udata(Udata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitImplodeFunctor2D_MHD functor(params, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
    
    real_t tmp = x+y;
    if (tmp > 0.5 && tmp < 1.5) {
      Udata(i,j , ID)  = 1.0;
      Udata(i,j , IP)  = 1.0/(gamma0-1.0);
      Udata(i,j , IU)  = 0.0;
      Udata(i,j , IV)  = 0.0;
      Udata(i,j , IW)  = 0.0;
      Udata(i,j , IBX) = 0.5;
      Udata(i,j , IBY) = 0.0;
      Udata(i,j , IBZ) = 0.0;
    } else {
      Udata(i,j , ID)  = 0.125;
      Udata(i,j , IP)  = 0.14/(gamma0-1.0);
      Udata(i,j , IU)  = 0.0;
      Udata(i,j , IV)  = 0.0;
      Udata(i,j , IW)  = 0.0;
      Udata(i,j , IBX) = 0.5;
      Udata(i,j , IBY) = 0.0;
      Udata(i,j , IBZ) = 0.0;
    }
    
  } // end operator ()

  DataArray2d Udata;

}; // InitImplodeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor2D_MHD : public MHDBaseFunctor2D {

public:
  InitBlastFunctor2D_MHD(HydroParams params,
			 BlastParams bParams,
			 DataArray2d Udata) :
    MHDBaseFunctor2D(params), bParams(bParams), Udata(Udata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    BlastParams bParams,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitBlastFunctor2D_MHD functor(params, bParams, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius      = bParams.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_density_in  = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out= bParams.blast_pressure_out;
  

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    real_t d2 = 
      (x-blast_center_x)*(x-blast_center_x)+
      (y-blast_center_y)*(y-blast_center_y);    
    
    if (d2 < radius2) {
      Udata(i,j , ID) = blast_density_in;
      Udata(i,j , IU) = 0.0;
      Udata(i,j , IV) = 0.0;
      Udata(i,j , IW) = 0.0;
      Udata(i,j , IA) = 0.5;
      Udata(i,j , IB) = 0.5;
      Udata(i,j , IC) = 0.5;
      Udata(i,j , IP) = blast_pressure_in/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j , IA)) +
	       SQR(Udata(i,j , IB)) +
	       SQR(Udata(i,j , IC)) );
    } else {
      Udata(i,j , ID) = blast_density_out;
      Udata(i,j , IU) = 0.0;
      Udata(i,j , IV) = 0.0;
      Udata(i,j , IW) = 0.0;
      Udata(i,j , IA) = 0.5;
      Udata(i,j , IB) = 0.5;
      Udata(i,j , IC) = 0.5;
      Udata(i,j , IP) = blast_pressure_out/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j , IA)) +
	       SQR(Udata(i,j , IB)) +
	       SQR(Udata(i,j , IC)) );
    }
    
  } // end operator ()
  
  BlastParams bParams;
  DataArray2d Udata;
  
}; // InitBlastFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
template<OrszagTang_init_type ot_type>
class InitOrszagTangFunctor2D : public MHDBaseFunctor2D {

public:
  InitOrszagTangFunctor2D(HydroParams params,
			  DataArray2d Udata) :
    MHDBaseFunctor2D(params), Udata(Udata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitOrszagTangFunctor2D<ot_type> functor(params, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    if (ot_type == INIT_ALL_VAR_BUT_ENERGY)
      init_all_var_but_energy(index);
    else if(ot_type == INIT_ENERGY)
      init_energy(index);

  } // end operator ()

  KOKKOS_INLINE_FUNCTION
  void init_all_var_but_energy(const int index) const
  {
    
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;

    const double xmin = params.xmin;
    const double ymin = params.ymin;
        
    const double dx = params.dx;
    const double dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = gamma0/(2.0*TwoPi);
    const double d0    = gamma0*p0;
    const double v0    = 1.0;

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    double xPos = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    double yPos = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
    
    if(j < jsize  &&
       i < isize ) {

      // density
      Udata(i,j,ID) = d0;
      
      // rho*vx
      Udata(i,j,IU)  = static_cast<real_t>(-d0*v0*sin(yPos*TwoPi));
      
      // rho*vy
      Udata(i,j,IV)  = static_cast<real_t>( d0*v0*sin(xPos*TwoPi));
      
      // rho*vz
      Udata(i,j,IW) =  ZERO_F;
      
      // bx, by, bz
      Udata(i,j, IBX) = -B0*sin(    yPos*TwoPi);
      Udata(i,j, IBY) =  B0*sin(2.0*xPos*TwoPi);
      Udata(i,j, IBZ) =  0.0;

    }
    
  } // init_all_var_but_energy

  KOKKOS_INLINE_FUNCTION
  void init_energy(const int index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    //const double xmin = params.xmin;
    //const double ymin = params.ymin;
    
    //const double dx = params.dx;
    //const double dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    //const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = gamma0/(2.0*TwoPi);
    //const double d0    = gamma0*p0;
    //const double v0    = 1.0;

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    //double xPos = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    //double yPos = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
        
    if (i<isize-1 and j<jsize-1) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(i+1,j,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i,j+1,IBY)) );
    } else if ( (i <isize-1) and (j==jsize-1)) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(i+1,j           ,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i  ,2*ghostWidth,IBY)) );
    } else if ( (i==isize-1) and (j <jsize-1)) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(2*ghostWidth,j  ,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i           ,j+1,IBY)) );
    } else if ( (i==isize-1) and (j==jsize-1) ) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(2*ghostWidth,j ,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i,2*ghostWidth ,IBY)) );
    }
    
  } // init_energy
  
  DataArray2d Udata;
  
}; // InitOrszagTangFunctor2D

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_INIT_FUNCTORS_2D_H_
