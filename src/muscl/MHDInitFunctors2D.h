#ifndef MHD_INIT_FUNCTORS_2D_H_
#define MHD_INIT_FUNCTORS_2D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "MHDBaseFunctor2D.h"

// init conditions
#include "shared/problems/BlastParams.h"
#include "shared/problems/ImplodeParams.h"
#include "shared/problems/KHParams.h"
#include "shared/problems/RotorParams.h"
#include "shared/problems/FieldLoopParams.h"
#include "shared/problems/OrszagTangParams.h"

// kokkos random numbers
#include <Kokkos_Random.hpp>

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
			   ImplodeParams iparams,
               DataArray2d Udata) :
    MHDBaseFunctor2D(params), iparams(iparams), Udata(Udata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    ImplodeParams iparams,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitImplodeFunctor2D_MHD functor(params, iparams, Udata);
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
    const real_t xmax = params.xmax;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    // outer parameters
    const real_t rho_out = this->iparams.rho_out;
    const real_t p_out = this->iparams.p_out;
    const real_t u_out = this->iparams.u_out;
    const real_t v_out = this->iparams.v_out;
    const real_t Bx_out = this->iparams.Bx_out;
    const real_t By_out = this->iparams.By_out;

    // inner parameters
    const real_t rho_in = this->iparams.rho_in;
    const real_t p_in = this->iparams.p_in;
    const real_t u_in = this->iparams.u_in;
    const real_t v_in = this->iparams.v_in;
    const real_t Bx_in = this->iparams.Bx_in;
    const real_t By_in = this->iparams.By_in;

    const int shape = this->iparams.shape;

    bool tmp;
    if (shape == 1)
      tmp = x+y > 0.5 && x+y < 2.5;
    else
      tmp = x+y > (xmin+xmax)/2. + ymin;

    if (tmp) {
      Udata(i,j , ID)  = rho_out;
      Udata(i,j , IP)  = p_out/(gamma0-1.0) +
        0.5 * rho_out * (u_out*u_out + v_out*v_out) +
        0.5 * (Bx_out*Bx_out + By_out*By_out);
      Udata(i,j , IU)  = u_out;
      Udata(i,j , IV)  = v_out;
      Udata(i,j , IW)  = 0.0;
      Udata(i,j , IBX) = Bx_out;
      Udata(i,j , IBY) = By_out;
      Udata(i,j , IBZ) = 0.0;
    } else {
      Udata(i,j , ID)  = rho_in;
      Udata(i,j , IP)  = p_in/(gamma0-1.0) +
        0.5 * rho_in * (u_in*u_in + v_in*v_in) +
        0.5 * (Bx_in*Bx_in + By_in*By_in);
      Udata(i,j , IU)  = u_in;
      Udata(i,j , IV)  = v_in;
      Udata(i,j , IW)  = 0.0;
      Udata(i,j , IBX) = Bx_in;
      Udata(i,j , IBY) = By_in;
      Udata(i,j , IBZ) = 0.0;
    }

  } // end operator ()

  ImplodeParams iparams;
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
class InitOrszagTangFunctor2D : public MHDBaseFunctor2D {

private:
  enum PhaseType {
    INIT_ALL_VAR_BUT_ENERGY = 0,
    INIT_ENERGY = 1
  };


public:
  InitOrszagTangFunctor2D(HydroParams params,
			  OrszagTangParams otParams,
                          DataArray2d Udata) :
    MHDBaseFunctor2D(params), otParams(otParams), Udata(Udata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
                    OrszagTangParams otParams,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitOrszagTangFunctor2D functor(params, otParams, Udata);

    functor.phase = INIT_ALL_VAR_BUT_ENERGY;
    Kokkos::parallel_for(nbCells, functor);

    functor.phase = INIT_ENERGY;
    Kokkos::parallel_for(nbCells, functor);

  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    if (phase == INIT_ALL_VAR_BUT_ENERGY)
      init_all_var_but_energy(index);
    else if(phase == INIT_ENERGY)
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
    //const int ghostWidth = params.ghostWidth;

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
    }

  } // init_energy

  OrszagTangParams otParams;
  DataArray2d Udata;
  PhaseType   phase;

}; // InitOrszagTangFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class InitKelvinHelmholtzFunctor2D_MHD : public MHDBaseFunctor2D {

private:

public:
  InitKelvinHelmholtzFunctor2D_MHD(HydroParams params,
				   KHParams    khParams,
				   DataArray2d Udata) :
    MHDBaseFunctor2D(params),
    khParams(khParams),
    Udata(Udata),
    rand_pool(khParams.seed) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    KHParams    khParams,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitKelvinHelmholtzFunctor2D_MHD functor(params, khParams, Udata);
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
    //const real_t xmax = params.xmax;
    const real_t ymax = params.ymax;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    const real_t d_in  = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    // normalized coordinates in [0,1]
    //real_t xn = (x-xmin)/(xmax-xmin);
    real_t yn = (y-ymin)/(ymax-ymin);

    if (khParams.p_rand) {

      // get random number state
      rand_type rand_gen = rand_pool.get_state();

      real_t d, u, v;

      if ( yn < 0.25 or yn > 0.75) {

	d = d_out;
	u = vflow_out;
	v = 0.0;

      } else {

	d = d_in;
	u = vflow_in;
	v = 0.0;

      }

      u += ampl * (rand_gen.drand() - 0.5);
      v += ampl * (rand_gen.drand() - 0.5);

      const real_t bx = 0.5;
      const real_t by = 0.0;
      const real_t bz = 0.0;

      Udata(i,j,ID) = d;
      Udata(i,j,IU) = d * u;
      Udata(i,j,IV) = d * v;

      Udata(i,j,IBX) = bx;
      Udata(i,j,IBY) = by;
      Udata(i,j,IBZ) = bz;

      Udata(i,j,IE) =
	pressure/(gamma0-1.0) +
	0.5*d*(u*u + v*v) +
	0.5*(bx*bx + by*by + bz*bz);

      // free random number
      rand_pool.free_state(rand_gen);

    } else if (khParams.p_sine_rob) {

      const int    n     = khParams.mode;
      const real_t w0    = khParams.w0;
      const real_t delta = khParams.delta;
      const double y1 = 0.25;
      const double y2 = 0.75;
      const double rho1 = d_in;
      const double rho2 = d_out;
      const double v1 = vflow_in;
      const double v2 = vflow_out;

      const double ramp =
	1.0 / ( 1.0 + exp( 2*(y-y1)/delta ) ) +
	1.0 / ( 1.0 + exp( 2*(y2-y)/delta ) );

      const real_t d = rho1 + ramp*(rho2-rho1);
      const real_t u = v1   + ramp*(v2-v1);
      const real_t v = w0 * sin(n*M_PI*x);

      const real_t bx = 0.5;
      const real_t by = 0.0;
      const real_t bz = 0.0;

      Udata(i,j,ID) = d;
      Udata(i,j,IU) = d * u;
      Udata(i,j,IV) = d * v;

      Udata(i,j,IBX) = bx;
      Udata(i,j,IBY) = by;
      Udata(i,j,IBZ) = bz;

      Udata(i,j,IP) =
	pressure / (gamma0-1.0) +
	0.5*d*(u*u + v*v) +
	0.5*(bx*bx + by*by + bz*bz);

    } else if (khParams.p_sine) {

      const int    n     = khParams.mode;
      const real_t w0    = khParams.w0;
      //const real_t delta = khParams.delta;
      const double y1 = 0.25;
      const double y2 = 0.75;

      const real_t d = (y>=y1 and y<=y2) ? d_in : d_out;
      const real_t u = (y>=y1 and y<=y2) ? vflow_in : vflow_out;
      const real_t v = w0 * sin(n*M_PI*x);

      const real_t bx = 0.5;
      const real_t by = 0.0;
      const real_t bz = 0.0;

      Udata(i,j,ID) = d;
      Udata(i,j,IU) = d * u;
      Udata(i,j,IV) = d * v;

      Udata(i,j,IBX) = bx;
      Udata(i,j,IBY) = by;
      Udata(i,j,IBZ) = bz;

      Udata(i,j,IP) =
	pressure / (gamma0-1.0) +
	0.5*d*(u*u + v*v) +
	0.5*(bx*bx + by*by + bz*bz);

    }

  } // end operator ()

  KHParams khParams;
  DataArray2d Udata;

  // random number generator
  Kokkos::Random_XorShift64_Pool<Device> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;

}; // InitKelvinHelmholtzFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * The two-dimensional MHD rotor problem. Some references:
 * - Balsara and Spicer, 1999, JCP, 149, 270.
 * - G. Toth, "The div(B)=0 constraint in shock-capturing MHD codes",
 *   JCP, 161, 605 (2000)
 *
 * Initial conditions are taken from Toth's paper.
 *
 */
class InitRotorFunctor2D_MHD : public MHDBaseFunctor2D {

public:
  InitRotorFunctor2D_MHD(HydroParams params,
			 RotorParams rParams,
			 DataArray2d Udata) :
    MHDBaseFunctor2D(params), rParams(rParams), Udata(Udata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    RotorParams rParams,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitRotorFunctor2D_MHD functor(params, rParams, Udata);
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
    const real_t xmax = params.xmax;
    const real_t ymax = params.ymax;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    // rotor problem parameters
    const real_t r0      = rParams.r0;
    const real_t r1      = rParams.r1;
    const real_t u0      = rParams.u0;
    const real_t p0      = rParams.p0;
    const real_t b0      = rParams.b0;

    const real_t xCenter = (xmax + xmin)/2;
    const real_t yCenter = (ymax + ymin)/2;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    real_t r = SQRT( (x-xCenter)*(x-xCenter) +
		     (y-yCenter)*(y-yCenter) );
    real_t f_r = (r1-r)/(r1-r0);

    if (r<=r0) {
      Udata(i,j,ID) =  10.0;
      Udata(i,j,IU) = -Udata(i,j,ID)*f_r*u0*(y-yCenter)/r0;
      Udata(i,j,IV) =  Udata(i,j,ID)*f_r*u0*(x-xCenter)/r0;
    } else if (r<=r1) {
      Udata(i,j,ID) = 1+9*f_r;
      Udata(i,j,IU) = -Udata(i,j,ID)*f_r*u0*(y-yCenter)/r;
      Udata(i,j,IV) =  Udata(i,j,ID)*f_r*u0*(x-xCenter)/r;
    } else {
      Udata(i,j,ID) = 1.0;
      Udata(i,j,IU) = 0.0;
      Udata(i,j,IV) = 0.0;
    }

    Udata(i,j,IW) = 0.0;
    Udata(i,j,IA) = b0; //5.0/SQRT(FourPi);
    Udata(i,j,IB) = 0.0;
    Udata(i,j,IC) = 0.0;
    Udata(i,j,IP) = p0/(gamma0-1.0) +
      0.5*( Udata(i,j,IU)*Udata(i,j,IU) +
	    Udata(i,j,IV)*Udata(i,j,IV) +
	    Udata(i,j,IW)*Udata(i,j,IW) ) / Udata(i,j,ID) +
      0.5*( Udata(i,j,IA)*Udata(i,j,IA) );


  } // end operator ()

  RotorParams rParams;
  DataArray2d Udata;

}; // InitRotorFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * The 2D/3D MHD field loop advection problem.
 *
 * Parameters that can be set in the ini file :
 * - radius       : radius of field loop
 * - amplitude    : amplitude of vector potential (and therefore B in loop)
 * - vflow        : flow velocity
 * - densityRatio : density ratio in loop.  Enables density advection and
 *                  thermal conduction tests.
 * The flow is automatically set to run along the diagonal.
 * - direction : integer
 *   direction 0 -> field loop in x-y plane (cylinder in 3D)
 *   direction 1 -> field loop in y-z plane (cylinder in 3D)
 *   direction 2 -> field loop in z-x plane (cylinder in 3D)
 *   direction 3 -> rotated cylindrical field loop in 3D.
 *
 * Reference :
 * - T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD
 *   via constrined transport", JCP, 205, 509 (2005)
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
 *
 */
class InitFieldLoopFunctor2D_MHD : public MHDBaseFunctor2D {

private:
  enum PhaseType {
    COMPUTE_VECTOR_POTENTIAL,
    DO_INIT_CONDITION,
    DO_INIT_ENERGY
  };

public:

  struct TagComputeVectorPotential {};
  struct TagInitCond {};
  struct TagInitEnergy {};

  InitFieldLoopFunctor2D_MHD(HydroParams     params,
			     FieldLoopParams flParams,
			     DataArray2d     Udata,
			     int             nbCells) :
    MHDBaseFunctor2D(params),
    flParams(flParams),
    Udata(Udata)
  {
    Az = DataArrayScalar("Az", params.isize, params.jsize);
  };

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    FieldLoopParams flParams,
                    DataArray2d Udata,
		    int         nbCells)
  {
    InitFieldLoopFunctor2D_MHD functor(params, flParams, Udata, nbCells);

    functor.phase = COMPUTE_VECTOR_POTENTIAL;
    Kokkos::parallel_for(nbCells, functor);

    functor.phase = DO_INIT_CONDITION;
    Kokkos::parallel_for(nbCells, functor);

    functor.phase = DO_INIT_ENERGY;
    Kokkos::parallel_for(nbCells, functor);

  } // apply

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    if ( phase == COMPUTE_VECTOR_POTENTIAL ) {
      compute_vector_potential(index);
    } else if (phase == DO_INIT_CONDITION) {
      do_init_condition(index);
    } else if (phase == DO_INIT_ENERGY) {
      do_init_energy(index);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_vector_potential(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    const int nx = params.nx;
    const int ny = params.ny;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    // field loop problem parameters
    const real_t radius    = flParams.radius;
    const real_t amplitude = flParams.amplitude;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

    real_t r = sqrt(x*x+y*y);
    if ( r < radius ) {
      Az(i,j) = amplitude * ( radius - r );
    } else {
      Az(i,j) = 0.0;
    }

  } // compute_vector_potential

  KOKKOS_INLINE_FUNCTION
  void do_init_condition(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    //const real_t gamma0 = params.settings.gamma0;

    // field loop problem parameters
    const real_t radius    = flParams.radius;
    const real_t density_in= flParams.density_in;
    const real_t vflow     = flParams.vflow;
    //const real_t amp       = flParams.amp;

    const real_t cos_theta = 2.0/sqrt(5.0);
    const real_t sin_theta = sqrt(1-cos_theta*cos_theta);


    int i,j;
    index2coord(index,i,j,isize,jsize);

    if (i>=ghostWidth and i<isize-ghostWidth and
	j>=ghostWidth and j<jsize-ghostWidth) {

      real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
      real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

      real_t diag = sqrt(1.0*(nx*nx + ny*ny + nz*nz));
      real_t r    = sqrt(x*x+y*y);

      // density
      if (r < radius)
	Udata(i,j,ID) = density_in;
      else
	Udata(i,j,ID) = 1.0;

      // rho*vx
      //Udata(i,j,IU) = Udata(i,j,ID)*vflow*nx/diag;
      Udata(i,j,IU) = Udata(i,j,ID)*vflow*cos_theta;

      // rho*vy
      //Udata(i,j,IV) = Udata(i,j,ID)*vflow*ny/diag;
      Udata(i,j,IV) = Udata(i,j,ID)*vflow*sin_theta;

      // rho*vz
      Udata(i,j,IW) = Udata(i,j,ID)*vflow*nz/diag; //ZERO_F;

      // bx
      Udata(i,j,IA) =   (Az(i  ,j+1) - Az(i,j))/dy; // + amp*(drand48()-0.5);

      // by
      Udata(i,j,IB) = - (Az(i+1,j  ) - Az(i,j))/dx; // + amp*(drand48()-0.5);

      // bz
      Udata(i,j,IC) = ZERO_F;

    }

  } // do_init_condition

  KOKKOS_INLINE_FUNCTION
  void do_init_energy(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    const real_t gamma0 = params.settings.gamma0;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    if (i>=ghostWidth and i<isize-ghostWidth and
	j>=ghostWidth and j<jsize-ghostWidth )
      {

	// total energy
	if (params.settings.cIso>0) {
	  Udata(i,j,IP) = ZERO_F;
	} else {
	  Udata(i,j,IP) = 1.0f/(gamma0-1.0) +
	    0.5 * ( 0.25*SQR(Udata(i,j,IA) + Udata(i+1,j,IA)) +
		    0.25*SQR(Udata(i,j,IB) + Udata(i,j+1,IB)) ) +
	    0.5 * ( Udata(i,j,IU) * Udata(i,j,IU) +
		    Udata(i,j,IV) * Udata(i,j,IV) )/Udata(i,j,ID);
	}

      }

  } // end do_init_energy

  FieldLoopParams flParams;
  DataArray2d     Udata;

  // vector potential
  DataArrayScalar Az;

  PhaseType       phase ;

}; // InitFieldLoopFunctor2D_MHD

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_INIT_FUNCTORS_2D_H_
