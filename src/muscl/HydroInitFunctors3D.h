#ifndef HYDRO_INIT_FUNCTORS_3D_H_
#define HYDRO_INIT_FUNCTORS_3D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"

// init conditions
#include "shared/problems/BlastParams.h"
#include "shared/problems/ImplodeParams.h"
#include "shared/problems/KHParams.h"

// kokkos random numbers
#include <Kokkos_Random.hpp>

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class InitFakeFunctor3D : public HydroBaseFunctor3D
{

public:
  InitFakeFunctor3D(HydroParams params, DataArray3d Udata)
    : HydroBaseFunctor3D(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray3d Udata, int nbCells)
  {
    InitFakeFunctor3D functor(params, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;

    int i, j, k;

    index2coord(index, i, j, k, isize, jsize, ksize);

    Udata(i, j, k, ID) = 0.0;
    Udata(i, j, k, IP) = 0.0;
    Udata(i, j, k, IU) = 0.0;
    Udata(i, j, k, IV) = 0.0;
    Udata(i, j, k, IW) = 0.0;

  } // end operator ()

  DataArray3d Udata;

}; // InitFakeFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor3D : public HydroBaseFunctor3D
{

public:
  InitImplodeFunctor3D(HydroParams params, ImplodeParams iparams, DataArray3d Udata)
    : HydroBaseFunctor3D(params)
    , iparams(iparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, ImplodeParams iparams, DataArray3d Udata, int nbCells)
  {
    InitImplodeFunctor3D functor(params, iparams, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const real_t xmin = params.xmin;
    const real_t xmax = params.xmax;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    real_t z = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    // outer parameters
    const real_t rho_out = this->iparams.rho_out;
    const real_t p_out = this->iparams.p_out;
    const real_t u_out = this->iparams.u_out;
    const real_t v_out = this->iparams.v_out;
    const real_t w_out = this->iparams.w_out;

    // inner parameters
    const real_t rho_in = this->iparams.rho_in;
    const real_t p_in = this->iparams.p_in;
    const real_t u_in = this->iparams.u_in;
    const real_t v_in = this->iparams.v_in;
    const real_t w_in = this->iparams.w_in;

    bool tmp;
    if (this->iparams.shape == 1)
      tmp = x + y + z > 0.5 && x + y + z < 2.5;
    else
      tmp = x + y + z > (xmin + xmax) / 2. + ymin + zmin;

    if (tmp)
    {
      Udata(i, j, k, ID) = rho_out;
      Udata(i, j, k, IP) =
        p_out / (gamma0 - 1.0) + 0.5 * rho_out * (u_out * u_out + v_out * v_out + w_out * w_out);
      Udata(i, j, k, IU) = u_out;
      Udata(i, j, k, IV) = v_out;
      Udata(i, j, k, IW) = w_out;
    }
    else
    {
      Udata(i, j, k, ID) = rho_in;
      Udata(i, j, k, IP) =
        p_in / (gamma0 - 1.0) + 0.5 * rho_in * (u_in * u_in + v_in * v_in + w_in * w_in);
      Udata(i, j, k, IU) = u_in;
      Udata(i, j, k, IV) = v_in;
      Udata(i, j, k, IW) = w_in;
    }

  } // end operator ()

  ImplodeParams iparams;
  DataArray3d   Udata;

}; // InitImplodeFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor3D : public HydroBaseFunctor3D
{

public:
  InitBlastFunctor3D(HydroParams params, BlastParams bParams, DataArray3d Udata)
    : HydroBaseFunctor3D(params)
    , bParams(bParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, BlastParams bParams, DataArray3d Udata, int nbCells)
  {
    InitBlastFunctor3D functor(params, bParams, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius = bParams.blast_radius;
    const real_t radius2 = blast_radius * blast_radius;
    const real_t blast_center_x = bParams.blast_center_x;
    const real_t blast_center_y = bParams.blast_center_y;
    const real_t blast_center_z = bParams.blast_center_z;
    const real_t blast_density_in = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out = bParams.blast_pressure_out;


    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    real_t z = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    real_t d2 = (x - blast_center_x) * (x - blast_center_x) +
                (y - blast_center_y) * (y - blast_center_y) +
                (z - blast_center_z) * (z - blast_center_z);

    if (d2 < radius2)
    {
      Udata(i, j, k, ID) = blast_density_in;
      Udata(i, j, k, IP) = blast_pressure_in / (gamma0 - 1.0);
      Udata(i, j, k, IU) = 0.0;
      Udata(i, j, k, IV) = 0.0;
      Udata(i, j, k, IW) = 0.0;
    }
    else
    {
      Udata(i, j, k, ID) = blast_density_out;
      Udata(i, j, k, IP) = blast_pressure_out / (gamma0 - 1.0);
      Udata(i, j, k, IU) = 0.0;
      Udata(i, j, k, IV) = 0.0;
      Udata(i, j, k, IW) = 0.0;
    }

  } // end operator ()

  BlastParams bParams;
  DataArray3d Udata;

}; // InitBlastFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class InitKelvinHelmholtzFunctor3D : public HydroBaseFunctor3D
{

public:
  InitKelvinHelmholtzFunctor3D(HydroParams params, KHParams khParams, DataArray3d Udata)
    : HydroBaseFunctor3D(params)
    , khParams(khParams)
    , Udata(Udata)
    , rand_pool(khParams.seed){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, KHParams khParams, DataArray3d Udata, int nbCells)
  {
    InitKelvinHelmholtzFunctor3D functor(params, khParams, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    // const real_t xmax = params.xmax;
    // const real_t ymax = params.ymax;
    const real_t zmax = params.zmax;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    const real_t d_in = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl = khParams.amplitude;
    const real_t pressure = khParams.pressure;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    real_t z = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    // normalized coordinates in [0,1]
    // real_t xn = (x-xmin)/(xmax-xmin);
    // real_t yn = (y-ymin)/(ymax-ymin);
    real_t zn = (z - zmin) / (zmax - zmin);

    if (khParams.p_rand)
    {

      // get random number generator state
      rand_type rand_gen = rand_pool.get_state();

      real_t d, u, v, w;

      if (zn < 0.25 or zn > 0.75)
      {

        d = d_out;
        u = vflow_out;
        v = 0.0;
        w = 0.0;
      }
      else
      {

        d = d_in;
        u = vflow_in;
        v = 0.0;
        w = 0.0;
      }

      u += ampl * (rand_gen.drand() - 0.5);
      v += ampl * (rand_gen.drand() - 0.5);
      w += ampl * (rand_gen.drand() - 0.5);

      Udata(i, j, k, ID) = d;
      Udata(i, j, k, IU) = d * u;
      Udata(i, j, k, IV) = d * v;
      Udata(i, j, k, IW) = d * w;
      Udata(i, j, k, IP) = pressure / (gamma0 - 1.0) + 0.5 * d * (u * u + v * v + w * w);

      // free random number
      rand_pool.free_state(rand_gen);
    }
    else if (khParams.p_sine_rob)
    {

      const int    n = khParams.mode;
      const real_t w0 = khParams.w0;
      const real_t delta = khParams.delta;

      const double z1 = 0.25;
      const double z2 = 0.75;

      const double rho1 = d_in;
      const double rho2 = d_out;

      const double v1x = vflow_in;
      const double v2x = vflow_out;

      const double v1y = vflow_in / 2;
      const double v2y = vflow_out / 2;

      const double ramp =
        1.0 / (1.0 + exp(2 * (z - z1) / delta)) + 1.0 / (1.0 + exp(2 * (z2 - z) / delta));

      const real_t d = rho1 + ramp * (rho2 - rho1);
      const real_t u = v1x + ramp * (v2x - v1x);
      const real_t v = v1y + ramp * (v2y - v1y);
      const real_t w = w0 * sin(n * M_PI * x) * sin(n * M_PI * y);

      Udata(i, j, k, ID) = d;
      Udata(i, j, k, IU) = d * u;
      Udata(i, j, k, IV) = d * v;
      Udata(i, j, k, IW) = d * w;
      Udata(i, j, k, IP) = pressure / (gamma0 - 1.0) + 0.5 * d * (u * u + v * v + w * w);
    }

  } // end operator ()

  KHParams    khParams;
  DataArray3d Udata;

  // random number generator
  Kokkos::Random_XorShift64_Pool<Device>                                  rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;

}; // InitKelvinHelmholtzFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class InitGreshoVortexFunctor3D : public HydroBaseFunctor3D
{

public:
  InitGreshoVortexFunctor3D(HydroParams params, GreshoParams gvParams, DataArray3d Udata)
    : HydroBaseFunctor3D(params)
    , gvParams(gvParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, GreshoParams gvParams, DataArray3d Udata, int nbCells)
  {
    InitGreshoVortexFunctor3D functor(params, gvParams, Udata);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    // const int k_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;
    // const int nz = params.nz;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    // const real_t zmin = params.zmin;

    // const real_t xmax = params.xmax;
    // const real_t ymax = params.ymax;
    // const real_t zmax = params.zmax;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    // const real_t dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    const real_t rho0 = gvParams.rho0;
    const real_t Ma = gvParams.Ma;

    const real_t p0 = rho0 / (gamma0 * Ma * Ma);

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    // real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;

    real_t r = sqrt(x * x + y * y);
    real_t theta = atan2(y, x);

    // polar coordinate
    real_t cosT = cos(theta);
    real_t sinT = sin(theta);

    real_t uphi, p;

    if (r < 0.2)
    {

      uphi = 5 * r;
      p = p0 + 25 / 2.0 * r * r;
    }
    else if (r < 0.4)
    {

      uphi = 2 - 5 * r;
      p = p0 + 25 / 2.0 * r * r + 4 * (1 - 5 * r - log(0.2) + log(r));
    }
    else
    {

      uphi = 0;
      p = p0 - 2 + 4 * log(2.0);
    }

    Udata(i, j, k, ID) = rho0;
    Udata(i, j, k, IU) = rho0 * (-sinT * uphi);
    Udata(i, j, k, IV) = rho0 * (cosT * uphi);
    Udata(i, j, k, IW) = rho0 * (cosT * uphi);
    Udata(i, j, k, IP) = p / (gamma0 - 1.0) + 0.5 *
                                                (Udata(i, j, k, IU) * Udata(i, j, k, IU) +
                                                 Udata(i, j, k, IV) * Udata(i, j, k, IV) +
                                                 Udata(i, j, k, IW) * Udata(i, j, k, IW)) /
                                                Udata(i, j, k, ID);


  } // end operator ()

  GreshoParams gvParams;
  DataArray3d  Udata;

}; // InitGreshoVortexFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Test of the Rayleigh-Taylor instability.
 * See
 * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 * for a description of such initial conditions
 */
class RayleighTaylorInstabilityFunctor3D : public HydroBaseFunctor3D
{

public:
  RayleighTaylorInstabilityFunctor3D(HydroParams                     params,
                                     RayleighTaylorInstabilityParams rtiparams,
                                     DataArray3d                     Udata,
                                     VectorField3d                   gravity)
    : HydroBaseFunctor3D(params)
    , rtiparams(rtiparams)
    , Udata(Udata)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams                     params,
        RayleighTaylorInstabilityParams rtiparams,
        DataArray3d                     Udata,
        VectorField3d                   gravity)
  {
    uint64_t                           nbCells = params.isize * params.jsize * params.ksize;
    RayleighTaylorInstabilityFunctor3D functor(params, rtiparams, Udata, gravity);
    Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IY];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    const real_t xmax = params.xmax;
    const real_t ymax = params.ymax;
    const real_t zmax = params.zmax;

    const real_t Lx = xmax - xmin;
    const real_t Ly = ymax - ymin;
    // const real_t Lz = zmax-zmin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    real_t z = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    /* initialize perturbation amplitude */
    real_t amplitude = rtiparams.amplitude;
    real_t d0 = rtiparams.d0;
    real_t d1 = rtiparams.d1;

    // bool  randomEnabled = rti.randomEnabled;


    /* uniform static gravity field */
    const real_t gravity_x = rtiparams.gx;
    const real_t gravity_y = rtiparams.gy;
    const real_t gravity_z = rtiparams.gz;

    real_t P0 = 1.0;


    // the initial condition must ensure the condition of
    // hydrostatic equilibrium for pressure P = P0 - 0.1*\rho*y

    // Athena initial conditions are
    // if ( y > 0.0 ) {
    //   h_U(i,j,ID) = 2.0;
    // } else {
    //   h_U(i,j,ID) = 1.0;
    // }
    // h_U(i,j,IP) = P0 + gravity_x*x + gravity_y*y;
    // h_U(i,j,IU) = 0.0;
    // h_U(i,j,IV) = amplitude*(1+cos(2*M_PI*x))*(1+cos(0.5*M_PI*y))/4;

    if (z > (zmin + zmax) / 2)
    {
      Udata(i, j, k, ID) = d1;
    }
    else
    {
      Udata(i, j, k, ID) = d0;
    }
    Udata(i, j, k, IU) = 0.0;
    Udata(i, j, k, IV) = 0.0;
    // if (randomEnabled)
    //   Udata(i,j,IV) = amplitude * ( rand() * 1.0 / RAND_MAX - 0.5);
    // else
    Udata(i, j, k, IW) =
      amplitude * (1 + cos(2 * M_PI * x / Lx)) * (1 + cos(2 * M_PI * y / Ly)) / 4;

    Udata(i, j, k, IE) =
      (P0 + Udata(i, j, k, ID) * (gravity_x * x + gravity_y * y + gravity_z * z)) / (gamma0 - 1);

    // init gravity field
    gravity(i, j, k, IX) = gravity_x;
    gravity(i, j, k, IY) = gravity_y;
    gravity(i, j, k, IZ) = gravity_z;

  } // end operator ()

  RayleighTaylorInstabilityParams rtiparams;
  DataArray3d                     Udata;
  VectorField3d                   gravity;

}; // class RayleighTaylorInstabilityFunctor3D

} // namespace  muscl

} // namespace euler_kokkos

#endif // HYDRO_INIT_FUNCTORS_3D_H_
