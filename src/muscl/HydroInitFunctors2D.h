#ifndef HYDRO_INIT_FUNCTORS_2D_H_
#define HYDRO_INIT_FUNCTORS_2D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include <shared/euler_kokkos_config.h>
#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor2D.h"

// init conditions
#include "shared/problems/BlastParams.h"
#include "shared/problems/ImplodeParams.h"
#include "shared/problems/KHParams.h"
#include "shared/problems/GreshoParams.h"
#include "shared/problems/IsentropicVortexParams.h"
#include "shared/problems/RayleighTaylorInstabilityParams.h"
#include "shared/problems/initRiemannConfig2d.h"

// kokkos random numbers
#include <Kokkos_Random.hpp>

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor2D : public HydroBaseFunctor2D
{

public:
  InitImplodeFunctor2D(HydroParams params, ImplodeParams iparams, DataArray2d Udata)
    : HydroBaseFunctor2D(params)
    , iparams(iparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, ImplodeParams iparams, DataArray2d Udata)
  {
    InitImplodeFunctor2D functor(params, iparams, Udata);
    Kokkos::parallel_for(
      "InitImplodeFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    // outer parameters
    const real_t rho_out = this->iparams.rho_out;
    const real_t p_out = this->iparams.p_out;
    const real_t u_out = this->iparams.u_out;
    const real_t v_out = this->iparams.v_out;

    // inner parameters
    const real_t rho_in = this->iparams.rho_in;
    const real_t p_in = this->iparams.p_in;
    const real_t u_in = this->iparams.u_in;
    const real_t v_in = this->iparams.v_in;

    const int shape = this->iparams.shape;

    bool tmp;
    if (shape == 1)
      tmp = x + y * y > 0.5 && x + y * y < 1.5;
    else
      tmp = x + y > (xmin + xmax) / 2. + ymin;

    if (tmp)
    {
      Udata(i, j, ID) = rho_out;
      Udata(i, j, IP) = p_out / (gamma0 - 1.0) + 0.5 * rho_out * (u_out * u_out + v_out * v_out);
      Udata(i, j, IU) = u_out;
      Udata(i, j, IV) = v_out;
    }
    else
    {
      Udata(i, j, ID) = rho_in;
      Udata(i, j, IP) = p_in / (gamma0 - 1.0) + 0.5 * rho_in * (u_in * u_in + v_in * v_in);
      Udata(i, j, IU) = u_in;
      Udata(i, j, IV) = v_in;
    }

  } // end operator ()

  ImplodeParams iparams;
  DataArray2d   Udata;

}; // InitImplodeFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor2D : public HydroBaseFunctor2D
{

public:
  InitBlastFunctor2D(HydroParams params, BlastParams bParams, DataArray2d Udata)
    : HydroBaseFunctor2D(params)
    , bParams(bParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, BlastParams bParams, DataArray2d Udata)
  {
    InitBlastFunctor2D functor(params, bParams, Udata);
    Kokkos::parallel_for(
      "InitBlastFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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
    const real_t blast_radius = bParams.blast_radius;
    const real_t radius2 = blast_radius * blast_radius;
    const real_t blast_center_x = bParams.blast_center_x;
    const real_t blast_center_y = bParams.blast_center_y;
    const real_t blast_density_in = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out = bParams.blast_pressure_out;

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    real_t d2 =
      (x - blast_center_x) * (x - blast_center_x) + (y - blast_center_y) * (y - blast_center_y);

    if (d2 < radius2)
    {
      Udata(i, j, ID) = blast_density_in;
      Udata(i, j, IP) = blast_pressure_in / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }
    else
    {
      Udata(i, j, ID) = blast_density_out;
      Udata(i, j, IP) = blast_pressure_out / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }

  } // end operator ()

  BlastParams bParams;
  DataArray2d Udata;

}; // InitBlastFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class InitKelvinHelmholtzFunctor2D : public HydroBaseFunctor2D
{

public:
  InitKelvinHelmholtzFunctor2D(HydroParams params, KHParams khParams, DataArray2d Udata)
    : HydroBaseFunctor2D(params)
    , khParams(khParams)
    , Udata(Udata)
    , rand_pool(khParams.seed){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, KHParams khParams, DataArray2d Udata)
  {
    InitKelvinHelmholtzFunctor2D functor(params, khParams, Udata);
    Kokkos::parallel_for(
      "InitKelvinHelmholtzFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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
    // const real_t xmax = params.xmax;
    const real_t ymax = params.ymax;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    const real_t d_in = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl = khParams.amplitude;
    const real_t pressure = khParams.pressure;

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    // normalized coordinates in [0,1]
    // real_t xn = (x-xmin)/(xmax-xmin);
    real_t yn = (y - ymin) / (ymax - ymin);

    if (khParams.p_rand)
    {

      // get random number state
      rand_type rand_gen = rand_pool.get_state();

      real_t d, u, v;

      if (yn < 0.25 or yn > 0.75)
      {

        d = d_out;
        u = vflow_out;
        v = 0.0;
      }
      else
      {

        d = d_in;
        u = vflow_in;
        v = 0.0;
      }

      u += ampl * (rand_gen.drand() - 0.5);
      v += ampl * (rand_gen.drand() - 0.5);

      Udata(i, j, ID) = d;
      Udata(i, j, IU) = d * u;
      Udata(i, j, IV) = d * v;
      Udata(i, j, IE) = pressure / (gamma0 - 1.0) + 0.5 * d * (u * u + v * v);

      // free random number
      rand_pool.free_state(rand_gen);
    }
    else if (khParams.p_sine_rob)
    {

      const int    n = khParams.mode;
      const real_t w0 = khParams.w0;
      const real_t delta = khParams.delta;
      const double y1 = 0.25;
      const double y2 = 0.75;
      const double rho1 = d_in;
      const double rho2 = d_out;
      const double v1 = vflow_in;
      const double v2 = vflow_out;

      const double ramp =
        1.0 / (1.0 + exp(2 * (y - y1) / delta)) + 1.0 / (1.0 + exp(2 * (y2 - y) / delta));

      const real_t d = rho1 + ramp * (rho2 - rho1);
      const real_t u = v1 + ramp * (v2 - v1);
      const real_t v = w0 * sin(n * M_PI * x);

      Udata(i, j, ID) = d;
      Udata(i, j, IU) = d * u;
      Udata(i, j, IV) = d * v;
      Udata(i, j, IP) = pressure / (gamma0 - 1.0) + 0.5 * d * (u * u + v * v);
    }

  } // end operator ()

  KHParams    khParams;
  DataArray2d Udata;

  // random number generator
  Kokkos::Random_XorShift64_Pool<Device>                                  rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;

}; // InitKelvinHelmholtzFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class InitGreshoVortexFunctor2D : public HydroBaseFunctor2D
{

public:
  InitGreshoVortexFunctor2D(HydroParams params, GreshoParams gvParams, DataArray2d Udata)
    : HydroBaseFunctor2D(params)
    , gvParams(gvParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, GreshoParams gvParams, DataArray2d Udata)
  {
    InitGreshoVortexFunctor2D functor(params, gvParams, Udata);
    Kokkos::parallel_for(
      "InitGreshoVortexFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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
    // const real_t xmax = params.xmax;
    // const real_t ymax = params.ymax;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    const real_t rho0 = gvParams.rho0;
    const real_t Ma = gvParams.Ma;

    const real_t p0 = rho0 / (gamma0 * Ma * Ma);

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

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

    Udata(i, j, ID) = rho0;
    Udata(i, j, IU) = rho0 * (-sinT * uphi);
    Udata(i, j, IV) = rho0 * (cosT * uphi);
    Udata(i, j, IP) = p / (gamma0 - 1.0) +
                      0.5 *
                        (Udata(i, j, IU) * Udata(i, j, IU) + Udata(i, j, IV) * Udata(i, j, IV)) /
                        Udata(i, j, ID);

  } // end operator ()

  GreshoParams gvParams;
  DataArray2d  Udata;

}; // InitGreshoVortexFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class InitFourQuadrantFunctor2D : public HydroBaseFunctor2D
{

public:
  InitFourQuadrantFunctor2D(HydroParams params,
                            DataArray2d Udata,
                            int         configNumber,
                            HydroState  U0,
                            HydroState  U1,
                            HydroState  U2,
                            HydroState  U3,
                            real_t      xt,
                            real_t      yt)
    : HydroBaseFunctor2D(params)
    , Udata(Udata)
    , U0(U0)
    , U1(U1)
    , U2(U2)
    , U3(U3)
    , xt(xt)
    , yt(yt){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray2d Udata,
        int         configNumber,
        HydroState  U0,
        HydroState  U1,
        HydroState  U2,
        HydroState  U3,
        real_t      xt,
        real_t      yt)
  {
    InitFourQuadrantFunctor2D functor(params, Udata, configNumber, U0, U1, U2, U3, xt, yt);
    Kokkos::parallel_for(
      "InitFourQuadrantFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    if (x < xt)
    {
      if (y < yt)
      {
        // quarter 2
        Udata(i, j, ID) = U2[ID];
        Udata(i, j, IP) = U2[IP];
        Udata(i, j, IU) = U2[IU];
        Udata(i, j, IV) = U2[IV];
      }
      else
      {
        // quarter 1
        Udata(i, j, ID) = U1[ID];
        Udata(i, j, IP) = U1[IP];
        Udata(i, j, IU) = U1[IU];
        Udata(i, j, IV) = U1[IV];
      }
    }
    else
    {
      if (y < yt)
      {
        // quarter 3
        Udata(i, j, ID) = U3[ID];
        Udata(i, j, IP) = U3[IP];
        Udata(i, j, IU) = U3[IU];
        Udata(i, j, IV) = U3[IV];
      }
      else
      {
        // quarter 0
        Udata(i, j, ID) = U0[ID];
        Udata(i, j, IP) = U0[IP];
        Udata(i, j, IU) = U0[IU];
        Udata(i, j, IV) = U0[IV];
      }
    }

  } // end operator ()

  DataArray2d  Udata;
  HydroState2d U0, U1, U2, U3;
  real_t       xt, yt;

}; // InitFourQuadrantFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
class InitIsentropicVortexFunctor2D : public HydroBaseFunctor2D
{

public:
  InitIsentropicVortexFunctor2D(HydroParams            params,
                                IsentropicVortexParams iparams,
                                DataArray2d            Udata)
    : HydroBaseFunctor2D(params)
    , iparams(iparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, IsentropicVortexParams iparams, DataArray2d Udata)
  {
    InitIsentropicVortexFunctor2D functor(params, iparams, Udata);
    Kokkos::parallel_for(
      "InitIsentropicVortexFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    // const real_t p_a   = this->iparams.p_a;
    const real_t T_a = this->iparams.T_a;
    const real_t u_a = this->iparams.u_a;
    const real_t v_a = this->iparams.v_a;
    // const real_t w_a   = this->iparams.w_a;

    // vortex center
    const real_t vortex_x = this->iparams.vortex_x;
    const real_t vortex_y = this->iparams.vortex_y;

    // relative coordinates versus vortex center
    real_t xp = x - vortex_x;
    real_t yp = y - vortex_y;
    real_t r = sqrt(xp * xp + yp * yp);

    const real_t beta = this->iparams.beta;

    real_t du = -yp * beta / (2 * M_PI) * exp(0.5 * (1.0 - r * r));
    real_t dv = xp * beta / (2 * M_PI) * exp(0.5 * (1.0 - r * r));

    real_t T = T_a - (gamma0 - 1) * beta * beta / (8 * gamma0 * M_PI * M_PI) * exp(1.0 - r * r);
    real_t rho = rho_a * pow(T / T_a, 1.0 / (gamma0 - 1));

    Udata(i, j, ID) = rho;
    Udata(i, j, IU) = rho * (u_a + du);
    Udata(i, j, IV) = rho * (v_a + dv);
    // Udata(i  ,j  , IP) = pow(rho,gamma0)/(gamma0-1.0) +
    Udata(i, j, IP) = rho * T / (gamma0 - 1.0) + 0.5 * rho * (u_a + du) * (u_a + du) +
                      0.5 * rho * (v_a + dv) * (v_a + dv);

  } // end operator ()

  IsentropicVortexParams iparams;
  DataArray2d            Udata;

}; // InitIsentropicVortexFunctor2D

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Test of the Rayleigh-Taylor instability.
 * See
 * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 * for a description of such initial conditions
 */
class RayleighTaylorInstabilityFunctor2D : public HydroBaseFunctor2D
{

public:
  RayleighTaylorInstabilityFunctor2D(HydroParams                     params,
                                     RayleighTaylorInstabilityParams rtiparams,
                                     DataArray2d                     Udata,
                                     VectorField2d                   gravity)
    : HydroBaseFunctor2D(params)
    , rtiparams(rtiparams)
    , Udata(Udata)
    , gravity(gravity){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams                     params,
        RayleighTaylorInstabilityParams rtiparams,
        DataArray2d                     Udata,
        VectorField2d                   gravity)
  {
    RayleighTaylorInstabilityFunctor2D functor(params, rtiparams, Udata, gravity);
    Kokkos::parallel_for(
      "RayleighTaylorInstabilityFunctor2D",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
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

    const real_t Lx = xmax - xmin;
    const real_t Ly = ymax - ymin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    /* initialize perturbation amplitude */
    real_t amplitude = rtiparams.amplitude;
    real_t d0 = rtiparams.d0;
    real_t d1 = rtiparams.d1;

    // bool  randomEnabled = rti.randomEnabled;


    /* uniform static gravity field */
    const real_t gravity_x = rtiparams.gx;
    const real_t gravity_y = rtiparams.gy;

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

    if (y > (ymin + ymax) / 2)
    {
      Udata(i, j, ID) = d1;
    }
    else
    {
      Udata(i, j, ID) = d0;
    }
    Udata(i, j, IU) = 0.0;
    // if (randomEnabled)
    //   Udata(i,j,IV) = amplitude * ( rand() * 1.0 / RAND_MAX - 0.5);
    // else
    Udata(i, j, IV) = amplitude * (1 + cos(2 * M_PI * x / Lx)) * (1 + cos(2 * M_PI * y / Ly)) / 4;

    // initial hydrostatic equilibrium :
    // -dP/dz + rho*g = 0
    // P = P0 + rho g z
    Udata(i, j, IE) = (P0 + Udata(i, j, ID) * (gravity_x * x + gravity_y * y)) / (gamma0 - 1.0);

    // init gravity field
    gravity(i, j, IX) = gravity_x;
    gravity(i, j, IY) = gravity_y;

  } // end operator ()

  RayleighTaylorInstabilityParams rtiparams;
  DataArray2d                     Udata;
  VectorField2d                   gravity;

}; // class RayleighTaylorInstabilityFunctor2D

} // namespace muscl

} // namespace euler_kokkos

#endif // HYDRO_INIT_FUNCTORS_2D_H_
