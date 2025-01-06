#ifndef MHD_INIT_FUNCTORS_3D_H_
#define MHD_INIT_FUNCTORS_3D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include <shared/euler_kokkos_config.h>
#include "shared/kokkos_shared.h"
#include "MHDBaseFunctor3D.h"

// some utils
#include "shared/utils.h"

// init conditions
#include "shared/problems/BlastParams.h"
#include "shared/problems/ImplodeParams.h"
#include "shared/problems/KHParams.h"
#include "shared/problems/RotorParams.h"
#include "shared/problems/OrszagTangParams.h"

// kokkos random numbers
#include <Kokkos_Random.hpp>

namespace euler_kokkos
{
namespace muscl
{

/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  InitImplodeFunctor3D_MHD(HydroParams params, ImplodeParams iparams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , iparams(iparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, ImplodeParams iparams, DataArray3d Udata)
  {
    InitImplodeFunctor3D_MHD functor(params, iparams, Udata);
    Kokkos::parallel_for("InitImplodeFunctor3D_MHD",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IT];
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

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    real_t z = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    // outer parameters
    const real_t rho_out = this->iparams.rho_out;
    const real_t p_out = this->iparams.p_out;
    const real_t u_out = this->iparams.u_out;
    const real_t v_out = this->iparams.v_out;
    const real_t w_out = this->iparams.w_out;
    const real_t Bx_out = this->iparams.Bx_out;
    const real_t By_out = this->iparams.By_out;
    const real_t Bz_out = this->iparams.Bz_out;

    // inner parameters
    const real_t rho_in = this->iparams.rho_in;
    const real_t p_in = this->iparams.p_in;
    const real_t u_in = this->iparams.u_in;
    const real_t v_in = this->iparams.v_in;
    const real_t w_in = this->iparams.w_in;
    const real_t Bx_in = this->iparams.Bx_in;
    const real_t By_in = this->iparams.By_in;
    const real_t Bz_in = this->iparams.Bz_in;

    const int shape = this->iparams.shape;

    bool tmp;
    if (shape == 1)
      tmp = x + y + z > 0.5 && x + y + z < 2.5;
    else
      tmp = x + y + z > (xmin + xmax) / 2. + ymin + zmin;

    if (tmp)
    {
      Udata(i, j, k, ID) = rho_out;
      Udata(i, j, k, IP) = p_out / (gamma0 - 1.0) +
                           0.5 * rho_out * (u_out * u_out + v_out * v_out + w_out * w_out) +
                           0.5 * (Bx_out * Bx_out + By_out * By_out + Bz_out * Bz_out);
      Udata(i, j, k, IU) = u_out;
      Udata(i, j, k, IV) = v_out;
      Udata(i, j, k, IW) = w_out;
      Udata(i, j, k, IA) = Bx_out;
      Udata(i, j, k, IB) = By_out;
      Udata(i, j, k, IC) = Bz_out;
    }
    else
    {
      Udata(i, j, k, ID) = rho_in;
      Udata(i, j, k, IP) = p_in / (gamma0 - 1.0) +
                           0.5 * rho_in * (u_in * u_in + v_in * v_in + w_in * w_in) +
                           0.5 * (Bx_in * Bx_in + By_in * By_in + Bz_in * Bz_in);
      Udata(i, j, k, IU) = u_in;
      Udata(i, j, k, IV) = v_in;
      Udata(i, j, k, IW) = w_in;
      Udata(i, j, k, IA) = Bx_in;
      Udata(i, j, k, IB) = By_in;
      Udata(i, j, k, IC) = Bz_in;
    }

  } // end operator ()

  ImplodeParams iparams;
  DataArray3d   Udata;

}; // InitImplodeFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class InitBrioWuFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  InitBrioWuFunctor3D_MHD(HydroParams params, BrioWuParams bwparams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , bwparams(bwparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, BrioWuParams bwparams, DataArray3d Udata)
  {
    InitBrioWuFunctor3D_MHD functor(params, bwparams, Udata);
    Kokkos::parallel_for("InitBrioWuFunctor3D_MHD",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    // const int j_mpi = params.myMpiPos[IT];
    // const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    // const int                  j_mpi = 0;
    // const int                  k_mpi = 0;
#endif

    const int nx = params.nx;
    // const int ny = params.ny;
    // const int nz = params.ny;

    const real_t xmin = params.xmin;
    // const real_t xmax = params.xmax;
    //  const real_t ymin = params.ymin;
    //  const real_t zmin = params.zmin;
    const real_t dx = params.dx;
    // const real_t dy = params.dy;
    // const real_t dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    // real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    // left state
    const real_t rhoL = this->bwparams.rhoL;
    const real_t pL = this->bwparams.pL;
    const real_t uL = this->bwparams.uL;
    const real_t ByL = this->bwparams.ByL;

    // right state
    const real_t rhoR = this->bwparams.rhoR;
    const real_t pR = this->bwparams.pR;
    const real_t uR = this->bwparams.uR;
    const real_t ByR = this->bwparams.ByR;

    const real_t Bx = this->bwparams.Bx;

    const real_t xd = this->bwparams.xd;

    // init cell centered values
    if (x <= xd)
    {
      Udata(i, j, k, ID) = rhoL;
      Udata(i, j, k, IP) = pL / (gamma0 - 1.0) + 0.5 * rhoL * uL * uL + 0.5 * (Bx * Bx + ByL * ByL);
      Udata(i, j, k, IU) = rhoL * uL;
      Udata(i, j, k, IV) = 0.0;
      Udata(i, j, k, IW) = 0.0;
      Udata(i, j, k, IA) = Bx;
      Udata(i, j, k, IB) = ByL;
      Udata(i, j, k, IC) = 0.0;
    }
    else
    {
      Udata(i, j, k, ID) = rhoR;
      Udata(i, j, k, IP) = pR / (gamma0 - 1.0) + 0.5 * rhoR * uR * uR + 0.5 * (Bx * Bx + ByR * ByR);
      Udata(i, j, k, IU) = rhoR * uR;
      Udata(i, j, k, IV) = 0.0;
      Udata(i, j, k, IW) = 0.0;
      Udata(i, j, k, IA) = Bx;
      Udata(i, j, k, IB) = ByR;
      Udata(i, j, k, IC) = 0.0;
    }

  } // end operator ()

  BrioWuParams bwparams;
  DataArray3d  Udata;

}; // InitBrioWuFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  InitBlastFunctor3D_MHD(HydroParams params, BlastParams bParams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , bParams(bParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, BlastParams bParams, DataArray3d Udata)
  {
    InitBlastFunctor3D_MHD functor(params, bParams, Udata);
    Kokkos::parallel_for("InitBlastFunctor3D_MHD",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IT];
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

    const real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    const real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    const real_t z = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    real_t d2 = (x - blast_center_x) * (x - blast_center_x) +
                (y - blast_center_y) * (y - blast_center_y) +
                (z - blast_center_z) * (z - blast_center_z);

    if (d2 < radius2)
    {
      Udata(i, j, k, ID) = blast_density_in;
      Udata(i, j, k, IU) = 0.0;
      Udata(i, j, k, IV) = 0.0;
      Udata(i, j, k, IW) = 0.0;
      Udata(i, j, k, IA) = bParams.blast_bx_in;
      Udata(i, j, k, IB) = bParams.blast_by_in;
      Udata(i, j, k, IC) = bParams.blast_bz_in;
      Udata(i, j, k, IP) =
        blast_pressure_in / (gamma0 - 1.0) +
        0.5 * (SQR(Udata(i, j, k, IA)) + SQR(Udata(i, j, k, IB)) + SQR(Udata(i, j, k, IC)));
    }
    else
    {
      Udata(i, j, k, ID) = blast_density_out;
      Udata(i, j, k, IU) = 0.0;
      Udata(i, j, k, IV) = 0.0;
      Udata(i, j, k, IW) = 0.0;
      Udata(i, j, k, IA) = bParams.blast_bx_out;
      Udata(i, j, k, IB) = bParams.blast_by_out;
      Udata(i, j, k, IC) = bParams.blast_bz_out;
      Udata(i, j, k, IP) =
        blast_pressure_out / (gamma0 - 1.0) +
        0.5 * (SQR(Udata(i, j, k, IA)) + SQR(Udata(i, j, k, IB)) + SQR(Udata(i, j, k, IC)));
    }

  } // end operator ()

  BlastParams bParams;
  DataArray3d Udata;

}; // InitBlastFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class InitOrszagTangFunctor3D : public MHDBaseFunctor3D
{

public:
  struct TagInitTotalEnergy
  {};
  struct TagInitOtherVars
  {};

  InitOrszagTangFunctor3D(HydroParams params, OrszagTangParams otParams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , otParams(otParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, OrszagTangParams otParams, DataArray3d Udata)
  {
    InitOrszagTangFunctor3D functor(params, otParams, Udata);

    Kokkos::parallel_for("InitOrszagTangFunctor3D",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>, TagInitOtherVars>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);

    Kokkos::parallel_for("InitOrszagTangFunctor3D - energy",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>, TagInitTotalEnergy>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitOtherVars const &, const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    [[maybe_unused]] const int i_mpi = params.myMpiPos[IX];
    [[maybe_unused]] const int j_mpi = params.myMpiPos[IT];
    [[maybe_unused]] const int k_mpi = params.myMpiPos[IZ];
#else
    [[maybe_unused]] const int i_mpi = 0;
    [[maybe_unused]] const int j_mpi = 0;
    [[maybe_unused]] const int k_mpi = 0;
#endif

    [[maybe_unused]] const int nx = params.nx;
    [[maybe_unused]] const int ny = params.ny;
    [[maybe_unused]] const int nz = params.nz;

    [[maybe_unused]] const real_t xmin = params.xmin;
    [[maybe_unused]] const real_t xmax = params.xmax;
    [[maybe_unused]] const real_t ymin = params.ymin;
    [[maybe_unused]] const real_t ymax = params.ymax;
    [[maybe_unused]] const real_t zmin = params.zmin;
    [[maybe_unused]] const real_t zmax = params.zmax;

    [[maybe_unused]] const double dx = params.dx;
    [[maybe_unused]] const double dy = params.dy;
    [[maybe_unused]] const double dz = params.dz;

    const real_t gamma0 = params.settings.gamma0;

    const double B0 = 1.0 / sqrt(4 * PI_v<double>);
    const double p0 = gamma0 / (4 * PI_v<double>);
    const double d0 = gamma0 * p0;
    const double v0 = 1.0;
    const double kt = otParams.kt;

    double xPos = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    double yPos = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    double zPos = zmin + dz / 2 + (k + nz * k_mpi - ghostWidth) * dz;

    if (otParams.vortex_dir == OrszagTangParams::VortexDir::Z)
    {
      // density
      Udata(i, j, k, ID) = d0;

      // rho*vx
      Udata(i, j, k, IU) = static_cast<real_t>(-d0 * v0 * sin(yPos * TWOPI_F));

      // rho*vy
      Udata(i, j, k, IV) = static_cast<real_t>(d0 * v0 * sin(xPos * TWOPI_F));

      // rho*vz
      Udata(i, j, k, IW) = ZERO_F;

      // bx, by, bz
      Udata(i, j, k, IA) =
        -B0 * cos(2 * TWOPI_F * kt * (zPos - zmin) / (zmax - zmin)) * sin(yPos * TWOPI_F);
      Udata(i, j, k, IB) =
        B0 * cos(2 * TWOPI_F * kt * (zPos - zmin) / (zmax - zmin)) * sin(2.0 * xPos * TWOPI_F);
      Udata(i, j, k, IC) = 0.0;
    }
    else if (otParams.vortex_dir == OrszagTangParams::VortexDir::X)
    {
      // density
      Udata(i, j, k, ID) = d0;

      // rho*vx
      Udata(i, j, k, IV) = static_cast<real_t>(-d0 * v0 * sin(zPos * TWOPI_F));

      // rho*vy
      Udata(i, j, k, IW) = static_cast<real_t>(d0 * v0 * sin(yPos * TWOPI_F));

      // rho*vz
      Udata(i, j, k, IU) = ZERO_F;

      // bx, by, bz
      Udata(i, j, k, IB) =
        -B0 * cos(2 * TWOPI_F * kt * (xPos - xmin) / (xmax - xmin)) * sin(zPos * TWOPI_F);
      Udata(i, j, k, IC) =
        B0 * cos(2 * TWOPI_F * kt * (xPos - xmin) / (xmax - xmin)) * sin(2.0 * yPos * TWOPI_F);
      Udata(i, j, k, IA) = 0.0;
    }
    else if (otParams.vortex_dir == OrszagTangParams::VortexDir::Y)
    {
      // density
      Udata(i, j, k, ID) = d0;

      // rho*vx
      Udata(i, j, k, IW) = static_cast<real_t>(-d0 * v0 * sin(xPos * TWOPI_F));

      // rho*vy
      Udata(i, j, k, IU) = static_cast<real_t>(d0 * v0 * sin(zPos * TWOPI_F));

      // rho*vz
      Udata(i, j, k, IV) = ZERO_F;

      // bx, by, bz
      Udata(i, j, k, IC) =
        -B0 * cos(2 * TWOPI_F * kt * (yPos - ymin) / (ymax - ymin)) * sin(xPos * TWOPI_F);
      Udata(i, j, k, IA) =
        B0 * cos(2 * TWOPI_F * kt * (yPos - ymin) / (ymax - ymin)) * sin(2.0 * zPos * TWOPI_F);
      Udata(i, j, k, IB) = 0.0;
    }

  } // init_all_var_but_energy

  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitTotalEnergy const &, const int & i, const int & j, const int & k) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //  const int ghostWidth = params.ghostWidth;

    const real_t gamma0 = params.settings.gamma0;

    // const double B0    = 1.0/sqrt(2.0*TWOPI_F);
    const double p0 = gamma0 / (4 * PI_v<double>);
    // const double d0    = gamma0*p0;
    // const double v0    = 1.0;

    // double xPos = xmin + dx/2 + (i-ghostWidth)*dx;
    // double yPos = ymin + dy/2 + (j-ghostWidth)*dy;
    // double zPos = zmin + dz/2 + (k-ghostWidth)*dz;

    if (i < isize - 1 and j < jsize - 1 and k < ksize - 1)
    {

      Udata(i, j, k, IP) = p0 / (gamma0 - 1.0) +
                           0.5 * (SQR(Udata(i, j, k, IU)) / Udata(i, j, k, ID) +
                                  SQR(Udata(i, j, k, IV)) / Udata(i, j, k, ID) +
                                  SQR(Udata(i, j, k, IW)) / Udata(i, j, k, ID)) +
                           0.5 * (SQR((Udata(i, j, k, IA) + Udata(i + 1, j, k, IA)) / 2) +
                                  SQR((Udata(i, j, k, IB) + Udata(i, j + 1, k, IB)) / 2) +
                                  SQR((Udata(i, j, k, IC) + Udata(i, j, k + 1, IC)) / 2));
    }

  } // init_energy

  const OrszagTangParams otParams;
  DataArray3d            Udata;

}; // InitOrszagTangFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class InitKelvinHelmholtzFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  InitKelvinHelmholtzFunctor3D_MHD(HydroParams params, KHParams khParams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , khParams(khParams)
    , Udata(Udata)
    , rand_pool(khParams.seed){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, KHParams khParams, DataArray3d Udata)
  {
    InitKelvinHelmholtzFunctor3D_MHD functor(params, khParams, Udata);
    Kokkos::parallel_for("InitKelvinHelmholtzFunctor3D_MHD",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IT];
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

      const real_t bx = 0.5;
      const real_t by = 0.0;
      const real_t bz = 0.0;

      Udata(i, j, k, ID) = d;
      Udata(i, j, k, IU) = d * u;
      Udata(i, j, k, IV) = d * v;
      Udata(i, j, k, IW) = d * w;

      Udata(i, j, k, IA) = bx;
      Udata(i, j, k, IB) = by;
      Udata(i, j, k, IC) = bz;

      Udata(i, j, k, IP) = pressure / (gamma0 - 1.0) + 0.5 * d * (u * u + v * v + w * w) +
                           0.5 * (bx * bx + by * by + bz * bz);
    }
    else if (khParams.p_sine)
    {

      const int    n = khParams.mode;
      const real_t w0 = khParams.w0;
      // const real_t delta = khParams.delta;
      const double z1 = 0.25;
      const double z2 = 0.75;

      const real_t d = (z >= z1 and z <= z2) ? d_in : d_out;
      const real_t u = (z >= z1 and z <= z2) ? vflow_in : vflow_out;
      const real_t v = 0;
      const real_t w = w0 * sin(n * M_PI * x);

      const real_t bx = 0.5;
      const real_t by = 0.0;
      const real_t bz = 0.0;

      Udata(i, j, k, ID) = d;
      Udata(i, j, k, IU) = d * u;
      Udata(i, j, k, IV) = d * v;
      Udata(i, j, k, IW) = d * w;

      Udata(i, j, k, IA) = bx;
      Udata(i, j, k, IB) = by;
      Udata(i, j, k, IC) = bz;

      Udata(i, j, k, IP) = pressure / (gamma0 - 1.0) + 0.5 * d * (u * u + v * v + w * w) +
                           0.5 * (bx * bx + by * by + bz * bz);
    }

  } // end operator ()

  KHParams    khParams;
  DataArray3d Udata;

  // random number generator
  Kokkos::Random_XorShift64_Pool<Device>                                  rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;

}; // InitKelvinHelmholtzFunctor3D_MHD

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
class InitRotorFunctor3D_MHD : public MHDBaseFunctor3D
{

public:
  InitRotorFunctor3D_MHD(HydroParams params, RotorParams rParams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , rParams(rParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, RotorParams rParams, DataArray3d Udata)
  {
    InitRotorFunctor3D_MHD functor(params, rParams, Udata);
    Kokkos::parallel_for("InitRotorFunctor3D_MHD",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IT];
    const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif
    UNUSED(k_mpi);

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;
    UNUSED(nz);

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t xmax = params.xmax;
    const real_t ymax = params.ymax;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    // rotor problem parameters
    const real_t r0 = rParams.r0;
    const real_t r1 = rParams.r1;
    const real_t u0 = rParams.u0;
    const real_t p0 = rParams.p0;
    const real_t b0 = rParams.b0;

    const real_t xCenter = (xmax + xmin) / 2;
    const real_t yCenter = (ymax + ymin) / 2;

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

    real_t r = sqrt((x - xCenter) * (x - xCenter) + (y - yCenter) * (y - yCenter));
    real_t f_r = (r1 - r) / (r1 - r0);

    if (r <= r0)
    {
      Udata(i, j, k, ID) = 10.0;
      Udata(i, j, k, IU) = -u0 * (y - yCenter) / r0;
      Udata(i, j, k, IV) = u0 * (x - xCenter) / r0;
    }
    else if (r <= r1)
    {
      Udata(i, j, k, ID) = 1 + 9 * f_r;
      Udata(i, j, k, IU) = -f_r * u0 * (y - yCenter) / r;
      Udata(i, j, k, IV) = f_r * u0 * (x - xCenter) / r;
    }
    else
    {
      Udata(i, j, k, ID) = 1.0;
      Udata(i, j, k, IU) = 0.0;
      Udata(i, j, k, IV) = 0.0;
    }

    Udata(i, j, k, IW) = 0.0;
    Udata(i, j, k, IA) = b0; // 5.0/sqrt(FourPi);
    Udata(i, j, k, IB) = 0.0;
    Udata(i, j, k, IC) = 0.0;
    Udata(i, j, k, IP) =
      p0 / (gamma0 - 1.0) +
      (Udata(i, j, k, IU) * Udata(i, j, k, IU) + Udata(i, j, k, IV) * Udata(i, j, k, IV) +
       Udata(i, j, k, IW) * Udata(i, j, k, IW)) /
        2 / Udata(i, j, k, ID) +
      (Udata(i, j, k, IA) * Udata(i, j, k, IA)) / 2;


  } // end operator ()

  RotorParams rParams;
  DataArray3d Udata;

}; // InitRotorFunctor3D_MHD

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
class InitFieldLoopFunctor3D_MHD : public MHDBaseFunctor3D
{

private:
  enum PhaseType
  {
    COMPUTE_VECTOR_POTENTIAL,
    DO_INIT_CONDITION,
    DO_INIT_ENERGY
  };

public:
  struct TagComputeVectorPotential
  {};
  struct TagInitCond
  {};
  struct TagInitEnergy
  {};

  InitFieldLoopFunctor3D_MHD(HydroParams params, FieldLoopParams flParams, DataArray3d Udata)
    : MHDBaseFunctor3D(params)
    , flParams(flParams)
    , Udata(Udata)
  {
    A = DataArrayVector3("A", params.isize, params.jsize, params.ksize);
  };

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, FieldLoopParams flParams, DataArray3d Udata)
  {
    InitFieldLoopFunctor3D_MHD functor(params, flParams, Udata);

    functor.phase = COMPUTE_VECTOR_POTENTIAL;
    Kokkos::parallel_for("InitFieldLoopFunctor3D_MHD - vector potenetial",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);

    functor.phase = DO_INIT_CONDITION;
    Kokkos::parallel_for("InitFieldLoopFunctor3D_MHD - init",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);

    functor.phase = DO_INIT_ENERGY;
    Kokkos::parallel_for("InitFieldLoopFunctor3D_MHD - init energy",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           { 0, 0, 0 }, { params.isize, params.jsize, params.ksize }),
                         functor);

  } // apply

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, const int & k) const
  {
    if (phase == COMPUTE_VECTOR_POTENTIAL)
    {
      compute_vector_potential(i, j, k);
    }
    else if (phase == DO_INIT_CONDITION)
    {
      do_init_condition(i, j, k);
    }
    else if (phase == DO_INIT_ENERGY)
    {
      do_init_energy(i, j, k);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void
  compute_vector_potential(const int & i, const int & j, const int & k) const
  {

    const int ghostWidth = params.ghostWidth;

    const int nx = params.nx;
    const int ny = params.ny;
    // const int nz = params.nz;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IT];
    // const int k_mpi = params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    // const int k_mpi = 0;
#endif

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    // const real_t zmin = params.zmin;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    // const real_t dz = params.dz;

    // field loop problem parameters
    const real_t radius = flParams.radius;
    const real_t amplitude = flParams.amplitude;

    real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
    // real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;

    A(i, j, k, 0) = ZERO_F;
    A(i, j, k, 1) = ZERO_F;
    A(i, j, k, 2) = ZERO_F;

    real_t r = sqrt(x * x + y * y);
    if (r < radius)
    {
      A(i, j, k, IZ) = amplitude * (radius - r);
    }

  } // compute_vector_potential

  KOKKOS_INLINE_FUNCTION
  void
  do_init_condition(const int & i, const int & j, const int & k) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

#ifdef EULER_KOKKOS_USE_MPI
    const int i_mpi = params.myMpiPos[IX];
    const int j_mpi = params.myMpiPos[IT];
    // const int k_mpi = params.myMpiPos[IZ];
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
    const real_t dz = params.dz;

    // const real_t gamma0 = params.settings.gamma0;

    // field loop problem parameters
    const real_t radius = flParams.radius;
    const real_t density_in = flParams.density_in;
    // const real_t amplitude = flParams.amplitude;
    const real_t vflow = flParams.vflow;

    const real_t cos_theta = 2.0 / sqrt(5.0);
    const real_t sin_theta = sqrt(1 - cos_theta * cos_theta);

    // clang-format off
    if (i >= ghostWidth and i < isize - ghostWidth and
        j >= ghostWidth and j < jsize - ghostWidth and
        k >= ghostWidth and k < ksize - ghostWidth)
    {

      real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
      real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;
      // real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;

      real_t r = sqrt(x * x + y * y);

      // density
      if (r < radius)
        Udata(i, j, k, ID) = density_in;
      else
        Udata(i, j, k, ID) = 1.0;

      // rho*vx
      Udata(i, j, k, IU) = Udata(i, j, k, ID) * vflow * cos_theta;
      // Udata(i,j,k,IU) = Udata(i,j,k,ID)*vflow*cos_theta*(1+amp*(drand48()-0.5));
      // Udata(i,j,k,IU) = Udata(i,j,k,ID)*vflow*nx/diag;

      // rho*vy
      Udata(i, j, k, IV) = Udata(i, j, k, ID) * vflow * sin_theta;
      // Udata(i,j,k,IV) = Udata(i,j,k,ID)*vflow*ny/diag;
      // Udata(i,j,k,IV) = Udata(i,j,k,ID)*vflow*sin_theta*(1+amp*(drand48()-0.5));

      // rho*vz
      Udata(i, j, k, IW) = Udata(i, j, k, ID) * vflow;
      // Udata(i,j,k,IW) = Udata(i,j,k,ID)*vflow*(1+amp*(drand48()-0.5));
      // ZERO_F; //Udata(i,j,k,ID)*vflow*nz/diag;

      // bx
      Udata(i, j, k, IA) = (A(i, j + 1, k, 2) - A(i, j, k, 2)) / dy -
                           (A(i, j, k + 1, 1) - A(i, j, k, 1)) / dz; //+ amp*(drand48()-0.5);

      // by
      Udata(i, j, k, IB) = (A(i, j, k + 1, 0) - A(i, j, k, 0)) / dz -
                           (A(i + 1, j, k, 2) - A(i, j, k, 2)) / dx; //+ amp*(drand48()-0.5);

      // bz
      Udata(i, j, k, IC) = (A(i + 1, j, k, 1) - A(i, j, k, 1)) / dx -
                           (A(i, j + 1, k, 0) - A(i, j, k, 0)) / dy; //+ amp*(drand48()-0.5);
    }
    // clang-format on

  } // end do_init_condition

  KOKKOS_INLINE_FUNCTION
  void
  do_init_energy(const int & i, const int & j, const int & k) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    const real_t gamma0 = params.settings.gamma0;

    // clang-format off
    if (i >= ghostWidth and i < isize - ghostWidth and
        j >= ghostWidth and j < jsize - ghostWidth and
        k >= ghostWidth and k < ksize - ghostWidth)
    {

      // total energy
      if (params.settings.cIso > 0)
      {
        Udata(i, j, k, IP) = ZERO_F;
      }
      else
      {
        Udata(i, j, k, IP) =
          1.0f / (gamma0 - 1.0) +
          0.5 * (0.25 * SQR(Udata(i, j, k, IA) + Udata(i + 1, j, k, IA)) +
                 0.25 * SQR(Udata(i, j, k, IB) + Udata(i, j + 1, k, IB)) +
                 0.25 * SQR(Udata(i, j, k, IC) + Udata(i, j, k + 1, IC))) +
          0.5 *
            (Udata(i, j, k, IU) * Udata(i, j, k, IU) + Udata(i, j, k, IV) * Udata(i, j, k, IV) +
             Udata(i, j, k, IW) * Udata(i, j, k, IW)) /
            Udata(i, j, k, ID);
      }
    }
    // clang-format on

  } // end do_init_energy

  FieldLoopParams flParams;
  DataArray3d     Udata;

  // vector potential
  DataArrayVector3 A;

  PhaseType phase;

}; // InitFieldLoopFunctor3D_MHD

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_INIT_FUNCTORS_3D_H_
