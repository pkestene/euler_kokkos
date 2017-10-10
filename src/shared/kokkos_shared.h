#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <impl/Kokkos_Error.hpp>

#include "shared/real_type.h"
#include "shared/utils.h"


#ifdef CUDA
# define DEVICE Kokkos::Cuda
#include <cuda.h>
#endif

#ifdef OPENMP
# define DEVICE Kokkos::OpenMP
#endif

#ifndef DEVICE
# define DEVICE Kokkos::OpenMP
#endif

enum KokkosLayout {
  KOKKOS_LAYOUT_LEFT,
  KOKKOS_LAYOUT_RIGHT
};

// last index is hydro variable
// n-1 first indexes are space (i,j,k,....)
typedef Kokkos::View<real_t***, DEVICE>   DataArray2d;
typedef DataArray2d::HostMirror           DataArray2dHost;

typedef Kokkos::View<real_t****, DEVICE>  DataArray3d;
typedef DataArray3d::HostMirror           DataArray3dHost;
//typedef DataArray2d     DataArray3d;
//typedef DataArray2dHost DataArray3dHost;

// for 2D
typedef Kokkos::View<real_t**,        DEVICE> DataArrayScalar;
typedef DataArrayScalar::HostMirror           DataArrayScalarHost;

// for 3D
typedef Kokkos::View<real_t***[3],     DEVICE> DataArrayVector3;
typedef DataArrayVector3::HostMirror           DataArrayVector3Host;

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a prefered layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 *
 * These function will eventually disappear.
 * We still need then as long as parallel_reduce does not accept MDRange policy.
 */

/* 2D */

KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny)
{
  UNUSED(Nx);
  UNUSED(Ny);
  
#ifdef CUDA
  j = index / Nx;
  i = index - j*Nx;
#else
  i = index / Ny;
  j = index - i*Ny;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i, int j, int Nx, int Ny)
{
  UNUSED(Nx);
  UNUSED(Ny);
#ifdef CUDA
  return i + Nx*j; // left layout
#else
  return j + Ny*i; // right layout
#endif
}

/* 3D */

KOKKOS_INLINE_FUNCTION
void index2coord(int index,
                 int &i, int &j, int &k,
                 int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef CUDA
  int NxNy = Nx*Ny;
  k = index / NxNy;
  j = (index - k*NxNy) / Nx;
  i = index - j*Nx - k*NxNy;
#else
  int NyNz = Ny*Nz;
  i = index / NyNz;
  j = (index - i*NyNz) / Nz;
  k = index - j*Nz - i*NyNz;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i,  int j,  int k,
                int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef CUDA
  return i + Nx*j + Nx*Ny*k; // left layout
#else
  return k + Nz*j + Nz*Ny*i; // right layout
#endif
}


#endif // KOKKOS_SHARED_H_
