#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>

#include "shared/real_type.h"
#include "shared/utils.h"

namespace euler_kokkos
{

using Device = Kokkos::DefaultExecutionSpace;

enum KokkosLayout
{
  KOKKOS_LAYOUT_LEFT,
  KOKKOS_LAYOUT_RIGHT
};

// last index is hydro variable
// n-1 first indexes are space (i,j,k,....)
using DataArray2d = Kokkos::View<real_t ***, Device>;
using DataArray2dHost = DataArray2d::HostMirror;

using DataArray3d = Kokkos::View<real_t ****, Device>;
using DataArray3dHost = DataArray3d::HostMirror;

// for 2D
using DataArrayScalar = Kokkos::View<real_t **, Device>;
using DataArrayScalarHost = DataArrayScalar::HostMirror;
using DataArrayVector2 = Kokkos::View<real_t ** [2], Device>;
using DataArrayVector2Host = DataArrayVector2::HostMirror;
using VectorField2d = DataArrayVector2;
using VectorField2dHost = DataArrayVector2::HostMirror;

// for 3D
using DataArrayVector3 = Kokkos::View<real_t *** [3], Device>;
using DataArrayVector3Host = DataArrayVector3::HostMirror;
using VectorField3d = DataArrayVector3;
using VectorField3dHost = DataArrayVector3::HostMirror;

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a preferred layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 *
 * These function will eventually disappear.
 * We still need then as long as parallel_reduce does not accept MDRange policy.
 */

/* 2D */

KOKKOS_INLINE_FUNCTION
void
index2coord(int64_t index, int & i, int & j, int Nx, int Ny)
{
  UNUSED(Nx);
  UNUSED(Ny);

#ifdef KOKKOS_ENABLE_CUDA
  j = index / Nx;
  i = index - j * Nx;
#else
  i = index / Ny;
  j = index - i * Ny;
#endif
}

KOKKOS_INLINE_FUNCTION
int64_t
coord2index(int i, int j, int Nx, int Ny)
{
  UNUSED(Nx);
  UNUSED(Ny);
#ifdef KOKKOS_ENABLE_CUDA
  int64_t res = i + Nx * j; // left layout
#else
  int64_t res = j + Ny * i; // right layout
#endif
  return res;
}

/* 3D */

KOKKOS_INLINE_FUNCTION
void
index2coord(int64_t index, int & i, int & j, int & k, int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  int NxNy = Nx * Ny;
  k = index / NxNy;
  j = (index - k * NxNy) / Nx;
  i = index - j * Nx - k * NxNy;
#else
  int     NyNz = Ny * Nz;
  i = index / NyNz;
  j = (index - i * NyNz) / Nz;
  k = index - j * Nz - i * NyNz;
#endif
}

KOKKOS_INLINE_FUNCTION
int64_t
coord2index(int i, int j, int k, int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  int64_t res = i + Nx * j + Nx * Ny * k; // left layout
#else
  int64_t res = k + Nz * j + Nz * Ny * i; // right layout
#endif
  return res;
}

} // namespace euler_kokkos

#endif // KOKKOS_SHARED_H_
