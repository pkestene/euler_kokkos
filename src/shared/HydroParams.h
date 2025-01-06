/**
 * \file HydroParams.h
 * \brief Hydrodynamics solver parameters.
 *
 * \date April, 16 2016
 * \author P. Kestener
 */
#ifndef HYDRO_PARAMS_H_
#define HYDRO_PARAMS_H_

#include <shared/euler_kokkos_config.h>
#include "shared/kokkos_shared.h"
#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"

#include <vector>
#include <stdbool.h>
#include <string>

#include "shared/enums.h"

#include <utils/mpi/ParallelEnv.h>
#ifdef EULER_KOKKOS_USE_MPI
#  include <utils/mpi/MpiCommCart.h>
#endif // EULER_KOKKOS_USE_MPI

namespace euler_kokkos
{

// ==================================================================================
// ==================================================================================
struct HydroSettings
{

  // hydro (numerical scheme) parameters
  real_t gamma0; /*!< specific heat capacity ratio (adiabatic index)*/
  real_t gamma6;
  real_t cfl;        /*!< Courant-Friedrich-Lewy parameter.*/
  real_t slope_type; /*!< type of slope computation (2 for second order scheme).*/
  int    iorder;     /*!< */
  real_t smallr;     /*!< small density cut-off*/
  real_t smallc;     /*!< small speed of sound cut-off*/
  real_t smallp;     /*!< small pressure cut-off*/
  real_t smallpp;    /*!< smallp times smallr*/
  real_t cIso;       /*!< if non zero, isothermal */
  real_t Omega0;     /*!< angular velocity */
  real_t cp;         /*!< specific heat (constant pressure) */
  real_t mu;         /*!< dynamic viscosity */
  real_t kappa;      /*!< thermal diffusivity */

  KOKKOS_INLINE_FUNCTION
  HydroSettings()
    : gamma0(1.4)
    , gamma6(1.0)
    , cfl(1.0)
    , slope_type(2.0)
    , iorder(1)
    , smallr(1e-8)
    , smallc(1e-8)
    , smallp(1e-6)
    , smallpp(1e-6)
    , cIso(0)
    , Omega0(0.0)
    , cp(0.0)
    , mu(0.0)
    , kappa(0.0)
  {}

}; // struct HydroSettings

// ==================================================================================
// ==================================================================================
DimensionType
get_dim(ConfigMap const & configMap);

// ==================================================================================
// ==================================================================================
/**
 * Hydro Parameters (declaration).
 */
class HydroParams
{
public:
  // run parameters
  int    nStepmax;     /*!< maximum number of time steps. */
  real_t tEnd;         /*!< end of simulation time. */
  int    nOutput;      /*!< number of time steps between 2 consecutive outputs. */
  bool   enableOutput; /*!< enable output file write. */
  bool   mhdEnabled;
  int    nlog; /*!<  number of time steps between 2 consecutive logs. */

  // geometry parameters
  int           nx; /*!< logical size along X (without ghost cells).*/
  int           ny; /*!< logical size along Y (without ghost cells).*/
  int           nz; /*!< logical size along Z (without ghost cells).*/
  int           ghostWidth;
  int           nbvar;   /*!< number of variables in HydroState / MHDState. */
  DimensionType dimType; //!< 2D or 3D.

  int imin; /*!< index minimum at X border*/
  int imax; /*!< index maximum at X border*/
  int jmin; /*!< index minimum at Y border*/
  int jmax; /*!< index maximum at Y border*/
  int kmin; /*!< index minimum at Z border*/
  int kmax; /*!< index maximum at Z border*/

  int isize; /*!< total size (in cell unit) along X direction with ghosts.*/
  int jsize; /*!< total size (in cell unit) along Y direction with ghosts.*/
  int ksize; /*!< total size (in cell unit) along Z direction with ghosts.*/

  real_t xmin; /*!< domain bound */
  real_t xmax; /*!< domain bound */
  real_t ymin; /*!< domain bound */
  real_t ymax; /*!< domain bound */
  real_t zmin; /*!< domain bound */
  real_t zmax; /*!< domain bound */
  real_t dx;   /*!< x resolution */
  real_t dy;   /*!< y resolution */
  real_t dz;   /*!< z resolution */

  BoundaryConditionType boundary_type_xmin; /*!< boundary condition */
  BoundaryConditionType boundary_type_xmax; /*!< boundary condition */
  BoundaryConditionType boundary_type_ymin; /*!< boundary condition */
  BoundaryConditionType boundary_type_ymax; /*!< boundary condition */
  BoundaryConditionType boundary_type_zmin; /*!< boundary condition */
  BoundaryConditionType boundary_type_zmax; /*!< boundary condition */

  // IO parameters
  bool ioVTK;  /*!< enable VTK  output file format (using VTI).*/
  bool ioHDF5; /*!< enable HDF5 output file format.*/

  //! hydro settings (gamma0, ...) to be passed to Kokkos device functions
  HydroSettings settings;

  int niter_riemann; /*!< number of iteration usd in quasi-exact riemann solver*/
  int riemannSolverType;

  // other parameters
  int implementationVersion = 0; /*!< triggers which implementation to use (currently 3 versions)*/

#ifdef EULER_KOKKOS_USE_MPI
  //! parallel environment
  ParallelEnv & par_env;

  //! size of the MPI cartesian grid
  int mx, my, mz;

  //! number of dimension
  int nDim;

  //! MPI rank of current process
  int myRank;

  //! number of MPI processes
  int nProcs;

  //! MPI cartesian coordinates inside MPI topology
  Kokkos::Array<int, 3> myMpiPos;

  //! number of MPI process neighbors (4 in 2D and 6 in 3D)
  int nNeighbors;

  //! MPI rank of adjacent MPI processes
  Kokkos::Array<int, 6> neighborsRank;

  //! boundary condition type with adjacent domains (corresponding to
  //! neighbor MPI processes)
  Kokkos::Array<BoundaryConditionType, 6> neighborsBC;

#endif // EULER_KOKKOS_USE_MPI

  HydroParams(ConfigMap const & configMap, ParallelEnv & par_env_);

  virtual ~HydroParams() {}

  void
  print();

private:
  void
  init();

#ifdef EULER_KOKKOS_USE_MPI
  //! Initialize MPI-specific parameters
  void
  setup_mpi(ConfigMap const & map);

public:
  MpiComm &
  communicator()
  {
    return par_env.comm();
  }
#endif // EULER_KOKKOS_USE_MPI


}; // struct HydroParams

} // namespace euler_kokkos

#endif // HYDRO_PARAMS_H_
