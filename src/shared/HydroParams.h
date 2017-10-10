/**
 * \file HydroParams.h
 * \brief Hydrodynamics solver parameters.
 *
 * \date April, 16 2016
 * \author P. Kestener
 */
#ifndef HYDRO_PARAMS_H_
#define HYDRO_PARAMS_H_

#include "shared/kokkos_shared.h"
#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"

#include <vector>
#include <stdbool.h>
#include <string>

#include "shared/enums.h"

#ifdef USE_MPI
#include "utils/mpiUtils/MpiCommCart.h"
#endif // USE_MPI

struct HydroSettings {

  // hydro (numerical scheme) parameters
  real_t gamma0;      /*!< specific heat capacity ratio (adiabatic index)*/
  real_t gamma6;
  real_t cfl;         /*!< Courant-Friedrich-Lewy parameter.*/
  real_t slope_type;  /*!< type of slope computation (2 for second order scheme).*/
  int    iorder;      /*!< */
  real_t smallr;      /*!< small density cut-off*/
  real_t smallc;      /*!< small speed of sound cut-off*/
  real_t smallp;      /*!< small pressure cut-off*/
  real_t smallpp;     /*!< smallp times smallr*/
  real_t cIso;        /*!< if non zero, isothermal */
  real_t Omega0;      /*!< angular velocity */
  real_t cp;          /*!< specific heat (constant pressure) */
  real_t mu;          /*!< dynamic viscosity */
  real_t kappa;       /*!< thermal diffusivity */
  
  KOKKOS_INLINE_FUNCTION
  HydroSettings() : gamma0(1.4), gamma6(1.0), cfl(1.0), slope_type(2.0),
		    iorder(1),
		    smallr(1e-8), smallc(1e-8), smallp(1e-6), smallpp(1e-6),
		    cIso(0), Omega0(0.0),
		    cp(0.0), mu(0.0), kappa(0.0) {}
  
}; // struct HydroSettings

/**
 * Hydro Parameters (declaration).
 */
struct HydroParams {
  
#ifdef USE_MPI
  using MpiCommCart = hydroSimu::MpiCommCart;
#endif // USE_MPI
  
  // run parameters
  int    nStepmax;   /*!< maximun number of time steps. */
  real_t tEnd;       /*!< end of simulation time. */
  int    nOutput;    /*!< number of time steps between 2 consecutive outputs. */
  bool   enableOutput; /*!< enable output file write. */
  bool   mhdEnabled;
  int    nlog;      /*!<  number of time steps between 2 consecutive logs. */
  
  // geometry parameters
  int nx;     /*!< logical size along X (without ghost cells).*/
  int ny;     /*!< logical size along Y (without ghost cells).*/
  int nz;     /*!< logical size along Z (without ghost cells).*/
  int ghostWidth;  
  int nbvar;  /*!< number of variables in HydroState / MHDState. */
  DimensionType dimType; //!< 2D or 3D.

  int imin;   /*!< index minimum at X border*/
  int imax;   /*!< index maximum at X border*/
  int jmin;   /*!< index minimum at Y border*/
  int jmax;   /*!< index maximum at Y border*/
  int kmin;   /*!< index minimum at Z border*/
  int kmax;   /*!< index maximum at Z border*/
  
  int isize;  /*!< total size (in cell unit) along X direction with ghosts.*/
  int jsize;  /*!< total size (in cell unit) along Y direction with ghosts.*/
  int ksize;  /*!< total size (in cell unit) along Z direction with ghosts.*/

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
  bool ioVTK;   /*!< enable VTK  output file format (using VTI).*/
  bool ioHDF5;  /*!< enable HDF5 output file format.*/
  
  //! hydro settings (gamma0, ...) to be passed to Kokkos device functions
  HydroSettings settings;
  
  int niter_riemann;  /*!< number of iteration usd in quasi-exact riemann solver*/
  int riemannSolverType;
    
  // other parameters
  int implementationVersion=0; /*!< triggers which implementation to use (currently 3 versions)*/

#ifdef USE_MPI
  //! runtime determination if we are using float ou double (for MPI communication)
  //! initialized in constructor to either MpiComm::FLOAT or MpiComm::DOUBLE
  int data_type;

  //! size of the MPI cartesian grid
  int mx,my,mz;
  
  //! MPI communicator in a cartesian virtual topology
  MpiCommCart *communicator;
  
  //! number of dimension
  int nDim;
  
  //! MPI rank of current process
  int myRank;
  
  //! number of MPI processes
  int nProcs;
  
  //! MPI cartesian coordinates inside MPI topology
  Kokkos::Array<int,3> myMpiPos;
  
  //! number of MPI process neighbors (4 in 2D and 6 in 3D)
  int nNeighbors;
  
  //! MPI rank of adjacent MPI processes
  Kokkos::Array<int,6> neighborsRank;
  
  //! boundary condition type with adjacent domains (corresponding to
  //! neighbor MPI processes)
  Kokkos::Array<BoundaryConditionType,6> neighborsBC;

#endif // USE_MPI
  
  HydroParams() :
    nStepmax(0), tEnd(0.0), nOutput(0), enableOutput(true), mhdEnabled(false),
    nlog(10),
    nx(0), ny(0), nz(0), ghostWidth(2), nbvar(4), dimType(TWO_D),
    imin(0), imax(0), jmin(0), jmax(0), kmin(0), kmax(0),
    isize(0), jsize(0), ksize(0),
    xmin(0.0), xmax(1.0), ymin(0.0), ymax(1.0), zmin(0.0), zmax(1.0),
    dx(0.0), dy(0.0), dz(0.0),
    boundary_type_xmin(BC_UNDEFINED),
    boundary_type_xmax(BC_UNDEFINED),
    boundary_type_ymin(BC_UNDEFINED),
    boundary_type_ymax(BC_UNDEFINED),
    boundary_type_zmin(BC_UNDEFINED),
    boundary_type_zmax(BC_UNDEFINED),
    ioVTK(true), ioHDF5(false),
    settings(),
    niter_riemann(10), riemannSolverType(),
    implementationVersion(0)
#ifdef USE_MPI
    // init MPI-specific parameters...
#endif // USE_MPI
  {}

  virtual ~HydroParams() {}

  //! This is the genuine initialiation / setup (fed by parameter file)
  virtual void setup(ConfigMap& map);

#ifdef USE_MPI
  //! Initialize MPI-specific parameters
  void setup_mpi(ConfigMap& map);
#endif // USE_MPI
  
  void init();
  void print();
  
}; // struct HydroParams


#endif // HYDRO_PARAMS_H_
