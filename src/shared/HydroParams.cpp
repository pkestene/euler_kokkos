#include "HydroParams.h"

#include <cstdlib> // for exit
#include <cstdio>  // for fprintf
#include <cstring> // for strcmp
#include <iostream>

#include "config/inih/ini.h" // our INI file reader

#ifdef USE_MPI
using namespace hydroSimu;
#endif // USE_MPI

// =======================================================
// =======================================================
/*
 * Hydro Parameters (read parameter file)
 */
void HydroParams::setup(ConfigMap &configMap)
{

  /* initialize RUN parameters */
  nStepmax = configMap.getInteger("run","nstepmax",1000);
  tEnd     = configMap.getFloat  ("run","tend",0.0);
  nOutput  = configMap.getInteger("run","noutput",100);
  if (nOutput == 0)
    enableOutput = false;

  nlog     = configMap.getInteger("run","nlog",10);
  
  std::string solver_name = configMap.getString("run", "solver_name", "unknown");

  if ( !solver_name.compare("Hydro_Muscl_2D") ) {
    
    dimType = TWO_D;
    nbvar = 4;
    ghostWidth = 2;
    
  } else if ( !solver_name.compare("Hydro_Muscl_3D")) {

    dimType = THREE_D;
    nbvar = 5;
    ghostWidth = 2;
    
  } else if ( !solver_name.compare("MHD_Muscl_2D") ) {

    dimType = TWO_D;
    nbvar = 8;
    ghostWidth = 3;
    mhdEnabled = true;
    
  } else if ( !solver_name.compare("MHD_Muscl_3D") ) {

    dimType = THREE_D;
    nbvar = 8;
    ghostWidth = 3;
    mhdEnabled = true;
    
  } else {

    // we should probably abort
    std::cerr << "Solver name not valid : " << solver_name << "\n";
    
  }
  
  /* initialize MESH parameters */
  nx = configMap.getInteger("mesh","nx", 1);
  ny = configMap.getInteger("mesh","ny", 1);
  nz = configMap.getInteger("mesh","nz", 1);

  xmin = configMap.getFloat("mesh", "xmin", 0.0);
  ymin = configMap.getFloat("mesh", "ymin", 0.0);
  zmin = configMap.getFloat("mesh", "zmin", 0.0);

  xmax = configMap.getFloat("mesh", "xmax", 1.0);
  ymax = configMap.getFloat("mesh", "ymax", 1.0);
  zmax = configMap.getFloat("mesh", "zmax", 1.0);

  boundary_type_xmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmin", BC_DIRICHLET));
  boundary_type_xmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmax", BC_DIRICHLET));
  boundary_type_ymin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymin", BC_DIRICHLET));
  boundary_type_ymax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymax", BC_DIRICHLET));
  boundary_type_zmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmin", BC_DIRICHLET));
  boundary_type_zmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmax", BC_DIRICHLET));

  settings.gamma0         = configMap.getFloat("hydro","gamma0", 1.4);
  settings.cfl            = configMap.getFloat("hydro", "cfl", 0.5);
  settings.iorder         = configMap.getInteger("hydro","iorder", 2);
  settings.slope_type     = configMap.getFloat("hydro","slope_type",1.0);
  settings.smallc         = configMap.getFloat("hydro","smallc", 1e-10);
  settings.smallr         = configMap.getFloat("hydro","smallr", 1e-10);

  // specific heat
  settings.cp             = configMap.getFloat("hydro", "cp", 0.0);

  // dynamic viscosity
  settings.mu             = configMap.getFloat("hydro", "mu", 0.0);

  // thermal diffusivity
  settings.kappa          = configMap.getFloat("hydro", "kappa", 0.0);
  
  niter_riemann  = configMap.getInteger("hydro","niter_riemann", 10);
  std::string riemannSolverStr = std::string(configMap.getString("hydro","riemann", "approx"));
  if ( !riemannSolverStr.compare("approx") ) {
    riemannSolverType = RIEMANN_APPROX;
  } else if ( !riemannSolverStr.compare("llf") ) {
    riemannSolverType = RIEMANN_LLF;
  } else if ( !riemannSolverStr.compare("hll") ) {
    riemannSolverType = RIEMANN_HLL;
  } else if ( !riemannSolverStr.compare("hllc") ) {
    riemannSolverType = RIEMANN_HLLC;
  } else if ( !riemannSolverStr.compare("hlld") ) {
    riemannSolverType = RIEMANN_HLLD;
  } else {
    std::cout << "Riemann Solver specified in parameter file is invalid\n";
    std::cout << "Use the default one : approx\n";
    riemannSolverType = RIEMANN_APPROX;
  }
  
  implementationVersion  = configMap.getFloat("OTHER","implementationVersion", 0);
  if (implementationVersion != 0 and
      implementationVersion != 1) {
    std::cout << "Implementation version is invalid (must be 0 or 1)\n";
    std::cout << "Use the default : 0\n";
    implementationVersion = 0;
  }

  init();

#ifdef USE_MPI
  setup_mpi(configMap);
#endif // USE_MPI
  
} // HydroParams::setup

#ifdef USE_MPI
// =======================================================
// =======================================================
void HydroParams::setup_mpi(ConfigMap& configMap)
{

  // runtime determination if we are using float ou double (for MPI communication)
  data_type = typeid(1.0f).name() == typeid((real_t)1.0f).name() ?
    hydroSimu::MpiComm::FLOAT : hydroSimu::MpiComm::DOUBLE;
  
  // MPI parameters :
  mx = configMap.getInteger("mpi", "mx", 1);
  my = configMap.getInteger("mpi", "my", 1);
  mz = configMap.getInteger("mpi", "mz", 1);
    
  // check that parameters are consistent
  bool error = false;
  error |= (mx < 1);
  error |= (my < 1);
  error |= (mz < 1);

  // get world communicator size and check it is consistent with mesh grid sizes
  nProcs = MpiComm::world().getNProc();
  if (nProcs != mx*my*mz) {
    std::cerr << "Inconsistent MPI cartesian virtual topology geometry; \n mx*my*mz must match with parameter given to mpirun !!!\n";
    
  }

  // create the MPI communicator for our cartesian mesh
  if (dimType == TWO_D) {
    communicator = new MpiCommCart(mx, my, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
    nDim = 2;
  } else {
    communicator = new MpiCommCart(mx, my, mz, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
    nDim = 3;
  }
  
  // get my MPI rank inside topology
  myRank = communicator->getRank();
  
  // get my coordinates inside topology
  // myMpiPos[0] is between 0 and mx-1
  // myMpiPos[1] is between 0 and my-1
  // myMpiPos[2] is between 0 and mz-1
  //myMpiPos.resize(nDim);
  {
    int mpiPos[3] = {0,0,0};
    communicator->getMyCoords(&mpiPos[0]);
    myMpiPos[0] = mpiPos[0];
    myMpiPos[1] = mpiPos[1];
    myMpiPos[2] = mpiPos[2];
  }
  
  /*
   * compute MPI ranks of our neighbors and 
   * set default boundary condition types
   */
  if (dimType == TWO_D) {
    nNeighbors = N_NEIGHBORS_2D;
    neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
    neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
    neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
    neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
    neighborsRank[Z_MIN] = 0;
    neighborsRank[Z_MAX] = 0;
    
    neighborsBC[X_MIN] = BC_COPY;
    neighborsBC[X_MAX] = BC_COPY;
    neighborsBC[Y_MIN] = BC_COPY;
    neighborsBC[Y_MAX] = BC_COPY;
    neighborsBC[Z_MIN] = BC_UNDEFINED;
    neighborsBC[Z_MAX] = BC_UNDEFINED;
  } else {
    nNeighbors = N_NEIGHBORS_3D;
    neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
    neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
    neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
    neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
    neighborsRank[Z_MIN] = communicator->getNeighborRank<Z_MIN>();
    neighborsRank[Z_MAX] = communicator->getNeighborRank<Z_MAX>();
    
    neighborsBC[X_MIN] = BC_COPY;
    neighborsBC[X_MAX] = BC_COPY;
    neighborsBC[Y_MIN] = BC_COPY;
    neighborsBC[Y_MAX] = BC_COPY;
    neighborsBC[Z_MIN] = BC_COPY;
    neighborsBC[Z_MAX] = BC_COPY;
  }
  
  /*
   * identify outside boundaries (no actual communication if we are
   * doing BC_DIRICHLET or BC_NEUMANN)
   *
   * Please notice the duality 
   * XMIN -- boundary_xmax
   * XMAX -- boundary_xmin
   *
   */
  
  // X_MIN boundary
  if (myMpiPos[DIR_X] == 0)
    neighborsBC[X_MIN] = boundary_type_xmin;
  
  // X_MAX boundary
  if (myMpiPos[DIR_X] == mx-1)
    neighborsBC[X_MAX] = boundary_type_xmax;
  
  // Y_MIN boundary
  if (myMpiPos[DIR_Y] == 0)
    neighborsBC[Y_MIN] = boundary_type_ymin;
  
  // Y_MAX boundary
  if (myMpiPos[DIR_Y] == my-1)
    neighborsBC[Y_MAX] = boundary_type_ymax;
  
  if (dimType == THREE_D) {
    
    // Z_MIN boundary
    if (myMpiPos[DIR_Z] == 0)
      neighborsBC[Z_MIN] = boundary_type_zmin;
    
    // Y_MAX boundary
    if (myMpiPos[DIR_Z] == mz-1)
      neighborsBC[Z_MAX] = boundary_type_zmax;
    
  } // end THREE_D
  
  /*
   * Initialize CUDA device if needed.
   * 
   * Let's assume hwloc is doing its job !
   *
   * Old comments from RamsesGPU:
   * When running on a Linux machine with mutiple GPU per node, it might be
   * very helpfull if admin has set the CUDA device compute mode to exclusive
   * so that a device is only attached to 1 host thread (i.e. 2 different host
   * thread can not communicate with the same GPU).
   *
   * As a sys-admin, just run for all devices command:
   *   nvidia-smi -g $(DEV_ID) -c 1
   *
   * If compute mode is set to normal mode, we need to use cudaSetDevice, 
   * so that each MPI device is mapped onto a different GPU device.
   * 
   * At CCRT, on machine Titane, each node (2 quadri-proc) "sees" only 
   * half a Tesla S1070, that means cudaGetDeviceCount should return 2.
   * If we want the ration 1 MPI process <-> 1 GPU, we need to allocate
   * N nodes and 2*N tasks (MPI process). 
   */
#ifdef CUDA
  // // get device count
  // int count;
  // cutilSafeCall( cudaGetDeviceCount(&count) );
  
  // int devId = myRank % count;
  // cutilSafeCall( cudaSetDevice(devId) );
  
  // cudaDeviceProp deviceProp;
  // int myDevId = -1;
  // cutilSafeCall( cudaGetDevice( &myDevId ) );
  // cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
  // // faire un cudaSetDevice et cudaGetDeviceProp et aficher le nom
  // // ajouter un booleen dans le constructeur pour savoir si on veut faire ca
  // // sachant que sur Titane, probablement que le mode exclusif est active
  // // a verifier demain
  
  // std::cout << "MPI process " << myRank << " is using GPU device num " << myDevId << std::endl;

#endif // CUDA

    // fix space resolution :
    // need to take into account number of MPI process in each direction
    dx = (xmax - xmin)/(nx*mx);
    dy = (ymax - ymin)/(ny*my);
    dz = (zmax - zmin)/(nz*mz);

    // print information about current setup
    if (myRank == 0) {
      std::cout << "We are about to start simulation with the following characteristics\n";

      std::cout << "Global resolution : " << 
	nx*mx << " x " << ny*my << " x " << nz*mz << "\n";
      std::cout << "Local  resolution : " << 
	nx << " x " << ny << " x " << nz << "\n";
      std::cout << "MPI Cartesian topology : " << mx << "x" << my << "x" << mz << std::endl;
    }
  
} // HydroParams::setup_mpi

#endif // USE_MPI

// =======================================================
// =======================================================
void HydroParams::init()
{

  // set other parameters
  imin = 0;
  jmin = 0;
  kmin = 0;

  imax = nx - 1 + 2*ghostWidth;
  jmax = ny - 1 + 2*ghostWidth;
  kmax = nz - 1 + 2*ghostWidth;
  
  isize = imax - imin + 1;
  jsize = jmax - jmin + 1;
  ksize = kmax - kmin + 1;

  dx = (xmax - xmin) / nx;
  dy = (ymax - ymin) / ny;
  dz = (zmax - zmin) / nz;
  
  settings.smallp  = settings.smallc*settings.smallc/
    settings.gamma0;
  settings.smallpp = settings.smallr*settings.smallp;
  settings.gamma6  = (settings.gamma0 + ONE_F)/(TWO_F * settings.gamma0);
  
  // check that given parameters are valid
  if ( (implementationVersion != 0) && 
       (implementationVersion != 1) && 
       (implementationVersion != 2) ) {
    fprintf(stderr, "The implementation version parameter should 0,1 or 2 !!!");
    fprintf(stderr, "Check your parameter file, section OTHER");
    exit(EXIT_FAILURE);
  } 

} // HydroParams::init


// =======================================================
// =======================================================
void HydroParams::print()
{
  
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  //printf( "Solver name: %s\n",solver_name.c_str());
  printf( "nx         : %d\n", nx);
  printf( "ny         : %d\n", ny);
  printf( "nz         : %d\n", nz);
  
  printf( "dx         : %f\n", dx);
  printf( "dy         : %f\n", dy);
  printf( "dz         : %f\n", dz);

  printf( "imin       : %d\n", imin);
  printf( "imax       : %d\n", imax);

  printf( "jmin       : %d\n", jmin);      
  printf( "jmax       : %d\n", jmax);      

  printf( "kmin       : %d\n", kmin);      
  printf( "kmax       : %d\n", kmax);      

  printf( "ghostWidth : %d\n", ghostWidth);
  printf( "nbvar      : %d\n", nbvar);
  printf( "nStepmax   : %d\n", nStepmax);
  printf( "tEnd       : %f\n", tEnd);
  printf( "nOutput    : %d\n", nOutput);
  printf( "gamma0     : %f\n", settings.gamma0);
  printf( "gamma6     : %f\n", settings.gamma6);
  printf( "cfl        : %f\n", settings.cfl);
  printf( "smallr     : %12.10f\n", settings.smallr);
  printf( "smallc     : %12.10f\n", settings.smallc);
  printf( "smallp     : %12.10f\n", settings.smallp);
  printf( "smallpp    : %g\n", settings.smallpp);
  printf( "cp (specific heat)          : %g\n", settings.cp);
  printf( "mu (dynamic visosity)       : %g\n", settings.mu);
  printf( "kappa (thermal diffusivity) : %g\n", settings.kappa);
  //printf( "niter_riemann : %d\n", niter_riemann);
  printf( "iorder     : %d\n", settings.iorder);
  printf( "slope_type : %f\n", settings.slope_type);
  printf( "riemann    : %d\n", riemannSolverType);
  //printf( "problem    : %d\n", problemStr);
  printf( "implementation version : %d\n",implementationVersion);
  printf( "##########################\n");
  
} // HydroParams::print
