/**
 * Hydro/MHD solver (Muscl-Hancock).
 *
 * \date April, 16 2016
 * \author P. Kestener
 */

#include <cstdlib>
#include <cstdio>

#include "shared/kokkos_shared.h"

#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/solver_utils.h" // print monitoring information

// solver
#include "shared/SolverFactory.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

#ifdef USE_HDF5
#include "utils/io/IO_HDF5.h"
#endif // USE_HDF5

#ifdef USE_FPE_DEBUG
// for catching floating point errors
#include <fenv.h>
#include <signal.h>

// signal handler for catching floating point errors
void fpehandler(int sig_num)
{
  signal(SIGFPE, fpehandler);
  printf("SIGFPE: floating point exception occured of type %d, exiting.\n",sig_num);
  abort();
}
#endif // USE_FPE_DEBUG

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char *argv[])
{

  using namespace euler_kokkos;

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI
  
  Kokkos::initialize(argc, argv);

  int rank=0;
  int nRanks=1;
  
  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef USE_FPE_DEBUG
    /*
     * Install a signal handler for floating point errors.
     * This only usefull when debugging, doing a backtrace in gdb,
     * tracking for NaN 
     */
    feenableexcept(FE_DIVBYZERO | FE_INVALID);
    signal(SIGFPE, fpehandler);
#endif // USE_FPE_DEBUG
    
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
# ifdef KOKKOS_ENABLE_CUDA
    {
      
      int cudaDeviceId;
      cudaGetDevice(&cudaDeviceId);
      std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")"
		<< " pinned to GPU #" << cudaDeviceId << "\n";
    }
# endif // KOKKOS_ENABLE_CUDA
#endif // USE_MPI

    
  }

  // if (argc != 2) {
  //   if (rank==0)
  //     fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter on the command line\n");
  //   exit(EXIT_FAILURE);
  // }

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // initialize workspace memory (U, U2, ...)
  SolverBase *solver = SolverFactory::Instance().create(solver_name,
							params,
							configMap);

  if (params.nOutput != 0)
    solver->save_solution();
  
  // start computation
  if (rank==0) std::cout << "Start computation....\n";
  solver->timers[TIMER_TOTAL]->start();

  // Hydrodynamics solver loop
  while ( ! solver->finished() ) {

    solver->next_iteration();

  } // end solver loop

  // end of computation
  solver->timers[TIMER_TOTAL]->stop();

  // save last time step
  if (params.nOutput != 0)
    solver->save_solution();

  // write Xdmf wrapper file if necessary
#ifdef USE_HDF5
  bool outputHdf5Enabled = configMap.getBool("output","hdf5_enabled",false);
  if (outputHdf5Enabled) {
    euler_kokkos::io::writeXdmfForHdf5Wrapper(params, configMap, solver->m_variables_names, solver->m_times_saved-1, false);
  }
#endif // USE_HDF5
  
  if (rank==0) printf("final time is %f\n", solver->m_t);
  
  print_solver_monitoring_info(solver);
  
  delete solver;

  Kokkos::finalize();
  
  return EXIT_SUCCESS;

} // end main
