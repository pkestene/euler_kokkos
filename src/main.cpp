/**
 * Hydro/MHD solver (Muscl-Hancock).
 *
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <shared/euler_kokkos_config.h>

#include <shared/kokkos_shared.h>
#include <shared/real_type.h>    // choose between single and double precision
#include <shared/HydroParams.h>  // read parameter file
#include <shared/solver_utils.h> // print monitoring information

// solver
#include <shared/SolverFactory.h>

#include <utils/mpi/ParallelEnv.h>

#ifdef EULER_KOKKOS_USE_MPI
#  include <mpi.h>
#endif // EULER_KOKKOS_USE_MPI

#ifdef EULER_KOKKOS_USE_HDF5
#  include <utils/io/IO_HDF5.h>
#endif // EULER_KOKKOS_USE_HDF5

// banner
#include <euler_kokkos_version.h>
#include <shared/euler_kokkos_git_info.h>
#include <shared/euler_kokkos_build_info.h>

#ifdef EULER_KOKKOS_USE_FPE_DEBUG
// for catching floating point errors
#  include <fenv.h>
#  include <signal.h>

// signal handler for catching floating point errors
void
fpehandler(int sig_num)
{
  signal(SIGFPE, fpehandler);
  printf("SIGFPE: floating point exception occurred of type %d, exiting.\n", sig_num);
  abort();
}
#endif // EULER_KOKKOS_USE_FPE_DEBUG

// ===============================================================
// ===============================================================
// ===============================================================
int
main(int argc, char * argv[])
{

  namespace ek = ::euler_kokkos;

  // Create MPI session if MPI enabled
  // initialize kokkos and print kokkos config
  auto par_env = ek::ParallelEnv(argc, argv);

  // read input parameter file
  // only MPI rank 0 actually reads input file
  std::string input_file = std::string(argv[1]);
  auto        configMap = ek::broadcast_parameters(input_file);

#ifdef EULER_KOKKOS_USE_MPI
  auto mx = configMap.getInteger("mpi", "mx", 1);
  auto my = configMap.getInteger("mpi", "my", 1);
  auto mz = configMap.getInteger("mpi", "mz", 1);
  auto dimType = get_dim(configMap);

  if (dimType == ek::TWO_D)
    par_env.setup_cartesian_topology(mx, my, ek::MPI_CART_PERIODIC_TRUE, ek::MPI_REORDER_TRUE);
  else if (dimType == ek::THREE_D)
    par_env.setup_cartesian_topology(mx, my, mz, ek::MPI_CART_PERIODIC_TRUE, ek::MPI_REORDER_TRUE);
#endif // EULER_KOKKOS_USE_MPI

#ifdef EULER_KOKKOS_USE_FPE_DEBUG
  /*
   * Install a signal handler for floating point errors.
   * This only useful when debugging, doing a backtrace in gdb,
   * tracking for NaN
   */
  feenableexcept(FE_DIVBYZERO | FE_INVALID);
  signal(SIGFPE, fpehandler);
#endif // EULER_KOKKOS_USE_FPE_DEBUG


  // banner
  if (par_env.rank() == 0)
  {
    ek::GitRevisionInfo::print();
    ek::BuildInfo::print();
  }

  // if (argc != 2) {
  //   if (par_env.rank()==0)
  //     fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter
  //     on the command line\n");
  //   exit(EXIT_FAILURE);
  // }

  /*
   * initialize a ConfigMap object
   */
  // test: create a HydroParams object
  ek::HydroParams params = ek::HydroParams(configMap, par_env);

  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // initialize workspace memory (U, U2, ...)
  ek::SolverBase * solver = ek::SolverFactory::Instance().create(solver_name, params, configMap);

  if (params.nOutput != 0)
    solver->save_solution();

  // start computation
  if (par_env.rank() == 0)
    std::cout << "Start computation....\n";
  solver->timers[TIMER_TOTAL]->start();

  // Hydrodynamics solver loop
  while (!solver->finished())
  {

    solver->next_iteration();

  } // end solver loop

  // end of computation
  solver->timers[TIMER_TOTAL]->stop();

  // save last time step
  if (params.nOutput != 0)
    solver->save_solution();

    // write Xdmf wrapper file if necessary
#ifdef EULER_KOKKOS_USE_HDF5
  bool outputHdf5Enabled = configMap.getBool("output", "hdf5_enabled", false);
  if (outputHdf5Enabled)
  {
    euler_kokkos::io::writeXdmfForHdf5Wrapper(
      params, configMap, solver->m_variables_names, solver->m_times_saved - 1, false);
  }
#endif // EULER_KOKKOS_USE_HDF5

  if (par_env.rank() == 0)
    printf("final time is %f\n", solver->m_t);

  ek::print_solver_monitoring_info(solver);

  delete solver;

  return EXIT_SUCCESS;

} // end main
