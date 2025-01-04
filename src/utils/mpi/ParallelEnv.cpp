/**
 * \file ParallelEnv.cpp
 */
#include "ParallelEnv.h"

#include <shared/euler_kokkos_config.h>

#include <cstring>
#include <iostream>

namespace euler_kokkos
{

// ===================================================================================
// ===================================================================================
ParallelEnv::ParallelEnv(int & argc, char **& argv)
{
  bool m_initialize_kokkos_before_mpi = false;

#ifdef KOKKOS_ENABLE_CUDA
  // when running on platform with Intel OmniPath interconnect and Nvidia GPUs, we may
  // need to initialize Kokkos (and Cuda context first)
  if ([[maybe_unused]] const char * env_value = std::getenv("PSM2_CUDA"))
  {
    std::cout << "PSM2_CUDA detected : Initializing Kokkos before MPI" << std::endl;
    m_initialize_kokkos_before_mpi = true;
  }
#endif

  if ([[maybe_unused]] const char * env_value = std::getenv("EULER_KOKKOS_INIT_KOKKOS_BEFORE_MPI"))
  {
    std::cout << "EULER_KOKKOS_INIT_KOKKOS_BEFORE_MPI detected : Initializing Kokkos before MPI"
              << std::endl;
    m_initialize_kokkos_before_mpi = true;
  }

  if (m_initialize_kokkos_before_mpi)
    Kokkos::initialize(argc, argv);

#ifdef EULER_KOKKOS_USE_MPI
  // Create MPI session if MPI enabled
  m_mpiSession = std::make_unique<GlobalMpiSession>(argc, argv);

  // create a communicator for MPI_COMM_WORLD
  m_comm_ptr = std::make_unique<MpiComm>();
#endif // EULER_KOKKOS_USE_MPI

  if (!m_initialize_kokkos_before_mpi)
    Kokkos::initialize(argc, argv);

  print_kokkos_config();

} // ParallelEnv::ParallelEnv

#ifdef EULER_KOKKOS_USE_MPI
// ===================================================================================
// ===================================================================================
ParallelEnv::ParallelEnv(int argc, char * argv[], const MPI_Comm & comm)
{
  // Create MPI session (check if MPI is already initialized, it should be)
  m_mpiSession = std::make_unique<GlobalMpiSession>(argc, argv);

  // create a communicator wrapping comm
  m_comm_ptr = std::make_unique<MpiComm>(comm, MpiComm::COMM_DUPLICATE);

  // initialize kokkos
  Kokkos::initialize(argc, argv);
  print_kokkos_config();

} // ParallelEnv::ParallelEnv
#endif // EULER_KOKKOS_USE_MPI


// ===================================================================================
// ===================================================================================
ParallelEnv::~ParallelEnv()
{

  // cleanup kokkos
  Kokkos::finalize();

} // ParallelEnv::~ParallelEnv

// ===================================================================================
// ===================================================================================
bool
ParallelEnv::MPI_enabled()
{
#ifdef EULER_KOKKOS_USE_MPI
  return true;
#else  // EULER_KOKKOS_USE_MPI
  return false;
#endif // EULER_KOKKOS_USE_MPI
} // ParallelEnv::MPI_enabled

// ===================================================================================
// ===================================================================================
void
ParallelEnv::print_kokkos_config()
{
  // only master MPI task print Kokkos config information
  if (rank() == 0)
  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
    std::cout << "END KOKKOS CONFIG         \n";
    std::cout << "##########################\n";
  }

#ifdef KOKKOS_ENABLE_CUDA
  if ([[maybe_unused]] const char * env_value = std::getenv("CUDA_VISIBLE_DEVICES"))
  {
    std::cout << "I'm MPI task #" << this->rank() << " CUDA_VISIBLE_DEVICES was set to "
              << std::string(env_value) << "\n";
  }
  else
  {
    std::cout << "I'm MPI task #" << this->rank() << " CUDA_VISIBLE_DEVICES was not set" << "\n";
  }

  if (MPI_enabled())
  {

    // To enable kokkos accessing multiple GPUs don't forget to
    // add option "--ndevices=X" where X is the number of GPUs
    // you want to use per node.

    // on a large cluster, the scheduler should assign resources
    // in a way that each MPI task is mapped to a different GPU
    // let's cross-checked that:

    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);
    std::cout << "I'm MPI task #" << this->rank() << " (out of " << this->nRanks() << ")"
              << " pinned to GPU #" << cudaDeviceId << "\n";
  }
#endif // KOKKOS_ENABLE_CUDA

} // ParallelEnv::print_kokkos_config

} // namespace euler_kokkos
