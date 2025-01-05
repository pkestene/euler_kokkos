/**
 * \file ParallelEnv.h
 * \brief Provide a class for initializing, finalizing environment for parallel
 * computation (MPI, Kokkos, ...)
 *
 * MPI is optional
 * Kokkos is mandatory
 *
 */
#ifndef EULER_KOKKOS_PARALLEL_ENV_H_
#define EULER_KOKKOS_PARALLEL_ENV_H_

#include <shared/euler_kokkos_config.h>

#ifdef EULER_KOKKOS_USE_MPI
#  include <utils/mpi/GlobalMpiSession.h>
#  include <utils/mpi/MpiComm.h>
#  include <utils/mpi/MpiCommCart.h>
#endif // EULER_KOKKOS_USE_MPI

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <cstdlib> // for std::getenv
#include <memory>  // for std::make_unique

namespace euler_kokkos
{

/**
 * \class ParallelEnv
 * Should initialize MPI and Kokkos.
 */
class ParallelEnv
{
private:
#ifdef EULER_KOKKOS_USE_MPI
  std::unique_ptr<euler_kokkos::GlobalMpiSession> m_mpiSession;
  std::unique_ptr<euler_kokkos::MpiComm>          m_comm_ptr;
#endif // EULER_KOKKOS_USE_MPI

public:
  /**
   * Default constructor.
   *
   * To be used in a standalone app.
   * MPI communicator will default to MPI_COMM_WORLD
   */
  ParallelEnv(int & argc, char **& argv);

#ifdef EULER_KOKKOS_USE_MPI
  /**
   * Additional constructor to be used when euler_kokkos is used as library.
   *
   * The calling code should provide a MPI communicator.
   * Here we either attach to the provided MPI communicator or duplicate it.
   *
   * In this case, we take responsibility to initialize kokkos unconditionally.
   */
  ParallelEnv(int argc, char * argv[], const MPI_Comm & comm);
#endif // EULER_KOKKOS_USE_MPI

  // Destructor.
  ~ParallelEnv();

  //! \return MPI rank
  inline int
  rank() const
  {
#ifdef EULER_KOKKOS_USE_MPI
    return m_comm_ptr->rank();
#else
    return 0;
#endif // EULER_KOKKOS_USE_MPI
  }

  //! \return MPI size
  inline int
  nRanks() const
  {
#ifdef EULER_KOKKOS_USE_MPI
    return m_comm_ptr->size();
#else
    return 1;
#endif //  EULER_KOKKOS_USE_MPI
  }

  //! \return MPI size
  inline int
  size() const
  {
#ifdef EULER_KOKKOS_USE_MPI
    return m_comm_ptr->size();
#else
    return 1;
#endif // EULER_KOKKOS_USE_MPI
  }

  //! 2d cartesian topology
  void
  setup_cartesian_topology(int mx, int my, int isPeriodic, int allowReorder);

  //! 3d cartesian topology
  void
  setup_cartesian_topology(int mx, int my, int mz, int isPeriodic, int allowReorder);

#ifdef EULER_KOKKOS_USE_MPI
  //! \return MPI communicator (see MPIComm)
  MpiComm &
  comm() const
  {
    return *m_comm_ptr;
  }

  //! return raw MPI communicator
  MPI_Comm
  mpi_comm() const
  {
    return m_comm_ptr->get_MPI_Comm();
  }
#endif // EULER_KOKKOS_USE_MPI

  //! \return boolean to indicated if MPI is enabled
  static bool
  MPI_enabled();

private:
  //! print Kokkos configuration (backend enabled, etc...)
  void
  print_kokkos_config();

}; // class ParallelEnv

} // namespace euler_kokkos

#endif // EULER_KOKKOS_PARALLEL_ENV_H_
