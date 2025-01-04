/**
 * \file ParallelEnv.h
 * \brief Provide a class for initializing, finalizing environment for parallel
 * computation (MPI, Kokkos, ...)
 *
 * MPI is optional
 * Kokkos is mandatory
 *
 */
#ifndef PARALLEL_ENV_H_
#define PARALLEL_ENV_H_

#include <utils/mpi/GlobalMpiSession.h>

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
  std::unique_ptr<euler_kokkos::GlobalMpiSession> m_mpiSession;
  std::unique_ptr<euler_kokkos::MpiComm>          m_comm_ptr;

public:
  /**
   * Default constructor.
   *
   * To be used in a standalone app.
   * MPI communicator will default to MPI_COMM_WORLD
   */
  ParallelEnv(int & argc, char **& argv);

  /**
   * Additional constructor to be used when euler_kokkos is used as library.
   *
   * The calling code should provide a MPI communicator.
   * Here we either attach to the provided MPI communicator or duplicate it.
   *
   * In this case, we take responsibility to initialize kokkos unconditionally.
   */
  ParallelEnv(int argc, char * argv[], const MPI_Comm & comm);

  // Destructor.
  ~ParallelEnv();

  //! \return MPI rank
  inline int
  rank() const
  {
    return m_comm_ptr->rank();
  }

  //! \return MPI size
  inline int
  nRanks() const
  {
    return m_comm_ptr->size();
  }

  //! \return MPI size
  inline int
  size() const
  {
    return m_comm_ptr->size();
  }

  //! \return MPI communicator (see MPIComm)
  const MpiComm &
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

  //! \return boolean to indicated if MPI is enabled
  static bool
  MPI_enabled();

private:
  //! print Kokkos configuration (backend enabled, etc...)
  void
  print_kokkos_config();

}; // class ParallelEnv

} // namespace euler_kokkos

#endif // PARALLEL_ENV_H_
