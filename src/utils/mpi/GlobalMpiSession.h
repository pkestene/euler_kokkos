/**
 * \file GlobalMpiSession.h
 * \brief A MPI utilities class, providing methods for initializing,
 *        finalizing, and querying the global MPI session
 *
 */
#ifndef EULER_KOKKOS_GLOBAL_MPI_SESSION_H_
#define EULER_KOKKOS_GLOBAL_MPI_SESSION_H_

#include <utils/mpi/MpiComm.h>

#include <utils/mpi/mpi_utils.h> // for macro CHECK_MPI_ERR

#include <utility> // for std::make_pair
#include <string>

namespace euler_kokkos
{

/**
 * A base class that makes derived class non copyable.
 * see https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Non-copyable_Mixin
 */
class NonCopyable
{
public:
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &
  operator=(const NonCopyable &) = delete;

protected:
  NonCopyable() = default;
  ~NonCopyable() = default; /// Protected non-virtual destructor
};

/**
 * \brief This class provides methods for initializing, finalizing, and
 * querying the global MPI session.
 *
 * Main behavior:
 * - if MPI is not already initialized, we use MPI_COMM_WORLD as MPI communicator
 * - if MPI is already initialized, e.g. euler_kokkos used as library in a thrird party application,
 * we expect the caller to give us a communicator from which either we attached to or we duplicate
 * from.
 *
 * This class is not copyable because the base class forbids it.
 * see https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Non-copyable_Mixin
 */
class GlobalMpiSession : public NonCopyable
{
public:
  //! @name Public constructor and destructor
  //@{

  /**
   * \brief Calls <tt>MPI_Init()</tt> if MPI is enabled and not already initialized.
   *
   * \param argc  [in] Argument passed into <tt>main(argc,argv)</tt>
   * \param argv  [in] Argument passed into <tt>main(argc,argv)</tt>
   */
  GlobalMpiSession(int & argc, char **& argv);

  /** Shuts down the MPI environment.
   *
   *  If this @c environment object was used to initialize the MPI
   *  environment, and the MPI environment has not already been shut
   *  down (finalized), this destructor will shut down the MPI
   *  environment. Under normal circumstances, this only involves
   *  invoking @c MPI_Finalize. However, if destruction is the result
   *  of an uncaught exception and the @c abort_on_exception parameter
   *  of the constructor had the value @c true, this destructor will
   *  invoke @c MPI_Abort with @c MPI_COMM_WORLD to abort the entire
   *  MPI program with a result code of -1.
   */
  ~GlobalMpiSession();

  //@}

  // //! Get MPI_COMM_WORLD communicator
  // static MpiComm &
  // get_comm_world()
  // {
  //   return MpiComm::world();
  // }

  /** Abort all MPI processes.
   *
   *  Aborts all MPI processes and returns to the environment. The
   *  precise behavior will be defined by the underlying MPI
   *  implementation. This is equivalent to a call to @c MPI_Abort
   *  with @c MPI_COMM_WORLD.
   *
   *  @param errcode The error code to return to the environment.
   *  @returns Will not return.
   */
  static void
  abort(int errcode);

  /** Determine if the MPI environment has already been initialized.
   *
   *  This routine is equivalent to a call to @c MPI_Initialized.
   *
   *  @returns @c true if the MPI environment has been initialized.
   */
  static bool
  initialized();

  /** Determine if the MPI environment has already been finalized.
   *
   *  The routine is equivalent to a call to @c MPI_Finalized.
   *
   *  @returns @c true if the MPI environment has been finalized.
   */
  static bool
  finalized();

  //! Get processor name
  static std::string
  processor_name();

  //! MPI library implementation version string.
  //! MPI_Get_library_version is a MPI-3 API
  static std::string
  mpi_library_version();

  //! return MPI standard version number as a pair (version, subversion).
  static std::pair<int, int>
  mpi_standard_version();

private:
  //! Whether this environment object called MPI_Init.
  //!
  //! if GlobalMpiSession called MPI_Init, it means we are responsible for calling MPI_Finalize.
  //! When euler_kokkos is used as library, MPI_Init/MPI_Finalize is called elsewhere, so MPI is not
  //! managed here, this variable must be false.
  bool i_initialized;

}; // class GlobalMpiSession

} // namespace euler_kokkos

#endif // EULER_KOKKOS_GLOBAL_MPI_SESSION_H_
