/**
 * \file GlobalMpiSession.cpp
 * \brief Implements class GlobalMpiSession.
 *
 */

#include <utils/mpi/GlobalMpiSession.h>

#include <mpi.h>

namespace euler_kokkos
{

// =======================================================
// =======================================================
GlobalMpiSession::GlobalMpiSession(int & argc, char **& argv)
  : i_initialized(false)
{
  if (!initialized())
  {
    CHECK_MPI_ERR(MPI_Init(&argc, &argv));
    i_initialized = true;
  }

} // GlobalMpiSession::GlobalMpiSession

// =======================================================
// =======================================================
GlobalMpiSession::~GlobalMpiSession()
{

  // calling MPI_Finalize only if MPI_Init was in the constructor
  if (i_initialized)
  {
    if (!finalized())
    {
      CHECK_MPI_ERR(MPI_Finalize());
    }
  }

} // GlobalMpiSession::~GlobalMpiSession

// ================================================================
// ================================================================
void
GlobalMpiSession::abort(int errcode)
{
  CHECK_MPI_ERR(MPI_Abort(MPI_COMM_WORLD, errcode));
} // GlobalMpiSession::abort

// ================================================================
// ================================================================
bool
GlobalMpiSession::initialized()
{
  int flag;
  CHECK_MPI_ERR(MPI_Initialized(&flag));
  return flag != 0;
} // GlobalMpiSession::initialized

// ================================================================
// ================================================================
bool
GlobalMpiSession::finalized()
{
  int flag;
  CHECK_MPI_ERR(MPI_Finalized(&flag));
  return flag != 0;
} // GlobalMpiSession::finalized

// ================================================================
// ================================================================
std::string
GlobalMpiSession::processor_name()
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int  len;

  CHECK_MPI_ERR(MPI_Get_processor_name(name, &len));
  return std::string(name, len);

} // GlobalMpiSession::processor_name

// ================================================================
// ================================================================
std::string
GlobalMpiSession::mpi_library_version()
{
#if (3 <= MPI_VERSION)
  char lib_version[MPI_MAX_LIBRARY_VERSION_STRING];
  int  len = 0;
  CHECK_MPI_ERR(MPI_Get_library_version(lib_version, &len));
  return std::string(lib_version, len);
#else
  return "";
#endif

} // GlobalMpiSession::mpi_library_version

// ================================================================
// ================================================================
std::pair<int, int>
GlobalMpiSession::mpi_standard_version()
{
  int version, subversion;
  CHECK_MPI_ERR(MPI_Get_version(&version, &subversion));
  return { version, subversion };

} // GlobalMpiSession::mpi_standard_version

} // namespace euler_kokkos
