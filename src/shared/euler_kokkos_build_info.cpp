#include <shared/euler_kokkos_config.h> // for EULER_KOKKOS_USE_MPI
#include <shared/euler_kokkos_build_info.h>

#include <iostream>

#ifdef EULER_KOKKOS_USE_MPI
#  include <mpi.h>
#  if EULER_KOKKOS_USE_MPI_EXT
#    include <mpi-ext.h>
#  endif
#endif

namespace euler_kokkos
{

std::string
BuildInfo::system_processor()
{
  return EULER_KOKKOS_SYSTEM_PROCESSOR;
}

std::string
BuildInfo::system_name()
{
  return EULER_KOKKOS_SYSTEM_NAME;
}

std::string
BuildInfo::build_type()
{
  return EULER_KOKKOS_BUILD_TYPE;
}

std::string
BuildInfo::compiler_id()
{
  return EULER_KOKKOS_BUILD_COMPILER_ID;
}

std::string
BuildInfo::compiler_version()
{
  return EULER_KOKKOS_BUILD_COMPILER_VERSION;
}

std::string
BuildInfo::compile_date()
{
  return EULER_KOKKOS_COMPILE_DATE;
}

std::string
BuildInfo::compile_time()
{
  return EULER_KOKKOS_COMPILE_TIME;
}

std::string
BuildInfo::mpi_runtime_config()
{
#ifdef EULER_KOKKOS_USE_MPI
  return []() {
    int  length;
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_library_version(mpi_version, &length);
    return std::string(mpi_version);
  }();
#else
  return "none";
#endif // EULER_KOKKOS_USE_MPI
}

std::string
BuildInfo::mpi_runtime_cuda_support()
{
#ifdef EULER_KOKKOS_USE_MPI
#  if EULER_KOKKOS_MPI_HAS_QUERY_CUDA_SUPPORT
  int result = MPIX_Query_cuda_support();
  return result == 1 ? std::string("yes") : std::string("no");
#  else
  return "unknown - not queryable";
#  endif
#else
  return "unknown - MPI not used";
#endif
}

std::string
BuildInfo::mpi_runtime_hip_support()
{
#ifdef EULER_KOKKOS_USE_MPI
#  if EULER_KOKKOS_MPI_HAS_QUERY_HIP_SUPPORT
  int result = MPIX_Query_hip_support();
  return result == 1 ? std::string("yes") : std::string("no");
#  else
  return "unknown - not queryable";
#  endif
#else
  return "unknown - MPI not used";
#endif
}

std::string
BuildInfo::mpi_runtime_ze_support()
{
#ifdef EULER_KOKKOS_USE_MPI
#  if EULER_KOKKOS_MPI_HAS_QUERY_ZE_SUPPORT
  int result = MPIX_Query_ze_support();
  return result == 1 ? std::string("yes") : std::string("no");
#  else
  return "unknown - not queryable";
#  endif
#else
  return "unknown - MPI not used";
#endif
}

bool
BuildInfo::hdf5_enabled()
{
#ifdef EULER_KOKKOS_USE_HDF5
  return true;
#else
  return false;
#endif
}

std::string
BuildInfo::hdf5_version()
{
#ifdef EULER_KOKKOS_USE_HDF5
  return EULER_KOKKOS_USE_HDF5_VERSION;
#else
  return "unknown";
#endif
}

void
BuildInfo::print()
{
  std::cout << "##############################################\n";
  std::cout << "euler_kokkos - build info\n";
  std::cout << "system name      : " << system_name() << "\n";
  std::cout << "system processor : " << system_processor() << "\n";
  std::cout << "compile date     : " << compile_date() << "\n";
  std::cout << "compile time     : " << compile_time() << "\n";
  std::cout << "build type       : " << build_type() << "\n";
  std::cout << "compiler id      : " << compiler_id() << "\n";
  std::cout << "compiler version : " << compiler_version() << "\n";
  std::cout << "MPI runtime config : " << mpi_runtime_config() << "\n";
  std::cout << "MPI runtime cuda support : " << mpi_runtime_cuda_support() << "\n";
  std::cout << "MPI runtime hip  support : " << mpi_runtime_hip_support() << "\n";
  std::cout << "MPI runtime ze   support : " << mpi_runtime_ze_support() << "\n";
  if (hdf5_enabled())
    std::cout << "HDF5 version             : " << hdf5_version() << "\n";
  std::cout << "##############################################\n";
}

} // namespace euler_kokkos
