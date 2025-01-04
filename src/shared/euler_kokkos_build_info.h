/**
 * \file euler_kokkos_build_info.h
 */
#ifndef EULER_KOKKOS_SHARED_EULER_KOKKOS_BUILD_INFO_H_
#define EULER_KOKKOS_SHARED_EULER_KOKKOS_BUILD_INFO_H_

#include <euler_kokkos_version.h>
#include <shared/euler_kokkos_config.h>

#include <string>

namespace euler_kokkos
{

struct BuildInfo
{
  static std::string
  system_processor();

  static std::string
  system_name();

  static std::string
  build_type();

  static std::string
  compiler_id();

  static std::string
  compiler_version();

  static std::string
  compile_date();
  static std::string
  compile_time();

  static std::string
  mpi_runtime_config();

  static std::string
  mpi_runtime_cuda_support();
  static std::string
  mpi_runtime_hip_support();
  static std::string
  mpi_runtime_ze_support();

  static bool
  hdf5_enabled();
  static std::string
  hdf5_version();

  static void
  print();
};

} // namespace euler_kokkos

#endif // EULER_KOKKOS_SHARED_EULER_KOKKOS_BUILD_INFO_H_
