#
# This file is borrowed and slightly modified from
# https://github.com/eschnett/MPIwrapper/blob/main/cmake/CheckMPIFeatures.cmake
#
# function check_mpi_features provides helper to check if MPI implementation has
# the runtime ability to probe GPU-awareness
#
# * cuda-aware (Nvidia GPU),
# * hip-aware (AMD GPU),
# * ze-aware (INTEL GPU)
#
# Apparently Intel MPI (as of version 2021.7.0) doesn't provide header
# mpi-ext.h.
#

include(CheckCSourceCompiles)

# check_mpi_features
macro(CHECK_MPI_FEATURES)
  if(NOT DEFINED EULER_KOKKOS_USE_MPI_EXT
     OR NOT DEFINED EULER_KOKKOS_MPI_HAS_QUERY_CUDA_SUPPORT)
    list(JOIN MPI_COMPILE_FLAGS " " cmake_required_flags)

    # set(cmake_required_includes ${MPI_INCLUDE_PATH})
    # set(cmake_required_libraries ${MPI_LIBRARIES})
    set(cmake_required_libraries MPI::MPI_C)

    # We cannot use check_include_file here as <mpi.h> needs to be included
    # before <mpi-ext.h>, and check_include_file doesn't support this.
    check_c_source_compiles(
      "
        #include <mpi.h>
        #include <mpi-ext.h>
        int main() {
          return 0;
        }
      "
      EULER_KOKKOS_USE_MPI_EXT)

    if(NOT EULER_KOKKOS_USE_MPI_EXT)
      set(EULER_KOKKOS_USE_MPI_EXT 0)
    else()
      set(EULER_KOKKOS_USE_MPI_EXT 1)
    endif()

    list(APPEND CMAKE_REQUIRED_DEFINITIONS
         -DEULER_KOKKOS_USE_MPI_EXT=${EULER_KOKKOS_USE_MPI_EXT})

    check_c_source_compiles(
      "
        #include <mpi.h>
        #if EULER_KOKKOS_USE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_cuda_support();
          return 0;
        }
        "
      EULER_KOKKOS_MPI_HAS_QUERY_CUDA_SUPPORT)

    if(NOT EULER_KOKKOS_MPI_HAS_QUERY_CUDA_SUPPORT)
      set(EULER_KOKKOS_MPI_HAS_QUERY_CUDA_SUPPORT 0)
    else()
      set(EULER_KOKKOS_MPI_HAS_QUERY_CUDA_SUPPORT 1)
    endif()

    check_c_source_compiles(
      "
        #include <mpi.h>
        #if EULER_KOKKOS_USE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_hip_support();
          return 0;
        }
        "
      EULER_KOKKOS_MPI_HAS_QUERY_HIP_SUPPORT)

    if(NOT EULER_KOKKOS_MPI_HAS_QUERY_HIP_SUPPORT)
      set(EULER_KOKKOS_MPI_HAS_QUERY_HIP_SUPPORT 0)
    else()
      set(EULER_KOKKOS_MPI_HAS_QUERY_HIP_SUPPORT 1)
    endif()

    check_c_source_compiles(
      "
        #include <mpi.h>
        #if EULER_KOKKOS_USE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_ze_support();
          return 0;
        }
        "
      EULER_KOKKOS_MPI_HAS_QUERY_ZE_SUPPORT)

    if(NOT EULER_KOKKOS_MPI_HAS_QUERY_ZE_SUPPORT)
      set(EULER_KOKKOS_MPI_HAS_QUERY_ZE_SUPPORT 0)
    else()
      set(EULER_KOKKOS_MPI_HAS_QUERY_ZE_SUPPORT 1)
    endif()

    list(REMOVE_ITEM CMAKE_REQUIRED_DEFINITIONS -DEULER_KOKKOS_USE_MPI_EXT)
  endif()
endmacro(CHECK_MPI_FEATURES)

check_mpi_features()
