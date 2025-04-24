# ##############################################################################
# MPI
# ##############################################################################
if(EULER_KOKKOS_USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})

    # set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    find_program(
      OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)

    # Full command line to probe if cuda support in MPI implementation is enabled ompi_info
    # --parsable --all | grep mpi_built_with_cuda_support:value
    if(OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO} OUTPUT_VARIABLE _output)
      if((_output MATCHES "smcuda") OR (EULER_KOKKOS_USE_MPI_CUDA_AWARE_ENFORCED))
        set(EULER_KOKKOS_USE_MPI_CUDA_AWARE_ENABLED ON)
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        set(EULER_KOKKOS_USE_MPI_CUDA_AWARE_ENABLED OFF)
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
        add_compile_options(-DMPI_CUDA_AWARE_OFF)
      endif()
    endif(OMPI_INFO)

  else(MPI_CXX_FOUND)
    message(WARNING "Not compiling with MPI. Suppress this warning with -DEULER_KOKKOS_USE_MPI=OFF")
    set(EULER_KOKKOS_USE_MPI OFF)
  endif(MPI_CXX_FOUND)

  # test if mpi-ext.h is available
  include(cmake/CheckMPIFeatures.cmake)

endif(EULER_KOKKOS_USE_MPI)
