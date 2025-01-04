# ##############################################################################
# HDF5
# ##############################################################################
# prefer using parallel HDF5 when build with mpi
if(EULER_KOKKOS_USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(EULER_KOKKOS_USE_MPI)

if(EULER_KOKKOS_USE_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    set(EULER_KOKKOS_USE_HDF5_VERSION ${HDF5_VERSION})
    include_directories(${HDF5_INCLUDE_DIRS})
    set(MY_HDF5_LIBS hdf5 hdf5_cpp)
    if(HDF5_IS_PARALLEL)
      set(EULER_KOKKOS_USE_HDF5_PARALLEL True)
    else()
      message(WARNING "PARALLEL HDF5 not found")
    endif()
  else()
    set(EULER_KOKKOS_USE_HDF5_VERSION "")
  endif(HDF5_FOUND)
endif(EULER_KOKKOS_USE_HDF5)
