# ##############################################################################
# HDF5
# ##############################################################################
# prefer using parallel HDF5 when build with mpi
if(EULER_KOKKOS_USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(EULER_KOKKOS_USE_MPI)

if(EULER_KOKKOS_USE_HDF5)
  set(HDF5_FIND_DEBUG True)
  find_package(
    HDF5
    COMPONENTS C CXX HL
    REQUIRED)
  if(HDF5_FOUND)
    set(EULER_KOKKOS_USE_HDF5_VERSION ${HDF5_VERSION})
    include_directories(${HDF5_INCLUDE_DIRS})
    set(MY_HDF5_LIBS hdf5 hdf5_cpp)
    if(HDF5_IS_PARALLEL)
      set(EULER_KOKKOS_USE_HDF5_PARALLEL True)
      message(WARNING "PARALLEL HDF5 found")
    else()
      message(WARNING "PARALLEL HDF5 not found")
    endif()
  else()
    set(EULER_KOKKOS_USE_HDF5_VERSION "")
    message(WARNING "HDF5 not found")
  endif(HDF5_FOUND)
endif(EULER_KOKKOS_USE_HDF5)
