add_library(euler_kokkos_io IO_common.cpp IO_ReadWrite.cpp IO_VTK.cpp)

if(EULER_KOKKOS_USE_HDF5)
  target_sources(euler_kokkos_io PRIVATE IO_HDF5.cpp)
endif(EULER_KOKKOS_USE_HDF5)

if(EULER_KOKKOS_USE_MPI)
  if(EULER_KOKKOS_USE_PNETCDF)
    target_sources(euler_kokkos_io PUBLIC IO_PNETCDF.h)
  endif(EULER_KOKKOS_USE_PNETCDF)
endif(EULER_KOKKOS_USE_MPI)

target_include_directories(euler_kokkos_io PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(euler_kokkos_io PUBLIC Kokkos::kokkos EulerKokkos::shared)

if(EULER_KOKKOS_USE_HDF5)
  target_link_libraries(euler_kokkos_io PUBLIC ${HDF5_LIBRARIES})
endif(EULER_KOKKOS_USE_HDF5)

if(EULER_KOKKOS_USE_PNETCDF)
  target_link_libraries(euler_kokkos_io PUBLIC ${PNETCDF_LIBRARIES})
endif(EULER_KOKKOS_USE_PNETCDF)

if(EULER_KOKKOS_USE_MPI)
  target_link_libraries(euler_kokkos_io PUBLIC MPI::MPI_CXX)
endif(EULER_KOKKOS_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::io ALIAS euler_kokkos_io)
