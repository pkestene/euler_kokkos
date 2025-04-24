configure_file(test_io_2d.ini test_io_2d.ini COPYONLY)
configure_file(test_io_3d.ini test_io_3d.ini COPYONLY)

# ##################################################################################################
# VTK test
# ##################################################################################################
add_executable(test_io_vtk "")

target_sources(test_io_vtk PUBLIC test_io_vtk.cpp)

target_link_libraries(
  test_io_vtk PUBLIC Kokkos::kokkos EulerKokkos::shared EulerKokkos::config EulerKokkos::monitoring
                     EulerKokkos::io EulerKokkos::mpiUtils)

# ##################################################################################################
# HDF5 test
# ##################################################################################################
if(EULER_KOKKOS_USE_HDF5)
  add_executable(test_io_hdf5 test_io_hdf5.cpp)

  target_link_libraries(
    test_io_hdf5
    PUBLIC Kokkos::kokkos
           EulerKokkos::shared
           EulerKokkos::config
           EulerKokkos::monitoring
           EulerKokkos::io
           EulerKokkos::mpiUtils
           ${HDF5_LIBRARIES})

endif(EULER_KOKKOS_USE_HDF5)

if(EULER_KOKKOS_USE_MPI)
  if(EULER_KOKKOS_USE_PNETCDF)

    add_executable(test_io_pnetcdf test_io_pnetcdf.cpp)

    target_link_libraries(
      test_io_pnetcdf
      PUBLIC Kokkos::kokkos
             EulerKokkos::shared
             EulerKokkos::config
             EulerKokkos::monitoring
             EulerKokkos::io
             EulerKokkos::mpiUtils
             ${PNETCDF_LIBRARIES})

  endif(EULER_KOKKOS_USE_PNETCDF)
endif(EULER_KOKKOS_USE_MPI)
