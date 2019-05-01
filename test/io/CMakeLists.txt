configure_file(test_io_2d.ini test_io_2d.ini COPYONLY)
configure_file(test_io_3d.ini test_io_3d.ini COPYONLY)

################################
# VTK test
################################
add_executable(test_io_vtk "")

target_sources(test_io_vtk
  PUBLIC
  test_io_vtk.cpp)

# target_include_directories(test_io_vtk
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src
#   ${CMAKE_SOURCE_DIR}/src/utils/io
#   ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
#   )
target_link_libraries(test_io_vtk
  PUBLIC
  kokkos 
  EulerKokkos::shared 
  EulerKokkos::config 
  EulerKokkos::monitoring 
  EulerKokkos::io)

if(USE_MPI)
  target_link_libraries(test_io_vtk
    PUBLIC
    EulerKokkos::mpiUtils)
endif(USE_MPI)

###############################
# HDF5 test
###############################
if(USE_HDF5)
  add_executable(test_io_hdf5
    test_io_hdf5.cpp
    #${BACKWARD_ENABLE}
    )
  
  target_link_libraries(test_io_hdf5
    PUBLIC
    kokkos 
    EulerKokkos::shared 
    EulerKokkos::config 
    EulerKokkos::monitoring 
    EulerKokkos::io 
    ${HDF5_LIBRARIES})
  
  if(USE_MPI)
    target_link_libraries(test_io_hdf5
      PUBLIC
      EulerKokkos::mpiUtils)
  endif(USE_MPI)

  #add_backward(test_io_hdf5)
  
endif(USE_HDF5)

if(USE_MPI)
  if(USE_PNETCDF)

    add_executable(test_io_pnetcdf
      test_io_pnetcdf.cpp
      #${BACKWARD_ENABLE}
      )
    
    target_link_libraries(test_io_pnetcdf
      PUBLIC
      kokkos
      EulerKokkos::shared
      EulerKokkos::config
      EulerKokkos::monitoring
      EulerKokkos::io
      EulerKokkos::mpiUtils
      ${PNETCDF_LIBRARIES})
    
    #add_backward(test_io_pnetcdf)
    
  endif(USE_PNETCDF)
endif(USE_MPI)
