configure_file(test_io_2d.ini test_io_2d.ini COPYONLY)
configure_file(test_io_3d.ini test_io_3d.ini COPYONLY)

################################
# VTK test
################################
add_executable(test_io_vtk
  test_io_vtk.cpp
  )

target_include_directories(test_io_vtk
  PUBLIC
  ${CMAKE_SOURCE_DIR}/kokkos/core/src
  ${CMAKE_SOURCE_DIR}/kokkos/containers/src
  ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
  ${CMAKE_BINARY_DIR}/kokkos
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils/io
  ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
  )
target_link_libraries(test_io_vtk
  kokkos shared config monitoring io)

if(USE_MPI)
  target_link_libraries(test_io_vtk
    mpiUtils)
endif(USE_MPI)

###############################
# HDF5 test
###############################
if(USE_HDF5)
  add_executable(test_io_hdf5
    test_io_hdf5.cpp
    ${BACKWARD_ENABLE}
    )
  
  target_include_directories(test_io_hdf5
    PUBLIC
    ${CMAKE_SOURCE_DIR}/kokkos/core/src
    ${CMAKE_SOURCE_DIR}/kokkos/containers/src
    ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
    ${CMAKE_BINARY_DIR}/kokkos
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils/io
    ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
    )
  target_link_libraries(test_io_hdf5
    kokkos shared config monitoring io ${HDF5_LIBRARIES})
  
  if(USE_MPI)
    target_link_libraries(test_io_hdf5
      mpiUtils)
  endif(USE_MPI)

  add_backward(test_io_hdf5)
  
endif(USE_HDF5)

if(USE_MPI)
  if(USE_PNETCDF)

    add_executable(test_io_pnetcdf
      test_io_pnetcdf.cpp
      ${BACKWARD_ENABLE}
      )
    
    target_include_directories(test_io_pnetcdf
      PUBLIC
      ${CMAKE_SOURCE_DIR}/kokkos/core/src
      ${CMAKE_SOURCE_DIR}/kokkos/containers/src
      ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
      ${CMAKE_BINARY_DIR}/kokkos
      ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/src/utils/io
      ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils
      )
    target_link_libraries(test_io_pnetcdf
      kokkos shared config monitoring io mpiUtils ${HDF5_PNETCDF})
    
    add_backward(test_io_pnetcdf)
    
  endif(USE_PNETCDF)
endif(USE_MPI)
