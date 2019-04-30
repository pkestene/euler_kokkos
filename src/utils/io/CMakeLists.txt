add_library(io "")

target_sources (io
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_ReadWriteBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_ReadWrite.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_ReadWrite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.cpp
  )

if(USE_HDF5)
  target_sources (io
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.cpp
    )
endif(USE_HDF5)

if(USE_MPI)
  if(USE_PNETCDF)
    target_sources (io
      PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/IO_PNETCDF.h
      )    
  endif(USE_PNETCDF)
endif(USE_MPI)

target_include_directories(io
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

target_link_libraries(io
  PUBLIC
  kokkos
  )

if (USE_HDF5)
  target_link_libraries(io
    PUBLIC
    ${HDF5_LIBRARIES}
    )
endif(USE_HDF5)

if (USE_PNETCDF)
  target_link_libraries(io
    PUBLIC
    ${PNETCDF_LIBRARIES}
    )
endif(USE_PNETCDF)

if (USE_MPI)
  target_compile_options    (io PUBLIC ${MPI_CXX_COMPILE_FLAGS})
  target_include_directories(io PUBLIC ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries     (io PUBLIC ${MPI_CXX_LIBRARIES})
endif(USE_MPI)
