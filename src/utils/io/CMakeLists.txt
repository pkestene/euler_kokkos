set (io_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_Writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK.cpp
  )

if(USE_HDF5)
  set (io_SRCS
    ${io_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_HDF5.cpp
    )
endif(USE_HDF5)

if(USE_MPI)
  if(USE_PNETCDF)
  set (io_SRCS
    ${io_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/IO_PNETCDF.h
    )    
  endif(USE_PNETCDF)
endif(USE_MPI)

add_library(io
  ${io_SRCS}
  )

target_include_directories(io
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(io
  kokkos
  )
if (USE_HDF5)
  target_link_libraries(io
    ${HDF5_LIBRARIES}
    )
endif()

if (USE_PNETCDF)
  target_link_libraries(io
    ${PNETCDF_LIBRARIES}
    )
endif()
  
# if (USE_MPI)
#   target_link_libraries(io
#     ${MPI_CXX_LIBRARIES}
#     )
# endif()
