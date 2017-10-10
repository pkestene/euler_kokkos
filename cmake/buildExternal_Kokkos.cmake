# allow a possible build of kokkos as an external project
include(ExternalProject)

set(DEPENDENCIES)

#
# Sets a default kokkos configure line options passed to generate_makefile.sh
#
if(NOT DEFINED KOKKOS_CONFIGURE_LINE)
  set ( KOKKOS_CONFIGURE_LINE "--with-cuda --arch=Kepler30 --with-cuda-options=enable_lambda --with-openmp --with-hwloc=/usr")
endif()

#
# Make sure the configure line is a CMake list (and not a string)
#
string(REPLACE " " ";" KOKKOS_CONFIGURE_LINE_LIST ${KOKKOS_CONFIGURE_LINE})

# build kokkos as an external project
list(APPEND DEPENDENCIES kokkos)
ExternalProject_Add(kokkos
  PREFIX ${CMAKE_BINARY_DIR}/external
  GIT_REPOSITORY https://github.com/pkestene/kokkos.git
  GIT_TAG output_kokkos_cmake
  
  CONFIGURE_COMMAND <SOURCE_DIR>/generate_makefile.bash ${KOKKOS_CONFIGURE_LINE_LIST}

  BUILD_COMMAND make kokkoslib
  INSTALL_COMMAND make install
  )

# build the main project
# set (EXTRA_CMAKE_ARGS)
# ist (APPEND EXTRA_CMAKE_ARGS
#   -DKOKKOS_PATH=${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Source/ep_boost
#   -DBoost_NO_SYSTEM_PATHS=ON)
#set (ENV{KOKKOS_PATH} "${CMAKE_BINARY_DIR}/external/kokkos/install")

ExternalProject_Add (mandelbrot_kokkos
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS -DBUILD_KOKKOS=OFF 
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
