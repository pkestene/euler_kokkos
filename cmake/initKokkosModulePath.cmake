#
# Here we decide to build kokkos or use an installed version of Kokkos.
#
# 1. KOKKOS_PATH env variable is defined:
#    it means we want to use an already installed version of Kokkos.
#
# 2. KOKKOS_PATH env variable is NOT defined:
#    we assume Kokkos has been built earlier using macro buildExternal_Kokkos.cmake
#
if(NOT DEFINED ENV{KOKKOS_PATH})

  # set CMAKE_MODULE_PATH to directory containing the generated kokkos.cmake
  #ExternalProject_Get_Property(kokkos install_dir)
  #set(CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/external/src/kokkos-build/install"  ${CMAKE_MODULE_PATH})
  list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_BINARY_DIR}/external/src/kokkos-build/install")
  message(STATUS "Just printing again CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
  
else()
  
  # just make sure kokkos install path contains a file named kokkos.cmake
  if (NOT EXISTS $ENV{KOKKOS_PATH}/kokkos.cmake)
    message(FATAL_ERROR "file kokkos.cmake does not exists in \"$ENV{KOKKOS_PATH}\" ! Check your kokkos installation")
  endif()

  # set CMAKE_MODULE_PATH to directory containing the generated kokkos.cmake
  set(CMAKE_MODULE_PATH "$ENV{KOKKOS_PATH}" ${CMAKE_MODULE_PATH})
  
endif()
