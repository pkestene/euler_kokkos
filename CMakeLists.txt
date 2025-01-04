# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.18)

#
# default local cmake macro repository
#
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# Create project version (using git info ?) TODO
#

#
# Init build type: Release, Debug, ...
#
include(initBuildType)

# options
option(EULER_KOKKOS_BUILD_DOC
       "Enable / disable documentation build (sphinx/html)" OFF)
option(EULER_KOKKOS_USE_SPHINX_EXHALE
       "Enable / disable building API documentation (very long)" OFF)
option(EULER_KOKKOS_USE_MPI "Activate / want MPI build" OFF)
option(EULER_KOKKOS_USE_VTK "Activate / want VTK build" OFF)
option(EULER_KOKKOS_USE_DOUBLE "build with double precision" ON)
option(EULER_KOKKOS_USE_HDF5 "build HDF5 input/output support" OFF)
option(EULER_KOKKOS_USE_PNETCDF
       "build PNETCDF input/output support (MPI required)" OFF)
option(EULER_KOKKOS_USE_FPE_DEBUG
       "build with floating point Nan tracing (signal handler)" OFF)
option(
  EULER_KOKKOS_USE_MPI_CUDA_AWARE_ENFORCED
  "Some MPI cuda-aware implementation are not well detected; use this to enforce"
  OFF)

# disable base languages
unset(PROJECT_LANGUAGES)

set(PROJECT_LANGUAGES ${PROJECT_LANGUAGES} C CXX)

set(EULER_KOKKOS_VERSION "0.9.0")

# deduce EULER_KOKKOS_SHORT_VERSION using regex
string(REGEX MATCH "^[0-9]+\.[0-9]+\.[0-9]+" EULER_KOKKOS_SHORT_VERSION
             ${EULER_KOKKOS_VERSION})
if("${EULER_KOKKOS_SHORT_VERSION}" STREQUAL "")
  message(
    FATAL_ERROR
      "Unable to compute short version from EULER_KOKKOS_VERSION=${EULER_KOKKOS_VERSION}"
  )
endif()

project(
  euler_kokkos
  VERSION ${EULER_KOKKOS_SHORT_VERSION}
  LANGUAGES ${PROJECT_LANGUAGES})

# Documentation type
if(EULER_KOKKOS_BUILD_DOC)
  set(EULER_KOKKOS_DOC_TYPE
      "Undefined"
      CACHE
        STRING
        "The documentation type to generate. Available values are html and doxygen"
  )
  # Set the possible values for documentation type
  set_property(CACHE EULER_KOKKOS_DOC_TYPE PROPERTY STRINGS "html" "doxygen"
                                                    "Undefined")
  add_subdirectory(doc)
  return()
endif()

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++17 is for Kokkos
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# ##############################################################################
#
# check MPI, VTK, HDF5, PNETCDF, ...
#
# ##############################################################################
include(cmake/config_mpi.cmake)

if(EULER_KOKKOS_USE_VTK)
  # look for VTK only if requested; VTK macro might even be not present on the
  # target platform
  find_package(
    VTK
    COMPONENTS CommonCore
               CommonDataModel
               CommonExecutionModel
               CommonMath
               CommonMisc
               CommonSystem
               CommonTransforms
               IOCore
               IOGeometry
               IOImage
               IOLegacy
               IOXML
               IOXMLParser
               ParallelCore
               ParallelMPI
               IOParallelXML)

  if(VTK_FOUND)
    message("VTK version: ${VTK_VERSION}")
  else()
    message("VTK NOT FOUND")
  endif()
endif(EULER_KOKKOS_USE_VTK)

# ##############################################################################
# HDF5
# ##############################################################################
include(cmake/config_hdf5.cmake)

# ##############################################################################
# PNETCDF
# ##############################################################################
if(EULER_KOKKOS_USE_MPI)
  if(EULER_KOKKOS_USE_PNETCDF)
    find_package(PNETCDF)
    if(PNETCDF_FOUND)
      add_compile_options(-DUSE_PNETCDF)
      include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(EULER_KOKKOS_USE_PNETCDF)
endif(EULER_KOKKOS_USE_MPI)

#
# Kokkos : https://github.com/kokkos/kokkos
#
include(build_or_find_kokkos)

#
# if MPI is enabled and kokkos backend is cuda, we need a cuda aware
# implementation
#
if(EULER_KOKKOS_USE_MPI)

  if(MPI_FOUND
     AND (EULER_KOKKOS_BACKEND MATCHES "Cuda")
     AND (NOT MPI_CUDA_AWARE_ENABLED))

    message(FATAL_ERROR "You must use a cuda-aware MPI implementation.")

  endif()

endif()

#
# Using flags -Wextra, it's to strong for Kokkos, too many warnings But -Wall is
# really a minimum
#
# add_definitions(-Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor
# -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual
# -pedantic ) add_definitions( -Wall -Wextra )
add_definitions(-Wall)

# Generate euler_kokkos_config.h and euler_kokkos_version.h (with git info and
# build date)
#
include(cmake/generate_config_h.cmake)

#
# sources
#
add_subdirectory(test)
add_subdirectory(src)

# #################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("")
message("  CMake version          : ${CMAKE_VERSION}")
if(NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CMake install prefix   : ${CMAKE_INSTALL_PREFIX}")
message("  CMake system processor : ${CMAKE_SYSTEM_PROCESSOR}")
message("  CMake system name (OS) : ${CMAKE_SYSTEM_NAME}")
message("")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION} " "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")

message("  MPI enabled                  = ${EULER_KOKKOS_USE_MPI}")

message("  Kokkos version               = ${Kokkos_VERSION}")
message("  Kokkos_CXX_COMPILER          = ${Kokkos_CXX_COMPILER}")
message("  Kokkos_CXX_COMPILER_ID       = ${Kokkos_CXX_COMPILER_ID}")
message("  Kokkos_CXX_STANDARD          = ${Kokkos_CXX_STANDARD}")
message("  Kokkos_OPTIONS               = ${Kokkos_OPTIONS}")
message("  Kokkos_TPLS                  = ${Kokkos_TPLS}")
message("  Kokkos_DIRS                  = ${Kokkos_DIR}")

if(Kokkos_ENABLE_OPENMP)
  message("  Kokkos_ENABLE_OPENMP         = ${Kokkos_ENABLE_OPENMP}")
endif()

if(Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA           = ${Kokkos_ENABLE_CUDA}")

  if((${Kokkos_CUDA_LAMBDA_ENABLED}) OR (${Kokkos_ENABLE_CUDA_LAMBDA}))
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = OFF")
  endif()

  if((${Kokkos_CUDA_CONSTEXPR_ENABLED}) OR (${Kokkos_ENABLE_CUDA_CONSTEXPR}))
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = OFF")
  endif()

  if((${Kokkos_CUDA_UVM_ENABLED}) OR (${Kokkos_ENABLE_CUDA_UVM}))
    message("  Kokkos_ENABLE_CUDA_UVM       = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_UVM       = OFF")
  endif()

  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
  # message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}") message("  CUDA
  # Compiler exec : ${CUDA_NVCC_EXECUTABLE}") message("  CUDA Compile flags :
  # ${CUDA_NVCC_FLAGS}")
endif(Kokkos_ENABLE_CUDA)

if(Kokkos_ENABLE_HIP)
  message("  Kokkos_ENABLE_HIP            = ${Kokkos_ENABLE_HIP}")
endif(Kokkos_ENABLE_HIP)

if((${Kokkos_TPLS_HWLOC_ENABLED}) OR (${Kokkos_ENABLE_HWLOC}))
  message("  Kokkos_ENABLE_HWLOC          = ON")
else()
  message("  Kokkos_ENABLE_HWLOC          = OFF")
endif()

message("  Kokkos architecture          = ${Kokkos_ARCH}")

if(HDF5_FOUND)
  message("  HDF5_VERSION                 = ${HDF5_VERSION}")
  message("  HDF5_DEFINITIONS             = ${HDF5_DEFINITIONS}")
  message("  HDF5_IS_PARALLEL             = ${HDF5_IS_PARALLEL}")
  message("  HDF5_INCLUDE_DIRS            = ${HDF5_INCLUDE_DIRS}")
  message("  HDF5_LIBRARIES               = ${HDF5_LIBRARIES}")
endif(HDF5_FOUND)

if(PNETCDF_FOUND)
  message("  PNETCDF_VERSION_STRING       = ${PNETCDF_VERSION_STRING}")
  message("  PNETCDF_INCLUDE_DIRS         = ${PNETCDF_INCLUDE_DIRS}")
  message("  PNETCDF_LIBRARIES            = ${PNETCDF_LIBRARIES}")
endif(PNETCDF_FOUND)
message("")
